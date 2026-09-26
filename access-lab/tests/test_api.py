import hashlib
from email.message import Message
import http.client
import io
import json
import threading
import uuid
from unittest.mock import Mock, patch

import pytest

from szl_access import api, engine, model

SOURCE = '<form><label>Email</label><input id="email" name="email" type="email"></form>'
KEY = "synthetic-api-tests-only-not-a-user-secret-20260912"


@pytest.fixture
def running(tmp_path):
    servers = []

    def start(name=None, directory=None):
        server = api.AccessServer(0, directory or tmp_path / str(uuid.uuid4()), KEY, name)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        servers.append((server, thread))
        return server

    yield start
    for server, thread in servers:
        server.shutdown()
        server.server_close()
        thread.join(3)


def request(server, path="/api/runs", body=None, *, headers=None, method=None):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    actual = {"Authorization": "Bearer " + KEY}
    actual.update(headers or {})
    if body is not None:
        actual["Content-Type"] = "application/json"
        encoded = body if isinstance(body, bytes) else json.dumps(body).encode()
    else:
        encoded = None
    connection.request(method or ("POST" if body is not None else "GET"), path, body=encoded, headers=actual)
    response = connection.getresponse()
    raw = response.read()
    connection.close()
    return response.status, raw, dict(response.getheaders())


def run_body(**updates):
    return {"source": SOURCE, "mode": "deterministic", "request_id": str(uuid.uuid4()), **updates}


def test_public_ui_and_health_but_no_private_data(running):
    server = running()
    status, raw, headers = request(server, "/", headers={"Authorization": ""})
    assert status == 200 and b"SZL Access" in raw
    assert "'sha256-" in headers["Content-Security-Policy"]
    assert "'unsafe-inline'" not in headers["Content-Security-Policy"]
    status, raw, _ = request(server, "/api/health", headers={"Authorization": ""})
    assert status == 200 and json.loads(raw)["model"]["enabled"] is False
    assert request(server, "/api/examples", headers={"Authorization": ""})[0] == 401


@pytest.mark.parametrize("headers", [{"Origin": "https://evil.example"}, {"Host": "evil.example"}])
def test_cross_origin_and_rebinding_are_rejected(running, headers):
    assert request(running(), body=run_body(), headers=headers)[0] == 403


def test_non_ascii_auth_is_a_denial_not_disconnect(running):
    status, raw, _ = request(running(), headers={"Authorization": "Bearer caf\xe9"})
    assert status == 401 and json.loads(raw)["error"] == "AUTHENTICATION_REQUIRED"


@pytest.mark.parametrize("headers,status", [({"Host": "evil.example"}, 403),
                                            ({"Origin": "https://evil.example"}, 403),
                                            ({"Authorization": "Bearer incorrect"}, 401)])
def test_early_post_denials_deliver_response_without_claiming(running, headers, status):
    server = running()
    for _ in range(8):
        assert request(server, body=run_body(source="x" * 16000), headers=headers)[0] == status
    assert json.loads(request(server)[1])["runs"] == []


def rejected_body_handler(headers, body=b"abcdmore"):
    handler = object.__new__(api.AccessHandler)
    handler.command = "POST"
    handler.headers = Message()
    for key, value in headers:
        handler.headers[key] = value
    handler.rfile = io.BytesIO(body)
    handler.connection = Mock()
    handler.connection.gettimeout.return_value = 10
    return handler


def test_rejected_body_drain_consumes_only_declared_bytes_once():
    handler = rejected_body_handler([("Content-Length", "4")])
    handler.discard_rejected_body()
    assert handler.rfile.tell() == 4
    handler.discard_rejected_body()
    assert handler.rfile.tell() == 4
    handler.connection.settimeout.assert_called_with(10)


@pytest.mark.parametrize("headers", [[], [("Content-Length", "4"), ("Content-Length", "4")],
                                     [("Content-Length", "4"), ("Transfer-Encoding", "chunked")],
                                     [("Content-Length", "65537")], [("Content-Length", "-1")]])
def test_rejected_body_drain_refuses_unsafe_framing(headers):
    handler = rejected_body_handler(headers)
    handler.discard_rejected_body()
    assert handler.rfile.tell() == 0
    handler.connection.settimeout.assert_not_called()


def test_rejected_body_drain_stops_at_total_deadline():
    handler = rejected_body_handler([("Content-Length", "4")])
    with patch.object(api.time, "monotonic", side_effect=[0, 1.1]):
        handler.discard_rejected_body()
    assert handler.rfile.tell() == 0
    handler.connection.settimeout.assert_called_once_with(10)


def test_repair_receipt_replay_history_and_no_overwrite(running):
    server = running()
    body = run_body()
    status, raw, _ = request(server, body=body)
    result = json.loads(raw)
    assert status == 200 and result["state"] == "VERIFIED_SCOPED_REPAIR"
    assert '<label for="email">' in result["output_html"]
    receipt = dict(result)
    saved_hash = receipt.pop("receipt_sha256")
    assert saved_hash == api.digest(receipt)
    assert result["model"]["trained_for_accessibility"] is False
    assert json.loads(request(server, body=body)[1]) == result
    assert json.loads(request(server, "/api/runs/" + result["run_id"])[1]) == result
    assert len(json.loads(request(server)[1])["runs"]) == 1
    assert request(server, body={**body, "source": SOURCE + " "})[0] == 409


@pytest.mark.parametrize("body", [b'{"source":"a","source":"b"}', b'{', b'['*5000+b'0'+b']'*5000,
                                  run_body(source="x"*32769), run_body(mode=[]), run_body(request_id="not-a-uuid")],
                         ids=["duplicate", "malformed", "deep", "oversize", "mode_type", "bad_uuid"])
def test_bad_inputs_never_claim_a_run(running, body):
    server = running()
    assert request(server, body=body)[0] == 400
    assert json.loads(request(server)[1])["runs"] == []


def test_unsupported_source_is_not_executed_or_repaired(running):
    source = '<script>alert("untrusted")</script>'
    status, raw, _ = request(running(), body=run_body(source=source))
    result = json.loads(raw)
    assert status == 200 and result["state"] == "REVIEW_REQUIRED"
    assert result["output_html"] == source


def test_disabled_model_and_busy_worker(running):
    server = running()
    assert request(server, body=run_body(mode="ollama"))[0] == 409
    server.work_lock.acquire()
    try:
        assert request(server, body=run_body())[0] == 429
    finally:
        server.work_lock.release()


def test_model_failure_persists_truth_without_fallback(running):
    server = running("fixture-model:1")
    body = run_body(mode="ollama")
    with patch.object(model, "propose", side_effect=model.ModelUnavailable("fixture offline")) as mock:
        status, raw, _ = request(server, body=body)
        result = json.loads(raw)
        assert status == 200 and result["state"] == "MODEL_UNAVAILABLE"
        assert result["output_html"] == SOURCE and result["applied_bindings"] == []
        assert json.loads(request(server, body=body)[1]) == result
        assert mock.call_count == 1


def test_model_unknown_candidate_is_held(running):
    server = running("fixture-model:1")
    proposal = {"source_sha256": hashlib.sha256(SOURCE.encode()).hexdigest(), "methodology": "model",
                "bindings": [{"candidate_id": "invented"}]}
    with patch.object(model, "propose", return_value=(proposal, {"name": "fixture", "trained_for_accessibility": False})):
        result = json.loads(request(server, body=run_body(mode="ollama"))[1])
    assert result["state"] == "REVIEW_REQUIRED" and result["output_html"] == SOURCE


@pytest.mark.parametrize("change", ["delete_digest", "replace_output"])
def test_corruption_is_denied_on_read_and_idempotent_replay(running, change):
    server = running()
    body = run_body()
    result = json.loads(request(server, body=body)[1])
    if change == "delete_digest":
        result.pop("receipt_sha256")
    else:
        result["output_html"] = "forged"
    with server.store.connect() as connection:
        connection.execute("UPDATE runs SET payload=?", (json.dumps(result),))
    assert request(server, "/api/runs/" + result["run_id"])[0] == 409
    assert request(server, body=body)[0] == 409


def test_restart_history_and_incomplete_operation(tmp_path):
    directory = tmp_path / "state"
    first = api.AccessServer(0, directory, KEY)
    result, _ = first.store.claim(str(uuid.uuid4()), SOURCE, "deterministic")
    with pytest.raises(ValueError, match="active writer"):
        api.AccessServer(0, directory, KEY)
    first.server_close()
    with api.AccessServer(0, directory, KEY) as second:
        interrupted = second.store.get(result["run_id"])
        assert interrupted["state"] == "INTERRUPTED"
        assert interrupted["receipt_sha256"]


def test_terminal_receipt_survives_restart(tmp_path):
    directory = tmp_path / "state"
    with api.AccessServer(0, directory, KEY) as first:
        result, _ = first.store.claim(str(uuid.uuid4()), SOURCE, "deterministic")
        result.update(engine.apply_proposal(SOURCE, engine.deterministic_proposal(SOURCE)))
        complete = first.store.finish(result)
    with api.AccessServer(0, directory, KEY) as second:
        assert second.store.get(complete["run_id"]) == complete


@pytest.mark.parametrize("corruption", ["array", "duplicate", "deep", "oversize", "identity", "unknown_state", "output_hash", "running_without_hash", "nonfinite"])
def test_invalid_stored_records_fail_closed_across_read_paths(running, corruption):
    server = running()
    body = run_body()
    result = json.loads(request(server, body=body)[1])
    if corruption == "array":
        raw = "[]"
    elif corruption == "duplicate":
        raw = '{"state":"RUNNING",' + json.dumps(result)[1:]
    elif corruption == "deep":
        raw = '[' * 5000 + '0' + ']' * 5000
    elif corruption == "oversize":
        raw = ' ' * 262145 + json.dumps(result)
    elif corruption == "nonfinite":
        raw = '{"unexpected":NaN,' + json.dumps(result)[1:]
    else:
        if corruption == "identity":
            result["run_id"] = str(uuid.uuid4())
        elif corruption == "unknown_state":
            result["state"] = "CERTIFIED_ACCESSIBLE"
        elif corruption == "output_hash":
            result["output_sha256"] = "0" * 64
        elif corruption == "running_without_hash":
            result["state"] = "RUNNING"
        result.pop("receipt_sha256")
        if corruption != "running_without_hash":
            result["receipt_sha256"] = api.digest(result)
        raw = json.dumps(result)
    with server.store.connect() as connection:
        original_id = connection.execute("SELECT run_id FROM runs").fetchone()[0]
        connection.execute("UPDATE runs SET payload=?", (raw,))
    for route in ("/api/runs", "/api/runs/" + original_id, "/api/runs/" + original_id + "/summary"):
        status, response, _ = request(server, route)
        assert status == 409
        assert json.loads(response)["error"] == "LOCAL_RECEIPT_INTEGRITY_FAILED"
    status, response, _ = request(server, body=body)
    assert status == 409
    assert json.loads(response)["error"] == "LOCAL_RECEIPT_INTEGRITY_FAILED"


def test_restart_refuses_corrupt_record_without_rewriting_or_leaking_lease(tmp_path):
    directory = tmp_path / "state"
    with api.AccessServer(0, directory, KEY) as first:
        original, _ = first.store.claim(str(uuid.uuid4()), SOURCE, "deterministic")
        with first.store.connect() as connection:
            connection.execute("UPDATE runs SET payload='[]'")
    for _ in range(2):
        with pytest.raises(api.RequestError, match="LOCAL_RECEIPT_INTEGRITY_FAILED"):
            api.AccessServer(0, directory, KEY)
    with first.store.connect() as connection:
        assert connection.execute("SELECT payload FROM runs").fetchone()[0] == "[]"
        connection.execute("UPDATE runs SET payload=?", (api.canonical(original),))
    with api.AccessServer(0, directory, KEY) as recovered:
        assert recovered.store.get(original["run_id"])["state"] == "INTERRUPTED"


def test_source_free_summary_export_has_its_own_hash_and_authentication(running):
    server = running()
    source = '<label>PrivateSyntheticName</label><input id="private-test-field">'
    result = json.loads(request(server, body=run_body(source=source))[1])
    path = "/api/runs/" + result["run_id"] + "/summary"
    assert request(server, path, headers={"Authorization": ""})[0] == 401
    status, raw, headers = request(server, path)
    assert status == 200
    summary = json.loads(raw)
    assert summary["schema"] == "szl-access-receipt-summary-v1"
    assert summary["receipt_sha256"] == result["receipt_sha256"]
    assert summary["source_sha256"] == result["source_sha256"]
    assert summary["output_sha256"] == result["output_sha256"]
    assert summary["contains_source"] is False and summary["independent_attestation"] is False
    assert summary["export_sha256"] == api.digest({key: value for key, value in summary.items() if key != "export_sha256"})
    assert b"PrivateSyntheticName" not in raw and b"private-test-field" not in raw
    assert b"output_html" not in raw and b"applied_bindings" not in raw
    assert headers["Content-Disposition"] == f'attachment; filename="szl-access-{result["run_id"]}-summary.json"'
    assert headers["Cache-Control"] == "no-store"
    assert json.loads(request(server, path)[1]) == summary


def test_weak_key_is_rejected_before_state_creation(tmp_path):
    directory = tmp_path / "state"
    with pytest.raises(ValueError):
        api.AccessServer(0, directory, "short")
    assert not directory.exists()
