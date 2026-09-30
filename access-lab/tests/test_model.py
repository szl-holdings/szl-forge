import json
import socket
import threading
import time
from contextlib import contextmanager
from copy import deepcopy
from http.client import IncompleteRead
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

import pytest

from szl_access import engine, model

NAME = "fixture:1"
MANIFEST = "a" * 64
BLOB = "b" * 64


def tags():
    return {"models": [{"name": NAME, "model": NAME, "digest": MANIFEST, "details": {"format": "gguf"}}]}


def details():
    return {"details": {"format": "gguf", "family": "fixture"},
            "model_info": {"general.architecture": "fixture", "general.parameter_count": 1000},
            "modelfile": f'# fixture only\nFROM C:\\PrivateUser\\.ollama\\models\\blobs\\sha256-{BLOB}\nTEMPLATE "{{{{ .Prompt }}}}"'}


def generated(bindings=None, **extra):
    if bindings is None:
        bindings = [{"candidate_id": analysis()["candidates"][0]["id"]}]
    return {"model": NAME, "done": True, "response": json.dumps({"bindings": bindings}), **extra}


def mock_sequence(opener, responses):
    opener.return_value.open.side_effect = [Reply(value) if not isinstance(value, Exception) else value for value in responses]


def called_paths(opener):
    return [call.args[0].full_url for call in opener.return_value.open.call_args_list]


def assert_no_prompt(opener):
    assert all(not path.endswith("/generate") for path in called_paths(opener))
    for call in opener.return_value.open.call_args_list:
        if call.args[0].data:
            assert "prompt" not in json.loads(call.args[0].data)


class Reply:
    def __init__(self, data):
        self.data = data if isinstance(data, bytes) else json.dumps(data).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read1(self, length):
        result, self.data = self.data[:length], self.data[length:]
        return result


def analysis():
    return engine.analyze('<form><label>Email</label><input id="email" type="email"></form>')


@pytest.mark.parametrize("raw", [b'{', b'['*5000+b'0'+b']'*5000, b'x'*65537,
                                 b'{"done":false,"response":"{}"}',
                                 b'{"done":true,"response":"{\\"bindings\\":[],\\"bindings\\":[]}"}',
                                 b'{"done":true,"response":"{\\"bindings\\":[],\\"code\\":\\"evil\\"}"}'],
                         ids=["malformed", "deep", "oversize", "incomplete", "duplicate", "extra"])
def test_bad_model_responses_are_unavailable(raw):
    with patch.object(model, "build_opener") as opener:
        mock_sequence(opener, [tags(), details(), tags(), raw])
        with pytest.raises(model.ModelUnavailable):
            model.propose(analysis(), "fixture:1")


def test_valid_selection_is_bound_and_endpoint_is_local():
    observed = analysis()
    candidate = observed["candidates"][0]["id"]
    raw = json.dumps({"model": NAME, "done": True, "response": json.dumps({"bindings": [{"candidate_id": candidate}]})}).encode()
    with patch.object(model, "build_opener") as opener:
        mock_sequence(opener, [tags(), details(), tags(), raw, tags()])
        proposal, evidence = model.propose(observed, "fixture:1")
        req = opener.return_value.open.call_args_list[3].args[0]
        assert req.full_url == "http://127.0.0.1:11434/api/generate"
        assert json.loads(req.data)["options"]["num_ctx"] == 4096
        assert called_paths(opener) == ["http://127.0.0.1:11434/api/" + path for path in ("tags", "show", "tags", "generate", "tags")]
        handlers = opener.call_args.args
        assert isinstance(handlers[0], model.ProxyHandler) and handlers[0].proxies == {}
        assert isinstance(handlers[1], model.NoRedirect)
    assert proposal["source_sha256"] == observed["source_sha256"]
    assert evidence["trained_for_accessibility"] is False
    assert evidence["local_admission"]["manifest_sha256"] == MANIFEST
    assert evidence["local_admission"]["blob_sha256"] == BLOB
    assert evidence["local_admission"]["server_cloud_disable_state"] == "NOT_VERIFIED"
    assert "PrivateUser" not in json.dumps(evidence)
    assert engine.apply_proposal('<form><label>Email</label><input id="email" type="email"></form>', proposal)["state"] == "VERIFIED_SCOPED_REPAIR"


def test_redirects_are_not_followed():
    assert model.NoRedirect().redirect_request(None, None, 302, "", {}, "https://external.example") is None


@pytest.mark.parametrize("name", ["fixture:cloud", "qwen:80b-cloud", "Cloud/innocent:1", "https://remote.example/model", "a cloud"])
def test_cloud_style_names_rejected_before_network(name):
    with patch.object(model, "build_opener") as opener:
        with pytest.raises(ValueError):
            model.model_name(name)
        with pytest.raises(model.ModelUnavailable):
            model.propose(analysis(), name)
        opener.assert_not_called()


@pytest.mark.parametrize("where", ["tags", "show", "nested_show"])
@pytest.mark.parametrize("key", ["remote_host", "remote_model"])
def test_innocent_alias_remote_metadata_never_receives_prompt(where, key):
    listed, shown = tags(), details()
    if where == "tags":
        listed["models"][0][key] = "remote-value"
    elif where == "show":
        shown[key] = "remote-value"
    else:
        shown["details"][key] = "remote-value"
    with patch.object(model, "build_opener") as opener:
        mock_sequence(opener, [listed, shown])
        with pytest.raises(model.ModelUnavailable):
            model.propose(analysis(), NAME)
        assert_no_prompt(opener)


@pytest.mark.parametrize("change", ["missing_info", "missing_architecture", "missing_modelfile", "not_gguf", "remote_from", "tag_from", "relative_from", "unc_from", "two_from", "traversal_from"])
def test_missing_or_nonlocal_metadata_refused(change):
    shown = details()
    if change == "missing_info":
        shown.pop("model_info")
    elif change == "missing_architecture":
        shown["model_info"] = {}
    elif change == "missing_modelfile":
        shown.pop("modelfile")
    elif change == "not_gguf":
        shown["details"]["format"] = "remote"
    else:
        shown["modelfile"] = {
            "remote_from": f"FROM https://example.invalid/blobs/sha256-{BLOB}",
            "tag_from": "FROM innocent:latest",
            "relative_from": f"FROM models/blobs/sha256-{BLOB}",
            "unc_from": f"FROM \\\\server\\models\\blobs\\sha256-{BLOB}",
            "two_from": f"FROM /models/blobs/sha256-{BLOB}\nFROM innocent:latest",
            "traversal_from": f"FROM /models/../blobs/sha256-{BLOB}",
        }[change]
    with patch.object(model, "build_opener") as opener:
        mock_sequence(opener, [tags(), shown])
        with pytest.raises(model.ModelUnavailable):
            model.propose(analysis(), NAME)
        assert_no_prompt(opener)


@pytest.mark.parametrize("raw", [b'{', b'['*5000+b'0'+b']'*5000, b'x'*262145, b'{"x":NaN}', b'{"models":[],"models":[]}', b'{"x":1e9999}'],
                         ids=["malformed", "deep", "oversize", "nonfinite", "duplicate", "floatoverflow"])
@pytest.mark.parametrize("where", ["tags", "show"])
def test_malformed_metadata_fails_closed_without_prompt(raw, where):
    with patch.object(model, "build_opener") as opener:
        mock_sequence(opener, [raw] if where == "tags" else [tags(), raw])
        with pytest.raises(model.ModelUnavailable):
            model.propose(analysis(), NAME)
        assert_no_prompt(opener)


@pytest.mark.parametrize("change", ["missing", "duplicate", "digest", "alias", "format"])
def test_exact_installed_inventory_required(change):
    listed = tags()
    if change == "missing":
        listed["models"] = []
    elif change == "duplicate":
        listed["models"] *= 2
    elif change == "digest":
        listed["models"][0]["digest"] = "short"
    elif change == "alias":
        listed["models"][0]["model"] = "other:1"
    else:
        listed["models"][0]["details"]["format"] = ""
    with patch.object(model, "build_opener") as opener:
        mock_sequence(opener, [listed])
        with pytest.raises(model.ModelUnavailable):
            model.propose(analysis(), NAME)
        assert_no_prompt(opener)


@pytest.mark.parametrize("when", ["before", "after"])
def test_manifest_drift_refused(when):
    changed = deepcopy(tags())
    changed["models"][0]["digest"] = "c" * 64
    responses = [tags(), details(), changed] if when == "before" else [tags(), details(), tags(), generated(), changed]
    with patch.object(model, "build_opener") as opener:
        mock_sequence(opener, responses)
        with pytest.raises(model.ModelUnavailable):
            model.propose(analysis(), NAME)
        if when == "before":
            assert_no_prompt(opener)


@pytest.mark.parametrize("bindings", [[{"candidate_id": "invented"}], [{"candidate_id": "x", "code": "evil"}], "all", [1], "duplicate"])
def test_invalid_candidate_selections_refused(bindings):
    if bindings == "duplicate":
        candidate = analysis()["candidates"][0]["id"]
        bindings = [{"candidate_id": candidate}] * 2
    with patch.object(model, "build_opener") as opener:
        mock_sequence(opener, [tags(), details(), tags(), generated(bindings)])
        with pytest.raises(model.ModelUnavailable):
            model.propose(analysis(), NAME)


def test_generation_with_remote_metadata_refused():
    with patch.object(model, "build_opener") as opener:
        mock_sequence(opener, [tags(), details(), tags(), generated(remote_host="https://remote.invalid")])
        with pytest.raises(model.ModelUnavailable):
            model.propose(analysis(), NAME)


def test_timeout_fails_closed_before_prompt():
    with patch.object(model, "build_opener") as opener:
        mock_sequence(opener, [TimeoutError("fixture")])
        with pytest.raises(model.ModelUnavailable):
            model.propose(analysis(), NAME, timeout=0.1)
        assert_no_prompt(opener)


def test_incomplete_http_response_is_unavailable():
    with patch.object(model, "build_opener") as opener:
        mock_sequence(opener, [IncompleteRead(b"fixture")])
        with pytest.raises(model.ModelUnavailable):
            model.propose(analysis(), NAME)
        assert_no_prompt(opener)


@contextmanager
def loopback_reply(prefix, tail, *, delay=0.05):
    """A real slow-drip HTTP peer; no model service or external network needed."""
    stop = threading.Event()
    completed = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            try:
                self.connection.sendall(prefix)
                for byte in tail:
                    if stop.wait(delay):
                        break
                    self.connection.sendall(bytes([byte]))
            except OSError:
                pass  # The deadline guard closes the connection mid-response.
            finally:
                completed.set()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    try:
        with patch.object(model, "_BASE", f"http://127.0.0.1:{server.server_port}"):
            yield
    finally:
        stop.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        assert not thread.is_alive()
        assert completed.wait(timeout=2)


@pytest.mark.parametrize("part", ["body", "chunk_body", "chunk_header", "chunk_trailer", "headers"])
def test_absolute_deadline_stops_real_slow_drip(part):
    body = b'{"models":[]}' + b" " * 80
    headers = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n"
    prefix, tail = {
        "body": (f"HTTP/1.1 200 OK\r\nContent-Length: {len(body)}\r\n\r\n".encode(), body),
        "chunk_body": (headers + f"{len(body):x}\r\n".encode(), body + b"\r\n0\r\n\r\n"),
        "chunk_header": (headers, b"2;padding=" + b"x" * 80 + b"\r\n{}\r\n0\r\n\r\n"),
        "chunk_trailer": (headers + b"2\r\n{}\r\n0\r\n", b"X-Padding: " + b"x" * 80 + b"\r\n\r\n"),
        "headers": (b"HTTP/1.1 200 OK\r\nX-Padding: ", b"x" * 80 + b"\r\nContent-Length: 2\r\n\r\n{}"),
    }[part]
    with loopback_reply(prefix, tail):
        started = time.monotonic()
        with pytest.raises(model.ModelUnavailable):
            model.propose(analysis(), NAME, timeout=0.2)
        elapsed = time.monotonic() - started
    # Generous scheduling allowance, but far below the >4s drip duration. The
    # invariant is an absolute deadline, not a reset-on-every-byte inactivity timer.
    assert elapsed < 1.0, f"{part} exceeded the absolute deadline: {elapsed:.3f}s"


@pytest.mark.parametrize("chunked", [False, True])
def test_fragmented_loopback_body_finishes_within_deadline(chunked):
    body = b'{"models": []}'
    if chunked:
        prefix = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n"
        tail = (b"4;fixture=yes\r\n" + body[:4] + f"\r\n{len(body[4:]):x}\r\n".encode()
                + body[4:] + b"\r\n0\r\nX-Fixture: yes\r\n\r\n")
    else:
        prefix = f"HTTP/1.1 200 OK\r\nContent-Length: {len(body)}\r\n\r\n".encode()
        tail = body
    with loopback_reply(prefix, tail, delay=0.001):
        deadline = time.monotonic() + 3
        opener = model.build_opener(model.ProxyHandler({}), model.NoRedirect(), model.DeadlineHTTPHandler(deadline))
        parsed, raw, encoded = model._request(opener, "/fixture", None, deadline, 1024)
    assert parsed == {"models": []}
    assert raw == body
    assert encoded is None
