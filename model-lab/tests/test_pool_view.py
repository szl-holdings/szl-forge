"""Synthetic schema fixtures, never observations of owner hardware."""
import copy
import hashlib
import json
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from szl_model_lab.app import Settings, create_app
from szl_model_lab.pool_view import load_pool_snapshot, project_pool

NOW = datetime(2026, 9, 13, 12, 0, tzinfo=timezone.utc)
TOKEN = "synthetic-unit-test-only-" + "x" * 32


def fixture_document():
    evidence = {"value": True, "evidence_class": "MEASURED", "observed_at": NOW.isoformat()}
    return {"schema_version": "szl.compute-pool/v1", "generated_at": NOW.isoformat(),
            "source_surface": "szl_backend_hardening.probe_fabric_pool",
            "control_plane_relation": "PYTHON_HF_PROJECTION_NOT_REPLIT_CONTROL_PLANE",
            "receipt_freshness_seconds": 300,
            "counts": dict(declared=1, configured=1, reachable=1, discovered=1, qualified=1, serving=1, ready=1),
            "nodes": [{"node_id": "synthetic-worker", "kind": "fixture", "sovereign": True,
                       "endpoint_label": "http://private-fixture.invalid", "state": "SERVING", "ready": True,
                       "evidence_class": "MEASURED", "configuration": copy.deepcopy(evidence),
                       "reachability": copy.deepcopy(evidence), "inference_receipt": {
                           "evidence_class": "MEASURED", "verified": True, "fresh": True, "bounded": True,
                           "observed_at": NOW.isoformat(), "model_id": "synthetic-fixture",
                           "model_digest_sha256": "a" * 64, "receipt_sha256": "b" * 64, "reason": None}}]}


def encode(doc):
    raw = json.dumps(doc).encode()
    return raw, hashlib.sha256(raw).hexdigest()


def project(doc, **kw):
    raw, digest = encode(doc)
    return project_pool(raw, expected_sha256=digest, now=kw.pop("now", NOW), **kw)


def test_reported_ready_never_authorizes():
    data = project(fixture_document())
    assert data["nodes"][0]["upstream_ready_reported"] is True
    assert data["nodes"][0]["ready"] is False
    assert data["pool_qualification_verified"] is False and data["signature_verified"] is False
    assert data["ready"] is False and data["schema_checked"] is True
    assert "private-fixture" not in json.dumps(data)


@pytest.mark.parametrize("seconds", [301, 3601, -31, -3600])
def test_stale_or_future_report_cannot_be_current(seconds):
    assert project(fixture_document(), now=NOW + timedelta(seconds=seconds))["state"] == "STALE_REPORTED_SNAPSHOT"


@pytest.mark.parametrize("seconds", [0, 299, 300, -30])
def test_snapshot_age_only_is_not_qualification(seconds):
    data = project(fixture_document(), now=NOW + timedelta(seconds=seconds))
    assert data["snapshot_fresh"] is True and data["ready"] is False


@pytest.mark.parametrize("key,value", [("schema_version", "wrong"), ("source_surface", "arbitrary"),
                                      ("control_plane_relation", "other"), ("receipt_freshness_seconds", True),
                                      ("receipt_freshness_seconds", 0), ("receipt_freshness_seconds", 3601),
                                      ("nodes", None), ("generated_at", "2026-09-13T12:00:00")])
def test_bad_top_level_rejected(key, value):
    doc = fixture_document(); doc[key] = value
    with pytest.raises(ValueError):
        project(doc)


@pytest.mark.parametrize("mutation", ["duplicate_node", "extra_key", "bad_count", "bool_count", "false_ready",
                                     "false_boolean", "bad_digest", "bad_evidence", "bad_receipt_boolean"])
def test_item_contract_and_counts(mutation):
    doc = fixture_document()
    node = doc["nodes"][0]
    if mutation == "duplicate_node": doc["nodes"].append(copy.deepcopy(node))
    elif mutation == "extra_key": node["authorization"] = True
    elif mutation == "bad_count": doc["counts"]["ready"] = 0
    elif mutation == "bool_count": doc["counts"]["ready"] = True
    elif mutation == "false_ready": node["ready"] = False
    elif mutation == "false_boolean": node["ready"] = "true"
    elif mutation == "bad_digest": node["inference_receipt"]["model_digest_sha256"] = "main"
    elif mutation == "bad_evidence": node["reachability"]["value"] = 1
    elif mutation == "bad_receipt_boolean": node["inference_receipt"]["verified"] = 1
    with pytest.raises(ValueError): project(doc)


def test_byte_pin_required_and_duplicate_json_rejected():
    raw, digest = encode(fixture_document())
    with pytest.raises(ValueError): project_pool(raw, expected_sha256="c"*64, now=NOW)
    with pytest.raises(ValueError): project_pool(raw, expected_sha256="main", now=NOW)
    raw = raw.replace(b'"schema_version":', b'"schema_version":"first","schema_version":', 1)
    with pytest.raises(ValueError, match="duplicate"): project_pool(raw, expected_sha256=hashlib.sha256(raw).hexdigest(), now=NOW)


def test_file_read_and_symlink(tmp_path):
    path = tmp_path / "snapshot.json"; raw, digest = encode(fixture_document()); path.write_bytes(raw)
    assert load_pool_snapshot(path, digest, now=NOW)["schema_checked"] is True
    link = tmp_path / "link.json"
    try: link.symlink_to(path)
    except OSError: pytest.skip("symlink creation unavailable")
    with pytest.raises(ValueError): load_pool_snapshot(link, digest, now=NOW)


def test_unconfigured_api_and_page_do_not_invent_zero():
    client = TestClient(create_app(Settings(TOKEN, {}, {})), base_url="http://localhost")
    for path in ("/api/pool", "/pool"):
        assert client.get(path).status_code == 401
        assert client.get(path, auth=("operator", TOKEN)).status_code == 200
    payload = client.get("/api/pool", auth=("operator", TOKEN)).json()
    assert payload["nodes"] is None and payload["counts_reported"] is None
    assert "Unknown inventory is not counted as zero" in client.get("/pool", auth=("operator", TOKEN)).text


def test_config_requires_external_digest_pair(tmp_path):
    with pytest.raises(ValueError): Settings(TOKEN, {}, {}, tmp_path / "x")
    with pytest.raises(ValueError): Settings(TOKEN, {}, {}, None, "a"*64)
    with pytest.raises(ValueError): Settings(TOKEN, {}, {}, tmp_path / "x", "bad")


def test_bad_snapshot_returns_explicit_empty_state_without_path_leak(tmp_path):
    client = TestClient(create_app(Settings(TOKEN, {}, {}, tmp_path / "private-nonexistent", "a"*64)), base_url="http://localhost")
    response = client.get("/api/pool", auth=("operator", TOKEN))
    assert response.json()["state"] == "INVALID_OR_UNAVAILABLE_SNAPSHOT"
    assert response.json()["nodes"] is None and "private-nonexistent" not in response.text


def test_pool_model_name_is_html_escaped(tmp_path):
    doc = fixture_document(); doc["nodes"][0]["inference_receipt"]["model_id"] = "<script>alert(1)</script>"
    raw, digest = encode(doc); path = tmp_path / "snapshot.json"; path.write_bytes(raw)
    client = TestClient(create_app(Settings(TOKEN, {}, {}, path, digest)), base_url="http://localhost")
    response = client.get("/pool", auth=("operator", TOKEN))
    assert response.status_code == 200 and "&lt;script&gt;" in response.text and "<script>" not in response.text


@pytest.mark.parametrize("value", [[], {}, True, 1, None])
def test_container_valued_enums_are_contract_errors(value):
    doc = fixture_document()
    doc["nodes"][0]["evidence_class"] = value
    with pytest.raises(ValueError):
        project(doc)


def test_unknown_and_explicit_empty_inventory_remain_distinct():
    doc = fixture_document()
    doc["nodes"] = []
    doc["counts"] = {key: 0 for key in doc["counts"]}
    output = project(doc)
    assert output["nodes"] == [] and output["counts_reported"]["declared"] == 0
    assert output["ready"] is False


def test_generated_timestamp_does_not_restamp_receipt_observation():
    doc = fixture_document()
    earlier = (NOW - timedelta(days=1)).isoformat()
    doc["nodes"][0]["inference_receipt"]["observed_at"] = earlier
    doc["nodes"][0]["inference_receipt"]["fresh"] = False
    output = project(doc)
    assert output["snapshot_fresh"] is True
    assert output["nodes"][0]["receipt_observed_at_reported"] == earlier
    assert output["nodes"][0]["receipt_fresh_reported"] is False
    assert output["nodes"][0]["ready"] is False


def test_return_value_cannot_mutate_schema_reference():
    result = project(fixture_document())
    result["schema_source_inspected"]["revision"] = "changed"
    assert project(fixture_document())["schema_source_inspected"]["revision"] != "changed"


def test_authentication_happens_before_snapshot_read(monkeypatch, tmp_path):
    from szl_model_lab import app
    def forbidden(*args, **kwargs):
        pytest.fail("unauthenticated request read the configured snapshot")
    monkeypatch.setattr(app, "load_pool_snapshot", forbidden)
    client = TestClient(create_app(Settings(TOKEN, {}, {}, tmp_path / "x", "a" * 64)), base_url="http://localhost")
    assert client.get("/api/pool").status_code == 401
    assert client.get("/pool").status_code == 401
    assert client.get("/api/pool", auth=("operator", "wrong")).status_code == 401
    assert client.get("/healthz").status_code == 200


def test_snapshot_pair_is_loaded_from_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("SZL_LAB_ACCESS_TOKEN", TOKEN)
    monkeypatch.setenv("SZL_LAB_OLLAMA_ENDPOINTS_JSON", "{}")
    monkeypatch.setenv("SZL_LAB_POOL_SNAPSHOT", str(tmp_path / "snapshot.json"))
    monkeypatch.setenv("SZL_LAB_POOL_SNAPSHOT_SHA256", "a" * 64)
    config = Settings.from_env()
    assert config.pool_snapshot == tmp_path / "snapshot.json"
    assert config.pool_snapshot_sha256 == "a" * 64


def test_out_of_range_utc_conversion_is_contract_error():
    doc = fixture_document()
    doc["generated_at"] = "9999-12-31T23:59:59-23:00"
    with pytest.raises(ValueError):
        project(doc)


def test_existing_duplicate_json_request_rejection_is_preserved():
    client = TestClient(create_app(Settings(TOKEN, {}, {})), base_url="http://localhost")
    result = client.post("/api/score/router", auth=("operator", TOKEN),
                         content='{"features":{},"features":{}}',
                         headers={"Content-Type": "application/json"})
    assert result.status_code == 400


def test_blueprint_allowlist_contains_pool_dependency_and_document():
    from szl_model_lab.blueprints import SOURCE_FILES
    assert "src/szl_model_lab/pool_view.py" in SOURCE_FILES
    assert "docs/POOL_VIEW.md" in SOURCE_FILES
    assert not any("snapshot.json" in name or name.endswith(".env") for name in SOURCE_FILES)


def test_pool_export_contains_exact_committed_module(tmp_path):
    import subprocess
    from szl_model_lab.blueprints import SOURCE_FILES, source_payload
    root = tmp_path / "source-contract-fixture"
    root.mkdir()
    for name in SOURCE_FILES:
        path = root / "model-lab" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"# exact synthetic source fixture\n")
    def git(*args):
        return subprocess.run(["git", "-C", str(root), *args], check=True,
                              capture_output=True, text=True).stdout.strip()
    git("init")
    git("config", "user.name", "Source Contract Fixture")
    git("config", "user.email", "fixture@example.invalid")
    git("add", ".")
    git("commit", "-m", "synthetic source fixture")
    payload = source_payload(root, git("rev-parse", "HEAD"), "router")
    module = "source/src/szl_model_lab/pool_view.py"
    assert payload[module] == b"# exact synthetic source fixture\n"
    binding = json.loads(payload["source-binding.json"])
    assert binding["files_sha256"][module] == hashlib.sha256(payload[module]).hexdigest()
    assert binding["weights_present"] is False
