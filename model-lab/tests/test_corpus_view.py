"""Synthetic report/UI contracts; no source trajectories, models or network jobs."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from szl_model_lab import corpus_view as view
from szl_model_lab.app import Settings, create_app

NOW = datetime(2026, 9, 13, 23, 0, tzinfo=timezone.utc)
TOKEN = "fixture-access-token-" + "x" * 32


def fixture():
    return {
        "schema": "szl.frontier.ultradata-pilot-review/v1",
        "generatedAt": "2026-09-13T22:59:00+00:00",
        "sourceRevisionDeclared": "a" * 40,
        "datasetRepoId": "openbmb/UltraData-SFT-Agent-2609",
        "datasetRevisionDeclared": "b" * 40,
        "inputSha256": "c" * 64, "inputHashMatched": True,
        "upstreamBytesVerified": False, "sourceExecutionVerified": False,
        "rows": [{"line": 1, "rowSha256": "d" * 64, "state": "FORMAT_CHECKED",
                  "reasonCode": None, "toolCalls": 1, "extraFieldsPresent": False}],
        "counts": {"total": 1, "formatChecked": 1, "reviewRequired": 0},
        "state": "FORMAT_CHECKED_NOT_ADMITTED",
        "authorityOrder": ["GitHub", "Hugging Face", "a-11-oy.com", "a11oy.net"],
        "authority": dict.fromkeys(("productionPromotion", "providerWrites", "trainingLaunch",
                                    "billableJobCreation", "weightPublication", "toolExecution"), False),
        "productionDisposition": "HOLD", "signatureState": "UNSIGNED",
        "reviews": dict.fromkeys(("rights", "privacy", "decontamination", "semanticDedup",
                                  "toolArgumentSchema", "outcomeQuality"), "NOT_PERFORMED"),
        "execution": "NONE", "rewardVerified": False, "successLabelsInferred": False,
        "bounds": ["PRIVATE_FIXTURE_EXPLANATION_DO_NOT_ECHO"],
    }


def wire(document):
    body = dict(document)
    body.pop("receiptSha256", None)
    canonical = json.dumps(body, ensure_ascii=False, allow_nan=False,
                           sort_keys=True, separators=(",", ":")).encode()
    body["receiptSha256"] = hashlib.sha256(canonical).hexdigest()
    return json.dumps(body, ensure_ascii=False, allow_nan=False).encode()


def project(document=None):
    raw = wire(fixture() if document is None else document)
    return view.project_report(raw, hashlib.sha256(raw).hexdigest(), now=NOW)


def test_valid_report_is_summary_not_permission():
    result = project()
    assert result["state"] == "REPORT_HASH_CHECKED_NOT_ADMITTED"
    assert result["counts"] == {"total": 1, "format_checked": 1, "review_required": 0}
    assert result["freshness"] == "RECENT_REPORT_NOT_RUNTIME_EVIDENCE"
    assert result["report_generated_at"] == fixture()["generatedAt"]
    for key in ("training_allowed", "execution_authority", "publication_authority",
                "source_verified", "signature_verified", "reward_verified"):
        assert result[key] is False
    encoded = json.dumps(result)
    for private in ("PRIVATE_FIXTURE", "rowSha256", "d" * 64, "bounds", "rows"):
        assert private not in encoded


def test_required_review_rows_count_without_source_text():
    doc = fixture()
    doc["rows"][0].update(state="REVIEW_REQUIRED", reasonCode="unsupported_or_invalid_record",
                           toolCalls=None, extraFieldsPresent=None)
    doc["counts"].update(formatChecked=0, reviewRequired=1)
    doc["state"] = "REVIEW_REQUIRED"
    assert project(doc)["counts"]["review_required"] == 1


@pytest.mark.parametrize("key,value", [
    ("schema", "wrong"), ("datasetRepoId", "unapproved/repo"),
    ("sourceRevisionDeclared", "main"), ("datasetRevisionDeclared", "A" * 40),
    ("inputSha256", "not-a-hash"), ("inputHashMatched", 1),
    ("upstreamBytesVerified", True), ("sourceExecutionVerified", True),
    ("productionDisposition", "PASS"), ("signatureState", "SIGNED"),
    ("execution", "RUNNING"), ("rewardVerified", True), ("successLabelsInferred", True),
    ("generatedAt", "2026-09-13T22:00:00"),
    ("generatedAt", "2026-09-14T22:00:00Z"), ("state", "TRAINED"),
    ("counts", {"total": True, "formatChecked": 1, "reviewRequired": 0}),
    ("counts", {"total": 2, "formatChecked": 1, "reviewRequired": 1}),
    ("rows", []), ("bounds", "PRIVATE_FIXTURE"),
])
def test_rehashed_false_or_malformed_claims_rejected(key, value):
    doc = fixture(); doc[key] = value
    with pytest.raises(ValueError):
        project(doc)


@pytest.mark.parametrize("key,value", [
    ("line", True), ("line", 2), ("rowSha256", "x"), ("state", "VERIFIED"),
    ("reasonCode", "PRIVATE_FIXTURE"), ("toolCalls", True), ("toolCalls", -1),
    ("extraFieldsPresent", 0), ("sourceText", "PRIVATE_FIXTURE"),
])
def test_bad_rows_rejected_even_if_rehashed(key, value):
    doc = fixture(); doc["rows"][0][key] = value
    with pytest.raises(ValueError):
        project(doc)


@pytest.mark.parametrize("section,key,value", [
    ("authority", "trainingLaunch", True), ("authority", "trainingLaunch", 0),
    ("authority", "newPermission", False), ("reviews", "rights", "APPROVED"),
    ("reviews", "unknown", "NOT_PERFORMED"),
])
def test_changed_authority_and_review_contracts_rejected(section, key, value):
    doc = fixture(); doc[section][key] = value
    with pytest.raises(ValueError):
        project(doc)


def test_stale_report_is_not_retimestamped_as_live():
    doc = fixture(); doc["generatedAt"] = "2026-09-10T00:00:00Z"
    result = project(doc)
    assert result["freshness"] == "STALE_REPORT"
    assert result["report_generated_at"] == doc["generatedAt"]


def test_wrong_external_pin_and_wrong_inner_receipt_rejected():
    raw = wire(fixture())
    with pytest.raises(ValueError):
        view.project_report(raw, "0" * 64, now=NOW)
    bad = json.loads(raw); bad["receiptSha256"] = "0" * 64
    raw = json.dumps(bad).encode()
    with pytest.raises(ValueError):
        view.project_report(raw, hashlib.sha256(raw).hexdigest(), now=NOW)


@pytest.mark.parametrize("raw", [b"", b"[]", b'{"x":1,"x":2}', b'{"x":NaN}',
                                 b'{"x":1e999}', b'{"x":"\\ud800"}', b"\xff"])
def test_invalid_json_rejected(raw):
    with pytest.raises(ValueError):
        view.project_report(raw, hashlib.sha256(raw).hexdigest(), now=NOW)


def test_oversize_rejected_before_json():
    with pytest.raises(ValueError):
        view.project_report(b"x" * (view.MAX_REPORT_BYTES + 1), "0" * 64, now=NOW)


def test_extra_top_level_data_is_not_accepted():
    doc = fixture(); doc["sourceText"] = "PRIVATE_FIXTURE"
    with pytest.raises(ValueError):
        project(doc)


def test_unknown_is_not_zero():
    result = view.unavailable_report()
    assert result["counts"] is None
    assert result["training_allowed"] is False


def test_file_loader_and_symlink_refusal(tmp_path):
    path = tmp_path / "report.json"; raw = wire(fixture()); path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    assert view.load_report(path, digest, now=NOW)["counts"]["total"] == 1
    alias = tmp_path / "alias.json"
    try:
        alias.symlink_to(path)
    except OSError:
        pytest.skip("host does not permit symlink creation")
    with pytest.raises(ValueError):
        view.load_report(alias, digest, now=NOW)


def test_settings_require_paired_path_and_digest(tmp_path):
    with pytest.raises(ValueError):
        Settings(TOKEN, {}, {}, corpus_report=tmp_path / "report.json")
    with pytest.raises(ValueError):
        Settings(TOKEN, {}, {}, corpus_report_sha256="a" * 64)
    with pytest.raises(ValueError):
        Settings(TOKEN, {}, {}, corpus_report="str-path", corpus_report_sha256="a" * 64)
    with pytest.raises(ValueError):
        Settings(TOKEN, {}, {}, corpus_report=tmp_path, corpus_report_sha256="main")


def test_environment_binding(tmp_path):
    path = tmp_path / "report.json"
    with patch.dict("os.environ", {"SZL_LAB_ACCESS_TOKEN": TOKEN,
                                   "SZL_LAB_CORPUS_REPORT": str(path),
                                   "SZL_LAB_CORPUS_REPORT_SHA256": "a" * 64}, clear=True):
        settings = Settings.from_env()
        assert isinstance(settings.corpus_report, Path)
        assert settings.corpus_report == path
        assert settings.corpus_report_sha256 == "a" * 64


def test_authenticated_real_asgi_routes_are_readonly_and_redacted(tmp_path):
    path = tmp_path / "PRIVATE_FIXTURE_PATH.json"; raw = wire(fixture()); path.write_bytes(raw)
    settings = Settings(TOKEN, {}, {}, corpus_report=path,
                        corpus_report_sha256=hashlib.sha256(raw).hexdigest())
    with TestClient(create_app(settings), base_url="http://127.0.0.1") as client:
        assert client.get("/api/corpus-review").status_code == 401
        assert client.get("/corpus-review").status_code == 401
        for target in ("/api/corpus-review", "/corpus-review", "/"):
            response = client.get(target, auth=("operator", TOKEN))
            assert response.status_code == 200
            assert response.headers["cache-control"] == "no-store"
            assert "PRIVATE_FIXTURE" not in response.text
            assert str(tmp_path) not in response.text
        assert "Corpus review" in client.get("/", auth=("operator", TOKEN)).text
        for method in ("POST", "PUT", "PATCH", "DELETE"):
            assert client.request(method, "/api/corpus-review", auth=("operator", TOKEN)).status_code == 405
        # Arbitrary query strings are not local file/URL inputs.
        assert client.get("/api/corpus-review?path=/etc/passwd&url=https://invalid.example",
                          auth=("operator", TOKEN)).json()["counts"]["total"] == 1
        path.write_text("PRIVATE_FIXTURE_CORRUPTION")
        bad = client.get("/api/corpus-review", auth=("operator", TOKEN))
        assert bad.status_code == 503 and bad.json()["counts"] is None
        assert "PRIVATE_FIXTURE" not in bad.text


def test_unconfigured_api_is_unavailable_not_empty():
    with TestClient(create_app(Settings(TOKEN, {}, {})), base_url="http://127.0.0.1") as client:
        result = client.get("/api/corpus-review", auth=("operator", TOKEN))
        assert result.status_code == 503
        assert result.json()["counts"] is None


def test_source_export_contains_view_and_docs_only():
    from szl_model_lab.blueprints import SOURCE_FILES
    assert "src/szl_model_lab/corpus_view.py" in SOURCE_FILES
    assert "docs/CORPUS_VIEW.md" in SOURCE_FILES
    assert not any("report.json" in name or "jsonl" in name for name in SOURCE_FILES)
