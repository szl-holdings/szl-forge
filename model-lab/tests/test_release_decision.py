"""Actual FastAPI/Jinja and read-only release rails; synthetic fixtures only."""

from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from szl_model_lab.app import Settings, create_app
from szl_model_lab.release_decision import (
    project_decision,
    MAX_BYTES,
    MAX_SNAPSHOT_AGE_SECONDS,
)

TOKEN = "synthetic-release-token-" + "x" * 32
AUTH = ("operator", TOKEN)


def sample():
    when = datetime.now(timezone.utc) - timedelta(minutes=2)
    return {
        "schema": "szl.release-decision-input/v1",
        "scope": "owner_local_unattested",
        "release_id": "synthetic-release-001",
        "observed_at": when.isoformat(timespec="seconds"),
        "rails": {
            "source": {
                "expected": {"repository": "szl-holdings/a11oy", "revision": "a" * 40},
                "observed": {"repository": "szl-holdings/a11oy", "revision": "a" * 40},
            },
            "space": {
                "expected": {
                    "space_id": "SZLHOLDINGS/a11oy",
                    "revision": "b" * 40,
                    "boot_id": "synthetic-boot-1",
                    "run_nonce": "synthetic-run-1",
                },
                "observed": None,
            },
            "gateway": {
                "expected": {
                    "repository": "szl-holdings/szl-router",
                    "revision": "c" * 40,
                    "route": "a11oy",
                },
                "observed": None,
            },
            "receipt": {
                "expected": {
                    "repository": "SZLHOLDINGS/frontier-54-runs",
                    "repo_type": "dataset",
                    "revision": "d" * 40,
                    "path": "receipts/synthetic/result.json",
                    "sha256": "e" * 64,
                },
                "observed": None,
            },
        },
    }


def wire(data=None):
    value = sample() if data is None else data
    raw = json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
    return raw, hashlib.sha256(raw).hexdigest()


def client(tmp_path, data=None):
    raw, digest = wire(data)
    path = tmp_path / "release.json"
    path.write_bytes(raw)
    app = create_app(
        Settings(TOKEN, {}, {}, release_snapshot=path, release_snapshot_sha256=digest)
    )
    return TestClient(app, base_url="http://localhost"), path


def project(data=None, **kwargs):
    raw, digest = wire(data)
    return project_decision(raw, digest, **kwargs)


@pytest.mark.parametrize("route", ["/release-decision", "/api/release-decision"])
def test_auth_required_even_if_unconfigured(route):
    with TestClient(
        create_app(Settings(TOKEN, {}, {})), base_url="http://localhost"
    ) as cli:
        assert cli.get(route).status_code == 401
        assert cli.get(route, auth=("operator", "wrong")).status_code == 401


@pytest.mark.parametrize("route", ["/release-decision", "/api/release-decision"])
def test_unconfigured_503_null_not_zero(route):
    with TestClient(
        create_app(Settings(TOKEN, {}, {})), base_url="http://localhost"
    ) as cli:
        res = cli.get(route, auth=AUTH)
        assert res.status_code == 503
        if route.startswith("/api"):
            assert res.json()["counts"] is None
        else:
            assert "UNCONFIGURED" in res.text


def test_backend_and_frontend_share_one_projection(tmp_path):
    cli, path = client(tmp_path)
    with cli:
        api = cli.get("/api/release-decision", auth=AUTH)
        html = cli.get("/release-decision", auth=AUTH)
        assert api.status_code == html.status_code == 200
        state = api.json()
        assert state["counts"] == {
            "rails_total": 4,
            "reported_matching": 1,
            "reported_conflicting": 0,
            "missing": 3,
        }
        assert state["snapshot_hash_matched"] is True
        assert "HOLD" in state["state"]
        assert state["snapshot_sha256"] in html.text
        for rail in state["rails"]:
            assert rail["identity_subject_sha256"] in html.text
            assert rail["name"] in html.text
        assert "ADMISSION: HOLD" in html.text
        assert "/release-decision" in cli.get("/", auth=AUTH).text
        assert "/release-decision" in cli.get("/evidence", auth=AUTH).text
        assert api.headers["cache-control"] == "no-store"
        assert "frame-ancestors 'none'" in html.headers["content-security-policy"]
        assert TOKEN not in html.text
    assert path.exists()


def test_all_four_matching_still_hold_not_authorized():
    data = sample()
    for item in data["rails"].values():
        item["observed"] = copy.deepcopy(item["expected"])
    report = project(data)
    assert report["counts"]["reported_matching"] == 4
    assert report["state"] == "HOLD_UNVERIFIED"
    for name in (
        "execution_authority",
        "publication_authority",
        "signature_verified",
        "independent_witness_verified",
        "source_byte_parity_verified",
        "live_readback_performed",
        "production_ready",
        "action_performed",
    ):
        assert report[name] is False


def test_conflict_reported_without_erasing_baseline():
    data = sample()
    data["rails"]["space"]["observed"] = copy.deepcopy(
        data["rails"]["space"]["expected"]
    )
    data["rails"]["space"]["observed"]["boot_id"] = "different-boot"
    report = project(data)
    assert report["counts"]["reported_conflicting"] == 1
    assert "SPACE_IDENTITY_CONFLICT" in report["blocking_reasons"]
    assert report["state"] == "HOLD_CONFLICT_REPORTED"


def test_what_if_is_reversible_and_does_not_modify_snapshot(tmp_path):
    cli, path = client(tmp_path)
    raw = path.read_bytes()
    with cli:
        baseline = cli.get("/api/release-decision", auth=AUTH).json()
        scenario = cli.get("/api/release-decision?invalidate=source", auth=AUTH)
        assert scenario.status_code == 200
        sim = scenario.json()
        assert sim["hypothetical"] is True
        assert sim["invalidated_rail"] == "source"
        assert sim["rails"][0]["state"] == "RECHECK_REQUIRED_SIMULATION"
        assert sim["rails"][0]["baseline_state"] == "MATCH_IN_SUPPLIED_SNAPSHOT"
        assert (
            "WHAT-IF ONLY"
            in cli.get("/release-decision?invalidate=source", auth=AUTH).text
        )
        assert (
            cli.get("/api/release-decision", auth=AUTH).json()["rails"]
            == baseline["rails"]
        )
    assert path.read_bytes() == raw


@pytest.mark.parametrize(
    "mutation",
    [
        lambda x: x.update(scope="authenticated_private"),
        lambda x: x.update(release_authorized=True),
        lambda x: x["rails"]["source"]["expected"].update(revision="main"),
        lambda x: x["rails"]["source"]["expected"].update(repository="evil/a11oy"),
        lambda x: x["rails"]["space"]["expected"].update(run_nonce="../out"),
        lambda x: x["rails"]["space"]["expected"].update(boot_id=True),
        lambda x: x["rails"]["receipt"]["expected"].update(repo_type="model"),
        lambda x: x["rails"]["receipt"]["expected"].update(path="../secret"),
        lambda x: x["rails"]["receipt"]["expected"].update(sha256="bad"),
        lambda x: x["rails"]["gateway"]["expected"].update(route="other"),
        lambda x: x["rails"]["gateway"]["expected"].update(repository="evil/router"),
        lambda x: x["rails"]["source"].update(observed={"revision": "a" * 40}),
        lambda x: x["rails"].pop("receipt"),
        lambda x: x.update(observed_at="9999-01-01T00:00:00Z"),
        lambda x: x.update(observed_at="2026-11-16T00:00:00"),
    ],
)
def test_invalid_or_privileged_input_fails_closed(mutation):
    data = sample()
    mutation(data)
    with pytest.raises(ValueError):
        project(data)


@pytest.mark.parametrize(
    "raw",
    [
        b'{"schema":"x","schema":"x"}',
        b'{"schema":NaN}',
        b'{"schema":1e9999}',
        b"\xff",
        b" " * (MAX_BYTES + 1),
    ],
)
def test_bad_wire_never_qualifies(raw):
    with pytest.raises(ValueError):
        project_decision(raw, hashlib.sha256(raw).hexdigest())


def test_digest_substitution_503_without_leaking_local_path(tmp_path):
    cli, path = client(tmp_path)
    path.write_bytes(path.read_bytes() + b" ")
    with cli:
        for route in ("/release-decision", "/api/release-decision"):
            res = cli.get(route, auth=AUTH)
            assert res.status_code == 503
            assert TOKEN not in res.text and str(path) not in res.text
            assert "INVALID_OR_UNAVAILABLE_SNAPSHOT" in res.text


def test_unknown_rail_is_404_and_no_write_routes(tmp_path):
    cli, path = client(tmp_path)
    before = path.read_bytes()
    with cli:
        assert (
            cli.get("/api/release-decision?invalidate=secrets", auth=AUTH).status_code
            == 404
        )
        assert (
            cli.get(
                "/api/release-decision?invalidate=" + "a" * 100, auth=AUTH
            ).status_code
            == 422
        )
        assert cli.post("/api/release-decision", auth=AUTH, json={}).status_code == 405
        assert (
            cli.get(
                "/release-decision", auth=AUTH, headers={"Host": "evil.test"}
            ).status_code
            == 400
        )
    assert path.read_bytes() == before


def test_age_is_clock_based_not_provider_freshness():
    now = datetime(2026, 11, 17, 4, 0, 0, tzinfo=timezone.utc)
    data = sample()
    data["observed_at"] = "2026-11-17T03:30:00Z"
    report = project(data, now=now)
    assert report["snapshot_age_seconds"] == 1800
    assert MAX_SNAPSHOT_AGE_SECONDS == 900
    assert "SUPPLIED_SNAPSHOT_OLDER_THAN_15_MINUTES" in report["blocking_reasons"]
    assert report["live_readback_performed"] is False


@pytest.mark.parametrize(
    "config",
    [
        {"release_snapshot": Path("unused")},
        {"release_snapshot_sha256": "f" * 64},
        {"release_snapshot": "string", "release_snapshot_sha256": "f" * 64},
        {"release_snapshot": Path("unused"), "release_snapshot_sha256": "main"},
    ],
)
def test_partial_settings_rejected(config):
    with pytest.raises(ValueError):
        Settings(TOKEN, {}, {}, **config)


def test_env_configuration_is_explicit(monkeypatch, tmp_path):
    monkeypatch.setenv("SZL_LAB_ACCESS_TOKEN", TOKEN)
    monkeypatch.setenv("SZL_LAB_RELEASE_SNAPSHOT", str(tmp_path / "snapshot.json"))
    monkeypatch.setenv("SZL_LAB_RELEASE_SNAPSHOT_SHA256", "f" * 64)
    settings = Settings.from_env()
    assert settings.release_snapshot == tmp_path / "snapshot.json"
    assert settings.release_snapshot_sha256 == "f" * 64


def test_unauthenticated_does_not_read_file(tmp_path):
    cli, _ = client(tmp_path)
    with (
        cli,
        patch(
            "szl_model_lab.release_web.load_decision",
            side_effect=AssertionError("must not read"),
        ),
    ):
        for path in ("/release-decision", "/api/release-decision"):
            assert cli.get(path).status_code == 401


def test_projection_does_not_perform_provider_calls():
    with patch(
        "socket.socket.connect", side_effect=AssertionError("network prohibited")
    ):
        report = project()
    assert report["live_readback_performed"] is False
    assert report["action_performed"] is False


def test_blueprint_includes_release_source_files():
    from szl_model_lab.blueprints import SOURCE_FILES

    for name in (
        "src/szl_model_lab/release_decision.py",
        "src/szl_model_lab/release_web.py",
        "src/szl_model_lab/templates/release-decision.html",
        "docs/RELEASE_DECISION.md",
    ):
        assert name in SOURCE_FILES
