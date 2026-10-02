from __future__ import annotations

import importlib.util
import io
import json
import sys
import urllib.error
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "hf_inference_selector",
    ROOT / "tools" / "acquire_hf_inference_token.py",
)
assert SPEC and SPEC.loader
selector = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = selector
SPEC.loader.exec_module(selector)


def test_select_tries_each_secret_independently(monkeypatch) -> None:
    calls = []

    def fake_validate(token: str, target_model: str, timeout: float):
        calls.append(token)
        return selector.Attempt(
            source="",
            present=True,
            valid=token == "hf_good",
            status_code=200 if token == "hf_good" else 403,
            target_model_listed=token == "hf_good",
        )

    monkeypatch.setattr(selector, "validate_token", fake_validate)
    token, source, attempts = selector.select(
        {
            "HF_INFERENCE_TOKEN_CANDIDATE": "hf_bad",
            "HF_ORG_TOKEN_CANDIDATE": "hf_good",
        },
        target_model="zai-org/GLM-5.3-Flash",
        timeout=1,
    )
    assert token == "hf_good"
    assert source == "HF_ORG_TOKEN"
    assert calls == ["hf_bad", "hf_good"]
    assert [attempt.source for attempt in attempts[:2]] == [
        "HF_INFERENCE_TOKEN",
        "HF_ORG_TOKEN",
    ]


def test_report_never_contains_token_bytes(tmp_path) -> None:
    path = tmp_path / "report.json"
    selector.write_report(
        path,
        target_model="zai-org/GLM-5.3-Flash",
        selected_source="HF_TOKEN",
        attempts=[
            selector.Attempt(
                source="HF_TOKEN",
                present=True,
                valid=True,
                status_code=200,
                target_model_listed=True,
                response_sha256="a" * 64,
            )
        ],
    )
    text = path.read_text()
    assert "hf_secret_value" not in text
    assert "selected_source" in text


def test_invalid_token_shape_is_rejected() -> None:
    for value in ("", "secret", "hf_has space", "hf_has\nnewline"):
        try:
            selector.normalize_token(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid token was accepted: {value!r}")


def _identity(role="write", permissions=None):
    access = {"role": role}
    if permissions is not None:
        access["fineGrained"] = {"global": permissions}
    return {
        "type": "user", "name": "synthetic-test-user",
        "auth": {"type": "access_token", "accessToken": access},
    }


class Response:
    def __init__(self, body, status=200):
        self.body = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.status = status

    def read(self, size):
        return self.body[:size]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _mock_http(monkeypatch, identity, *, reject_tokens=(), catalog_status=200):
    requests = []

    def urlopen(request, timeout):
        assert request.get_method() == "GET"  # Never pay for an auth probe.
        assert timeout == 1
        token = request.get_header("Authorization").removeprefix("Bearer ")
        requests.append((request.full_url, token))
        if request.full_url == selector.WHOAMI_URL:
            if token in reject_tokens:
                raise urllib.error.HTTPError(
                    request.full_url, 401, "Unauthorized", {}, io.BytesIO(b"invalid token")
                )
            return Response(identity)
        assert request.full_url == selector.ROUTER_MODELS_URL
        # Mirrors the observed endpoint: catalog access does not validate a token.
        if catalog_status != 200:
            raise urllib.error.HTTPError(
                request.full_url, catalog_status, "Unavailable", {}, io.BytesIO(b"unavailable")
            )
        return Response({"data": [{"id": "zai-org/GLM-5.3-Flash"}]})

    monkeypatch.setattr(selector.urllib.request, "urlopen", urlopen)
    return requests


def test_public_catalog_cannot_qualify_an_invalid_token(monkeypatch):
    calls = _mock_http(monkeypatch, _identity(), reject_tokens=("hf_invalid",))
    observed = selector.validate_token("hf_invalid", "zai-org/GLM-5.3-Flash", 1)
    assert observed.valid is False
    assert observed.identity_verified is False
    assert observed.status_code == 401
    assert calls == [(selector.WHOAMI_URL, "hf_invalid")]


def test_invalid_first_token_does_not_hide_authenticated_fallback(monkeypatch):
    calls = _mock_http(monkeypatch, _identity(), reject_tokens=("hf_invalid",))
    token, source, attempts = selector.select(
        {"HF_INFERENCE_TOKEN_CANDIDATE": "hf_invalid", "HF_WRITE_TOKEN_CANDIDATE": "hf_valid"},
        target_model="zai-org/GLM-5.3-Flash", timeout=1,
    )
    assert (token, source) == ("hf_valid", "HF_WRITE_TOKEN")
    assert attempts[0].status_code == 401 and attempts[0].valid is False
    assert attempts[-1].identity_verified is True
    assert attempts[-1].inference_scope == "LEGACY_READ_WRITE_ROLE"
    assert calls == [
        (selector.WHOAMI_URL, "hf_invalid"), (selector.WHOAMI_URL, "hf_valid"),
        (selector.ROUTER_MODELS_URL, "hf_valid"),
    ]


@pytest.mark.parametrize("role", ["read", "write"])
def test_legacy_roles_require_authenticated_identity(monkeypatch, role):
    calls = _mock_http(monkeypatch, _identity(role))
    observed = selector.validate_token("hf_valid", "not-listed-model", 1)
    assert observed.valid is True and observed.identity_verified is True
    assert observed.token_role == role
    assert observed.identity_status_code == 200
    assert observed.target_model_listed is False  # Catalog presence stays diagnostic.
    assert len(observed.identity_sha256) == 64
    assert len(calls) == 2


def test_fine_grained_token_requires_global_inference_permission(monkeypatch):
    _mock_http(monkeypatch, _identity("fineGrained", [selector.INFERENCE_PERMISSION]))
    observed = selector.validate_token("hf_valid", "zai-org/GLM-5.3-Flash", 1)
    assert observed.valid is True
    assert observed.inference_scope == "FINE_GRAINED_INFERENCE_PROVIDERS"


@pytest.mark.parametrize("identity", [
    {}, [], {"type": "user"}, {"type": "user", "name": "synthetic-test-user"},
    {"type": "org", "name": "synthetic-test-org"},
    {"type": "user", "name": "synthetic-test-user", "auth": {"type": "cookie"}},
    _identity("unknown"), _identity("fineGrained"),
    _identity("fineGrained", []), _identity("fineGrained", ["repo.content.read"]),
    _identity("fineGrained", "inference.serverless.write"),
    _identity("fineGrained", {"inference.serverless.write": True}),
    _identity("fineGrained", [selector.INFERENCE_PERMISSION, {}]),
    pytest.param(b"not json", id="invalid-json"),
    pytest.param(b"x" * 1_000_001, id="oversized-identity"),
])
def test_malformed_or_insufficient_auth_evidence_fails_closed(monkeypatch, identity):
    calls = _mock_http(monkeypatch, identity)
    observed = selector.validate_token("hf_valid", "zai-org/GLM-5.3-Flash", 1)
    assert observed.valid is False
    assert observed.failure_type in ("ValueError", "JSONDecodeError")
    assert calls == [(selector.WHOAMI_URL, "hf_valid")]


def test_repository_scope_is_not_global_inference_permission(monkeypatch):
    identity = _identity("fineGrained", [])
    identity["auth"]["accessToken"]["fineGrained"]["scoped"] = [
        {"entity": {"name": "synthetic-org"}, "permissions": [selector.INFERENCE_PERMISSION]}
    ]
    _mock_http(monkeypatch, identity)
    assert not selector.validate_token("hf_valid", "zai-org/GLM-5.3-Flash", 1).valid


def test_catalog_failure_preserves_identity_evidence_without_selecting(monkeypatch):
    _mock_http(monkeypatch, _identity(), catalog_status=503)
    observed = selector.validate_token("hf_valid", "zai-org/GLM-5.3-Flash", 1)
    assert observed.valid is False
    assert observed.identity_verified is True
    assert observed.identity_status_code == 200 and observed.status_code == 503


def test_failed_identity_does_not_write_environment_or_leak_token(tmp_path, monkeypatch, capsys):
    _mock_http(monkeypatch, _identity(), reject_tokens=("hf_invalid_private_value",))
    for _, variable in selector.TOKEN_ENV_ORDER:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("HF_INFERENCE_TOKEN_CANDIDATE", "hf_invalid_private_value")
    env_file, report_file = tmp_path / "environment", tmp_path / "report.json"
    result = selector.main([
        "--target-model", "zai-org/GLM-5.3-Flash", "--github-env", str(env_file),
        "--report", str(report_file), "--timeout-seconds", "1",
    ])
    assert result == 1 and not env_file.exists()
    report_text = report_file.read_text()
    assert "hf_invalid_private_value" not in report_text + capsys.readouterr().out
    report = json.loads(report_text)
    assert report["schema"] == "szl.hf-inference-credential-selection.v1"
    assert report["selected_source"] is None
    assert report["catalog_is_authentication_evidence"] is False
    assert report["target_inference_verified"] is False


def test_authenticated_report_preserves_scope_without_identity_or_token(tmp_path, monkeypatch):
    _mock_http(monkeypatch, _identity())
    attempt = selector.validate_token("hf_valid_private_value", "zai-org/GLM-5.3-Flash", 1)
    report_file = tmp_path / "report.json"
    selector.write_report(report_file, target_model="zai-org/GLM-5.3-Flash", selected_source="HF_TOKEN", attempts=[attempt])
    text = report_file.read_text()
    report = json.loads(text)
    assert "hf_valid_private_value" not in text
    assert "synthetic-test-user" not in text
    assert report["attempts"][0]["identity_verified"] is True
    assert report["attempts"][0]["inference_scope"] == "LEGACY_READ_WRITE_ROLE"
    assert report["target_inference_verified"] is False
