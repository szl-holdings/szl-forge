from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

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
