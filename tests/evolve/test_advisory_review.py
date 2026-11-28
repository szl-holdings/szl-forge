"""Advisory reviewer: second opinion only. Never a pass/fail gate.

All Ollama calls are injected via a fake transport. These tests make no
network requests.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from urllib.error import URLError

from szl_evolve import advisory_review as ar
from szl_evolve import card_lint as cl
from szl_evolve import decon_ext as d

GOOD = """---
license: apache-2.0
base_model: Qwen/Qwen3.5-0.8B
datasets: [SZLHOLDINGS/x]
tags: [sft, lora]
---
# M
## Intended use
Internal. Data revision a1b2c3d4e5f6.
## Limitations
n=6 probes, 1/6 abstained.
## Bias and risk
None known beyond training data.
## Training data provenance
x@a1b2c3d4e5f6
"""

MATCHING_DIGEST = "sha256:ca1b2de39499aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
FIXED_NOW = lambda: datetime(2026, 11, 26, 21, 30, 0, tzinfo=timezone.utc)


def _dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=True, separators=(",", ":"), sort_keys=False)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class FakeOllama:
    """In-process stand-in for Ollama /api/show, /api/tags and /api/generate."""

    def __init__(
        self,
        *,
        digest=MATCHING_DIGEST,
        name=ar.DEFAULT_MODEL,
        show=True,
        tags=True,
        response=None,
        show_error=None,
        tags_error=None,
        generate_error=None,
        generate_raw=None,
    ):
        self.digest = digest
        self.name = name
        self.show = show
        self.tags = tags
        self.show_error = show_error
        self.tags_error = tags_error
        self.generate_error = generate_error
        self.generate_raw = generate_raw
        self.response = response if response is not None else {
            "lint_opinions": [{"code": "CL001", "genuine": True, "rationale": "front matter is missing"}],
            "near_dup_opinions": [{"left_index": 0, "right_index": 0, "true_duplicate": True, "rationale": "high jaccard"}],
        }
        self.calls = []

    def __call__(self, method, url, body=None):
        self.calls.append((method, url, body))
        if url.endswith("/api/show"):
            if self.show_error:
                raise self.show_error
            if not self.show:
                raise URLError("show disabled")
            return {"name": self.name, "digest": self.digest}
        if url.endswith("/api/tags"):
            if self.tags_error:
                raise self.tags_error
            if not self.tags:
                raise URLError("tags disabled")
            return {"models": [{"name": self.name, "model": self.name, "digest": self.digest}]}
        if url.endswith("/api/generate"):
            if self.generate_error:
                raise self.generate_error
            if self.generate_raw is not None:
                return self.generate_raw
            text = self.response if isinstance(self.response, str) else json.dumps(self.response)
            return {"response": text, "done": True}
        raise AssertionError(f"unexpected url {url}")

    def generate_called(self) -> bool:
        return any(url.endswith("/api/generate") for _, url, _ in self.calls)


# ---------------------------------------------------------------------------
# digest check
# ---------------------------------------------------------------------------

def test_digest_matches_prefix_with_and_without_sha256():
    assert ar.digest_matches("sha256:ca1b2de39499deadbeef", "ca1b2de39499")
    assert ar.digest_matches("CA1B2DE39499ffff", "ca1b2de39499")
    assert not ar.digest_matches("sha256:deadbeef0000", "ca1b2de39499")
    assert not ar.digest_matches("", "ca1b2de39499")


def test_verify_accepts_matching_digest_via_show():
    fake = FakeOllama()
    got = ar.verify_model(transport=fake)
    assert got["ok"] is True and got["status"] == "OK"
    assert ar.digest_matches(got["digest"], ar.EXPECTED_DIGEST_PREFIX)
    assert got["model"] == ar.DEFAULT_MODEL


def test_verify_accepts_matching_digest_via_tags_only():
    fake = FakeOllama(show=False)
    got = ar.verify_model(transport=fake)
    assert got["ok"] is True and got["status"] == "OK"
    assert ar.digest_matches(got["digest"])


def test_digest_mismatch_is_refused_and_does_not_generate():
    fake = FakeOllama(digest="sha256:ffffffffffffffffffffffffffffffffffffffffffffffff")
    lint_res = cl.lint("# no front matter\n", "model")
    out = ar.review(lint_res, transport=fake, now=FIXED_NOW)
    assert out["advisory"]["status"] == "REFUSED"
    assert out["advisory"]["reason"] == "digest_mismatch"
    assert out["receipt"]["advisory"] is True and out["receipt"]["gate"] is False
    assert not fake.generate_called()
    # Pass/fail payload is unchanged.
    assert _dumps(out["lint"]) == _dumps(lint_res)


# ---------------------------------------------------------------------------
# :cloud / -cloud refusal
# ---------------------------------------------------------------------------

def test_cloud_tag_colon_refused_without_generate():
    fake = FakeOllama(name="kimi-estate-ops:cloud", digest=MATCHING_DIGEST)
    out = ar.review(cl.lint(GOOD, "model", "x-sft"), model="kimi-estate-ops:cloud", transport=fake)
    assert out["advisory"]["status"] == "REFUSED"
    assert out["advisory"]["reason"] == "cloud_model"
    assert out["receipt"]["advisory"] is True and out["receipt"]["gate"] is False
    assert not fake.generate_called()
    # Cloud refusal happens before any HTTP, so the fake should not have been hit.
    assert fake.calls == []


def test_cloud_tag_hyphen_refused_without_generate():
    fake = FakeOllama()
    out = ar.review([], model="kimi-estate-ops-cloud", transport=fake)
    assert out["advisory"]["status"] == "REFUSED"
    assert out["advisory"]["reason"] == "cloud_model"
    assert not fake.generate_called()
    assert fake.calls == []


def test_resolved_cloud_tag_from_api_is_refused():
    # Caller asked for a local-looking name; the daemon resolved it to a cloud tag.
    fake = FakeOllama(name="kimi-estate-ops:cloud", digest=MATCHING_DIGEST)
    out = ar.review(model="kimi-estate-ops:latest", transport=fake)
    assert out["advisory"]["status"] == "REFUSED"
    assert out["advisory"]["reason"] == "cloud_model"
    assert not fake.generate_called()


# ---------------------------------------------------------------------------
# pass-through invariance
# ---------------------------------------------------------------------------

def _sample_inputs():
    lint_res = cl.lint("# no front matter\n", "model", "broken")
    a = "the quick brown fox jumps over the lazy dog near the quiet river bank at dawn today"
    b = a.replace("quiet", "calm")
    c = "orthogonal content about gradient checkpointing optimizer state and learning rate schedules in training"
    pairs = d.near_duplicates([a], [b, c], threshold=0.6)
    assert lint_res["errors"] > 0 and pairs
    return lint_res, pairs


def test_passthrough_invariance_with_and_without_reviewer():
    lint_res, pairs = _sample_inputs()
    lint_bytes_before = _dumps(lint_res)
    pair_bytes_before = _dumps(pairs)
    lint_snapshot = json.loads(_dumps(lint_res))
    pair_snapshot = json.loads(_dumps(pairs))

    fake = FakeOllama()
    out = ar.review(lint_res, pairs, transport=fake, now=FIXED_NOW)

    # Results with the reviewer attached are byte-identical to results without it.
    assert _dumps(out["lint"]) == lint_bytes_before
    assert _dumps(out["near_duplicates"]) == pair_bytes_before
    # Inputs themselves are not mutated.
    assert lint_res == lint_snapshot
    assert pairs == pair_snapshot
    # Advisory lives in a separate field and does not alter error counts.
    assert out["lint"]["errors"] == lint_snapshot["errors"]
    assert out["lint"]["warnings"] == lint_snapshot["warnings"]
    assert out["advisory"]["status"] == "OK"
    assert out["advisory"]["lint_opinions"]
    assert "advisory" not in (lint_res if isinstance(lint_res, dict) else {})


def test_passthrough_holds_when_reviewer_is_unavailable():
    lint_res, pairs = _sample_inputs()
    lint_bytes_before = _dumps(lint_res)
    pair_bytes_before = _dumps(pairs)
    fake = FakeOllama(show_error=URLError("connection refused"), tags_error=URLError("connection refused"))
    out = ar.review(lint_res, pairs, transport=fake, now=FIXED_NOW)
    assert out["advisory"]["status"] == "UNAVAILABLE"
    assert _dumps(out["lint"]) == lint_bytes_before
    assert _dumps(out["near_duplicates"]) == pair_bytes_before


# ---------------------------------------------------------------------------
# receipt fields
# ---------------------------------------------------------------------------

def test_receipt_fields_on_successful_review(tmp_path):
    lint_res, pairs = _sample_inputs()
    fake = FakeOllama()
    receipt_path = tmp_path / "receipt.json"
    out = ar.review(lint_res, pairs, transport=fake, now=FIXED_NOW, receipt_path=str(receipt_path))
    rec = out["receipt"]
    assert rec["model"] == ar.DEFAULT_MODEL
    assert rec["digest"] == MATCHING_DIGEST
    assert rec["advisory"] is True
    assert rec["gate"] is False
    assert rec["timestamp"] == "2026-11-26T21:30:00Z"
    prompt = ar.build_prompt(lint_res, pairs)
    response = json.dumps(fake.response)
    assert rec["prompt_sha256"] == _sha(prompt)
    assert rec["response_sha256"] == _sha(response)
    assert len(rec["prompt_sha256"]) == 64 and len(rec["response_sha256"]) == 64
    # Written receipt has the same required fields and never stores bodies.
    written = json.loads(receipt_path.read_text())
    for key in ("model", "digest", "prompt_sha256", "response_sha256", "timestamp", "advisory", "gate"):
        assert written[key] == rec[key]
    assert "prompt" not in written and "response" not in written
    body = receipt_path.read_text()
    assert prompt not in body and "front matter" not in body.lower()


def test_receipt_fields_on_unavailable():
    fake = FakeOllama(show_error=URLError("timed out"), tags_error=URLError("timed out"))
    out = ar.review({"findings": []}, transport=fake, now=FIXED_NOW)
    rec = out["receipt"]
    assert rec["advisory"] is True and rec["gate"] is False
    assert rec["status"] == "UNAVAILABLE"
    assert rec["timestamp"] == "2026-11-26T21:30:00Z"
    assert rec["model"] == ar.DEFAULT_MODEL
    assert rec["prompt_sha256"] == _sha(ar.build_prompt({"findings": []}, None))
    assert rec["response_sha256"] == _sha("")


# ---------------------------------------------------------------------------
# unavailable / fail-open
# ---------------------------------------------------------------------------

def test_unavailable_when_daemon_unreachable():
    lint_res, pairs = _sample_inputs()
    fake = FakeOllama(show_error=URLError("connection refused"), tags_error=URLError("connection refused"))
    out = ar.review(lint_res, pairs, transport=fake)
    assert out["advisory"]["status"] == "UNAVAILABLE"
    assert out["advisory"]["reason"] == "unreachable"
    assert out["advisory"]["lint_opinions"] == []
    assert out["advisory"]["near_dup_opinions"] == []
    assert not fake.generate_called()


def test_unavailable_when_generate_fails_after_verify():
    lint_res, pairs = _sample_inputs()
    fake = FakeOllama(generate_error=URLError("reset by peer"))
    out = ar.review(lint_res, pairs, transport=fake)
    assert out["advisory"]["status"] == "UNAVAILABLE"
    assert _dumps(out["lint"]) == _dumps(lint_res)
    assert _dumps(out["near_duplicates"]) == _dumps(pairs)


def test_host_defaults_to_localhost_and_respects_env(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    assert ar.ollama_host() == "http://127.0.0.1:11434"
    monkeypatch.setenv("OLLAMA_HOST", "10.0.0.9:11434")
    assert ar.ollama_host() == "http://10.0.0.9:11434"
    assert ar.ollama_host("https://example.invalid:1/") == "https://example.invalid:1"


def test_prompt_contains_only_safe_finding_fields():
    # A finding dict might be sloppy and carry a card excerpt; the prompt must not.
    findings = {"name": "x", "kind": "model", "errors": 1, "warnings": 0,
                "findings": [{"code": "CL040", "severity": "ERROR",
                              "message": "secret-looking value in card (not shown)",
                              "card_text": "hf_" + "Q" * 34}]}
    prompt = ar.build_prompt(findings, None)
    assert "CL040" in prompt
    assert "hf_" not in prompt
    assert "QQQQ" not in prompt
