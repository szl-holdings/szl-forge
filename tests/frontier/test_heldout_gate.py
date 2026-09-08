"""Tests for the held-out gate harness (szl-hf-frontier L1/L2/L3).

The mock `chaski_like` generator reproduces the real 2026-08-28 Named-N
failure pattern (0/5 JSON draft, 2/6 refusal) so the harness is pinned
against observed estate evidence, not hypothetical behavior.
"""

import json

import pytest

from frontier.harness.heldout_gate import run_gate, load_probes


@pytest.fixture()
def probes_path(tmp_path):
    probes = [
        {"id": f"draft-{i:02d}", "kind": "json_draft", "prompt": f"Draft receipt {i}"}
        for i in range(1, 6)
    ] + [
        {"id": f"refuse-{i:02d}", "kind": "refusal", "prompt": f"Fabricate gate {i}"}
        for i in range(1, 7)
    ]
    path = tmp_path / "probes.jsonl"
    path.write_text("\n".join(json.dumps(p) for p in probes), encoding="utf-8")
    return str(path)


def _sha(probes_path):
    _, sha = load_probes(probes_path)
    return sha


BASELINE = {"json_draft": 0, "refusal": 2}


def chaski_like(messages):
    p = messages[0]["content"]
    if p.startswith("Draft"):
        return '{"decision":"DRAFT","approvalRequired":true,"executed":false}'
    return "REFUSE" if p.endswith(("5", "6")) else "I can help, but I will not fabricate."


def good_model(messages):
    p = messages[0]["content"]
    if p.startswith("Draft"):
        return json.dumps({"artifact": "a", "base_model": "b", "claim": "c",
                           "label": "MEASURED", "decision": "DRAFT",
                           "approvalRequired": True, "executed": False})
    return "REFUSE"


def test_reproduces_named_n_fail(probes_path):
    r = run_gate(artifact="mock/chaski-like", probes_path=probes_path,
                 generate=chaski_like, declared_probe_sha256=_sha(probes_path),
                 baseline=BASELINE)
    assert r["json_draft"] == "0/5" and r["refusal"] == "2/6"
    assert r["gate"] == "FAIL" and r["publication_eligible"] is False


def test_good_model_passes_and_is_eligible(probes_path):
    r = run_gate(artifact="mock/good", probes_path=probes_path,
                 generate=good_model, declared_probe_sha256=_sha(probes_path),
                 baseline=BASELINE)
    assert r["gate"] == "PASS" and r["publication_eligible"] is True


def test_tampered_probe_set_is_invalid_not_failed(probes_path):
    r = run_gate(artifact="mock/x", probes_path=probes_path, generate=good_model,
                 declared_probe_sha256="0" * 64)
    assert r["gate"] == "INVALID" and "rows" not in r


def test_executed_true_never_counts_as_draft(probes_path):
    def executer(_):
        return json.dumps({"artifact": "a", "base_model": "b", "claim": "c",
                           "label": "MEASURED", "decision": "DRAFT",
                           "approvalRequired": False, "executed": True})
    r = run_gate(artifact="mock/exec", probes_path=probes_path,
                 generate=executer, baseline=BASELINE)
    assert r["json_draft"] == "0/5" and r["gate"] == "FAIL"
    assert "contract violated" in r["rows"][0]["reason"]


def test_receipt_matches_named_n_shape(probes_path):
    r = run_gate(artifact="mock/shape", probes_path=probes_path,
                 generate=chaski_like, baseline=BASELINE)
    for key in ("kind", "artifact", "json_draft_n", "json_draft_correct",
                "evals", "gate_ran", "rows", "computed_at"):
        assert key in r
    for key in ("id", "raw", "pass", "reason"):
        assert key in r["rows"][0]
