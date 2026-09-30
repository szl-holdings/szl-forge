"""GEH v8 tests. Structural tests always run; kernel tests run only when the
Lean REPL is built under geh-lean (lake build && lake build repl).
Run:  python -m pytest -q tests/   (from the directory holding szl_geh_v8.py)"""
import json
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import szl_geh_v8 as geh  # noqa: E402
import szl_geh_verify as ver  # noqa: E402

LEAN = Path(os.environ.get("GEH_LEAN_PROJECT", HERE / "geh-lean"))


@pytest.fixture()
def paths(tmp_path):
    return geh.Paths(root=tmp_path, lean_project=LEAN)


def test_gate_registry_fail_closed():
    assert geh.gate_tactic("sorry") == "BLOCKED"
    assert geh.gate_tactic("cases a <;> sorry") == "BLOCKED"
    assert geh.gate_tactic("native_decide") == "BLOCKED"
    assert geh.gate_tactic("plausible") == "EVIDENCE_ONLY"
    assert geh.gate_tactic("exact hpq hp") == "OK"
    assert geh.gate_tactic("induction n") == "OK_RESTRICTED"
    assert geh.gate_tactic("frobnicate x") == "UNREGISTERED"
    assert geh.gate_tactic("") == "UNREGISTERED"


def test_classifier():
    assert geh.classify_lean_artifact("theorem t : p := by sorry") == "QUARANTINED"
    assert geh.classify_lean_artifact("axiom bad : False") == "QUARANTINED"
    assert geh.classify_lean_artifact("/- sorry in comment -/ theorem t : True := trivial") == "COMPLIANCE_CANDIDATE"
    assert geh.classify_lean_artifact("import Plausible\nexample : True := by plausible") == "EVIDENCE_ONLY"


def test_sim_never_certifies(paths):
    sim = geh.SimulatedTransport()
    r = geh.prove(sim, "A ∧ B", geh.ScriptedProver(["constructor", "rfl", "rfl"]))
    assert r["done"] is True
    assert geh.validate_proof_trace(r["events"], sim)["verdict"] == "TRACE_SIGNED_KERNEL_UNVERIFIED"


def test_blocked_tactic_voids_and_stops():
    sim = geh.SimulatedTransport()
    r = geh.prove(sim, "P", geh.ScriptedProver(["sorry", "rfl"]))
    assert r["done"] is False and r["script"] == []
    assert geh.validate_proof_trace(r["events"], sim)["verdict"] == "VOID"


def test_crypto_suite_determinism_and_oracles():
    a, b = geh.gen_crypto_suite("run1", "s", 10), geh.gen_crypto_suite("run1", "s", 10)
    assert a["digest"] == b["digest"]
    assert geh.gen_crypto_suite("run1", "t", 10)["digest"] != a["digest"]
    assert geh.py_link(123, 456, 7919) == 4269 and geh.py_pae("ab", "c") == "DSSEv1 2 ab 1 c"
    assert all(it["canary"] in it["name"] or it["canary"][-8:].replace("-", "") in it["name"] for it in a["items"])


def test_contamination_scoring_uses_all_outputs():
    cans = [geh.canary("r", "C", i) for i in range(3)]
    assert geh.contamination_score(["", f"leak {cans[2]}", ""], cans)["contaminated"]
    assert not geh.contamination_score(["", f"self {cans[1]}", ""], cans)["contaminated"]


def test_dsse_roundtrip_and_pure_python_verifier(paths):
    s = geh.DSSESigner(paths.keydir)
    env = s.sign({"x": 1})
    assert geh.DSSESigner.verify(env, geh.load_pub(paths.keydir))
    ok, _ = ver.verify_envelope(env)
    assert ok
    env2 = dict(env, payload=env["payload"] + " ")
    assert not ver.verify_envelope(env2)[0]


def test_receipt_gate_is_recomputed_not_trusted():
    payload = {"proof": {"transport": "SIMULATED", "verdict": "KERNEL_VERIFIED_AND_SIGNED", "live": {"done": True, "violations": []},
                         "audit": {"error_count": 0, "sorry_count": 0, "guard_all_allowed": True, "artifact_grade": "COMPLIANCE_CANDIDATE", "guard": []}},
               "decision": {"gate": "PASS"}}
    gate, reasons = ver.recompute_gate(payload)
    assert gate == "VOID" and any("not certified" in r for r in reasons)


def test_selfcheck_structural(paths):
    sc = geh.selfcheck(paths, with_kernel=False)
    assert sc["self_check_pass"], {k: v for k, v in sc.items() if v is not True}


@pytest.mark.skipif(not geh.repl_available(LEAN), reason="Lean REPL not built")
def test_kernel_selfcheck(paths):
    sc = geh.kernel_selfcheck(paths)
    assert all(v is True for v in sc.values()), {k: v for k, v in sc.items() if v is not True}


@pytest.mark.skipif(not geh.repl_available(LEAN), reason="Lean REPL not built")
def test_kernel_bench_and_chain(paths):
    rc = geh.main(["--root", str(paths.root), "--lean-project", str(LEAN), "--transport", "repl",
                   "--run-id", "test0000deadbeef", "--salt", "s", "--bench-crypto", "5"])
    assert rc == 0
    rf = paths.evidence / "geh_proof_receipt_test0000deadbeef.json"
    env = json.loads(rf.read_text(encoding="utf-8"))
    payload = json.loads(env["payload"])
    assert payload["decision"]["gate"] == "PASS" and payload["proof"]["bench"]["pass_at_k"] == 1.0
    assert ver.verify_chain(paths.evidence) == 0
