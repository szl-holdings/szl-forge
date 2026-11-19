"""Focused tests for the leader-adaptation CP admission module.

Mirrors the estate test style of tests/test_szl_math_edge_admission.py:
strict contract checks, HOLD code checks, bound-math property checks.
All values are deterministic; no network, no randomness.
"""
import hashlib
import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MOD = os.path.join(HERE, "..", "tools", "szl_math_edge_admission_cp.py")
spec = importlib.util.spec_from_file_location("szl_math_edge_admission_cp", os.path.abspath(MOD))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def hs(tag):
    return hashlib.sha256(tag.encode()).hexdigest()


def valid_data(n=1000, k_cand=50, k_base=90):
    return {
        "schema": m.SCHEMA,
        "source_pins": dict(m.PINNED_SOURCES),
        "scenario_set": {"sha256": hs("ds"), "heldout_count": n, "iid_declared": True,
                          "pre_registered": True, "independent_evaluator": True},
        "baseline": {"model_sha256": hs("bm"), "receipt_sha256": hs("br"), "dataset_sha256": hs("ds"),
                      "errors": k_base, "safety_violations": 0,
                      "flash_bytes": 900000, "ram_bytes": 500000,
                      "latency_p99_ms": 200.0, "energy_j": 5.0},
        "candidate": {"model_sha256": hs("cm"), "receipt_sha256": hs("cr"), "dataset_sha256": hs("ds"),
                       "errors": k_cand, "safety_violations": 0,
                       "flash_bytes": 800000, "ram_bytes": 400000,
                       "latency_p99_ms": 100.0, "energy_j": 2.5},
        "constraints": {"flash_max_bytes": 1000000, "ram_max_bytes": 600000,
                         "latency_p99_max_ms": 250.0, "energy_max_j": 6.0,
                         "risk_upper_max": 0.1, "delta": 0.05},
        "governance": {"rights_approved": True, "source_verified": True, "weights_verified": True,
                        "measured_on_target": True, "independent_receipt_verified": True,
                        "zero_observed_safety_violations": True},
    }


def test_schema_contract_holds():
    d = valid_data()
    d["schema"] = "wrong"
    assert m.evaluate(d)["reasons"] == ["SCHEMA"]

def test_extra_key_holds():
    d = valid_data()
    d["extra"] = 1
    assert m.evaluate(d)["reasons"] == ["SCHEMA"]

def test_source_pin_mismatch_holds():
    d = valid_data()
    d["source_pins"]["szl-formulas"] = "0" * 40
    assert m.evaluate(d)["reasons"] == ["SOURCE_PIN_MISMATCH"]

def test_governance_hold():
    d = valid_data()
    d["governance"]["rights_approved"] = False
    assert m.evaluate(d)["reasons"] == ["GOVERNANCE_HOLD"]

def test_constraint_domain_holds():
    d = valid_data()
    d["constraints"]["delta"] = 1.0
    assert m.evaluate(d)["reasons"] == ["RISK_CONSTRAINTS_INVALID"]
    d = valid_data()
    d["constraints"]["delta"] = -0.05
    assert m.evaluate(d)["reasons"] == ["CONSTRAINTS_INVALID"]

def test_observed_safety_violation_holds():
    d = valid_data(k_cand=50)
    d["candidate"]["safety_violations"] = 1
    assert m.evaluate(d)["reasons"] == ["OBSERVED_SAFETY_VIOLATION"]

def test_unpaired_dataset_holds():
    d = valid_data()
    d["candidate"]["dataset_sha256"] = hs("other")
    assert m.evaluate(d)["reasons"] == ["UNPAIRED_DATASET"]

def test_counts_exceed_denominator_holds():
    d = valid_data(n=1000, k_cand=50)
    d["candidate"]["errors"] = 1001
    assert m.evaluate(d)["reasons"] == ["COUNT_EXCEEDS_DENOMINATOR"]

def test_pareto_regression_holds():
    d = valid_data(k_cand=20, k_base=90)
    d["candidate"]["flash_bytes"] = 950000
    assert m.evaluate(d)["reasons"] == ["NOT_PARETO_NONREGRESSIVE"]

def test_no_improvement_holds():
    d = valid_data(k_cand=20, k_base=20)
    d["candidate"]["flash_bytes"] = 900000
    d["candidate"]["ram_bytes"] = 500000
    d["candidate"]["latency_p99_ms"] = 200.0
    d["candidate"]["energy_j"] = 5.0
    assert m.evaluate(d)["reasons"] == ["NO_MEASURED_IMPROVEMENT"]

def test_risk_gate_holds_heavy_tail():
    d = valid_data(n=1000, k_cand=900, k_base=990)
    assert m.evaluate(d)["reasons"] == ["RISK_UPPER_BOUND_EXCEEDED"]

def test_admissible_case_passes_with_engine_marker():
    out = m.evaluate(valid_data(n=1000, k_cand=50, k_base=90))
    assert out["state"] == "STRUCTURALLY_ADMISSIBLE_UNVERIFIED"
    assert out["reasons"] == []
    assert out["production_authorized"] is False
    assert out["proof_refinement"] is False
    assert out["bound_engine"] == m.BOUND_ENGINE
    assert out["candidate_risk_upper_cp_exact"] <= 0.1

def test_cp_admits_where_hoeffding_blocks_measured_gain():
    # n=1000, k=60: empirical 0.06; Hoeffding(0.05/2) = 0.06+0.0429 > 0.1 but
    # exact CP bound stays below 0.1: the leader-adaptation admits, v1 holds.
    d = valid_data(n=1000, k_cand=60, k_base=90)
    ho = m.hoeffding_risk_upper(60, 1000, 0.025)
    cp = m.cp_risk_upper(60, 1000, 0.025)
    assert ho > 0.1 and cp <= 0.1
    assert m.evaluate(d)["state"] == "STRUCTURALLY_ADMISSIBLE_UNVERIFIED"

def test_cp_zero_errors_closed_form():
    b = m.cp_risk_upper(0, 30, 0.025)
    assert b == round(1.0 - 0.025 ** (1.0 / 30), 12)
    assert b > 0.0

def test_cp_bound_dominates_empirical_and_hoeffding_grid():
    for n in (30, 100, 1000):
        for k in (0, 1, n // 4, n // 2):
            for delta in (0.005, 0.025, 0.1):
                b = m.cp_risk_upper(k, n, delta)
                assert b >= k / n - 1e-12
                assert b <= m.hoeffding_risk_upper(k, n, delta) + 1e-12

def test_cp_bound_deterministic_and_quantized():
    a = m.cp_risk_upper(37, 500, 0.025)
    b = m.cp_risk_upper(37, 500, 0.025)
    assert a == b
    assert a == round(a, 12)

def test_calibrate_lambda_exact_toy():
    uppers = [0.10, 0.20, 0.30]
    flags = [False, True, True]
    # at lam=0.10: loss=0 -> (0+1)/4 = 0.25 <= alpha only if alpha >= 0.25
    assert m.calibrate_lambda(uppers, flags, 0.25, [0.10, 0.20]) == 0.10
    # alpha=0.2: (0+1)/4 = 0.25 > 0.2; at 0.20: (1+1)/4=0.5 > 0.2 -> None
    assert m.calibrate_lambda(uppers, flags, 0.2, [0.10, 0.20]) is None
    assert m.calibrate_lambda([0.5] * 10, [True] * 10, 0.05, [0.4]) is None

def test_canonical_hash_deterministic_and_order_independent():
    out = m.evaluate(valid_data())
    a = {"x": 1, "y": 2}
    b = {"y": 2, "x": 1}
    assert m.canonical_hash(a) == m.canonical_hash(b)
    assert m.canonical_hash(out) == m.canonical_hash(dict(reversed(list(out.items()))))
