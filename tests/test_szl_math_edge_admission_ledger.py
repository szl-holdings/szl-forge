"""Tests for SZL edge-math admission evolution: append-only receipts (F22) and Khipu lineage (F4).

Structural contract tests only: no hardware, model execution, theorem or
independent verification is claimed. The ledger proves chain structure and
ordering, never the truth of a recorded claim.
"""

import importlib.util
import os
import random

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_TOOL = os.path.join(_HERE, "..", "tools")


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_TOOL, name + ".py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


base = _load("szl_math_edge_admission")
ledger_mod = _load("szl_math_edge_admission_ledger")


def _synthetic_case(i):
    n = 200 + (i % 300)
    return {
        "schema": base.SCHEMA,
        "source_pins": dict(base.PINNED_SOURCES),
        "scenario_set": {
            "sha256": base.canonical_hash({"i": i}),
            "heldout_count": n,
            "iid_declared": True,
            "pre_registered": True,
            "independent_evaluator": True,
        },
        "baseline": {
            "model_sha256": "0" * 64,
            "receipt_sha256": "0" * 64,
            "dataset_sha256": base.canonical_hash({"i": i}),
            "errors": n // 4,
            "safety_violations": 0,
            "flash_bytes": 500000,
            "ram_bytes": 400000,
            "latency_p99_ms": 100.0,
            "energy_j": 2.0,
        },
        "candidate": {
            "model_sha256": "1" * 64,
            "receipt_sha256": "1" * 64,
            "dataset_sha256": base.canonical_hash({"i": i}),
            # Low candidate error rate: n//4 would exceed the pre-registered
            # Hoeffding risk_upper_max=0.2 at these sample sizes, and HOLD is
            # the correct outcome there. These cases are built to be admissible.
            "errors": n // 40,
            "safety_violations": 0,
            "flash_bytes": 400000,
            "ram_bytes": 300000,
            "latency_p99_ms": 90.0,
            "energy_j": 1.5,
        },
        "constraints": {
            "flash_max_bytes": 1000000,
            "ram_max_bytes": 600000,
            "latency_p99_max_ms": 250.0,
            "energy_max_j": 6.0,
            "risk_upper_max": 0.2,
            "delta": 0.05,
        },
        "governance": {
            "rights_approved": True,
            "source_verified": True,
            "weights_verified": True,
            "measured_on_target": True,
            "independent_receipt_verified": True,
            "zero_observed_safety_violations": True,
        },
    }


# --- append-only ledger (F22) -------------------------------------------------


def test_genesis_ledger_verifies():
    lg = ledger_mod.AppendOnlyLedger()
    assert lg.head == ledger_mod.GENESIS
    assert lg.length == 0
    assert lg.verify() is True


def test_append_returns_linked_receipt():
    lg = ledger_mod.AppendOnlyLedger()
    out = lg.append({"state": "HOLD", "reasons": ["SCHEMA"]})
    assert out["receipt"]["seq"] == 0
    assert out["receipt"]["prev_hash"] == ledger_mod.GENESIS
    assert out["ledger_head"] == out["receipt"]["entry_hash"]
    assert lg.verify() is True


def test_chain_hashes_link_sequentially():
    lg = ledger_mod.AppendOnlyLedger()
    h0 = lg.append({"a": 1})["ledger_head"]
    h1 = lg.append({"b": 2})["receipt"]
    assert h1["prev_hash"] == h0
    h2 = lg.append({"c": 3})["receipt"]
    assert h2["prev_hash"] == h1["entry_hash"]
    assert lg.verify() is True
    assert lg.length == 3


def test_tampered_entry_breaks_verification():
    lg = ledger_mod.AppendOnlyLedger()
    lg.append({"x": 1})
    lg.append({"x": 2})
    lg._entries[0][0]["record"].update({"x": 999})  # simulate corruption
    assert lg.verify() is False


def test_reordered_entry_breaks_verification():
    lg = ledger_mod.AppendOnlyLedger()
    lg.append({"x": 1})
    lg.append({"x": 2})
    lg._entries.reverse()
    assert lg.verify() is False


def test_append_rejects_non_dict():
    lg = ledger_mod.AppendOnlyLedger()
    with pytest.raises(TypeError):
        lg.append(["not", "a", "dict"])


def test_hold_decisions_are_receipted():
    lg = ledger_mod.AppendOnlyLedger()
    data = _synthetic_case(0)
    data["schema"] = "wrong"
    out = ledger_mod.admit_with_receipt(base.evaluate, data, lg)
    assert out["decision"]["state"] == "HOLD"
    assert out["decision"]["reasons"] == ["SCHEMA"]
    assert out["ledger_head"] == out["receipt"]["entry_hash"]
    assert lg.verify() is True


def test_admissible_decisions_are_receipted_with_state():
    lg = ledger_mod.AppendOnlyLedger()
    out = ledger_mod.admit_with_receipt(base.evaluate, _synthetic_case(1), lg)
    assert out["decision"]["state"] == "STRUCTURALLY_ADMISSIBLE_UNVERIFIED"
    assert out["decision"]["production_authorized"] is False
    assert lg.verify() is True


def test_ledger_determinism_across_instances():
    a, b = ledger_mod.AppendOnlyLedger(), ledger_mod.AppendOnlyLedger()
    for i in range(5):
        a.append({"i": i})
        b.append({"i": i})
    assert a.head == b.head
    assert a.verify() and b.verify()


def test_entries_view_is_a_copy_not_live():
    lg = ledger_mod.AppendOnlyLedger()
    lg.append({"x": 1})
    view = lg.entries()
    view[0]["record"]["x"] = 999
    assert lg._entries[0][0]["record"]["x"] == 1
    assert lg.verify() is True


def test_thousand_decision_stress_chain_holds():
    lg = ledger_mod.AppendOnlyLedger()
    rng = random.Random(20261118)
    for i in range(1000):
        data = _synthetic_case(i)
        if rng.random() < 0.5:
            data["schema"] = "wrong"
        ledger_mod.admit_with_receipt(base.evaluate, data, lg)
    assert lg.length == 1000
    assert lg.verify() is True


# --- Khipu lineage (F4) --------------------------------------------------------


def test_lineage_valid_backward_chain():
    lin = {
        "nodes": [
            {"id": "root", "parents": []},
            {"id": "mid", "parents": ["root"]},
            {"id": "leaf", "parents": ["root", "mid"]},
        ]
    }
    ok, reason = ledger_mod.validate_lineage(lin)
    assert ok and reason == "OK"
    roots = ledger_mod.lineage_root_causes(lin)
    assert roots["roots"] == ["root"]


def test_lineage_rejects_forward_edge():
    lin = {
        "nodes": [
            {"id": "a", "parents": ["b"]},
            {"id": "b", "parents": []},
        ]
    }
    ok, reason = ledger_mod.validate_lineage(lin)
    assert not ok and reason == "NOT_BACKWARD_EDGE"


def test_lineage_cycles_structurally_impossible():
    lin = {
        "nodes": [
            {"id": "a", "parents": ["b"]},
            {"id": "b", "parents": ["a"]},
        ]
    }
    ok, reason = ledger_mod.validate_lineage(lin)
    assert not ok and reason == "NOT_BACKWARD_EDGE"


def test_lineage_rejects_duplicate_ids_and_bad_contracts():
    assert ledger_mod.validate_lineage(
        {"nodes": [{"id": "a", "parents": []}, {"id": "a", "parents": []}]}
    ) == (False, "DUP_OR_INVALID_ID")
    assert ledger_mod.validate_lineage({"nodes": [{"id": "", "parents": []}]}) == (
        False,
        "DUP_OR_INVALID_ID",
    )
    assert ledger_mod.validate_lineage({}) == (False, "LINEAGE_CONTRACT")
    assert ledger_mod.validate_lineage({"nodes": [{"id": "a", "parents": "b"}]}) == (
        False,
        "NODE_CONTRACT",
    )


def test_admit_with_receipt_records_lineage_verdict():
    lg = ledger_mod.AppendOnlyLedger()
    good = {"nodes": [{"id": "n1", "parents": []}]}
    bad = {"nodes": [{"id": "n1", "parents": ["later"]}]}
    out_good = ledger_mod.admit_with_receipt(
        base.evaluate, _synthetic_case(2), lg, lineage=good
    )
    out_bad = ledger_mod.admit_with_receipt(
        base.evaluate, _synthetic_case(3), lg, lineage=bad
    )
    assert out_good["decision"]["state"] == "STRUCTURALLY_ADMISSIBLE_UNVERIFIED"
    assert out_bad["decision"]["state"] == "STRUCTURALLY_ADMISSIBLE_UNVERIFIED"
    assert lg.length == 2
    assert lg.verify() is True


def test_receipt_never_grants_production_authorization():
    lg = ledger_mod.AppendOnlyLedger()
    out = ledger_mod.admit_with_receipt(base.evaluate, _synthetic_case(4), lg)
    assert out["receipt"]["chain_verified_after_append"] is True
    assert out["decision"]["production_authorized"] is False
    entries = lg.entries()
    assert all(e["record"]["production_authorized"] is False for e in entries)
