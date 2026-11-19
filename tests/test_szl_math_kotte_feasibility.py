"""Tests for szl_math_kotte_feasibility (battery v2.2, fail-closed doctrine)."""
import hashlib
import importlib.util
import os
from decimal import Decimal, localcontext
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SPEC = importlib.util.spec_from_file_location(
    "kotte_mod",
    os.path.normpath(os.path.join(_HERE, "..", "tools", "szl_math_kotte_feasibility.py")))
kotte = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(kotte)


def test_feasible_basic():
    r = kotte.kotte_feasibility(0.05, 0.02)
    assert r["feasible"] is True
    assert r["abstention_floor"] == 0.0
    assert r["verdict"] == "FEASIBLE"


def test_infeasible_floor_formula():
    r = kotte.kotte_feasibility(0.05, 0.125)
    assert r["feasible"] is False
    assert r["abstention_floor"] == pytest.approx((0.125 - 0.05) / 0.95, abs=1e-12)
    assert r["verdict"] == "INFEASIBLE_PUBLISH_FLOOR"


def test_cycle0_leak_measured_numbers():
    # cycle-0 receipts: SCRC-I quantile transfer realized rates (estate-measured)
    assert kotte.kotte_certify(0.05, 0.125)["decision"] == "REFUSE"
    assert kotte.kotte_certify(0.1, 0.086)["decision"] == "CERTIFY"
    assert kotte.kotte_certify(0.2, 0.0094)["decision"] == "CERTIFY"
    # fixed-band control: zero realized violations (cycle-0/2 receipts)
    for a in (0.05, 0.1):
        assert kotte.kotte_certify(a, 0.0)["decision"] == "CERTIFY"


def test_floor_recovers_alpha():
    for a, m in ((0.05, 0.125), (0.1, 0.9), (0.2, 0.5)):
        f = kotte.kotte_feasibility(a, m)
        residual = (m - f["abstention_floor"]) / (1.0 - f["abstention_floor"])
        assert residual == pytest.approx(a, abs=1e-11)


def test_boundary_mu_equals_alpha():
    r = kotte.kotte_feasibility(0.05, 0.05)
    assert r["feasible"] is True and r["abstention_floor"] == 0.0


def test_quantization_12dp():
    r = kotte.kotte_feasibility(0.05, 0.123456789123456)
    assert r["mu"] == round(0.123456789123456, 12)
    assert r["abstention_floor"] == kotte._ceil_floor_12(0.05, 0.123456789123456)


def test_abstention_floor_ceil_is_not_rounded_below_idealized_minimum():
    with localcontext() as ctx:
        ctx.prec = 60
        for a, m in ((0.05, 0.125), (0.05, 0.123456789123456),
                     (0.1, 0.9), (0.2, 0.5)):
            out = kotte.kotte_feasibility(a, m)
            idealized = (Decimal(str(m)) - Decimal(str(a))) / (Decimal(1) - Decimal(str(a)))
            assert Decimal(str(out["abstention_floor"])) >= idealized


def test_determinism():
    a = kotte.kotte_feasibility(0.05, 0.125)
    b = kotte.kotte_feasibility(0.05, 0.125)
    assert a == b


@pytest.mark.parametrize("alpha,mu", [
    (0.0, 0.1), (1.0, 0.1), (-0.1, 0.1), (0.5, -0.01), (0.5, 1.01),
    (float("nan"), 0.1), (0.5, float("nan")), (float("inf"), 0.1),
])
def test_validation_fail_closed(alpha, mu):
    with pytest.raises(ValueError):
        kotte.kotte_feasibility(alpha, mu)


def test_bool_rejected():
    with pytest.raises(ValueError):
        kotte.kotte_feasibility(True, 0.1)


def test_audit_batch_and_receipt():
    entries = [
        {"alpha": 0.05, "mu": 0.125, "label": "cycle0_scrc_i_leak_a005"},
        {"alpha": 0.1, "mu": 0.086, "label": "cycle0_scrc_i_a010"},
        {"alpha": 0.2, "mu": 0.0094, "label": "cycle0_scrc_i_a020"},
        {"alpha": 0.05, "mu": 0.0, "label": "cycle2_exact_transfer_a005"},
        {"alpha": 0.1, "mu": 0.0, "label": "cycle2_exact_transfer_a010"},
    ]
    rec = kotte.audit(entries)
    assert rec["batch_size"] == 5 and rec["refused"] == 1 and rec["certified"] == 4
    assert rec["fail_closed"] is True
    assert kotte.audit_receipt_ok(rec) is True
    assert rec["production_authorized"] is False


def test_audit_tamper_detected():
    rec = kotte.audit([{"alpha": 0.05, "mu": 0.125, "label": "x"}])
    bad = dict(rec)
    bad["body"] = list(bad["body"])
    bad["body"][0]["mu"] = 0.0
    assert kotte.audit_receipt_ok(bad) is False


def test_audit_entry_validation():
    with pytest.raises(ValueError):
        kotte.audit([{"alpha": 0.05, "mu": 0.1}])


def test_companion_pin_matches_actual_cp_source_bytes():
    path = os.path.normpath(os.path.join(_HERE, "..", "tools", "szl_math_edge_admission_cp.py"))
    with open(path, "rb") as stream:
        canonical = stream.read().replace(b"\r\n", b"\n")
    assert hashlib.sha256(canonical).hexdigest() == kotte.PINNED_COMPANIONS["cp_module_sha256"]


def test_schema_and_pins():
    assert kotte.SCHEMA == "szl.edge-math-evidence/kotte-v1"
    assert kotte.PINNED_COMPANIONS["cp_module_sha256"] == (
        "de8dbea415c6a3082f13eb355adfc451b7de5395b366bfa951a2647e75c4c40e")
    assert kotte.PINNED_COMPANIONS["v1_gate_sha256"] == (
        "3aebebab39c5ab73a2a8aa11ec907ebc129c64a40d1816262d71af951ad908b0")


def test_certify_reason_published_on_refuse():
    r = kotte.kotte_certify(0.05, 0.125)
    assert r["decision"] == "REFUSE"
    assert r["feasibility"]["abstention_floor"] == pytest.approx(0.078947368421, abs=1e-11)
    assert r["production_authorized"] is False


def test_huge_integer_inputs_raise_value_error_not_overflow():
    with pytest.raises(ValueError):
        kotte.kotte_certify(10 ** 400, 0.1)
    with pytest.raises(ValueError):
        kotte.kotte_certify(0.05, 10 ** 400)
