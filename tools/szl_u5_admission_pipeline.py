"""szl_u5_admission_pipeline.py - U5 admission-gated proving pipeline (v1).

Implements the staged spec collaboration/u5-admission-gated-proving-20261119.json.
Zero-claims: this module makes nothing live. It wires the already-merged estate
modules (szl_math_edge_admission_cp, szl_math_kotte_feasibility) into a
per-problem fail-closed pipeline with a certified-spend ledger convention.

Gate order per spec:
  A. CP admission: calibrate_lambda over the calibration arrays (None = no
     feasible lambda -> HOLD; nothing downstream runs).
  B. Kotte CB: kotte_certify(alpha, mu_hat) with mu_hat = realized true-violation
     rate among calibration-admitted cases (the cycle-0 audit convention).
     REFUSE stops the pipeline and publishes the abstention floor.
  C. Lean gates supplied by the PG harness: FAIL stop, FAIL_SKETCH lands in the
     sketch tier (receipted, NEVER certified), only PASS_CLEAN certifies.

The GPU release stays owner-gated: every receipt carries
production_authorized=False and gpu_release="PENDING_OWNER_ORDER". This module
starts nothing, spends nothing, performs no network I/O and touches no secret.

Module shas of the imports are recorded at call time and annotated against the
cycle-receipt pins (cp cf56cd36..., v1 3aebebab...). Drift is an annotation, not
a failure: the modules that run are the ones imported here, while the cycle
receipts pin the frozen artifacts that produced the evidence.
"""
import hashlib
import importlib.util
import json
import os

SCHEMA = "szl.edge-math-evidence/u5-pipeline-v1"
HERE = os.path.dirname(os.path.abspath(__file__))
CP_PATH = os.path.join(HERE, "szl_math_edge_admission_cp.py")
KOTTE_PATH = os.path.join(HERE, "szl_math_kotte_feasibility.py")
CYCLE_PINS = {
    "cp_module_sha256_receipts": "cf56cd363ee06595598be97aa3286c06dbcc95f4bfee0818c46a692a6835caac",
    "v1_module_sha256_receipts": "3aebebab39c5ab73a2a8aa11ec907ebc129c64a40d1816262d71af951ad908b0",
}
LEAN_VERIFY_VERDICTS = ("PASS", "FAIL")
HYGIENE_VERDICTS = ("PASS_CLEAN", "FAIL_SKETCH", "FAIL_SKETCH_AND_COMPILE", "FAIL", "HOLD")
GPU_RELEASE = "PENDING_OWNER_ORDER"

_CACHE = {}


def _load(path, name):
    if name not in _CACHE:
        if not os.path.exists(path):
            raise FileNotFoundError(name + " not found beside this module")
        spec = importlib.util.spec_from_file_location(name, os.path.abspath(path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _CACHE[name] = mod
    return _CACHE[name]


def _cp():
    return _load(CP_PATH, "szl_math_edge_admission_cp_u5")


def _kotte():
    return _load(KOTTE_PATH, "szl_math_kotte_feasibility_u5")


def module_pinnings():
    """Current sha256 of the imported modules, annotated vs cycle pins."""
    rows = []
    for path, key, pin in ((CP_PATH, "cp_module", CYCLE_PINS["cp_module_sha256_receipts"]),
                           (KOTTE_PATH, "kotte_module", None)):
        sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
        rows.append({
            "module": key,
            "sha256": sha,
            "cycle_receipt_pin": pin,
            "matches_cycle_pin": (sha == pin) if pin else None,
        })
    return rows


def _sha_record(rec):
    body = {k: v for k, v in rec.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()


def _receipt(kind, problem, alpha, gates, tier, verdict):
    rec = {
        "schema": SCHEMA,
        "problem": problem,
        "alpha": float(alpha),
        "verdict": verdict,
        "tier": tier,
        "gate_trace": gates,
        "production_authorized": False,
        "gpu_release": GPU_RELEASE,
        "fail_closed": True,
        "kind": kind,
    }
    rec["sha256"] = _sha_record(rec)
    return rec


def _hold(problem, alpha, code, detail=None):
    gates = [{"gate": code, "outcome": "HOLD"}] + ([{"detail": detail}] if detail else [])
    return _receipt("hold", problem, alpha, gates, "none", "HOLD_" + code)


def admit(spec):
    """Run the A -> B -> C pipeline for one problem. Pure; no side effects."""
    if not isinstance(spec, dict) or not {"problem", "alpha", "calibration", "lean"} <= set(spec):
        return _hold("(malformed)", 0.5, "CONTRACT")
    if not isinstance(spec["problem"], str) or not spec["problem"]:
        return _hold("(malformed)", 0.5, "CONTRACT")
    alpha = spec["alpha"]
    if not isinstance(alpha, float) or not (0.0 < alpha < 1.0):
        return _hold(spec["problem"], 0.5 if not isinstance(alpha, float) else alpha, "CONTRACT_ALPHA")
    cal = spec["calibration"]
    if not isinstance(cal, dict) or not {"certified_uppers", "true_flags", "grid"} <= set(cal):
        return _hold(spec["problem"], alpha, "CONTRACT_CALIBRATION")
    uppers, flags, grid = cal["certified_uppers"], cal["true_flags"], cal["grid"]
    if (not isinstance(uppers, list) or not isinstance(flags, list)
            or len(uppers) != len(flags) or not uppers
            or any((not isinstance(u, (int, float))) or (not isinstance(f, bool)) for u, f in zip(uppers, flags))
            or not isinstance(grid, list) or not grid):
        return _hold(spec["problem"], alpha, "CONTRACT_CALIBRATION")
    lean = spec["lean"]
    if (not isinstance(lean, dict)
            or {"lean_verify", "hygiene"} - set(lean)
            or lean.get("lean_verify") not in LEAN_VERIFY_VERDICTS
            or lean.get("hygiene") not in HYGIENE_VERDICTS):
        return _hold(spec["problem"], alpha, "CONTRACT_LEAN")
    try:
        cp, kotte = _cp(), _kotte()
    except Exception:
        return _hold(spec["problem"], alpha, "MODULE_IMPORT")

    gates = []
    lam = cp.calibrate_lambda(list(uppers), list(flags), alpha, list(grid))
    if lam is None:
        gates.append({"gate": "cp_admission", "outcome": "HOLD_NO_FEASIBLE_LAMBDA"})
        rec = _receipt("hold", spec["problem"], alpha, gates, "none", "HOLD_ADMISSION")
        rec["module_pinnings"] = module_pinnings()
        rec["sha256"] = _sha_record(rec)
        return rec
    gates.append({"gate": "cp_admission", "outcome": "ADMIT", "lambda_hat": lam})

    admitted = sum(1 for u in uppers if u <= lam)
    if admitted == 0:
        gates.append({"gate": "kotte_cb", "outcome": "HOLD_NO_ADMITTED_CASES"})
        rec = _receipt("hold", spec["problem"], alpha, gates, "none", "HOLD_KOTTE")
        rec["module_pinnings"] = module_pinnings()
        rec["sha256"] = _sha_record(rec)
        return rec
    mu_hat = sum(1.0 for u, f in zip(uppers, flags) if u <= lam and f) / admitted
    kc = kotte.kotte_certify(alpha, mu_hat)
    gates.append({"gate": "kotte_cb", "outcome": kc["decision"],
                  "mu_hat": kc["feasibility"]["mu"],
                  "abstention_floor": kc["feasibility"]["abstention_floor"]})
    if kc["decision"] != "CERTIFY":
        rec = _receipt("refused", spec["problem"], alpha, gates, "none", "REFUSED_KOTTE")
        rec["mu_hat"] = mu_hat
        rec["module_pinnings"] = module_pinnings()
        rec["sha256"] = _sha_record(rec)
        return rec

    gates.append({"gate": "lean_verify", "outcome": lean["lean_verify"]})
    if lean["lean_verify"] != "PASS":
        rec = _receipt("stopped", spec["problem"], alpha, gates, "none", "STOP_LEAN_VERIFY_FAIL")
        rec["module_pinnings"] = module_pinnings()
        rec["sha256"] = _sha_record(rec)
        return rec
    gates.append({"gate": "lean_hygiene", "outcome": lean["hygiene"]})
    if lean["hygiene"] == "PASS_CLEAN":
        verdict, tier, kind = "CERTIFIED_SPEND_READY", "certified_spend", "certified"
    elif lean["hygiene"] == "FAIL_SKETCH":
        verdict, tier, kind = "SKETCH_TIER_READY_NOT_CERTIFIED", "sketch", "sketch"
    else:
        rec = _receipt("stopped", spec["problem"], alpha, gates, "none", "STOP_HYGIENE")
        rec["module_pinnings"] = module_pinnings()
        rec["sha256"] = _sha_record(rec)
        return rec
    rec = _receipt(kind, spec["problem"], alpha, gates, tier, verdict)
    rec["lambda_hat"] = lam
    rec["mu_hat"] = mu_hat
    rec["module_pinnings"] = module_pinnings()
    rec["sha256"] = _sha_record(rec)
    return rec


def receipt_ok(rec):
    """Fail-closed receipt self-check."""
    if not isinstance(rec, dict) or "sha256" not in rec or rec.get("schema") != SCHEMA:
        return False
    return _sha_record(rec) == rec["sha256"]


def ledger_append(ledger, receipt, prev_entry=None):
    """Append a certified-spend ledger entry linked kit-style (prev = sha256(prev full + newline))."""
    if not receipt_ok(receipt):
        raise ValueError("refusing to append an unverifiable receipt")
    entry = {"receipt": receipt, "kind": "certified_spend"}
    if prev_entry is None:
        entry["prev"] = "0" * 64
    else:
        pb = {k: v for k, v in prev_entry.items() if k != "sha256"}
        if hashlib.sha256(json.dumps(pb, sort_keys=True).encode("utf-8")).hexdigest() != prev_entry.get("sha256"):
            raise ValueError("previous ledger entry fails verification")
        entry["prev"] = hashlib.sha256(
            (json.dumps(prev_entry, sort_keys=True) + "\n").encode("utf-8")).hexdigest()
    entry["sha256"] = hashlib.sha256(
        json.dumps({k: v for k, v in entry.items() if k != "sha256"}, sort_keys=True).encode("utf-8")).hexdigest()
    ledger.append(entry)
    return entry


def ledger_ok(entries):
    """Verify a certified-spend ledger chain (genesis 64-zeros, kit prev formula)."""
    prev_digest = None
    for i, e in enumerate(entries):
        if not isinstance(e, dict) or "sha256" not in e or "receipt" not in e:
            return False
        body = {k: v for k, v in e.items() if k != "sha256"}
        if hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest() != e["sha256"]:
            return False
        expect_prev = "0" * 64 if i == 0 else prev_digest
        if e.get("prev") != expect_prev:
            return False
        prev_digest = hashlib.sha256((json.dumps(e, sort_keys=True) + "\n").encode("utf-8")).hexdigest()
        if not receipt_ok(e["receipt"]):
            return False
    return True
