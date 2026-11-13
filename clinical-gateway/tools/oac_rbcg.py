#!/usr/bin/env python3
"""Receipt-bound certified gate (RBCG) for oac-system-health-v1 (synthetic only).

Standard library only, self-contained (does not import oac_ltt.py, so the
verifier is an independent recomputation).

  issue     bind sha256 of model.json, artifact_receipt.json, the kernel and the
            calibration split, plus alpha/delta/threshold and a recomputable
            Learn-then-Test missed-alert (FNR) certificate, into a receipt JSON.
  verify    ACCEPT, or REFUSE:<REASON> for: RECEIPT_DIGEST, POLICY,
            HASH_MISMATCH:<file>, KERNEL, CERTIFICATE, DELTA,
            INSUFFICIENT_DEPLOYMENT, SHIFT (calibration-vs-deployment score
            distribution: KS, Mann-Whitney, alert-rate Fisher; Bonferroni x3).
  selftest  known-answer and forgery cases; exit 0 iff all pass, else 3.
  mutants   apply deliberate code mutants to a copy of this file and require
            each mutant's selftest to fail.

The receipt digest is a sha256 over canonical JSON. It is an integrity check,
NOT a signature: anyone who can edit the receipt can recompute it. Security
against forgery comes from recomputation (hashes, certificate, delta, policy),
not from the digest.

Paths default to locations relative to this file; no absolute paths are
hard-coded.
"""
from __future__ import annotations

import argparse
import bisect
import copy
import hashlib
import importlib.util
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = HERE.parent / "huggingface" / "model" / "oac-system-health-v1"
DEFAULT_DATA_DIR = (
    HERE.parent / "huggingface" / "dataset" / "oac-clinical-transport-observability-synthetic" / "data"
)
RECEIPT_SCHEMA = "szl-oac/rbcg-receipt/v1"
MODEL_FILES = ("model.json", "artifact_receipt.json", "oac_operational_health.py")
GRID_STEP = 0.0005
GRID = [round(i * GRID_STEP, 4) for i in range(int(round(1 / GRID_STEP)) + 1)]
POLICY_MAX_ALPHA = 0.20
POLICY_MAX_DELTA = 0.10
POLICY_SHIFT_LEVEL = 0.10
POLICY_MIN_DEPLOY = 30
N_SHIFT_TESTS = 3
SCORE_KEY = "operator_attention_score"
LABEL_KEY = "operator_attention_required"


class Refuse(Exception):
    pass


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    return sha256_bytes(Path(p).read_bytes())


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def receipt_digest(body: dict) -> str:
    b = {k: v for k, v in body.items() if k != "receipt_sha256"}
    return sha256_bytes(canonical(b))


# ----------------------------------------------------------------- stats
def binom_cdf(k, n, p):
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return min(1.0, sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1)))


def misses_at(pos_sorted, lam):
    return bisect.bisect_left(pos_sorted, lam)


def ltt_certify(pos_scores, alpha, delta):
    ps = sorted(pos_scores)
    n = len(ps)
    cert, steps, first_fail = None, 0, None
    for lam in GRID:
        k = misses_at(ps, lam)
        p = binom_cdf(k, n, alpha)
        steps += 1
        if p <= delta:
            cert = {"lambda": lam, "misses": k, "p_value": p}
        else:
            first_fail = {"lambda": lam, "misses": k, "p_value": p}
            break
    return cert, first_fail, steps


def ks_2samp(a, b):
    a, b = sorted(a), sorted(b)
    n, m = len(a), len(b)
    d = 0.0
    for x in sorted(set(a) | set(b)):
        fa = bisect.bisect_right(a, x) / n
        fb = bisect.bisect_right(b, x) / m
        d = max(d, abs(fa - fb))
    en = math.sqrt(n * m / (n + m))
    lam = (en + 0.12 + 0.11 / en) * d
    if lam < 1e-3:
        return d, 1.0
    s = 0.0
    for j in range(1, 101):
        s += 2 * (-1) ** (j - 1) * math.exp(-2 * j * j * lam * lam)
    return d, max(0.0, min(1.0, s))


def mann_whitney(a, b):
    n, m = len(a), len(b)
    allv = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    ranks = [0.0] * len(allv)
    ties = 0.0
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        t = j - i + 1
        ties += t**3 - t
        for k in range(i, j + 1):
            ranks[k] = (i + j) / 2 + 1
        i = j + 1
    ra = sum(r for r, (_, g) in zip(ranks, allv) if g == 0)
    u = ra - n * (n + 1) / 2
    N = n + m
    var = n * m / 12 * ((N + 1) - ties / (N * (N - 1)))
    if var <= 0:
        return u, 1.0
    z = (u - n * m / 2) / math.sqrt(var)
    return u, math.erfc(abs(z) / math.sqrt(2))


def fisher_two_sided(x1, n1, x2, n2):
    k, N = x1 + x2, n1 + n2

    def pmf(x):
        return math.comb(n1, x) * math.comb(n2, k - x) / math.comb(N, k)

    lo, hi = max(0, k - n2), min(n1, k)
    p_obs = pmf(x1)
    return min(1.0, sum(pmf(x) for x in range(lo, hi + 1) if pmf(x) <= p_obs * (1 + 1e-9)))


def shift_tests(cal, dep, lam):
    d, p_ks = ks_2samp(cal, dep)
    u, p_mw = mann_whitney(cal, dep)
    xa = sum(1 for s in cal if s >= lam)
    xb = sum(1 for s in dep if s >= lam)
    p_f = fisher_two_sided(xa, len(cal), xb, len(dep))
    pmin = min(p_ks, p_mw, p_f)
    return {
        "ks": {"D": round(d, 12), "p": p_ks},
        "mann_whitney": {"U": u, "p": p_mw},
        "alert_rate_fisher": {"cal_rate": xa / len(cal), "dep_rate": xb / len(dep), "p": p_f},
        "p_min": pmin,
        "p_bonferroni": min(1.0, N_SHIFT_TESTS * pmin),
    }


# ----------------------------------------------------------------- model io
def load_kernel(model_dir: Path):
    kpath = Path(model_dir) / "oac_operational_health.py"
    name = "oac_kernel_rbcg_" + sha256_file(kpath)[:12]
    spec = importlib.util.spec_from_file_location(name, kpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod.OperationalHealthKernel(Path(model_dir) / "model.json", Path(model_dir) / "artifact_receipt.json")


def read_jsonl(path: Path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def score_all(kernel, rows):
    return [float(kernel.score(r["features"])[SCORE_KEY]) for r in rows]


def score_digest(scores):
    return sha256_bytes(canonical([round(s, 12) for s in scores]))


# ----------------------------------------------------------------- issue
def issue(model_dir: Path, calibration: Path, alpha=0.20, delta=0.10):
    kernel = load_kernel(model_dir)
    rows = read_jsonl(calibration)
    scores = score_all(kernel, rows)
    labels = [bool(r["label"][LABEL_KEY]) for r in rows]
    pos = [s for s, y in zip(scores, labels) if y]
    cert, first_fail, steps = ltt_certify(pos, alpha, delta)
    if cert is None:
        raise SystemExit("no threshold certified at this alpha/delta")
    body = {
        "schema": RECEIPT_SCHEMA,
        "authority": "advisory_only_no_ack_release_or_clinical_authority",
        "status": "research_only_synthetic_not_qualification",
        "bound_files": {f: sha256_file(Path(model_dir) / f) for f in MODEL_FILES},
        "calibration": {"name": Path(calibration).name, "sha256": sha256_file(calibration),
                        "rows": len(rows), "positives": len(pos), "score_digest": score_digest(scores)},
        "policy": {"alpha": alpha, "delta": delta, "pvalue": "binomial_exact", "risk": "missed_alert_rate_fnr",
                   "procedure": "learn_then_test_fixed_sequence_ascending",
                   "grid": {"start": 0.0, "step": GRID_STEP, "points": len(GRID)},
                   "alert_rule": "alert iff operator_attention_score >= threshold",
                   "shift": {"tests": ["ks", "mann_whitney", "alert_rate_fisher"], "correction": "bonferroni",
                             "level": POLICY_SHIFT_LEVEL, "min_deployment_rows": POLICY_MIN_DEPLOY}},
        "threshold": cert["lambda"],
        "certificate": {"n_pos": len(pos), "misses": cert["misses"], "p_value": cert["p_value"],
                        "steps": steps, "first_not_rejected": first_fail},
        "issuer_sha256_informational": sha256_file(Path(__file__)),
    }
    body["receipt_sha256"] = receipt_digest(body)
    return body


# ----------------------------------------------------------------- verify
def _verify(receipt: dict, model_dir: Path, calibration: Path, deployment_rows):
    if not isinstance(receipt, dict) or receipt.get("schema") != RECEIPT_SCHEMA:
        raise Refuse("RECEIPT_SCHEMA")
    if receipt.get("receipt_sha256") != receipt_digest(receipt):
        raise Refuse("RECEIPT_DIGEST")
    pol = receipt["policy"]
    if not (0 < pol["alpha"] <= POLICY_MAX_ALPHA and 0 < pol["delta"] <= POLICY_MAX_DELTA
            and pol["pvalue"] == "binomial_exact" and pol["grid"]["step"] == GRID_STEP
            and pol["shift"]["level"] <= POLICY_SHIFT_LEVEL
            and pol["shift"]["min_deployment_rows"] >= POLICY_MIN_DEPLOY
            and pol["shift"]["correction"] == "bonferroni"):
        raise Refuse("POLICY")
    for f in MODEL_FILES:
        if sha256_file(Path(model_dir) / f) != receipt["bound_files"][f]:
            raise Refuse(f"HASH_MISMATCH:{f}")
    if sha256_file(calibration) != receipt["calibration"]["sha256"]:
        raise Refuse("HASH_MISMATCH:calibration")
    try:
        kernel = load_kernel(model_dir)
        rows = read_jsonl(calibration)
        cal_scores = score_all(kernel, rows)
        dep_scores = score_all(kernel, deployment_rows)
    except Exception as exc:  # kernel refuses its own receipt or an input
        raise Refuse(f"KERNEL:{type(exc).__name__}")
    if score_digest(cal_scores) != receipt["calibration"]["score_digest"]:
        raise Refuse("CERTIFICATE")
    labels = [bool(r["label"][LABEL_KEY]) for r in rows]
    pos = sorted(s for s, y in zip(cal_scores, labels) if y)
    lam = receipt["threshold"]
    c = receipt["certificate"]
    k = misses_at(pos, lam)
    p = binom_cdf(k, len(pos), pol["alpha"])
    if c["n_pos"] != len(pos) or c["misses"] != k or abs(c["p_value"] - p) > 1e-12:
        raise Refuse("CERTIFICATE")
    if lam not in GRID:
        raise Refuse("CERTIFICATE")
    # fixed-sequence validity: every grid point up to and including lambda must reject at delta
    for g in GRID:
        if g > lam:
            break
        if binom_cdf(misses_at(pos, g), len(pos), pol["alpha"]) > pol["delta"]:
            raise Refuse("DELTA")
    if len(dep_scores) < pol["shift"]["min_deployment_rows"]:
        raise Refuse("INSUFFICIENT_DEPLOYMENT")
    st = shift_tests(cal_scores, dep_scores, lam)
    if st["p_bonferroni"] < pol["shift"]["level"]:
        raise Refuse("SHIFT")
    return st


def verify(receipt, model_dir, calibration, deployment_rows):
    try:
        st = _verify(receipt, model_dir, calibration, deployment_rows)
        return "ACCEPT", st
    except Refuse as r:
        return f"REFUSE:{r}", None
    except (KeyError, TypeError, ValueError) as exc:
        return f"REFUSE:MALFORMED:{type(exc).__name__}", None


def verify_detail(receipt, model_dir, calibration, deployment_rows):
    """Verdict plus the shift statistics even when the verdict is a refusal."""
    verdict, st = verify(receipt, model_dir, calibration, deployment_rows)
    if st is None and verdict == "REFUSE:SHIFT":
        k = load_kernel(model_dir)
        st = shift_tests(score_all(k, read_jsonl(calibration)), score_all(k, deployment_rows), receipt["threshold"])
    return verdict, st


# ----------------------------------------------------------------- selftest
def _reseal(r):
    r = copy.deepcopy(r)
    r["receipt_sha256"] = receipt_digest(r)
    return r


def _consistent_forgery(r, model_dir, calibration, lam, delta=None):
    f = copy.deepcopy(r)
    if delta is not None:
        f["policy"]["delta"] = delta
    k = load_kernel(model_dir)
    rows = read_jsonl(calibration)
    sc = score_all(k, rows)
    pos = sorted(s for s, row in zip(sc, rows) if row["label"][LABEL_KEY])
    m = misses_at(pos, lam)
    f["threshold"] = lam
    f["certificate"].update({"misses": m, "p_value": binom_cdf(m, len(pos), f["policy"]["alpha"])})
    return _reseal(f)


def selftest(model_dir: Path, data_dir: Path):
    model_dir, data_dir = Path(model_dir).resolve(), Path(data_dir).resolve()
    cal = data_dir / "validation.jsonl"
    val_rows = read_jsonl(cal)
    results = []

    def case(name, got, want):
        results.append({"case": name, "expected": want, "got": got, "pass": got == want})

    with tempfile.TemporaryDirectory(prefix="rbcg-selftest-") as td:
        td = Path(td)
        r = issue(model_dir, cal)
        case("01_genuine_receipt_validation_as_deployment", verify(r, model_dir, cal, val_rows)[0], "ACCEPT")
        c = r["certificate"]
        case("02_known_answer_certificate_pin",
             [r["threshold"], c["n_pos"], c["misses"], round(c["p_value"], 12)],
             [0.1375, 33, 3, 0.080812725765])
        e = copy.deepcopy(r)
        e["policy"]["alpha"] = 0.19
        case("03_receipt_edited_without_reseal", verify(e, model_dir, cal, val_rows)[0], "REFUSE:RECEIPT_DIGEST")
        md2 = td / "model_tampered"
        shutil.copytree(model_dir, md2)
        (md2 / "model.json").write_bytes((md2 / "model.json").read_bytes() + b"\n")
        case("04_model_json_tampered", verify(r, md2, cal, val_rows)[0], "REFUSE:HASH_MISMATCH:model.json")
        md3 = td / "kernel_tampered"
        shutil.copytree(model_dir, md3)
        (md3 / "oac_operational_health.py").write_bytes((md3 / "oac_operational_health.py").read_bytes() + b"# x\n")
        case("05_kernel_tampered", verify(r, md3, cal, val_rows)[0], "REFUSE:HASH_MISMATCH:oac_operational_health.py")
        cal2 = td / "validation.jsonl"
        cal2.write_text("".join(json.dumps(x) + "\n" for x in val_rows[1:]))
        case("06_calibration_tampered_row_dropped", verify(r, model_dir, cal2, val_rows)[0], "REFUSE:HASH_MISMATCH:calibration")
        t = copy.deepcopy(r)
        t["threshold"] = 0.145
        case("07_threshold_raised_resealed_certificate_stale", verify(_reseal(t), model_dir, cal, val_rows)[0], "REFUSE:CERTIFICATE")
        fg = _consistent_forgery(r, model_dir, cal, 0.16)
        case("08_consistent_forgery_at_card_threshold_fails_delta", verify(fg, model_dir, cal, val_rows)[0], "REFUSE:DELTA")
        fp = _consistent_forgery(r, model_dir, cal, 0.16, delta=0.9)
        case("09_consistent_forgery_with_relaxed_delta_fails_policy", verify(fp, model_dir, cal, val_rows)[0], "REFUSE:POLICY")
        shifted = copy.deepcopy(val_rows)
        for row in shifted:
            row["features"]["consecutive_failures"] = min(20, row["features"]["consecutive_failures"] + 5)
        case("10_shifted_deployment", verify(r, model_dir, cal, shifted)[0], "REFUSE:SHIFT")
        case("11_tiny_deployment", verify(r, model_dir, cal, val_rows[:10])[0], "REFUSE:INSUFFICIENT_DEPLOYMENT")
        test_rows = read_jsonl(data_dir / "test.jsonl")
        _, st = verify_detail(r, model_dir, cal, test_rows)
        case("12_bonferroni_arithmetic_on_test_split",
             round(st["p_bonferroni"], 12), round(min(1.0, 3 * min(st["ks"]["p"], st["mann_whitney"]["p"], st["alert_rate_fisher"]["p"])), 12))
        case("13_ks_known_answer", [round(x, 6) for x in ks_2samp([1, 2, 3, 4], [5, 6, 7, 8])][:1] +
             [round(ks_2samp([1, 2, 3], [1, 2, 3])[0], 6)], [1.0, 0.0])
        case("14_issue_is_deterministic", canonical(issue(model_dir, cal)) == canonical(r), True)
    ok = all(x["pass"] for x in results)
    return ok, results


# ----------------------------------------------------------------- mutants
MUTANTS = [
    ("M1_skip_receipt_digest", 'if receipt.get("receipt_sha256") != receipt_digest(receipt):',
     'if False and receipt.get("receipt_sha256") != receipt_digest(receipt):'),
    ("M2_skip_model_file_hashes", 'if sha256_file(Path(model_dir) / f) != receipt["bound_files"][f]:',
     'if False and sha256_file(Path(model_dir) / f) != receipt["bound_files"][f]:'),
    ("M3_skip_calibration_hash", 'if sha256_file(calibration) != receipt["calibration"]["sha256"]:',
     'if False and sha256_file(calibration) != receipt["calibration"]["sha256"]:'),
    ("M4_delta_check_disabled", 'pol["alpha"]) > pol["delta"]:', 'pol["alpha"]) > 1.0:'),
    ("M5_policy_delta_bound_disabled", '0 < pol["delta"] <= POLICY_MAX_DELTA', '0 < pol["delta"] <= 1.0'),
    ("M6_no_bonferroni", '"p_bonferroni": min(1.0, N_SHIFT_TESTS * pmin)', '"p_bonferroni": min(1.0, pmin)'),
    ("M7_binomial_cdf_off_by_one", "(1 - p) ** (n - i) for i in range(k + 1)))", "(1 - p) ** (n - i) for i in range(k)))"),
    ("M8_min_deployment_rows_ignored", 'if len(dep_scores) < pol["shift"]["min_deployment_rows"]:',
     'if len(dep_scores) < 0:'),
]


def run_mutants(model_dir: Path, data_dir: Path):
    src = Path(__file__).read_text(encoding="utf-8")
    marker = "# " + "-" * 65 + " mutants"
    head, tail = src.split(marker, 1)  # mutate only code above the mutant table
    out = []
    with tempfile.TemporaryDirectory(prefix="rbcg-mutants-") as td:
        for name, old, new in [("M0_control_unmutated", "", "")] + MUTANTS:
            if old:
                n = head.count(old)
                if n != 1:
                    out.append({"mutant": name, "error": f"pattern found {n} times", "caught": False})
                    continue
                msrc = head.replace(old, new) + marker + tail
            else:
                msrc = src
            mp = Path(td) / f"{name}.py"
            mp.write_text(msrc, encoding="utf-8")
            pr = subprocess.run([sys.executable, "-B", str(mp), "selftest", "--model-dir", str(model_dir),
                                 "--data-dir", str(data_dir)], capture_output=True, text=True, timeout=600)
            failed = []
            try:
                failed = [c["case"] for c in json.loads(pr.stdout)["cases"] if not c["pass"]]
            except Exception:
                failed = ["<unparseable selftest output>"]
            caught = pr.returncode != 0
            out.append({"mutant": name, "exit_code": pr.returncode,
                        "caught": caught if old else None, "control_passed": (pr.returncode == 0) if not old else None,
                        "failing_cases": failed})
    control_ok = out[0].get("control_passed") is True
    caught = sum(1 for o in out[1:] if o.get("caught"))
    return control_ok and caught == len(MUTANTS), {"control_passed": control_ok, "caught": caught,
                                                   "total": len(MUTANTS), "mutants": out}


# ----------------------------------------------------------------- cli
def main(argv=None):
    ap = argparse.ArgumentParser(description="Receipt-bound certified gate for oac-system-health-v1")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("issue", "verify", "selftest", "mutants"):
        p = sub.add_parser(name)
        p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
        p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
        p.add_argument("--output", type=Path)
        if name == "issue":
            p.add_argument("--calibration", type=Path)
            p.add_argument("--alpha", type=float, default=0.20)
            p.add_argument("--delta", type=float, default=0.10)
        if name == "verify":
            p.add_argument("--receipt", type=Path, required=True)
            p.add_argument("--calibration", type=Path)
            p.add_argument("--deployment", type=Path, required=True, help="JSONL of rows with a features object")
    a = ap.parse_args(argv)
    md, dd = a.model_dir.resolve(), a.data_dir.resolve()

    def emit(obj):
        txt = json.dumps(obj, indent=2, sort_keys=True) + "\n"
        if getattr(a, "output", None):
            a.output.write_text(txt)
        sys.stdout.write(txt)

    if a.cmd == "issue":
        emit(issue(md, (a.calibration or dd / "validation.jsonl").resolve(), a.alpha, a.delta))
        return 0
    if a.cmd == "verify":
        receipt = json.loads(a.receipt.read_text())
        verdict, st = verify_detail(receipt, md, (a.calibration or dd / "validation.jsonl").resolve(),
                                    read_jsonl(a.deployment))
        emit({"verdict": verdict, "deployment": a.deployment.name, "shift_tests": st})
        return 0 if verdict == "ACCEPT" else 1
    if a.cmd == "selftest":
        ok, res = selftest(md, dd)
        emit({"selftest_passed": ok, "passed": sum(r["pass"] for r in res), "total": len(res), "cases": res})
        return 0 if ok else 3
    if a.cmd == "mutants":
        ok, res = run_mutants(md, dd)
        emit({"all_caught": ok, **res})
        return 0 if ok else 5
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
