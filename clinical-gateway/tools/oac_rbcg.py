#!/usr/bin/env python3
"""Receipt-bound missed-alert guarantee for oac-system-health-v1 (stdlib). Commands: issue | verify RECEIPT DEPLOY.jsonl | selftest.
A receipt certifies, via Learn-then-Test on the calibration split, that P(missed-alert rate > alpha) <= delta at `threshold`.
verify() refuses (fail-closed) on: any bound-file hash change, a certificate that does not recompute, or a calibration->deployment
score shift (two-sample KS permutation test, p < shift_alpha). Synthetic operations telemetry only."""
import hashlib, json, math, os, random, shutil, sys
F = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # repo root; this file lives at clinical-gateway/tools/
MD = f"{F}/clinical-gateway/huggingface/model/oac-system-health-v1"; DD = f"{F}/ops-health/v2/data/legacy_v1"
sys.path.insert(0, MD)
from oac_operational_health import OperationalHealthKernel
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
def binom_cdf(k, n, p): return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))
def kernel(md): return OperationalHealthKernel(f"{md}/model.json", f"{md}/artifact_receipt.json")
def scored(K, path):
    out = []
    for l in open(path):
        if l.strip():
            r = json.loads(l); out.append((K.score(r["features"])["operator_attention_score"], int(bool(r["label"]["operator_attention_required"]))))
    return out
def ks(a, b):
    A, B = sorted(a), sorted(b); i = j = 0; m = 0.0
    for x in sorted(set(a) | set(b)):
        while i < len(A) and A[i] <= x: i += 1
        while j < len(B) and B[j] <= x: j += 1
        m = max(m, abs(i / len(A) - j / len(B)))
    return m
def shift_p(a, b, B=2000, seed=11):
    rng = random.Random(seed); obs = ks(a, b); pool = a + b; c = 0
    for _ in range(B):
        rng.shuffle(pool); c += ks(pool[:len(a)], pool[len(a):]) >= obs - 1e-12
    return round(obs, 6), (c + 1) / (B + 1)
def ltt(cal, alpha, delta):
    pos = [s for s, y in cal if y]; best = None
    for t in sorted({s for s, _ in cal}):
        k = sum(1 for s in pos if s < t); p = binom_cdf(k, len(pos), alpha)
        if p <= delta: best = (t, k, len(pos), p)
        else: break
    return best
def issue(md=MD, cal_path=f"{DD}/validation.jsonl", alpha=0.20, delta=0.10, shift_alpha=0.05):
    K = kernel(md); cal = scored(K, cal_path); t, k, n1, p = ltt(cal, alpha, delta)
    files = {n: sha(f"{md}/{n}") for n in ("model.json", "artifact_receipt.json", "oac_operational_health.py")}
    files["calibration.jsonl"] = sha(cal_path)
    body = {"schema": "szl-oac/receipt-bound-missed-alert-guarantee/v0", "files": files, "alpha": alpha, "delta": delta,
            "threshold": t, "cal_misses": k, "cal_positives": n1, "p_value": p, "shift_test": "two-sample KS on model scores, permutation B=2000 seed 11",
            "shift_alpha": shift_alpha, "claim": f"P(missed-alert rate > {alpha}) <= {delta} for deployment batches exchangeable with calibration"}
    body["receipt_sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    return body
def verify(rc, deploy_path, md=MD, cal_path=f"{DD}/validation.jsonl"):
    b = {k: v for k, v in rc.items() if k != "receipt_sha256"}
    if hashlib.sha256(json.dumps(b, sort_keys=True).encode()).hexdigest() != rc.get("receipt_sha256"): return "REFUSE:RECEIPT_TAMPERED"
    now = {n: sha(f"{md}/{n}") for n in ("model.json", "artifact_receipt.json", "oac_operational_health.py")}
    now["calibration.jsonl"] = sha(cal_path)
    if now != rc["files"]: return "REFUSE:HASH"
    K = kernel(md); cal = scored(K, cal_path); pos = [s for s, y in cal if y]
    k = sum(1 for s in pos if s < rc["threshold"]); p = binom_cdf(k, len(pos), rc["alpha"])
    if k != rc["cal_misses"] or p > rc["delta"]: return "REFUSE:CERT"
    dep = [s for s, _ in scored(K, deploy_path)]
    if shift_p([s for s, _ in cal], dep)[1] < rc["shift_alpha"]: return "REFUSE:SHIFT"
    return "ACCEPT"
def selftest():
    W = os.path.abspath("_rb_fx"); shutil.rmtree(W, ignore_errors=True); os.makedirs(W)
    ok = {}
    try:
        rc = issue(); ok["issue_reproduces_ltt_0.2"] = abs(rc["threshold"] - 0.137721) < 1e-6 and rc["cal_misses"] == 3
        ok["kat_binom"] = abs(binom_cdf(0, 33, 0.1) - 0.9 ** 33) < 1e-15 and abs(binom_cdf(1, 33, 0.1) - (0.9 ** 33 + 33 * 0.1 * 0.9 ** 32)) < 1e-15
        ok["accept_exchangeable"] = verify(rc, f"{DD}/validation.jsonl") == "ACCEPT"           # deployment == calibration
        ok["refuse_shifted_test"] = verify(rc, f"{DD}/test.jsonl") == "REFUSE:SHIFT"            # real val->test shift (p~0.03)
        # tamper calibration data: flip one label in a copy
        lines = open(f"{DD}/validation.jsonl").read().splitlines(True); r0 = json.loads(lines[0])
        r0["label"]["operator_attention_required"] = not r0["label"]["operator_attention_required"]; lines[0] = json.dumps(r0) + "\n"
        open(f"{W}/cal.jsonl", "w").writelines(lines)
        ok["refuse_tampered_calibration"] = verify(rc, f"{DD}/validation.jsonl", cal_path=f"{W}/cal.jsonl") == "REFUSE:HASH"
        bad = dict(rc); bad["threshold"] = 0.2; ok["refuse_edited_threshold"] = verify(bad, f"{DD}/validation.jsonl") == "REFUSE:RECEIPT_TAMPERED"
        bad["receipt_sha256"] = hashlib.sha256(json.dumps({k: v for k, v in bad.items() if k != "receipt_sha256"}, sort_keys=True).encode()).hexdigest()
        ok["refuse_resealed_uncertifiable_threshold"] = verify(bad, f"{DD}/validation.jsonl") == "REFUSE:CERT"
        # internally consistent forgery: true miss count at an uncertifiable threshold, resealed -> only the delta check can refuse
        Kc = kernel(MD); calp = [x for x, y in scored(Kc, f"{DD}/validation.jsonl") if y]
        forg = dict(rc); forg["threshold"] = 0.2; forg["cal_misses"] = sum(1 for x in calp if x < 0.2)
        forg["p_value"] = binom_cdf(forg["cal_misses"], len(calp), forg["alpha"])
        forg["receipt_sha256"] = hashlib.sha256(json.dumps({k: v for k, v in forg.items() if k != "receipt_sha256"}, sort_keys=True).encode()).hexdigest()
        ok["refuse_consistent_forgery_failing_delta"] = forg["p_value"] > rc["delta"] and verify(forg, f"{DD}/validation.jsonl") == "REFUSE:CERT"
        # synthetic shift: deployment = top-scored half of test
        Kx = kernel(MD); rows = [l for l in open(f"{DD}/test.jsonl") if l.strip()]
        rows.sort(key=lambda l: -Kx.score(json.loads(l)["features"])["operator_attention_score"])
        open(f"{W}/hot.jsonl", "w").writelines(rows[:120])
        ok["refuse_strong_shift"] = verify(rc, f"{W}/hot.jsonl") == "REFUSE:SHIFT"
    finally:
        shutil.rmtree(W, ignore_errors=True)
    res = {"cases": ok, "result": "PASS" if all(ok.values()) else "FAIL"}
    print(json.dumps(res)); return 0 if res["result"] == "PASS" else 3
if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if cmd == "issue":
        rc = issue(); json.dump(rc, open("oac_rbcg_receipt.json", "w"), indent=1); print(json.dumps({"threshold": rc["threshold"], "receipt": rc["receipt_sha256"]}))
    elif cmd == "verify":
        print(verify(json.load(open(sys.argv[2])), sys.argv[3]))
    else:
        sys.exit(selftest())
