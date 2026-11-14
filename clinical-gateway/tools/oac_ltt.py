#!/usr/bin/env python3
"""FX-2 + RBCG on oac-system-health-v1 (szl-forge main model files). Stdlib only. Synthetic ops telemetry only."""
import hashlib, json, math, os, random, sys
F = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # repo root; this file lives at clinical-gateway/tools/
MD = f"{F}/clinical-gateway/huggingface/model/oac-system-health-v1"; DD = f"{F}/ops-health/v2/data/legacy_v1"
sys.path.insert(0, MD)
from oac_operational_health import OperationalHealthKernel
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
K = OperationalHealthKernel(f"{MD}/model.json", f"{MD}/artifact_receipt.json")
def load(s):
    rows = [json.loads(l) for l in open(f"{DD}/{s}.jsonl") if l.strip()]
    return [(K.score(r["features"])["operator_attention_score"], int(bool(r["label"]["operator_attention_required"])), r["features"]) for r in rows]
VAL, TEST = load("validation"), load("test")
def auc(d):
    pos = [s for s, y, _ in d if y]; neg = [s for s, y, _ in d if not y]
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))
def conf(d, t, key=None):
    tp = fp = tn = fn = 0
    for s, y, f in d:
        a = (f["consecutive_failures"] > 0) if key == "rule" else (False if key == "majority" else s >= t)
        tp += a and y; fp += a and not y; tn += (not a) and not y; fn += (not a) and y
    n1 = tp + fn
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn, "accuracy": round((tp + tn) / len(d), 4),
            "missed_alert_rate": round(fn / n1, 4) if n1 else None, "alert_rate": round((tp + fp) / len(d), 4),
            "alert_precision": round(tp / (tp + fp), 4) if tp + fp else None}
def binom_cdf(k, n, p):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(0, k + 1))
def ltt_fnr(d, alpha, delta):
    """Fixed-sequence LTT over thresholds from low (FNR~0) to high; certify H0: FNR(t)>alpha rejected while p<=delta.
    Returns the HIGHEST certified threshold (fewest alerts), or None (refuse)."""
    pos = sorted(s for s, y, _ in d if y); n1 = len(pos)
    cands = sorted({s for s, _, _ in d})
    best = None
    for t in cands:
        k = sum(1 for s in pos if s < t)
        p = binom_cdf(k, n1, alpha)
        if p <= delta: best = (t, k, n1, round(p, 6))
        else: break
    return best
def ltt_select(d, alpha, delta, grid=24):
    """Abstention band (a<=b): predict 0 if s<a, 1 if s>=b, else defer. Risk = P(answer and wrong).
    Bonferroni over the band grid; among certified bands maximise coverage."""
    qs = sorted(s for s, _, _ in d); n = len(d)
    cut = sorted({qs[min(n - 1, int(i * n / grid))] for i in range(grid + 1)})
    bands = [(a, b) for i, a in enumerate(cut) for b in cut[i:]]
    dl = delta / len(bands); best = None
    for a, b in bands:
        err = sum(1 for s, y, _ in d if (s < a and y) or (s >= b and not y))
        cov = sum(1 for s, _, _ in d if s < a or s >= b) / n
        if binom_cdf(err, n, alpha) <= dl and (best is None or cov > best["coverage"]):
            best = {"a": a, "b": b, "coverage": round(cov, 4), "errors": err, "n": n, "bands_tested": len(bands)}
    return best
def band_eval(d, a, b):
    ans = [(s, y) for s, y, _ in d if s < a or s >= b]
    wrong = sum(1 for s, y in ans if (s < a and y) or (s >= b and not y))
    return {"coverage": round(len(ans) / len(d), 4), "answered": len(ans), "wrong": wrong,
            "joint_risk": round(wrong / len(d), 4), "selective_accuracy": round(1 - wrong / len(ans), 4) if ans else None}
def ks(a, b):
    xs = sorted(set(a) | set(b)); A, B = sorted(a), sorted(b); i = j = 0; m = 0.0
    for x in xs:
        while i < len(A) and A[i] <= x: i += 1
        while j < len(B) and B[j] <= x: j += 1
        m = max(m, abs(i / len(A) - j / len(B)))
    return m
def perm_p(a, b, stat, B=2000, seed=11):
    rng = random.Random(seed); obs = stat(a, b); pool = a + b; na = len(a); c = 0
    for _ in range(B):
        rng.shuffle(pool); c += stat(pool[:na], pool[na:]) >= obs - 1e-12
    return round(obs, 4), round((c + 1) / (B + 1), 4)
out = {"files": {"model.json": sha(f"{MD}/model.json"), "artifact_receipt.json": sha(f"{MD}/artifact_receipt.json"),
                 "oac_operational_health.py": sha(f"{MD}/oac_operational_health.py"),
                 "validation.jsonl": sha(f"{DD}/validation.jsonl"), "test.jsonl": sha(f"{DD}/test.jsonl")}}
# 0. reproduction gate
c16 = conf(TEST, 0.16); out["gate"] = {"test_auc": round(auc(TEST), 6), "test_at_0.16": c16}
out["gate"]["reproduced"] = out["gate"]["test_auc"] == 0.830169 and (c16["TP"], c16["FP"], c16["TN"], c16["FN"]) == (40, 57, 132, 11)
if not out["gate"]["reproduced"]:
    json.dump(out, open("oac_ltt.json", "w"), indent=1); print("GATE FAILED"); sys.exit(4)
out["val"] = {"n": len(VAL), "positives": sum(y for _, y, _ in VAL), "auc": round(auc(VAL), 4)}
# 1. baselines on test
out["test_baselines"] = {"majority": conf(TEST, 0, "majority"), "rule_consecutive_failures_gt_0": conf(TEST, 0, "rule"),
                         "model_at_card_threshold_0.16": c16}
# 2. LTT missed-alert thresholds
out["ltt_fnr"] = {}
for alpha in (0.10, 0.20):
    r = ltt_fnr(VAL, alpha, 0.10)
    out["ltt_fnr"][str(alpha)] = {"certified": r is not None} | ({"threshold": round(r[0], 6), "val_misses": r[1], "val_positives": r[2],
                                   "p_value": r[3], "test": conf(TEST, r[0])} if r else {"verdict": "REFUSED: no threshold certifiable"})
# 3. abstention band
sb = ltt_select(VAL, 0.10, 0.10)
out["abstention_alpha0.10"] = ({"band": sb, "test": band_eval(TEST, sb["a"], sb["b"])} if sb else {"verdict": "REFUSED"})
# 4. validity simulation (pool val+test as the population; resample calibration sets of size 192)
pool = VAL + TEST; ppos = sorted(s for s, y, _ in pool if y)
true_fnr = lambda t: sum(1 for s in ppos if s < t) / len(ppos)
rng = random.Random(2026); R = 2000; viol = {"ltt": 0, "naive": 0}; refused = 0
for _ in range(R):
    cal = [pool[rng.randrange(len(pool))] for _ in range(len(VAL))]
    r = ltt_fnr(cal, 0.20, 0.10)
    if r is None: refused += 1
    elif true_fnr(r[0]) > 0.20: viol["ltt"] += 1
    pos = [s for s, y, _ in cal if y]; cands = sorted({s for s, _, _ in cal}); nt = None
    for t in cands:
        if sum(1 for s in pos if s < t) / len(pos) <= 0.20: nt = t
        else: break
    if nt is not None and true_fnr(nt) > 0.20: viol["naive"] += 1
def wci(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]
out["validity_sim"] = {"alpha": 0.20, "delta": 0.10, "reps": R, "ltt_violation_rate": round(viol["ltt"] / R, 4),
                       "ltt_violation_ci95": wci(viol["ltt"], R), "naive_violation_rate": round(viol["naive"] / R, 4),
                       "naive_violation_ci95": wci(viol["naive"], R), "ltt_refused": refused,
                       "note": "population = pooled val+test rows; calibration sets resampled with replacement"}
# 5. exchangeability val vs test (scores, label rate, consecutive_failures)
vs, ts = [s for s, _, _ in VAL], [s for s, _, _ in TEST]
out["shift"] = {"score_ks": perm_p(vs, ts, ks),
                "label_rate": perm_p([y for _, y, _ in VAL], [y for _, y, _ in TEST], lambda a, b: abs(sum(a) / len(a) - sum(b) / len(b))),
                "consecutive_failures_mean": perm_p([f["consecutive_failures"] for *_, f in VAL], [f["consecutive_failures"] for *_, f in TEST],
                                                    lambda a, b: abs(sum(a) / len(a) - sum(b) / len(b)))}
json.dump(out, open("oac_ltt.json", "w"), indent=1)
print(json.dumps({"gate": out["gate"]["reproduced"], "ltt": {k: v.get("threshold") for k, v in out["ltt_fnr"].items()},
                  "validity": out["validity_sim"], "shift": out["shift"]}))
