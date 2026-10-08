"""Stage-1 review, lens (b): independent numpy/scipy references for the v2 statistics.

SYNTHETIC; DEV data only (generator.dev_rows, split name "dev"; no registered split is read and
nothing under v2/data/sealed/ is touched).  Run with the Python 3.11 interpreter that has
numpy + scipy (`python`), from ROOT:

    PYTHONUTF8=1 python -B logs/v2/review/stats_reference.py

The v2 code under test is standard-library only; every reference below is computed with
numpy/scipy (and one sklearn cross-check) from the raw DEV feature dicts, not from v2 code.
Every value printed is a DEV value used only to compare implementations; none is a result.
"""

from __future__ import annotations

import math
import random
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import scipy
from scipy import optimize, stats

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from v2.research import (  # noqa: E402
    bootstrap, calibration, gates, lr, metrics, pipeline,
)
from v2.tests import _support as S  # noqa: E402

K = S.kernel()
NAMES = K.FEATURE_NAMES
SPECS = {name: (lo, hi, kind) for name, lo, hi, kind in K.FEATURE_SPECS}
RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def np_design(rows) -> np.ndarray:
    """Independent v1 normalization: binary as 0/1, continuous min-max over declared ranges."""
    X = np.empty((len(rows), len(NAMES)))
    for j, name in enumerate(NAMES):
        lo, hi, kind = SPECS[name]
        col = np.array([float(r["features"][name]) for r in rows])
        X[:, j] = col if kind == "binary" else (col - lo) / (hi - lo)
    return X


def np_labels(rows) -> np.ndarray:
    return np.array([1 if r["label"]["operator_attention_required"] else 0 for r in rows])


def logistic_ref(X: np.ndarray, y: np.ndarray, l2: float, penalize_first: bool = False):
    """scipy trust-exact on mean log-loss + l2/2 ||w||^2 (intercept unpenalized)."""
    A = np.hstack([np.ones((X.shape[0], 1)), X])
    n = A.shape[0]
    pen = np.full(A.shape[1], l2)
    pen[0] = l2 if penalize_first else 0.0

    def f(t):
        z = A @ t
        return np.mean(np.logaddexp(0.0, z) - y * z) + 0.5 * np.sum(pen * t * t)

    def g(t):
        p = 1.0 / (1.0 + np.exp(-(A @ t)))
        return A.T @ (p - y) / n + pen * t

    def h(t):
        p = 1.0 / (1.0 + np.exp(-(A @ t)))
        return (A.T * (p * (1 - p))) @ A / n + np.diag(pen)

    res = optimize.minimize(f, np.zeros(A.shape[1]), jac=g, hess=h, method="trust-exact",
                            options={"gtol": 1e-13, "maxiter": 1000})
    return res.x, res, float(np.max(np.abs(g(res.x))))


def main() -> int:
    t_start = time.perf_counter()
    print("# v2 statistics review (lens b) -- SYNTHETIC, DEV data only")
    print(f"python {sys.version.split()[0]}  numpy {np.__version__}  scipy {scipy.__version__}")
    dev = list(S.dev_rows())
    train, cal, conf_rows, thr, ev = (dev[S.TRAIN], dev[S.CALIBRATION], dev[S.CONFORMAL],
                                      dev[S.THRESHOLD], dev[S.EVAL])
    print(f"DEV rows: train {len(train)}, calibration {len(cal)}, conformal {len(conf_rows)}, "
          f"threshold {len(thr)}, eval {len(ev)}")

    # 0. normalization parity
    Xtr = np_design(train)
    v2_vecs = np.array(pipeline.normalized_vectors(train))
    check("normalization_parity", np.array_equal(Xtr, v2_vecs),
          f"max|diff| = {np.max(np.abs(Xtr - v2_vecs)):.3g} over {Xtr.size} values")
    ytr = np_labels(train)

    # 1. IRLS vs scipy
    for l2 in (0.0, 1e-4, 1e-3, 1e-2):
        fit = lr.fit(lr.columns_from_vectors(v2_vecs.tolist()), ytr.tolist(), l2)
        theta_v2 = np.array([fit["intercept"], *fit["weights"]])
        theta_ref, res, gref = logistic_ref(Xtr, ytr, l2)
        diff = float(np.max(np.abs(theta_v2 - theta_ref)))
        check(f"irls_vs_scipy_l2={l2:g}", diff < 1e-6,
              f"max|theta_v2 - theta_scipy| = {diff:.3e}; v2 iters {fit['iterations']} "
              f"{fit['stop_reason']}, v2 grad {fit['gradient_max_abs']:.2e}; scipy "
              f"{res.message!r} grad {gref:.2e}; objective v2 {fit['objective']:.15f} "
              f"scipy {res.fun:.15f}")

    # 2. Platt vs scipy
    art_none = S.candidate(calibrator="none")
    scorer_none = K.ArtifactScorer(art_none)
    Xcal = np_design(cal)
    ycal = np_labels(cal)
    raw_cal = [scorer_none.evaluate(v)["raw_score"] for v in pipeline.normalized_vectors(cal)]
    b0 = art_none["intercept"]
    w0 = np.array([art_none["weights"][n] for n in NAMES])
    raw_np = 1.0 / (1.0 + np.exp(-(b0 + Xcal @ w0)))
    check("raw_score_parity_numpy", np.max(np.abs(raw_np - np.array(raw_cal))) < 1e-13,
          f"max|raw_numpy - raw_kernel| = {np.max(np.abs(raw_np - np.array(raw_cal))):.3e}")
    platt, _diag = calibration.fit_platt(raw_cal, ycal.tolist())
    pc = np.clip(np.array(raw_cal), 1e-15, 1 - 1e-15)
    u = np.log(pc) - np.log1p(-pc)
    theta_ref, res, gref = logistic_ref(u.reshape(-1, 1), ycal, 0.0)
    ab_v2 = np.array([platt["params"]["b"], platt["params"]["a"]])
    diff = float(np.max(np.abs(ab_v2 - theta_ref)))
    check("platt_vs_scipy", diff < 1e-6,
          f"v2 (a,b)=({platt['params']['a']}, {platt['params']['b']}); scipy "
          f"(a,b)=({theta_ref[1]:.12f}, {theta_ref[0]:.12f}); max|diff| = {diff:.3e}")

    # 3. Isotonic vs numpy/scipy PAV with ties
    def iso_check(label, scores, ys):
        iso, diag = calibration.fit_isotonic(scores, ys)
        keys = np.round(np.array(scores, dtype=float), 12)
        uniq, inv = np.unique(keys, return_inverse=True)
        cnt = np.bincount(inv).astype(float)
        mean = np.bincount(inv, weights=np.array(ys, dtype=float)) / cnt
        ref = optimize.isotonic_regression(mean, weights=cnt, increasing=True).x
        # independent textbook PAV on the tie-pooled blocks
        blocks = []
        for m, c in zip(mean, cnt):
            blocks.append([m * c, c, 1])
            while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
                s2, c2, k2 = blocks.pop()
                blocks[-1][0] += s2
                blocks[-1][1] += c2
                blocks[-1][2] += k2
        pav = np.concatenate([np.full(k, s / c) for s, c, k in blocks])
        v2_vals = np.array([K.apply_calibrator(iso, float(x)) for x in uniq])
        d1 = float(np.max(np.abs(v2_vals - ref)))
        d2 = float(np.max(np.abs(v2_vals - pav)))
        check(f"isotonic_{label}", d1 <= 5e-13 and d2 <= 5e-13,
              f"{len(scores)} rows, {len(uniq)} tie blocks ({diag['tie_blocks']} in v2), "
              f"{diag['blocks']} fitted blocks; max|v2 - scipy| = {d1:.2e}, max|v2 - PAV| = {d2:.2e}")
        # step prediction between keys: value of the last block whose min key <= s
        probe = np.concatenate([uniq[:-1] + np.diff(uniq) / 2, [uniq[0] / 2, 1.0]])
        idx = np.searchsorted(uniq, np.round(probe, 12), side="right") - 1
        exp_vals = ref[np.maximum(idx, 0)]
        got = np.array([K.apply_calibrator(iso, float(x)) for x in probe])
        d3 = float(np.max(np.abs(got - exp_vals)))
        check(f"isotonic_step_prediction_{label}", d3 <= 5e-13,
              f"{len(probe)} between-key probes; max|diff| = {d3:.2e}")

    iso_check("dev_calibration_raw", raw_cal, ycal.tolist())
    iso_check("dev_rounded_2dp_heavy_ties", [round(s, 2) for s in raw_cal], ycal.tolist())
    iso_check("hand_ties", [0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.4, 0.4, 0.5],
              [1, 0, 0, 0, 1, 0, 1, 1, 0])

    # 4. ECE edges
    edges = np.array([k / 10 for k in range(11)])

    def ece_ref(y, s):
        s = np.asarray(s, dtype=float)
        b = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, 9)
        tot = 0.0
        for k in range(10):
            m = b == k
            if m.any():
                tot += abs(np.sum(s[m] - y[m]))
        return tot / len(s), b

    edge_scores = []
    for k in range(11):
        e = k / 10
        edge_scores += [e, float(np.nextafter(e, -1.0)), float(np.nextafter(e, 2.0))]
    edge_scores = [min(max(v, 0.0), 1.0) for v in edge_scores]
    ybin = [i % 2 for i in range(len(edge_scores))]
    ref_e, ref_b = ece_ref(np.array(ybin), edge_scores)
    v2_b = [metrics.ece_bin(v) for v in edge_scores]
    check("ece_bin_edges", list(ref_b) == v2_b,
          f"{len(edge_scores)} edge/nextafter scores; bins agree: {list(ref_b) == v2_b}; "
          f"score 1.0 -> bin {metrics.ece_bin(1.0)}, 0.3 -> {metrics.ece_bin(0.3)}, "
          f"nextafter(0.3,-) -> {metrics.ece_bin(float(np.nextafter(0.3, -1)))}")
    check("ece_edges_value", abs(ref_e - metrics.ece(ybin, edge_scores)) < 1e-15,
          f"ref {ref_e!r} v2 {metrics.ece(ybin, edge_scores)!r}")
    art_p = S.candidate(calibrator="platt")
    scorer_p = K.ArtifactScorer(art_p)
    yev = np_labels(ev)
    ev_eval = [scorer_p.evaluate(v) for v in pipeline.normalized_vectors(ev)]
    s_ev = [e["calibrated_score"] for e in ev_eval]
    ref_e, _ = ece_ref(yev, s_ev)
    check("ece_dev_eval", abs(ref_e - metrics.ece(yev.tolist(), s_ev)) < 1e-15,
          f"ref {ref_e:.15f} v2 {metrics.ece(yev.tolist(), s_ev):.15f}")

    # 5. AUROC vs Mann-Whitney with ties
    def auc_checks(label, y, s):
        y = np.asarray(y)
        s = np.asarray(s, dtype=float)
        P, N = int(y.sum()), int(len(y) - y.sum())
        U = stats.mannwhitneyu(s[y == 1], s[y == 0], alternative="two-sided").statistic
        ref = U / (P * N)
        from sklearn.metrics import roc_auc_score
        sk = roc_auc_score(y, s)
        v2 = metrics.roc_auc(y.tolist(), s.tolist())
        ss = bootstrap.ScoreStats(y.tolist(), s.tolist())
        v2w = ss.auroc([1] * len(y))
        ties = len(s) - len(np.unique(s))
        check(f"auroc_{label}", abs(v2 - ref) < 1e-12 and abs(v2w - ref) < 1e-12 and abs(sk - ref) < 1e-12,
              f"{ties} tied values; scipy MWU {ref:.15f}; sklearn {sk:.15f}; metrics.py {v2:.15f}; "
              f"weighted(ones) {v2w:.15f}")
        rng = np.random.default_rng(7)
        worst = 0.0
        for _ in range(5):
            c = rng.integers(0, 4, size=len(y))
            yy, sv = np.repeat(y, c), np.repeat(s, c)
            Pw, Nw = int(yy.sum()), int(len(yy) - yy.sum())
            Uw = stats.mannwhitneyu(sv[yy == 1], sv[yy == 0], alternative="two-sided").statistic
            worst = max(worst, abs(ss.auroc(c.tolist()) - Uw / (Pw * Nw)))
        check(f"auroc_weighted_{label}", worst < 1e-12,
              f"5 random count vectors (materialized rows) max|diff| = {worst:.2e}")

    auc_checks("dev_eval", yev, s_ev)
    auc_checks("dev_eval_rounded_2dp", yev, np.round(np.array(s_ev), 2))

    # 6. bootstrap: percentile order statistic (B = 2000), draws, pairing, strata
    d_v2 = [1 if e["threshold_decision"] else 0 for e in ev_eval]
    d_rule = [1 if r["features"]["consecutive_failures"] > 0 else 0 for r in ev]
    bs_a = bootstrap.BinaryStats(yev.tolist(), d_v2)
    bs_b = bootstrap.BinaryStats(yev.tolist(), d_rule)
    seen_a, seen_b = [], []

    def stat_a(w):
        seen_a.append(hash(tuple(w)))
        return bs_a.balanced_accuracy(w)

    def stat_b(w):
        seen_b.append(hash(tuple(w)))
        return bs_b.balanced_accuracy(w)

    n = len(ev)
    t0 = time.perf_counter()
    vals = bootstrap.paired_bootstrap(n, {"a": stat_a, "b": stat_b}, seed=20260927)
    t_v2 = time.perf_counter() - t0
    check("bootstrap_pairing", seen_a == seen_b and len(seen_a) == 2000,
          f"both contenders saw identical count vectors in all {len(seen_a)} resamples")
    rng = random.Random(20260927)
    ya, da, db = yev, np.array(d_v2), np.array(d_rule)
    worst = 0.0
    sums_ok = True
    ref_a, ref_b = [], []
    for bidx in range(2000):
        idx = rng.choices(range(n), k=n)
        c = np.bincount(idx, minlength=n)
        sums_ok &= int(c.sum()) == n
        for d, out in ((da, ref_a), (db, ref_b)):
            tp = int(np.sum(c * ya * d)); fn = int(np.sum(c * ya * (1 - d)))
            tn = int(np.sum(c * (1 - ya) * (1 - d))); fp = int(np.sum(c * (1 - ya) * d))
            out.append((tp / (tp + fn) + tn / (tn + fp)) / 2)
    worst = max(max(abs(x - y) for x, y in zip(vals["a"], ref_a)),
                max(abs(x - y) for x, y in zip(vals["b"], ref_b)))
    check("bootstrap_draws_equal_random_choices", worst == 0.0 and sums_ok,
          f"2000 resamples re-drawn with random.Random(20260927).choices(range(n), k=n); "
          f"max|BA_v2 - BA_ref| = {worst:.1e} (both contenders)")
    for label, v in (("a", vals["a"]), ("b", vals["b"]),
                     ("a-b", bootstrap.differences(vals["a"], vals["b"]))):
        iv = bootstrap.interval(v)
        ref = np.percentile(np.array(v), [2.5, 97.5], method="linear")
        srt = sorted(v)
        check(f"bootstrap_percentile_{label}",
              abs(iv["lower"] - ref[0]) < 1e-15 and abs(iv["upper"] - ref[1]) < 1e-15,
              f"v2 [{iv['lower']:.15f}, {iv['upper']:.15f}] numpy linear [{ref[0]:.15f}, {ref[1]:.15f}]; "
              f"h_lo = 1999*0.025 = 49.975 between order stats #50 {srt[49]:.12f} and #51 {srt[50]:.12f}; "
              f"h_hi = 1949.025 between #1950 {srt[1949]:.12f} and #1951 {srt[1950]:.12f}")
    # strata
    strata = [list(range(i, i + 5000)) for i in range(0, 20000, 5000)]
    rs = bootstrap.Resampler(n, seed=11, resamples=50, strata=strata)
    rng = random.Random(11)
    ok = True
    for w in rs:
        c = np.zeros(n, dtype=int)
        for st in strata:
            c += np.bincount(rng.choices(st, k=len(st)), minlength=n)
        ok &= list(c) == w and all(sum(w[s[0]:s[-1] + 1]) == 5000 for s in strata)
    check("bootstrap_strata", ok, "50 stratified resamples (4 x 5000) equal per-stratum "
          "random.choices draws; every stratum keeps its size")
    print(f"(bootstrap v2 time for 2000 resamples x 2 BA stats on {n} rows: {t_v2:.1f} s)")

    # 7. Mondrian quantile rank and empirical coverage
    for kind in ("none", "platt", "isotonic"):
        art = S.candidate(calibrator=kind)
        sc = K.ArtifactScorer(art)
        p_conf = np.array([sc.evaluate(v)["calibrated_score"] for v in pipeline.normalized_vectors(conf_rows)])
        y_conf = np_labels(conf_rows)
        detail = []
        ok = True
        for cls in (0, 1):
            sc_y = np.sort(p_conf[y_conf == 0]) if cls == 0 else np.sort(1.0 - p_conf[y_conf == 1])
            n_y = len(sc_y)
            k = math.ceil((n_y + 1) * (1 - Fraction(1, 10)))
            q_exact = sc_y[k - 1]
            q_np = np.quantile(sc_y, k / n_y, method="inverted_cdf")
            stored = art["conformal"]["q_hat"][str(cls)]
            ok &= (q_np == q_exact) and (stored >= q_exact) and (stored - q_exact < 1.01e-12)
            ok &= art["conformal"]["n_calibration"][str(cls)] == n_y
            detail.append(f"class {cls}: n={n_y} k=ceil((n+1)*0.9)={k} q_exact={q_exact:.15f} "
                          f"np.inverted_cdf={q_np:.15f} stored={stored!r}")
        # empirical coverage on 20,000 DEV eval rows, independent numpy scoring for the sets
        e_list = [sc.evaluate(v) for v in pipeline.normalized_vectors(ev)]
        sets = [e["prediction_set"] for e in e_list]
        cov = metrics.coverage(yev.tolist(), sets)
        p_ev = np.array([e["calibrated_score"] for e in e_list])
        q0, q1 = art["conformal"]["q_hat"]["0"], art["conformal"]["q_hat"]["1"]
        in0 = p_ev <= q0
        in1 = (1.0 - p_ev) <= q1
        hit = np.where(yev == 1, in1, in0)
        ref_cov = (hit.mean(), hit[yev == 0].mean(), hit[yev == 1].mean())
        ok &= abs(ref_cov[0] - cov["marginal"]) < 1e-15
        ok &= abs(ref_cov[1] - cov["per_class"]["0"]) < 1e-15 and abs(ref_cov[2] - cov["per_class"]["1"]) < 1e-15
        check(f"mondrian_{kind}", ok,
              "; ".join(detail) + f"; DEV eval coverage marginal {cov['marginal']:.4f}, class0 "
              f"{cov['per_class']['0']:.4f}, class1 {cov['per_class']['1']:.4f} (ref "
              f"{ref_cov[0]:.4f}/{ref_cov[1]:.4f}/{ref_cov[2]:.4f})")

    # 8. INVALID_BASELINE vs PREREGISTRATION section 10
    y_list = yev.tolist()
    const = [1] * len(y_list)
    v = metrics.baseline_validity(y_list, const, receipt_verified=True, probes_refused=len(gates.FAILCLOSED_PROBES),
                                  probes_total=len(gates.FAILCLOSED_PROBES))
    rep = metrics.binary_report(y_list, const, v)
    check("invalid_baseline_constant_alert", rep["per_class_rates"] == metrics.INVALID_BASELINE
          and v["failed_preconditions"] == ["iii_decisions_not_constant"],
          f"failed {v['failed_preconditions']}; per_class_rates={rep['per_class_rates']}")
    major = metrics.binary_report(y_list, [0] * len(y_list), metrics.baseline_validity(
        y_list, [0] * len(y_list), receipt_verified=True, probes_refused=1, probes_total=1))
    check("majority_ba_0_5_and_invalid", major["balanced_accuracy"] == 0.5
          and major["per_class_rates"] == metrics.INVALID_BASELINE,
          f"BA {major['balanced_accuracy']}, per_class_rates {major['per_class_rates']}")
    small = metrics.baseline_validity(y_list[:40], d_v2[:40], receipt_verified=False,
                                      probes_refused=3, probes_total=4)
    check("invalid_baseline_i_ii_iv", set(small["failed_preconditions"]) >= {"i_receipt_verifies", "iv_refused_all_probes"},
          f"first 40 rows, receipt False, 3/4 probes: failed {small['failed_preconditions']} "
          f"(class counts {small['class_counts']})")
    # leak test: a non-constant fixture that fails (iv) -- are sensitivity/specificity derivable?
    fo = metrics.baseline_validity(y_list, d_rule, receipt_verified=True, probes_refused=10,
                                   probes_total=len(gates.FAILCLOSED_PROBES))
    rep = metrics.binary_report(y_list, d_rule, fo)
    ba, acc, prev = rep["balanced_accuracy"], rep.get("accuracy"), rep["prevalence"]
    true_c = metrics.confusion(y_list, d_rule)
    true_sens = true_c["tp"] / (true_c["tp"] + true_c["fn"])
    if acc is not None and prev is not None and abs(prev - 0.5) > 1e-12:
        # acc = prev*sens + (1-prev)*spec ; ba = (sens+spec)/2
        sens = (acc - (1 - prev) * 2 * ba) / (prev - (1 - prev))
        spec = 2 * ba - sens
        leaked = abs(sens - true_sens) < 1e-9
        detail = (f"INVALID fixture report exposes BA, accuracy and prevalence; solving gives "
                  f"sensitivity {sens:.12f} (true {true_sens:.12f}), specificity {spec:.12f}")
    else:
        leaked = False
        detail = "accuracy/prevalence not both reported for INVALID fixtures; rates not derivable"
    check("invalid_baseline_rates_not_derivable", not leaked, detail)

    # 9. threshold selection vs brute-force numpy reference (tie rules)
    s_thr = [e["calibrated_score"] for e in (scorer_p.evaluate(v) for v in pipeline.normalized_vectors(thr))]
    y_thr = np_labels(thr)
    chosen = pipeline.select_threshold(y_thr.tolist(), s_thr)
    best = None
    st = np.array(s_thr)
    P, N = y_thr.sum(), len(y_thr) - y_thr.sum()
    for kk in range(1, 100):
        t = kk / 100
        d = st >= t
        tp = int(np.sum(d & (y_thr == 1))); tn = int(np.sum(~d & (y_thr == 0)))
        fp, fn = int(N - tn), int(P - tp)
        key = (Fraction(tp, int(P)) + Fraction(tn, int(N)), Fraction(2 * tp, 2 * tp + fp + fn), -abs(kk - 50), -kk)
        if best is None or key > best[0]:
            best = (key, kk)
    check("threshold_selection_bruteforce", best[1] == chosen["grid_index"],
          f"numpy brute force picks k={best[1]}, v2 picks k={chosen['grid_index']} "
          f"(BA {chosen['balanced_accuracy']:.12f})")

    print(f"\nSUMMARY: {sum(ok for _n, ok, _d in RESULTS)}/{len(RESULTS)} checks PASS; "
          f"FAIL: {[n for n, ok, _d in RESULTS if not ok]}  ({time.perf_counter() - t_start:.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
