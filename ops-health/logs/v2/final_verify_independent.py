"""Independent check of the final verdict (post-hoc; written AFTER the single opening).  SYNTHETIC.

Run with Python 3.11 + numpy (already installed; no package installs), from the repository root:
    PYTHONUTF8=1 python -B logs/v2/final_verify_independent.py

It reads ONLY v2/results/final/MANIFEST.json, the score files it lists (re-hashed) and
v2/results/final_evaluation.json.  It opens no data split.  It does not import v2.research.

Part A (MEASURED, independent code): point estimates of the primary metric (BA of every
  contender) on nominal test and on the pooled shift suite, and Brier and ECE (10 equal-width
  bins, last closed, count-weighted) of v2 and v1 on nominal test, recomputed with numpy.
Part B (MEASURED, independent code, registered stream): the registered paired percentile
  bootstrap re-implemented from its written definition (PREREGISTRATION section 7 and the
  v2/research/bootstrap.py docstring): one random.Random(20260927) stream per evaluation set; per
  resample, for each stratum in manifest order, m draws int(m * random()); type-7 2.5th and
  97.5th percentiles of 2,000 values.  The W1/W2 intervals and the v2 - v1 paired differences
  are compared with final_evaluation.json.
Part C (post-hoc, NOT registered, descriptive only): Monte Carlo sensitivity of the W1
  comparison v2 versus v1 on nominal test.  The same percentile rule (2,000 resamples, type 7)
  is repeated with 100 other resample streams (numpy PCG64, seeds 1..100).  It reports how often
  lower(BA v2) > upper(BA v1) and the spread of that margin.  It cannot change the registered
  verdict, which is defined by the single registered stream (seed 20260927).
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / "v2" / "results" / "final"
EVAL = ROOT / "v2" / "results" / "final_evaluation.json"
OUT = ROOT / "v2" / "results" / "final_robustness.json"
SEED = 20260927
B = 2000
CONTENDERS = ("v2", "v1", "majority", "rule", "ABL-data", "ABL-optimizer")
SENS_SEEDS = range(1, 101)
CHUNK = 25
MEM_RETRIES = 40
TOL = 1e-12


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load():
    mbytes = (FINAL / "MANIFEST.json").read_bytes()
    manifest = json.loads(mbytes)
    recs = {}
    for split in manifest["splits"]:
        e = manifest["files"][split]
        data = (FINAL / e["file"]).read_bytes()
        assert sha(data) == e["sha256"] and len(data) == e["bytes"], split
        rows = [json.loads(line) for line in data.decode("utf-8").splitlines()]
        assert len(rows) == e["rows"], split
        recs[split] = rows
    return manifest, sha(mbytes), recs


def arrays(rows):
    y = np.array([r["label"] for r in rows], dtype=np.int64)
    d = {c: np.array([r[c]["decision"] for r in rows], dtype=np.int64) for c in CONTENDERS}
    s = {c: np.array([r[c]["score"] for r in rows], dtype=np.float64) for c in ("v2", "v1")}
    return y, d, s


def ba_weighted(w, y, d):
    """w: (R, n) count weights; returns (R,) BA."""
    pos = w @ y
    neg = w.sum(axis=1) - pos
    tp = w @ (y * d)
    tn = w @ ((1 - y) * (1 - d))
    return 0.5 * (tp / pos + tn / neg)


def ece_bins(s):
    edges = np.arange(10) / 10.0
    return np.searchsorted(edges, s, side="right") - 1  # largest k with s >= k/10


def brier_ece_weighted(w, y, s):
    tot = w.sum(axis=1).astype(np.float64)
    brier = (w @ ((s - y) ** 2)) / tot
    k = ece_bins(s)
    diff = s - y
    gaps = np.zeros(w.shape[0])
    for b in range(10):
        m = k == b
        if m.any():
            gaps += np.abs(w[:, m] @ diff[m])
    return brier, gaps / tot


def type7(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q * 100.0, method="linear"))


def registered_counts(n, strata, chunk):
    """Yield (R, n) count matrices following the registered stream, chunk resamples at a time."""
    rng = random.Random(SEED)
    rnd = rng.random
    done = 0
    while done < B:
        r = min(chunk, B - done)
        w = np.zeros((r, n), dtype=np.int64)
        for i in range(r):
            if strata is None:
                u = np.array([rnd() for _ in range(n)])
                idx = (u * float(n)).astype(np.int64)
            else:
                parts = []
                for start, m in strata:
                    u = np.array([rnd() for _ in range(m)])
                    parts.append(start + (u * float(m)).astype(np.int64))
                idx = np.concatenate(parts)
            w[i] = np.bincount(idx, minlength=n)
        done += r
        yield w


def main() -> int:
    t0 = time.time()
    manifest, manifest_sha, recs = load()
    ev_bytes = EVAL.read_bytes()
    ev = json.loads(ev_bytes)
    assert ev["inputs"]["score_manifest_sha256"] == manifest_sha
    out = {"schema": "szl-oac/ops-health-final-robustness/v2", "synthetic": True,
           "post_hoc": True,
           "note": "written after the single opening; reads only the saved score files and "
                   "final_evaluation.json; Parts A and B are independent re-computations of "
                   "registered quantities, Part C is a NOT-registered descriptive sensitivity "
                   "analysis that cannot change the registered verdict",
           "inputs": {"score_manifest_sha256": manifest_sha,
                      "final_evaluation_sha256": sha(ev_bytes),
                      "script": "logs/v2/final_verify_independent.py",
                      "script_sha256": sha(Path(__file__).read_bytes()),
                      "python": sys.version.split()[0], "numpy": np.__version__}}
    sets = {"test": (recs["test"], None)}
    pooled, strata = [], []
    for name in manifest["pooled_shift_suite"]:
        strata.append((len(pooled), len(recs[name])))
        pooled.extend(recs[name])
    sets["pooled_shift"] = (pooled, strata)

    # ---- Part A and B ------------------------------------------------------------------
    worst = 0.0
    part_ab = {}
    for set_name, (rows, st) in sets.items():
        y, d, s = arrays(rows)
        n = len(y)
        ones = np.ones((1, n), dtype=np.int64)
        res = {"rows": n, "points": {}, "intervals": {}, "max_abs_diff_vs_evaluation": None}
        diffs = []
        evset = ev["sets"][set_name]
        for c in CONTENDERS:
            p = float(ba_weighted(ones, y, d[c])[0])
            res["points"][f"ba:{c}"] = p
            diffs.append(abs(p - evset["bootstrap"]["intervals"][f"ba:{c}"]["point"]))
        for c in ("v2", "v1"):
            br, ec = brier_ece_weighted(ones, y, s[c])
            res["points"][f"brier:{c}"] = float(br[0])
            res["points"][f"ece:{c}"] = float(ec[0])
            diffs.append(abs(float(br[0]) - evset["contenders"][c]["score_report"]["brier"]))
            diffs.append(abs(float(ec[0]) - evset["contenders"][c]["score_report"]["ece"]))
        vals = {k: [] for k in ("ba:v2", "ba:v1", "ba:majority", "ba:rule", "brier:v2",
                                "brier:v1", "ece:v2", "ece:v1")}
        for w in registered_counts(n, st, chunk=20):
            for c in ("v2", "v1", "majority", "rule"):
                vals[f"ba:{c}"].extend(ba_weighted(w, y, d[c]).tolist())
            for c in ("v2", "v1"):
                br, ec = brier_ece_weighted(w, y, s[c])
                vals[f"brier:{c}"].extend(br.tolist())
                vals[f"ece:{c}"].extend(ec.tolist())
        for key, v in vals.items():
            lo, hi = type7(v, 0.025), type7(v, 0.975)
            e = evset["bootstrap"]["intervals"][key]
            res["intervals"][key] = [lo, hi]
            diffs += [abs(lo - e["lower"]), abs(hi - e["upper"])]
        for stat in ("ba", "brier", "ece"):
            dv = np.array(vals[f"{stat}:v2"]) - np.array(vals[f"{stat}:v1"])
            lo, hi = type7(dv, 0.025), type7(dv, 0.975)
            e = evset["paired_differences"]["v2_minus_v1"][stat]
            res["intervals"][f"diff_v2_minus_v1:{stat}"] = [lo, hi]
            diffs += [abs(lo - e["lower"]), abs(hi - e["upper"])]
        res["max_abs_diff_vs_evaluation"] = max(diffs)
        worst = max(worst, max(diffs))
        iv = res["intervals"]
        res["w_rule"] = {c: {"v2_lower": iv["ba:v2"][0], "other_upper": iv[f"ba:{c}"][1],
                             "margin": iv["ba:v2"][0] - iv[f"ba:{c}"][1],
                             "holds": iv["ba:v2"][0] > iv[f"ba:{c}"][1]}
                         for c in ("v1", "majority", "rule")}
        part_ab[set_name] = res
        print(f"[A+B] {set_name}: rows {n}; max |independent - final_evaluation| = {max(diffs):.3e}; "
              f"W rule vs v1 margin {res['w_rule']['v1']['margin']:.9f} "
              f"holds={res['w_rule']['v1']['holds']}; "
              f"vs rule holds={res['w_rule']['rule']['holds']}; "
              f"vs majority holds={res['w_rule']['majority']['holds']} ({time.time() - t0:.0f} s)",
              flush=True)
    y, d, s = arrays(recs["test"])
    w3 = {"brier_v2": part_ab["test"]["points"]["brier:v2"], "brier_v1": part_ab["test"]["points"]["brier:v1"],
          "ece_v2": part_ab["test"]["points"]["ece:v2"], "ece_v1": part_ab["test"]["points"]["ece:v1"]}
    w3["holds"] = w3["brier_v2"] <= w3["brier_v1"] and w3["ece_v2"] <= w3["ece_v1"]
    verdict = ("WIN" if all(part_ab[x]["w_rule"][c]["holds"] for x in ("test", "pooled_shift")
                            for c in ("v1", "majority", "rule")) and w3["holds"] else "DOES NOT WIN")
    out["part_A_B"] = {"sets": part_ab, "W3": w3, "independent_verdict": verdict,
                       "matches_final_evaluation": verdict == ev["verdict"]["result"],
                       "max_abs_diff_vs_evaluation": worst, "tolerance": TOL,
                       "within_tolerance": worst <= TOL}
    print(f"[A+B] independent verdict {verdict}; final_evaluation verdict "
          f"{ev['verdict']['result']}; max abs diff {worst:.3e} (tolerance {TOL})", flush=True)

    # ---- Part C (post-hoc, not registered) ----------------------------------------------
    # Memory-light (int32 indices, int8 indicator vectors, 25 resamples per chunk) and
    # checkpointed per stream in scratch/ (git-ignored), because the host has been short of disk
    # and commit memory.
    n = len(y)
    y8 = y.astype(np.int8)
    tp2, tp1 = (y * d["v2"]).astype(np.int8), (y * d["v1"]).astype(np.int8)
    tn2, tn1 = ((1 - y) * (1 - d["v2"])).astype(np.int8), ((1 - y) * (1 - d["v1"])).astype(np.int8)
    ckpt = ROOT / "scratch" / "v2_final_robustness_partC.jsonl"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    key = {"manifest": manifest_sha, "B": B, "rng": "numpy-PCG64-integers-int32", "chunk": CHUNK}
    done_margins: dict[int, float] = {}
    if ckpt.exists():
        for line in ckpt.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec.get("key") == key:
                done_margins[rec["seed"]] = rec["margin"]
    print(f"[C] reusing {len(done_margins)} checkpointed streams", flush=True)
    margins, holds = [], 0
    for seed in SENS_SEEDS:
        if seed in done_margins:
            m = done_margins[seed]
        else:
            for attempt in range(MEM_RETRIES + 1):
                try:  # a stream is a pure function of its seed, so a retry recomputes it exactly
                    rng = np.random.default_rng(seed)
                    b2, b1 = [], []
                    for _ in range(B // CHUNK):
                        idx = rng.integers(0, n, size=(CHUNK, n), dtype=np.int32)
                        pos = y8[idx].sum(axis=1, dtype=np.int64)
                        neg = n - pos
                        b2.append(0.5 * (tp2[idx].sum(axis=1, dtype=np.int64) / pos
                                         + tn2[idx].sum(axis=1, dtype=np.int64) / neg))
                        b1.append(0.5 * (tp1[idx].sum(axis=1, dtype=np.int64) / pos
                                         + tn1[idx].sum(axis=1, dtype=np.int64) / neg))
                        del idx
                    m = type7(np.concatenate(b2), 0.025) - type7(np.concatenate(b1), 0.975)
                    break
                except MemoryError:
                    b2 = b1 = None
                    if attempt == MEM_RETRIES:
                        raise
                    print(f"[C] stream {seed}: MemoryError (host short of commit memory); "
                          f"retry {attempt + 1}/{MEM_RETRIES} in 30 s", flush=True)
                    time.sleep(30)
            with open(ckpt, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"key": key, "seed": seed, "margin": m}) + "\n")
        margins.append(m)
        holds += m > 0
        if seed % 10 == 0:
            print(f"[C] stream {seed}: holds so far {holds}/{len(margins)} "
                  f"({time.time() - t0:.0f} s)", flush=True)
    margins_a = np.array(margins)
    out["part_C_post_hoc_sensitivity"] = {
        "registered": False,
        "what": "W1 comparison v2 vs v1 on nominal test (lower 2.5th percentile of BA v2 minus "
                "upper 97.5th percentile of BA v1), 2,000 resamples, type 7, repeated with other "
                "resample streams",
        "streams": f"numpy.random.default_rng(seed) PCG64, seeds {SENS_SEEDS.start}..{SENS_SEEDS.stop - 1}",
        "streams_count": len(margins),
        "streams_where_w1_v1_holds": int(holds),
        "fraction_holds": holds / len(margins),
        "margin_min": float(margins_a.min()),
        "margin_median": float(np.median(margins_a)),
        "margin_max": float(margins_a.max()),
        "margin_mean": float(margins_a.mean()),
        "margin_sd": float(margins_a.std(ddof=1)),
        "registered_stream_margin": part_ab["test"]["w_rule"]["v1"]["margin"],
    }
    pc = out["part_C_post_hoc_sensitivity"]
    print(f"[C, post-hoc, NOT registered] W1 v2-vs-v1 holds in {pc['streams_where_w1_v1_holds']}/"
          f"{pc['streams_count']} other streams; margin min {pc['margin_min']:.6f} median "
          f"{pc['margin_median']:.6f} max {pc['margin_max']:.6f} sd {pc['margin_sd']:.6f}; "
          f"registered stream margin {pc['registered_stream_margin']:.6f} ({time.time() - t0:.0f} s)",
          flush=True)
    data = (json.dumps(out, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    OUT.write_bytes(data)
    assert OUT.read_bytes() == data
    print(f"wrote {OUT.relative_to(ROOT).as_posix()}: {len(data)} bytes sha256 {sha(data)}", flush=True)
    return 0 if out["part_A_B"]["matches_final_evaluation"] and out["part_A_B"]["within_tolerance"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
