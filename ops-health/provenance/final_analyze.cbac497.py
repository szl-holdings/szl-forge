"""Final analysis from the saved score files (PREREGISTRATION sections 7, 8, 9 and 10).  SYNTHETIC.

Reads ONLY v2/results/final/MANIFEST.json and the score files it lists (each re-hashed and
re-counted against the manifest before use), plus the frozen model files for generator-truth
recovery (each re-hashed against the manifest's contender digests).  It opens no data split of
any kind, so it is safe to re-run; its output is a deterministic function of those files.

Registered analysis (section 7)
  Primary metric: balanced accuracy (BA) of each contender's binary decision, on the nominal test
  split and on the pooled shift suite (the four shift splits concatenated, equal regime weights).
  Uncertainty: paired percentile bootstrap, 2,000 resamples, seed 20260927, 95% interval
  (type-7 2.5th and 97.5th percentiles; ``v2.research.bootstrap``).  One resample stream per
  evaluation set; every contender and statistic of a set is evaluated on the SAME count vector.
  The pooled shift suite is resampled within each regime (strata in manifest order).
  W1: on nominal test, lower(BA v2) > upper(BA c) for c in v1, majority, rule (strict).
  W2: the same on the pooled shift suite.
  W3: on nominal test, Brier(v2) <= Brier(v1) and ECE(v2) <= ECE(v1), point estimates, ECE over
      10 equal-width bins (last closed), count-weighted.
  v2 WINS only if W1, W2 and W3 all hold; otherwise it DOES NOT WIN.
Secondary (section 8), on test, each shift regime, the pooled suite and legacy_v1_test:
  Brier, ECE with reliability table, log loss, ROC AUC (scored contenders); precision, recall,
  specificity, F1, false-alert share, only through ``metrics.binary_report`` gated by the
  section 10 preconditions (INVALID_BASELINE otherwise); v2 conformal marginal and per-class
  coverage, abstention (BOTH / EMPTY) and selective BA; paired-bootstrap intervals of v2 - v1 in
  BA, Brier and ECE (same resample stream as the set's primary); BA by regime and, on test, by
  tls_enabled and listener_running; generator-truth recovery (sign k/8, Spearman, relative L2).
Ablations (section 9): ABL-data and ABL-optimizer BA and v2 - ablation BA differences on the same
  streams; ABL-calibration NOT_APPLICABLE; ABL-conformal = v2 with versus without abstention.

Outputs: v2/results/final_evaluation.json, v2/results/FINAL_RESULTS.md, v2/results/reliability.svg.
Each evaluation set's result is saved to scratch/v2_final_analysis/<set>.json (git-ignored) as
soon as it completes, and reused on a re-run only if its input digest matches.

Usage (repository root):
    PYTHONUTF8=1 py -3.12 -B -m v2.research.final_analyze
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import bootstrap, metrics, registry, truth

SEED = 20260927
RESAMPLES = bootstrap.RESAMPLES
LEVEL = bootstrap.LEVEL
SCORE_DIR = registry.V2 / "results" / "final"
OUT_JSON = registry.V2 / "results" / "final_evaluation.json"
OUT_MD = registry.V2 / "results" / "FINAL_RESULTS.md"
OUT_SVG = registry.V2 / "results" / "reliability.svg"
PARTS_DIR = registry.ROOT / "scratch" / "v2_final_analysis"
MANIFEST_NAME = "MANIFEST.json"
SCRIPT_REL = "v2/research/final_analyze.py"
EVAL_SCHEMA = "szl-oac/ops-health-final-evaluation/v2"

CONTENDERS = ("v2", "v1", "majority", "rule", "ABL-data", "ABL-optimizer")
SCORED = ("v2", "v1", "ABL-data", "ABL-optimizer")
BASELINES_W = ("v1", "majority", "rule")
DIFF_BA_AGAINST = ("v1", "majority", "rule", "ABL-data", "ABL-optimizer")
TEST = "test"
POOLED = "pooled_shift"
LEGACY = "legacy_v1_test"
MODEL_FILES = {
    "v2": "v2/ops-health/model.json",
    "v1": "hub/oac-v1/model.json",
    "ABL-data": "v2/results/contenders/ABL-data/model.json",
    "ABL-optimizer": "v2/results/contenders/ABL-optimizer/params.json",
}


class AnalysisError(RuntimeError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


# --------------------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------------------
def load_scores(score_dir: Path) -> tuple[dict, str, dict[str, list[dict]]]:
    """Manifest, its sha256, and {split: records}; every file re-hashed and re-counted."""
    manifest_bytes = (score_dir / MANIFEST_NAME).read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    records: dict[str, list[dict]] = {}
    for split in manifest["splits"]:
        entry = manifest["files"][split]
        data = (score_dir / entry["file"]).read_bytes()
        if sha256_bytes(data) != entry["sha256"] or len(data) != entry["bytes"]:
            raise AnalysisError(f"{entry['file']}: bytes differ from the manifest")
        rows = [json.loads(line) for line in data.decode("utf-8").splitlines()]
        if len(rows) != entry["rows"] or sum(r["label"] for r in rows) != entry["positives"]:
            raise AnalysisError(f"{entry['file']}: row or positive count differs from the manifest")
        if any(r["split"] != split for r in rows):
            raise AnalysisError(f"{entry['file']}: a record names another split")
        records[split] = rows
    return manifest, sha256_bytes(manifest_bytes), records


def evaluation_sets(manifest: Mapping, records: Mapping[str, list[dict]]) -> dict[str, dict]:
    """{set: {"records", "strata"}} in reporting order: test, pooled, each shift, legacy."""
    pooled_names = list(manifest["pooled_shift_suite"])
    pooled: list[dict] = []
    strata = []
    for name in pooled_names:
        start = len(pooled)
        pooled.extend(records[name])
        strata.append(list(range(start, len(pooled))))
    sets = {TEST: {"records": records[TEST], "strata": None, "members": [TEST]},
            POOLED: {"records": pooled, "strata": strata, "members": pooled_names}}
    for name in pooled_names:
        sets[name] = {"records": records[name], "strata": None, "members": [name]}
    sets[LEGACY] = {"records": records[LEGACY], "strata": None, "members": [LEGACY]}
    return sets


# --------------------------------------------------------------------------------------
# per-set analysis
# --------------------------------------------------------------------------------------
def _validity(labels, decisions, inputs: Mapping) -> dict:
    return metrics.baseline_validity(labels, decisions,
                                     receipt_verified=inputs["receipt_verified"],
                                     probes_refused=inputs["probes"]["refused"],
                                     probes_total=inputs["probes"]["total"])


def analyze_set(name: str, spec: Mapping, validity_inputs: Mapping, *,
                resamples: int = RESAMPLES, seed: int = SEED) -> dict:
    recs = spec["records"]
    y = [r["label"] for r in recs]
    decisions = {c: [r[c]["decision"] for r in recs] for c in CONTENDERS}
    scores = {c: [r[c]["score"] for r in recs] for c in SCORED}
    out: dict[str, Any] = {
        "set": name,
        "members": list(spec["members"]),
        "rows": len(y),
        "positives": sum(y),
        "stratified_by_regime": spec["strata"] is not None,
        "contenders": {},
    }
    for c in CONTENDERS:
        block = {"binary_report": metrics.binary_report(y, decisions[c],
                                                        _validity(y, decisions[c],
                                                                  validity_inputs[c]))}
        if c in SCORED:
            block["score_report"] = metrics.score_report(y, scores[c])
        out["contenders"][c] = block
    v2 = [r["v2"] for r in recs]
    out["v2_conformal"] = {
        "coverage": metrics.coverage(y, [v["prediction_set"] for v in v2]),
        "selective": metrics.selective(y, [v["advisory"] for v in v2],
                                       [v["abstain_reason"] for v in v2]),
        "note": "the class-conditional guarantee holds under exchangeability with the nominal "
                "conformal split only; under shift and on legacy_v1_test coverage is descriptive",
    }
    # paired bootstrap: one stream for every statistic of this set
    t0 = time.perf_counter()
    stats = {}
    for c in CONTENDERS:
        stats[f"ba:{c}"] = bootstrap.BinaryStats(y, decisions[c]).balanced_accuracy
    for c in ("v2", "v1"):
        ss = bootstrap.ScoreStats(y, scores[c])
        stats[f"brier:{c}"] = ss.brier
        stats[f"ece:{c}"] = ss.ece
    values = bootstrap.paired_bootstrap(len(y), stats, seed=seed, resamples=resamples,
                                        strata=spec["strata"])
    points = {
        **{f"ba:{c}": out["contenders"][c]["binary_report"]["balanced_accuracy"] for c in CONTENDERS},
        **{f"brier:{c}": out["contenders"][c]["score_report"]["brier"] for c in ("v2", "v1")},
        **{f"ece:{c}": out["contenders"][c]["score_report"]["ece"] for c in ("v2", "v1")},
    }
    out["bootstrap"] = {
        "resamples": resamples,
        "seed": seed,
        "level": LEVEL,
        "strata": [len(s) for s in spec["strata"]] if spec["strata"] is not None else None,
        "seconds": round(time.perf_counter() - t0, 2),
        "intervals": {key: bootstrap.summarize(points[key], values[key]) for key in stats},
    }

    def diff(key_a: str, key_b: str) -> dict:
        pa, pb = points[key_a], points[key_b]
        point = None if pa is None or pb is None else pa - pb
        return bootstrap.summarize(point, bootstrap.differences(values[key_a], values[key_b]))

    out["paired_differences"] = {
        "v2_minus_v1": {"ba": diff("ba:v2", "ba:v1"), "brier": diff("brier:v2", "brier:v1"),
                        "ece": diff("ece:v2", "ece:v1")},
        "ba_v2_minus": {c: diff("ba:v2", f"ba:{c}") for c in DIFF_BA_AGAINST},
    }
    return out


def disaggregate_test(records: Sequence[Mapping]) -> dict:
    """BA of every contender on nominal test by tls_enabled and by listener_running."""
    out = {}
    for feature in ("tls_enabled", "listener_running"):
        block = {}
        for value in (0, 1):
            sub = [r for r in records if int(r["features"][feature]) == value]
            y = [r["label"] for r in sub]
            block[str(value)] = {
                "rows": len(sub),
                "positives": sum(y),
                "balanced_accuracy": {c: (metrics.balanced_accuracy(y, [r[c]["decision"] for r in sub])
                                          if sub else None) for c in CONTENDERS},
            }
        out[feature] = block
    return out


# --------------------------------------------------------------------------------------
# win condition (section 7)
# --------------------------------------------------------------------------------------
def interval_rule(set_result: Mapping) -> dict:
    iv = set_result["bootstrap"]["intervals"]
    v2 = iv["ba:v2"]
    comparisons = {}
    for c in BASELINES_W:
        other = iv[f"ba:{c}"]
        holds = (v2["lower"] is not None and other["upper"] is not None
                 and v2["lower"] > other["upper"])
        margin = (v2["lower"] - other["upper"]
                  if v2["lower"] is not None and other["upper"] is not None else None)
        comparisons[c] = {"v2_lower": v2["lower"], "other_upper": other["upper"],
                          "margin": margin, "holds": bool(holds)}
    return {"holds": all(v["holds"] for v in comparisons.values()), "comparisons": comparisons,
            "v2_interval": [v2["lower"], v2["upper"]], "v2_point": v2["point"]}


def win_conditions(test_result: Mapping, pooled_result: Mapping) -> dict:
    w1 = interval_rule(test_result)
    w2 = interval_rule(pooled_result)
    s2 = test_result["contenders"]["v2"]["score_report"]
    s1 = test_result["contenders"]["v1"]["score_report"]
    brier_ok = s2["brier"] is not None and s1["brier"] is not None and s2["brier"] <= s1["brier"]
    ece_ok = s2["ece"] is not None and s1["ece"] is not None and s2["ece"] <= s1["ece"]
    w3 = {"holds": bool(brier_ok and ece_ok),
          "brier": {"v2": s2["brier"], "v1": s1["brier"], "holds": bool(brier_ok)},
          "ece": {"v2": s2["ece"], "v1": s1["ece"], "holds": bool(ece_ok)}}
    wins = w1["holds"] and w2["holds"] and w3["holds"]
    failed = [name for name, w in (("W1", w1), ("W2", w2), ("W3", w3)) if not w["holds"]]
    return {"result": "WIN" if wins else "DOES NOT WIN", "failed": failed,
            "W1": w1, "W2": w2, "W3": w3,
            "rule": "v2 wins only if W1 (nominal test), W2 (pooled shift) and W3 (calibration on "
                    "nominal test) all hold (PREREGISTRATION section 7)"}


# --------------------------------------------------------------------------------------
# generator truth, validity summary, ablations
# --------------------------------------------------------------------------------------
def truth_recovery(manifest: Mapping) -> dict:
    out = {"truth": {"intercept": truth.TRUE_INTERCEPT, "weights": dict(truth.TRUE_WEIGHTS)}}
    for c, rel in MODEL_FILES.items():
        data = (registry.ROOT / rel).read_bytes()
        pinned = manifest["contenders"][c]["files"].get(rel)
        if pinned != sha256_bytes(data):
            raise AnalysisError(f"{rel}: sha256 differs from the score manifest")
        model = json.loads(data.decode("utf-8"))
        out[c] = {"file": rel, "sha256": pinned, "intercept": model["intercept"],
                  "weights": {n: model["weights"][n] for n in truth.FEATURE_NAMES},
                  **truth.recovery_report(model["intercept"], model["weights"])}
    return out


def validity_summary(manifest: Mapping, sets: Mapping[str, Mapping]) -> dict:
    out = {}
    for c in CONTENDERS:
        vin = manifest["validity_inputs"][c]
        out[c] = {
            "receipt_verified": vin["receipt_verified"],
            "receipt_check": vin["receipt_check"],
            "probes_refused": vin["probes"]["refused"],
            "probes_total": vin["probes"]["total"],
            "probes_escaped": vin["probes"]["escaped"],
            "probe_method": vin["probes"]["method"],
            "status_by_set": {s: {"status": r["contenders"][c]["binary_report"]["baseline_validity"]["status"],
                                  "failed": r["contenders"][c]["binary_report"]["baseline_validity"]["failed_preconditions"]}
                              for s, r in sets.items()},
        }
    return out


def ablations(sets: Mapping[str, Mapping], manifest: Mapping) -> dict:
    out = {}
    for c in ("ABL-data", "ABL-optimizer"):
        out[c] = {s: {"ba": sets[s]["bootstrap"]["intervals"][f"ba:{c}"],
                      "ba_v2_minus": sets[s]["paired_differences"]["ba_v2_minus"][c],
                      "brier": sets[s]["contenders"][c]["score_report"]["brier"],
                      "ece": sets[s]["contenders"][c]["score_report"]["ece"],
                      "v2_brier": sets[s]["contenders"]["v2"]["score_report"]["brier"],
                      "v2_ece": sets[s]["contenders"]["v2"]["score_report"]["ece"]}
                  for s in (TEST, POOLED)}
    out["ABL-calibration"] = {"status": manifest["contenders"]["ABL-calibration"]["status"],
                              "reason": manifest["contenders"]["ABL-calibration"]["reason"]}
    out["ABL-conformal"] = {
        "validation_decision_keep": manifest["contenders"]["ABL-conformal"]["keep"],
        "by_set": {s: {"ba_without_abstention": r["contenders"]["v2"]["binary_report"]["balanced_accuracy"],
                       "selective_ba_with_abstention": r["v2_conformal"]["selective"]["selective_balanced_accuracy"],
                       "abstention_rate": r["v2_conformal"]["selective"]["abstention_rate"],
                       "coverage_marginal": r["v2_conformal"]["coverage"]["marginal"]}
                   for s, r in sets.items()},
    }
    return out


# --------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------
def _git_head() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(registry.ROOT), "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def analyze(score_dir: Path = SCORE_DIR, *, parts_dir: Path | None = PARTS_DIR,
            resamples: int = RESAMPLES, log=print) -> dict:
    manifest, manifest_sha, records = load_scores(score_dir)
    script_sha = sha256_bytes((registry.ROOT / SCRIPT_REL).read_bytes())
    sets_spec = evaluation_sets(manifest, records)
    input_digest = sha256_bytes(f"{manifest_sha}:{script_sha}:{resamples}:{SEED}".encode())
    results: dict[str, dict] = {}
    for name, spec in sets_spec.items():
        part = parts_dir / f"{name}.json" if parts_dir is not None else None
        if part is not None and part.exists():
            cached = json.loads(part.read_text(encoding="utf-8"))
            if cached.get("input_digest") == input_digest:
                results[name] = cached["result"]
                log(f"set {name}: reused saved result")
                continue
        result = analyze_set(name, spec, manifest["validity_inputs"], resamples=resamples)
        results[name] = result
        if part is not None:
            part.parent.mkdir(parents=True, exist_ok=True)
            part.write_bytes(_json_bytes({"input_digest": input_digest, "result": result}))
        log(f"set {name}: rows {result['rows']} bootstrap {result['bootstrap']['seconds']} s")
    verdict = win_conditions(results[TEST], results[POOLED])
    marker = manifest["opening"]["marker"]
    evaluation = {
        "schema": EVAL_SCHEMA,
        "synthetic": True,
        "truth_labels": {
            "numbers": "MEASURED: computed by v2/research/final_analyze.py from the saved score "
                       "files; SYNTHETIC data, never generalized",
            "v1_published_legacy_counts": "REPORTED from the v1 receipt",
        },
        "registered_protocol": resamples == RESAMPLES,
        "mode": manifest["mode"],
        "inputs": {
            "score_manifest_sha256": manifest_sha,
            "score_files": {s: {"file": manifest["files"][s]["file"],
                                "sha256": manifest["files"][s]["sha256"],
                                "rows": manifest["files"][s]["rows"],
                                "positives": manifest["files"][s]["positives"]}
                            for s in manifest["splits"]},
            "opening": {"marker_file": manifest["opening"]["marker_file"],
                        "marker_sha256": manifest["opening"]["marker_sha256"],
                        "opened_utc": marker["opened_utc"], "git_head_at_opening": marker["git_head"],
                        "purpose": marker["purpose"],
                        "sealed_splits_sha256": marker["sealed_splits_sha256"]},
            "scorer": manifest["script"],
            "analysis_script": {"path": SCRIPT_REL, "sha256": script_sha, "git_head": _git_head(),
                                "python": platform.python_version()},
        },
        "bootstrap_protocol": {"resamples": resamples, "seed": SEED, "level": LEVEL,
                               "interval": "percentile, type-7 (2.5th, 97.5th)",
                               "paired": "one count-vector stream per evaluation set, shared by "
                                         "every contender and statistic",
                               "pooled_shift_strata": list(manifest["pooled_shift_suite"])},
        "contenders": manifest["contenders"],
        "verdict": verdict,
        "primary": {s: {c: results[s]["bootstrap"]["intervals"][f"ba:{c}"] for c in CONTENDERS}
                    for s in (TEST, POOLED)},
        "sets": results,
        "disaggregation": {
            "ba_by_regime": {s: {c: results[s]["bootstrap"]["intervals"][f"ba:{c}"]
                                 for c in CONTENDERS} for s in results},
            "test": disaggregate_test(sets_spec[TEST]["records"]),
        },
        "truth_recovery": truth_recovery(manifest),
        "validity": validity_summary(manifest, results),
        "ablations": ablations(results, manifest),
        "generated_utc": _utc(),
    }
    return evaluation


# --------------------------------------------------------------------------------------
# rendering: reliability.svg and FINAL_RESULTS.md
# --------------------------------------------------------------------------------------
def _f(value, digits: int = 4) -> str:
    if value is None:
        return "–"
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return f"{value:.{digits}f}"


def _ci(block: Mapping, digits: int = 4) -> str:
    return f"{_f(block['point'], digits)} [{_f(block['lower'], digits)}, {_f(block['upper'], digits)}]"


def render_svg(evaluation: Mapping) -> str:
    """Reliability diagrams of v2 and v1 on nominal test and on the pooled shift suite."""
    panels = [(TEST, "Nominal test"), (POOLED, "Pooled shift suite")]
    width, height, pad_l, pad_t, size, gap = 760, 420, 56, 64, 300, 80
    series = (("v2", "var(--series-1)", "v2 (T07, uncalibrated LR)"),
              ("v1", "var(--series-2)", "v1 as-is (Hub kernel score)"))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" '
        f'height="{height}" role="img" aria-labelledby="t d">',
        '<title id="t">Reliability of v2 and v1 (SYNTHETIC)</title>',
        '<desc id="d">Mean predicted score against observed fraction positive in 10 equal-width '
        'bins, for v2 and v1, on the nominal test split and the pooled shift suite. SYNTHETIC '
        'data. Values are in v2/results/final_evaluation.json.</desc>',
        '<style>svg{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--grid:#d9d8d4;'
        '--series-1:#2a78d6;--series-2:#eb6834;font-family:system-ui,-apple-system,"Segoe UI",'
        'sans-serif}@media (prefers-color-scheme:dark){svg{--surface:#1a1a19;--ink:#ffffff;'
        '--ink2:#c3c2b7;--grid:#3a3a37;--series-1:#3987e5;--series-2:#d95926}}'
        '.t{fill:var(--ink);font-size:14px;font-weight:600}.l{fill:var(--ink2);font-size:11px}'
        '.a{stroke:var(--grid);stroke-width:1}.d{stroke:var(--ink2);stroke-width:1;'
        'stroke-dasharray:4 4}</style>',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="var(--surface)"/>',
        f'<text class="t" x="{pad_l}" y="24">Reliability, 10 equal-width bins (SYNTHETIC)</text>',
    ]
    lx = pad_l
    for _c, color, label in series:
        parts.append(f'<line x1="{lx}" y1="42" x2="{lx + 18}" y2="42" stroke="{color}" '
                     'stroke-width="2"/>')
        parts.append(f'<circle cx="{lx + 9}" cy="42" r="4" fill="{color}"/>')
        parts.append(f'<text class="l" x="{lx + 24}" y="46">{label}</text>')
        lx += 230
    parts.append(f'<line class="d" x1="{lx}" y1="42" x2="{lx + 18}" y2="42"/>'
                 f'<text class="l" x="{lx + 24}" y="46">perfect calibration</text>')
    for k, (set_name, title) in enumerate(panels):
        x0 = pad_l + k * (size + gap)
        y0 = pad_t + 16
        sx = lambda v: x0 + v * size  # noqa: E731
        sy = lambda v: y0 + size - v * size  # noqa: E731
        parts.append(f'<text class="t" x="{x0}" y="{y0 - 8}">{title}</text>')
        for tick in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
            parts.append(f'<line class="a" x1="{sx(0):.1f}" y1="{sy(tick):.1f}" x2="{sx(1):.1f}" '
                         f'y2="{sy(tick):.1f}"/>')
            parts.append(f'<text class="l" x="{sx(0) - 6:.1f}" y="{sy(tick) + 4:.1f}" '
                         f'text-anchor="end">{tick:.1f}</text>')
            parts.append(f'<text class="l" x="{sx(tick):.1f}" y="{sy(0) + 16:.1f}" '
                         f'text-anchor="middle">{tick:.1f}</text>')
        parts.append(f'<line class="d" x1="{sx(0):.1f}" y1="{sy(0):.1f}" x2="{sx(1):.1f}" '
                     f'y2="{sy(1):.1f}"/>')
        parts.append(f'<text class="l" x="{sx(0.5):.1f}" y="{sy(0) + 32:.1f}" '
                     'text-anchor="middle">mean predicted score in bin</text>')
        parts.append(f'<text class="l" transform="translate({sx(0) - 38:.1f},{sy(0.5):.1f}) '
                     'rotate(-90)" text-anchor="middle">observed fraction positive</text>')
        set_result = evaluation["sets"][set_name]
        for c, color, label in series:
            table = set_result["contenders"][c]["score_report"]["reliability_table"]
            pts = [(b["mean_score"], b["fraction_positive"], b) for b in table if b["count"]]
            if len(pts) > 1:
                path = " ".join(f"{'M' if i == 0 else 'L'}{sx(m):.1f},{sy(p):.1f}"
                                for i, (m, p, _b) in enumerate(pts))
                parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2"/>')
            for m, p, b in pts:
                parts.append(
                    f'<circle cx="{sx(m):.1f}" cy="{sy(p):.1f}" r="4.5" fill="{color}" '
                    f'stroke="var(--surface)" stroke-width="2"><title>{label}, bin '
                    f'[{b["lower"]:.1f}, {b["upper"]:.1f}{"]" if b["upper_closed"] else ")"}: '
                    f'n={b["count"]}, mean score {m:.4f}, fraction positive {p:.4f}</title></circle>')
        e2 = set_result["contenders"]["v2"]["score_report"]["ece"]
        e1 = set_result["contenders"]["v1"]["score_report"]["ece"]
        parts.append(f'<text class="l" x="{sx(0) + 6:.1f}" y="{sy(1) + 14:.1f}">ECE v2 {_f(e2)}, '
                     f'v1 {_f(e1)}</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def render_md(evaluation: Mapping) -> str:  # noqa: C901 -- a linear report
    v = evaluation["verdict"]
    sets = evaluation["sets"]
    prim = evaluation["primary"]
    inputs = evaluation["inputs"]
    w1, w2, w3 = v["W1"], v["W2"], v["W3"]
    lines: list[str] = []
    add = lines.append
    dry = evaluation["mode"] != "FINAL"
    add("# OAC Ops Health v2: final evaluation on the sealed splits")
    add("")
    if dry:
        add("> **DEV DRY RUN.** These numbers come from DEV stand-in rows, not from the sealed "
            "splits. They are not results.")
        add("")
    if not evaluation["registered_protocol"]:
        add(f"> **Not the registered protocol:** {evaluation['bootstrap_protocol']['resamples']} "
            "bootstrap resamples instead of 2,000.")
        add("")

    def wline(name: str, w: Mapping, where: str) -> str:
        parts = [f"{c}: {_f(x['v2_lower'])} vs upper {_f(x['other_upper'])} "
                 f"({'holds' if x['holds'] else 'FAILS'})" for c, x in w["comparisons"].items()]
        return (f"- **{name} ({where}) {'HOLDS' if w['holds'] else 'FAILS'}.** v2 BA "
                f"{_f(w['v2_point'])} [{_f(w['v2_interval'][0])}, {_f(w['v2_interval'][1])}]; "
                f"v2 lower bound against each baseline's upper bound: " + "; ".join(parts) + ".")

    add(f"## Verdict: v2 **{v['result']}** (MEASURED, SYNTHETIC)")
    add("")
    if v["result"] == "WIN":
        add("All three registered conditions hold. On this synthetic benchmark only, v2's "
            "balanced accuracy interval lies strictly above v1's, the majority baseline's and the "
            "rule baseline's, on nominal test and on the pooled shift suite, and its Brier score "
            "and ECE on nominal test are no worse than v1's. This says nothing about any real "
            "system.")
    else:
        add(f"v2 does **not** win: {', '.join(v['failed'])} "
            f"{'fails' if len(v['failed']) == 1 else 'fail'}. Nothing may describe v2 as better "
            "(PREREGISTRATION section 7).")
    add("")
    add(wline("W1", w1, "nominal test"))
    add(wline("W2", w2, "pooled shift suite"))
    add(f"- **W3 (calibration, nominal test) {'HOLDS' if w3['holds'] else 'FAILS'}.** Brier v2 "
        f"{_f(w3['brier']['v2'], 6)} vs v1 {_f(w3['brier']['v1'], 6)} "
        f"({'<=' if w3['brier']['holds'] else '>'}); ECE v2 {_f(w3['ece']['v2'], 6)} vs v1 "
        f"{_f(w3['ece']['v1'], 6)} ({'<=' if w3['ece']['holds'] else '>'}).")
    add("")
    add("Everything here is **SYNTHETIC** and says nothing about any real transport, device, site, "
        "population or workflow. Labels: **MEASURED** (computed here from the saved score files, "
        "cited below), **REPORTED** (read from a cited file), **MODELED** (reasoning without a "
        "measurement). Unlabeled numbers in tables are MEASURED.")
    add("")
    add("## Provenance (MEASURED)")
    add("")
    op = inputs["opening"]
    add(f"- Opening: `{op['marker_file']}` sha256 `{op['marker_sha256']}`, opened "
        f"{op['opened_utc']} at git HEAD `{op['git_head_at_opening']}`, purpose "
        f"`{op['purpose']}`. The opening happened once, inside `v2/research/final_score.py`.")
    add(f"- Scorer: `{inputs['scorer']['path']}` sha256 `{inputs['scorer']['sha256']}` at "
        f"`{inputs['scorer']['git_head_at_run']}`, Python {inputs['scorer']['python']}. Score "
        f"manifest sha256 `{inputs['score_manifest_sha256']}`.")
    add(f"- Analysis: `{inputs['analysis_script']['path']}` sha256 "
        f"`{inputs['analysis_script']['sha256']}`, reading only the score files below.")
    add("")
    add("| split | rows | positives | score file sha256 | source split sha256 |")
    add("| --- | ---: | ---: | --- | --- |")
    for s, e in inputs["score_files"].items():
        src = op["sealed_splits_sha256"].get(s, "legacy_v1/test.jsonl (not sealed)")
        add(f"| {s} | {e['rows']} | {e['positives']} | `{e['sha256'][:16]}…` | `{src[:16]}…` |"
            if s in op["sealed_splits_sha256"] else
            f"| {s} | {e['rows']} | {e['positives']} | `{e['sha256'][:16]}…` | {src} |")
    add("")
    bp = evaluation["bootstrap_protocol"]
    add(f"Bootstrap: {bp['resamples']} resamples, seed {bp['seed']}, {bp['interval']} 95% "
        f"interval; {bp['paired']}; the pooled shift suite is resampled within each regime "
        f"({', '.join(bp['pooled_shift_strata'])}).")
    add("")
    add("## Primary metric: balanced accuracy, point [95% interval] (MEASURED)")
    add("")
    add("| contender | nominal test | pooled shift suite |")
    add("| --- | --- | --- |")
    for c in CONTENDERS:
        add(f"| {c} | {_ci(prim[TEST][c])} | {_ci(prim[POOLED][c])} |")
    add("")
    add("W1 and W2 compare v2 with v1, majority and rule only. ABL-data and ABL-optimizer are "
        "section 9 ablations (confirmation only).")
    add("")
    add("## Paired differences v2 − v1 (point [95% paired interval], MEASURED)")
    add("")
    add("| set | BA | Brier | ECE |")
    add("| --- | --- | --- | --- |")
    for s, r in sets.items():
        d = r["paired_differences"]["v2_minus_v1"]
        add(f"| {s} | {_ci(d['ba'])} | {_ci(d['brier'])} | {_ci(d['ece'])} |")
    add("")
    add("## Score metrics of the scored contenders (MEASURED)")
    add("")
    add("| set | contender | Brier | ECE | log loss | ROC AUC |")
    add("| --- | --- | ---: | ---: | ---: | ---: |")
    for s, r in sets.items():
        for c in SCORED:
            sr = r["contenders"][c]["score_report"]
            add(f"| {s} | {c} | {_f(sr['brier'])} | {_f(sr['ece'])} | {_f(sr['log_loss'])} | "
                f"{_f(sr['roc_auc'])} |")
    add("")
    add("v1's score is the Hub kernel's `operator_attention_score` as emitted, without "
        "recalibration (AM-5); v1's own card calls it not production calibrated (REPORTED, AM-5).")
    add("")
    add("## Decision metrics and section 10 validity (MEASURED)")
    add("")
    add("Per-class rates are shown only where the four section 10 preconditions hold; otherwise "
        "the contender is `INVALID_BASELINE` and only BA and prevalence are reported.")
    add("")
    add("| set | contender | BA | status | precision | recall | specificity | F1 | false-alert share |")
    add("| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |")
    for s in sets:
        for c in CONTENDERS:
            br = sets[s]["contenders"][c]["binary_report"]
            status = br["baseline_validity"]["status"]
            if br["per_class_rates"] == metrics.INVALID_BASELINE:
                failed = ", ".join(br["baseline_validity"]["failed_preconditions"])
                add(f"| {s} | {c} | {_f(br['balanced_accuracy'])} | INVALID_BASELINE ({failed}) "
                    "| – | – | – | – | – |")
            else:
                pc = br["per_class_rates"]
                add(f"| {s} | {c} | {_f(br['balanced_accuracy'])} | {status} | {_f(pc['precision'])} "
                    f"| {_f(pc['recall'])} | {_f(pc['specificity'])} | {_f(pc['f1'])} | "
                    f"{_f(pc['false_alert_share'])} |")
    add("")
    add("### Validity inputs per contender (MEASURED before the opening, in the scoring process)")
    add("")
    add("| contender | receipt verified | fail-closed probes refused | escaped probes |")
    add("| --- | --- | ---: | --- |")
    for c, vv in evaluation["validity"].items():
        esc = ", ".join(vv["probes_escaped"][:6]) + (" …" if len(vv["probes_escaped"]) > 6 else "")
        add(f"| {c} | {vv['receipt_verified']} | {vv['probes_refused']}/{vv['probes_total']} | "
            f"{esc or 'none'} |")
    add("")
    add("The probe suite is `gates.FAILCLOSED_PROBES` (v2's own suite). v1 was probed through its "
        "own CLI input path; AM-6 records that the two contenders' probe behavior differs by "
        "construction, and the v1 audit separately MEASURED that v1 accepts duplicate JSON keys "
        "(REPORTED from `audit/oac_v1_audit.md`, finding B-7a). Receipt and probe results do not "
        "enter W1–W3.")
    add("")
    add("## v2 conformal layer (Mondrian, alpha 0.10; MEASURED)")
    add("")
    add("| set | marginal coverage | class 0 | class 1 | abstention | BOTH | EMPTY | selective BA | BA without abstention |")
    add("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for s, r in sets.items():
        cov, sel = r["v2_conformal"]["coverage"], r["v2_conformal"]["selective"]
        add(f"| {s} | {_f(cov['marginal'])} | {_f(cov['per_class']['0'])} | "
            f"{_f(cov['per_class']['1'])} | {_f(sel['abstention_rate'])} | "
            f"{_f(sel['abstention_rate_both'])} | {_f(sel['abstention_rate_empty'])} | "
            f"{_f(sel['selective_balanced_accuracy'])} | "
            f"{_f(r['contenders']['v2']['binary_report']['balanced_accuracy'])} |")
    add("")
    add("The class-conditional coverage guarantee (at least 0.90 per class) holds only under "
        "exchangeability with the nominal conformal split (MODELED, section 6). Under shift and on "
        "legacy_v1_test the coverage figures are descriptive.")
    add("")
    add("## Disaggregation (MEASURED)")
    add("")
    add("BA by regime, point [95% interval]:")
    add("")
    add("| set | " + " | ".join(CONTENDERS) + " |")
    add("| --- |" + " --- |" * len(CONTENDERS))
    for s, block in evaluation["disaggregation"]["ba_by_regime"].items():
        add(f"| {s} | " + " | ".join(_ci(block[c]) for c in CONTENDERS) + " |")
    add("")
    add("Nominal test, BA by feature value (point estimates):")
    add("")
    add("| feature = value | rows | positives | " + " | ".join(CONTENDERS) + " |")
    add("| --- | ---: | ---: |" + " ---: |" * len(CONTENDERS))
    for feature, block in evaluation["disaggregation"]["test"].items():
        for value, b in block.items():
            add(f"| {feature} = {value} | {b['rows']} | {b['positives']} | "
                + " | ".join(_f(b["balanced_accuracy"][c]) for c in CONTENDERS) + " |")
    add("")
    add("## legacy_v1_test (v1's public 240-row test split; secondary, REPORTED comparability)")
    add("")
    lg = sets[LEGACY]
    for c in CONTENDERS:
        br = lg["contenders"][c]["binary_report"]
        add(f"- {c}: BA {_ci(lg['bootstrap']['intervals'][f'ba:{c}'])}"
            + (f", Brier {_f(lg['contenders'][c]['score_report']['brier'])}, ECE "
               f"{_f(lg['contenders'][c]['score_report']['ece'])}" if c in SCORED else "")
            + f" ({br['baseline_validity']['status']})")
    add("")
    add("Before the opening, the scorer checked that v1 reproduces its published confusion on "
        "this split (tp 40, fp 57, tn 132, fn 11; published values REPORTED from the v1 receipt, "
        "reproduction MEASURED).")
    add("")
    add("## Generator-truth recovery (MEASURED from the frozen parameters)")
    add("")
    tr = evaluation["truth_recovery"]
    add("| model | sign agreement | Spearman vs truth | relative L2 (intercept, weights) | disagreeing |")
    add("| --- | ---: | ---: | ---: | --- |")
    for c in MODEL_FILES:
        t = tr[c]
        add(f"| {c} | {t['sign_agreement_k']}/{t['sign_agreement_of']} | "
            f"{_f(t['spearman_weights_vs_truth'])} | {_f(t['relative_l2_intercept_and_weights'])} | "
            f"{', '.join(t['sign_disagreeing']) or 'none'} |")
    add("")
    add("## Ablations (section 9; test values confirm the validation decisions)")
    add("")
    ab = evaluation["ablations"]
    add("| ablation | set | ablation BA | v2 − ablation BA | ablation Brier | v2 Brier | ablation ECE | v2 ECE |")
    add("| --- | --- | --- | --- | ---: | ---: | ---: | ---: |")
    for c in ("ABL-data", "ABL-optimizer"):
        for s in (TEST, POOLED):
            a = ab[c][s]
            add(f"| {c} | {s} | {_ci(a['ba'])} | {_ci(a['ba_v2_minus'])} | {_f(a['brier'])} | "
                f"{_f(a['v2_brier'])} | {_f(a['ece'])} | {_f(a['v2_ece'])} |")
    add("")
    add(f"- ABL-calibration: {ab['ABL-calibration']['status']} ({ab['ABL-calibration']['reason']}).")
    t_conf = ab["ABL-conformal"]["by_set"][TEST]
    add(f"- ABL-conformal (kept on validation: {ab['ABL-conformal']['validation_decision_keep']}): "
        f"on nominal test, selective BA {_f(t_conf['selective_ba_with_abstention'])} on the "
        f"non-abstained rows at abstention rate {_f(t_conf['abstention_rate'])}, against BA "
        f"{_f(t_conf['ba_without_abstention'])} without abstention. Selective BA is computed on a "
        "different, easier subset of rows and is not comparable to the primary metric.")
    add("")
    add("## Reliability (nominal test, v2 and v1; MEASURED)")
    add("")
    add("![Reliability diagrams](reliability.svg)")
    add("")
    add("| bin | v2 n | v2 mean score | v2 fraction positive | v1 n | v1 mean score | v1 fraction positive |")
    add("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    t2 = sets[TEST]["contenders"]["v2"]["score_report"]["reliability_table"]
    t1 = sets[TEST]["contenders"]["v1"]["score_report"]["reliability_table"]
    for b2, b1 in zip(t2, t1, strict=True):
        rng = f"[{b2['lower']:.1f}, {b2['upper']:.1f}{']' if b2['upper_closed'] else ')'}"
        add(f"| {rng} | {b2['count']} | {_f(b2['mean_score'])} | {_f(b2['fraction_positive'])} | "
            f"{b1['count']} | {_f(b1['mean_score'])} | {_f(b1['fraction_positive'])} |")
    add("")
    add("Every other reliability table (each set, each scored contender) is in "
        "`v2/results/final_evaluation.json`.")
    add("")
    add("## What this does not show")
    add("")
    add("- Nothing about any real system: the data, labels and shifts are synthetic, generated "
        "from a logistic rule that the v2 model class matches exactly (MODELED, section 2).")
    add("- The shift regimes are four pre-registered synthetic perturbations. Other shifts were "
        "not tested (MODELED).")
    add("- Receipts attest integrity and origin, not quality.")
    if v["result"] != "WIN":
        add("- v2 is not better than v1 under the registered rule, and no statement here may "
            "describe it as better.")
    add("")
    add(f"Generated {evaluation['generated_utc']} by `v2/research/final_analyze.py` from "
        "`v2/results/final/` (re-runnable; it opens no data split).")
    return "\n".join(lines) + "\n"


def write_outputs(evaluation: Mapping, *, out_json: Path, out_md: Path, out_svg: Path) -> dict:
    written = {}
    for path, data in ((out_json, _json_bytes(evaluation)),
                       (out_svg, render_svg(evaluation).encode("utf-8")),
                       (out_md, render_md(evaluation).encode("utf-8"))):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        if path.read_bytes() != data:
            raise OSError(f"{path.name}: re-read bytes differ from the bytes written")
        written[path.name] = {"sha256": sha256_bytes(data), "bytes": len(data)}
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--score-dir", type=Path, default=SCORE_DIR)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="write the three outputs here instead of v2/results (DEV use)")
    parser.add_argument("--resamples", type=int, default=RESAMPLES,
                        help="DEV use only; the registered value is 2,000")
    args = parser.parse_args(argv)
    real = args.score_dir.resolve() == SCORE_DIR.resolve()
    if real and (args.resamples != RESAMPLES or args.out_dir is not None):
        parser.error("the real score directory is analyzed only with the registered protocol "
                     "and the registered output paths")
    evaluation = analyze(args.score_dir, parts_dir=PARTS_DIR if real else None,
                         resamples=args.resamples, log=lambda s: print(s, flush=True))
    if real and evaluation["mode"] != "FINAL":
        raise AnalysisError("v2/results/final holds a non-final manifest")
    out = args.out_dir
    written = write_outputs(
        evaluation,
        out_json=(out / OUT_JSON.name) if out else OUT_JSON,
        out_md=(out / OUT_MD.name) if out else OUT_MD,
        out_svg=(out / OUT_SVG.name) if out else OUT_SVG,
    )
    for name, info in written.items():
        print(f"wrote {name}: {info['bytes']} bytes sha256 {info['sha256']}", flush=True)
    print(f"verdict: {evaluation['verdict']['result']} (failed: {evaluation['verdict']['failed']})",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
