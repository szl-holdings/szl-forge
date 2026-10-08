"""Timing benchmark of the paired bootstrap on 50,000 DEV rows (SYNTHETIC; DEV data only).

This is a feasibility/timing measurement for stage 2, not an evaluation: the values it prints
come from DEV rows (split name "dev"), a DEV fixture model with arbitrary fixed settings, and must
never be cited as results.

Layout (fixed): DEV rows [0:4096] train, [4096:6144] calibration, [6144:8192] conformal,
[8192:10240] threshold, [10240:60240] the 50,000 benchmark rows.  Contenders: the DEV fixture
(l2 = 1e-3, Platt), v1 as published (embedded weights, threshold 0.16), majority (always 0),
rule (consecutive_failures > 0).  Statistics per resample: BA x 4 contenders, and Brier, log
loss, ECE and AUROC for the two scored contenders (12 statistics).  Runs 2,000 resamples with
seed 20260927 unstratified, then with 4 equal strata (the pooled-shift layout).

Usage: PYTHONUTF8=1 py -3.12 -B -m v2.research.bench_bootstrap
"""

from __future__ import annotations

import json
import platform
import sys
import time

from . import bootstrap, generator, metrics, pipeline, v1_gd
from .dataio import labels_of

SEED = 20260927
EVAL = slice(10240, 60240)


def main() -> int:
    started = time.perf_counter()
    dev = generator.dev_rows(60240)
    artifact, _diag = pipeline.build_candidate(
        train_rows=dev[0:4096],
        calibration_rows=dev[4096:6144],
        conformal_rows=dev[6144:8192],
        threshold_rows=dev[8192:10240],
        l2=1e-3,
        calibrator="platt",
        split_names={"train": "dev[0:4096]", "calibration": "dev[4096:6144]",
                     "conformal": "dev[6144:8192]", "threshold": "dev[8192:10240]"},
    )
    rows = dev[EVAL]
    y = labels_of(rows)
    from . import kernel_bridge

    scorer = kernel_bridge.kernel().ArtifactScorer(artifact)
    vectors = pipeline.normalized_vectors(rows)
    evals = [scorer.evaluate(v) for v in vectors]
    s_dev = [e["calibrated_score"] for e in evals]
    d_dev = [1 if e["threshold_decision"] else 0 for e in evals]
    s_v1 = [v1_gd.v1_committed_score(v) for v in vectors]
    d_v1 = [1 if s >= v1_gd.V1_DECISION_THRESHOLD else 0 for s in s_v1]
    d_major = [0] * len(rows)
    d_rule = [1 if row["features"]["consecutive_failures"] > 0 else 0 for row in rows]
    prep_seconds = time.perf_counter() - started

    t0 = time.perf_counter()
    binary = {name: bootstrap.BinaryStats(y, d) for name, d in
              (("dev_fixture", d_dev), ("v1", d_v1), ("majority", d_major), ("rule", d_rule))}
    scored = {name: bootstrap.ScoreStats(y, s) for name, s in (("dev_fixture", s_dev), ("v1", s_v1))}
    statistics = {f"ba:{name}": stats.balanced_accuracy for name, stats in binary.items()}
    for name, stats in scored.items():
        statistics[f"brier:{name}"] = stats.brier
        statistics[f"log_loss:{name}"] = stats.log_loss
        statistics[f"ece:{name}"] = stats.ece
        statistics[f"auroc:{name}"] = stats.auroc
    setup_seconds = time.perf_counter() - t0

    ones = [1] * len(rows)
    point_check = {
        "ba:dev_fixture": statistics["ba:dev_fixture"](ones) == metrics.balanced_accuracy(y, d_dev),
        "brier:v1": statistics["brier:v1"](ones) == metrics.brier(y, s_v1),
        "log_loss:v1": statistics["log_loss:v1"](ones) == metrics.log_loss(y, s_v1),
        "ece:v1": statistics["ece:v1"](ones) == metrics.ece(y, s_v1),
        "auroc:v1": statistics["auroc:v1"](ones) == metrics.roc_auc(y, s_v1),
    }

    runs = {}
    strata = [list(range(i, i + 12500)) for i in range(0, 50000, 12500)]
    for label, kwargs in (("unstratified", {}), ("stratified_4x12500", {"strata": strata})):
        t = time.perf_counter()
        values = bootstrap.paired_bootstrap(len(rows), statistics, seed=SEED, **kwargs)
        seconds = time.perf_counter() - t
        diff = bootstrap.differences(values["ba:dev_fixture"], values["ba:v1"])
        runs[label] = {
            "seconds": round(seconds, 2),
            "resamples": bootstrap.RESAMPLES,
            "statistics_per_resample": len(statistics),
            "ms_per_resample": round(1000 * seconds / bootstrap.RESAMPLES, 2),
            "undefined_values": sum(v is None for vals in values.values() for v in vals),
            "dev_only_intervals_not_results": {
                "ba:dev_fixture": bootstrap.interval(values["ba:dev_fixture"]),
                "ba:v1": bootstrap.interval(values["ba:v1"]),
                "ba:dev_fixture-v1": bootstrap.interval(diff),
            },
        }
    report = {
        "what": "paired bootstrap timing benchmark on 50,000 DEV rows (SYNTHETIC, DEV only)",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "rows": len(rows),
        "seed": SEED,
        "prep_seconds": round(prep_seconds, 2),
        "setup_seconds": round(setup_seconds, 2),
        "weighted_equals_metrics_py_at_unit_weights": point_check,
        "runs": runs,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if all(point_check.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
