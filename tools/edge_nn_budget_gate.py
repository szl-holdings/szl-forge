#!/usr/bin/env python3
"""Source-only, fail-closed edge neural-network deployment feasibility calculator.

All budgets are user-supplied claims; no firmware, hardware, accuracy, timing,
flash, power or model weight has been independently measured by this program.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

SCHEMA = "szl.edge-nn-budget-input/v1"
MAX_BYTES = 65536


def _positive_int(x):
    return type(x) is int and 0 < x <= 10**9


def assess(spec):
    if (
        type(spec) is not dict
        or set(spec) != {"schema", "layers", "device", "timing", "artifact", "accuracy"}
        or spec.get("schema") != SCHEMA
    ):
        return {"state": "HOLD", "reasons": ["INVALID_SCHEMA"]}
    layers = spec["layers"]
    if (
        type(layers) is not list
        or not 2 <= len(layers) <= 16
        or not all(_positive_int(x) and x <= 65536 for x in layers)
    ):
        return {"state": "HOLD", "reasons": ["INVALID_TOPOLOGY"]}
    expected = {"flash_bytes", "ram_bytes", "clock_hz", "cycles_per_mac_upper"}
    dev = spec["device"]
    timing = spec["timing"]
    artifact = spec["artifact"]
    accuracy = spec["accuracy"]
    if (
        type(dev) is not dict
        or set(dev) != expected
        or not all(_positive_int(dev[k]) for k in expected)
    ):
        return {"state": "HOLD", "reasons": ["INVALID_DEVICE_BUDGET"]}
    if (
        type(timing) is not dict
        or set(timing) != {"decision_deadline_ms", "non_mac_cycles_upper"}
        or type(timing["decision_deadline_ms"]) not in (float, int)
        or not math.isfinite(timing["decision_deadline_ms"])
        or timing["decision_deadline_ms"] <= 0
        or type(timing["non_mac_cycles_upper"]) is not int
        or timing["non_mac_cycles_upper"] < 0
    ):
        return {"state": "HOLD", "reasons": ["INVALID_TIMING"]}
    if type(artifact) is not dict or set(artifact) != {
        "weight_bits",
        "code_bytes",
        "activation_bytes_per_value",
        "scratch_bytes",
        "weight_sha256",
    }:
        return {"state": "HOLD", "reasons": ["INVALID_ARTIFACT"]}
    if (
        artifact["weight_bits"] not in (4, 8, 16, 32)
        or type(artifact["weight_bits"]) is not int
        or any(
            type(artifact[k]) is not int or artifact[k] < 0
            for k in ("code_bytes", "scratch_bytes")
        )
        or artifact["activation_bytes_per_value"] not in (1, 2, 4)
        or type(artifact["activation_bytes_per_value"]) is not int
        or type(artifact["weight_sha256"]) is not str
        or len(artifact["weight_sha256"]) != 64
        or any(c not in "0123456789abcdef" for c in artifact["weight_sha256"])
    ):
        return {"state": "HOLD", "reasons": ["INVALID_ARTIFACT"]}
    if (
        type(accuracy) is not dict
        or set(accuracy)
        != {
            "baseline_correct",
            "candidate_correct",
            "heldout_count",
            "paired_and_independent",
        }
        or any(
            type(accuracy[k]) is not int or accuracy[k] < 0
            for k in ("baseline_correct", "candidate_correct", "heldout_count")
        )
        or type(accuracy["paired_and_independent"]) is not bool
    ):
        return {"state": "HOLD", "reasons": ["INVALID_ACCURACY"]}
    macs = sum(a * b for a, b in zip(layers, layers[1:]))
    params = sum((a + 1) * b for a, b in zip(layers, layers[1:]))
    packed_weights = (params * artifact["weight_bits"] + 7) // 8
    flash = packed_weights + artifact["code_bytes"]
    # Two live layer buffers; no assumption of in-place safe activation.
    ram = (
        2 * max(layers) * artifact["activation_bytes_per_value"]
        + artifact["scratch_bytes"]
    )
    cycles = macs * dev["cycles_per_mac_upper"] + timing["non_mac_cycles_upper"]
    ms = 1000 * cycles / dev["clock_hz"]
    n = accuracy["heldout_count"]
    reasons = []
    if flash > dev["flash_bytes"]:
        reasons.append("FLASH_BUDGET_EXCEEDED")
    if ram > dev["ram_bytes"]:
        reasons.append("RAM_BUDGET_EXCEEDED")
    if ms > timing["decision_deadline_ms"]:
        reasons.append("LATENCY_BUDGET_EXCEEDED")
    if (
        not accuracy["paired_and_independent"]
        or n < 30
        or accuracy["baseline_correct"] > n
        or accuracy["candidate_correct"] > n
    ):
        reasons.append("ACCURACY_EVIDENCE_INSUFFICIENT")
    elif accuracy["candidate_correct"] < accuracy["baseline_correct"]:
        reasons.append("ACCURACY_REGRESSION")
    # Even structurally admissible inputs are only claimed budgets.
    return {
        "state": "HOLD" if reasons else "STRUCTURALLY_FEASIBLE_UNVERIFIED",
        "reasons": sorted(reasons),
        "parameters": params,
        "macs": macs,
        "packed_weights_bytes": packed_weights,
        "flash_estimate_bytes": flash,
        "ram_estimate_bytes": ram,
        "latency_upper_estimate_ms": ms,
        "physical_hardware_tested": False,
        "weight_bytes_verified": False,
        "deployment_qualified": False,
        "real_power_measured": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.input.is_symlink() or args.output.exists() or args.output.is_symlink():
            raise ValueError("UNSAFE_PATH")
        raw = args.input.read_bytes()
        if len(raw) > MAX_BYTES:
            raise ValueError("INPUT_SIZE")
        result = assess(json.loads(raw))
        result["schema"] = "szl.edge-nn-budget-result/v1"
        result["input_sha256"] = hashlib.sha256(raw).hexdigest()
        b = (
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode()
        with args.output.open("xb") as stream:
            stream.write(b)
        print(
            json.dumps(
                {
                    "state": result["state"],
                    "output_sha256": hashlib.sha256(b).hexdigest(),
                }
            )
        )
        return 0 if result["state"] == "STRUCTURALLY_FEASIBLE_UNVERIFIED" else 2
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(
            json.dumps({"state": "HOLD", "reason": type(exc).__name__}), file=sys.stderr
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
