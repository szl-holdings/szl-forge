"""Lane L2 runner — Khipu abstention bench (szl-hf-frontier#9).

Same held-out gate as L3, `abstain`-kind probes. The disclosed operating
evidence today: KHIPU-R2 abstain 3/6 MEASURED, declared not a pass;
khipu-r3 abstain 0/6. This runner records numerical comparisons, not bound
candidate qualification. Freezing an operating point requires separately
governed hidden-set evidence — never train loss or a public smoke run. The committed
`probes/khipu_abstain_smoke_v1.jsonl` wires CI; the promotion handle
set stays private, only its sha256 publishes.
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys

from frontier.harness.heldout_gate import run_gate

# KHIPU-R2: abstain 3/6 MEASURED, declared not a pass. A candidate must
# strictly beat the disclosed line before an operating point can freeze.
KHIPU_BASELINE = {"abstain": 3}


def mock_generate(messages):
    """CI-only synthetic abstention; no model or Brain content is loaded."""
    return json.dumps({
        "planId": "plan-synthetic-smoke-abstain",
        "capabilityProfile": "SZL-Khipu-1.5B-BrainNavigator",
        "provenance": "SYNTHETIC",
        "query": messages[0]["content"],
        "contentAccess": "HANDLES_ONLY",
        "candidates": [{
            "nodeId": "node://khipu-synthetic/smoke-unrelated",
            "nodeKind": "INDEX",
            "label": "SIMULATED",
            "note": "Synthetic nonmatching handle; no node content is embedded.",
        }],
        "decision": "ABSTAIN",
        "steps": [],
        "citedNodeIds": [],
        "groundedOnly": True,
        "brainBinding": {
            "protocol": "khipu-retrieval",
            "status": "NOT_RESOLVED",
            "note": "No node content was resolved by this software-only mock.",
        },
        "controllerBoundary": "Proposal only; the A11oy controller validates and resolves handles outside the model.",
        "abstainReason": "No offered handle supports this query.",
    })


def load_generate(spec):
    module_name, _, func_name = spec.partition(":")
    if not module_name or not func_name:
        raise ValueError("generate plugin must be 'module:function'")
    return getattr(importlib.import_module(module_name), func_name)


def main(argv=None):
    ap = argparse.ArgumentParser(description="L2 Khipu abstention bench runner")
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--probes", required=True)
    ap.add_argument("--probe-sha256", default=None,
                    help="declared probe set hash; mismatch -> INVALID")
    ap.add_argument("--generate", default=None,
                    help="module:function providing generate(messages) -> str")
    ap.add_argument("--mock", action="store_true", help="CI smoke mode")
    ap.add_argument("--method", default="")
    ap.add_argument("--out", default=None, help="write receipt JSON here")
    args = ap.parse_args(argv)

    if not args.mock and not args.generate:
        ap.error("provide --generate module:function or --mock")
    generate = mock_generate if args.mock else load_generate(args.generate)

    receipt = run_gate(
        artifact=args.artifact, probes_path=args.probes, generate=generate,
        declared_probe_sha256=args.probe_sha256, baseline=KHIPU_BASELINE,
        method=args.method or ("mock smoke" if args.mock else args.generate),
        env={"python": platform.python_version()}, mock=args.mock)

    text = json.dumps(receipt, indent=2, sort_keys=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    print(receipt["gate"])
    return 0 if receipt["gate"] in ("PASS", "FAIL") else 2


if __name__ == "__main__":
    sys.exit(main())
