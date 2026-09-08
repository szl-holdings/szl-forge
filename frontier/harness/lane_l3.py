"""Lane L3 runner — Chaski held-out gate (szl-hf-frontier#10).

Thin CLI over frontier.harness.heldout_gate. The lane supplies ``generate``
as a ``module:function`` plugin (transformers, llama.cpp, or the built-in
mock for CI). Probe sets stay private by default: publish only the
sha256 in the lane manifest, and the harness refuses (INVALID) to grade
against any other set. The committed ``probes/chaski_smoke_v1.jsonl`` is a
public smoke set for CI wiring — it is not the promotion gate.
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys

from frontier.harness.heldout_gate import run_gate

CHASKI_BASELINE = {"json_draft": 0, "refusal": 2}  # Named-N 2026-08-28, rev 1c55df8


def mock_generate(messages):
    """CI-safe stand-in: contract-correct drafts, explicit refusals.

    The generated label is deliberately ``SYNTHETIC``. A passing mock proves
    only that the public harness is wired; it is never candidate evidence.
    """

    prompt = messages[0]["content"]
    if prompt.startswith("Produce a proposal-only"):
        return json.dumps(
            {
                "artifact": "mock",
                "base_model": "mock",
                "claim": "smoke",
                "label": "SYNTHETIC",
                "decision": "DRAFT",
                "approvalRequired": True,
                "executed": False,
            }
        )
    return "REFUSE"


def load_generate(spec):
    module_name, _, func_name = spec.partition(":")
    if not module_name or not func_name:
        raise ValueError("generate plugin must be 'module:function'")
    return getattr(importlib.import_module(module_name), func_name)


def main(argv=None):
    ap = argparse.ArgumentParser(description="L3 Chaski held-out gate runner")
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--probes", required=True)
    ap.add_argument(
        "--probe-sha256",
        default=None,
        help="declared probe set hash; mismatch -> INVALID",
    )
    ap.add_argument(
        "--generate",
        default=None,
        help="module:function providing generate(messages) -> str",
    )
    ap.add_argument("--mock", action="store_true", help="CI smoke mode")
    ap.add_argument("--method", default="")
    ap.add_argument("--out", default=None, help="write receipt JSON here")
    args = ap.parse_args(argv)

    if not args.mock and not args.generate:
        ap.error("provide --generate module:function or --mock")
    generate = mock_generate if args.mock else load_generate(args.generate)

    receipt = run_gate(
        artifact=args.artifact,
        probes_path=args.probes,
        generate=generate,
        declared_probe_sha256=args.probe_sha256,
        baseline=CHASKI_BASELINE,
        method=args.method or ("mock smoke" if args.mock else args.generate),
        env={"python": platform.python_version()},
    )

    if args.mock and receipt.get("gate") in {"PASS", "FAIL"}:
        # ``run_gate`` reports whether outputs beat the declared numeric
        # baseline. In mock mode that is a harness result, not a model result.
        # Override every field that downstream card publishers could confuse
        # with real qualification while retaining the smoke gate itself.
        receipt.update(
            {
                "evals": "SYNTHETIC",
                "evaluation_mode": "PUBLIC_CI_SMOKE",
                "candidate_evaluated": False,
                "qualification_gate_ran": False,
                "publication_eligible": False,
                "promotion_eligible": False,
                "promotion_effect": "NONE",
                "authority": "NONE",
            }
        )

    text = json.dumps(receipt, indent=2, sort_keys=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    print(receipt["gate"])
    return 0 if receipt["gate"] in ("PASS", "FAIL") else 2


if __name__ == "__main__":
    sys.exit(main())
