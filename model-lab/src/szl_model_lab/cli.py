"""Explicit local commands. No command pushes, merges, downloads or launches paid jobs."""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path
from .catalog import TRACKS, catalog


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="szl-model-lab")
    s = p.add_subparsers(dest="command", required=True)
    s.add_parser("plan", help="Print research tracks; no side effects")
    v = s.add_parser("validate", help="Validate a local, split-labeled dataset")
    v.add_argument("--track", choices=TRACKS, required=True)
    v.add_argument("--data", type=Path, required=True)
    t = s.add_parser("train", help="Explicit CPU research run, never publication")
    t.add_argument("--track", choices=TRACKS, required=True)
    t.add_argument("--data", type=Path, required=True)
    t.add_argument("--output", type=Path, required=True)
    t.add_argument("--source-revision", required=True)
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--seed", type=int, default=20260913)
    t.add_argument("--batch-size", type=int, default=32)
    t.add_argument("--acknowledge-research-only", action="store_true", required=True)
    e = s.add_parser("evaluate-test", help="Explicit frozen-test observation; no promotion")
    e.add_argument("--artifact", type=Path, required=True)
    e.add_argument("--data", type=Path, required=True)
    e.add_argument("--output", type=Path, required=True)
    e.add_argument("--acknowledge-heldout-evaluation", action="store_true", required=True)
    serve = s.add_parser("serve", help="Loopback-only authenticated read-only workbench")
    serve.add_argument("--port", type=int, default=8765)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "plan":
            result = {"tracks": catalog(), "training_started": False, "hf_publication": False}
        elif args.command == "validate":
            from .data import Dataset
            result = Dataset.read(args.data, args.track).summary()
        elif args.command == "train":
            import torch
            from .training import train_candidate
            torch.set_num_threads(1)
            result = train_candidate(args.data, args.track, args.output, args.source_revision,
                                     args.epochs, args.seed, args.batch_size)
        elif args.command == "evaluate-test":
            from .training import evaluate_test
            from .safeio import canonical_bytes, write_new
            if args.output.exists():
                raise ValueError("evaluation_output_must_not_exist")
            result = evaluate_test(args.artifact, args.data)
            write_new(args.output, canonical_bytes(result))
        else:
            import uvicorn
            from .app import Settings, create_app
            if not 1024 <= args.port <= 65535:
                raise ValueError("unprivileged_port_required")
            # Avoid proxy-header trust and body-bearing access logs. No public bind flag.
            uvicorn.run(create_app(Settings.from_env()), host="127.0.0.1", port=args.port,
                        access_log=False, proxy_headers=False)
            return 0
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        # Local CLI errors never print tokens, request bodies, or model outputs.
        print(json.dumps({"status": "BLOCKED", "error_type": type(exc).__name__}), file=sys.stderr)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
