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
    resource = s.add_parser("storage-plan", help="Offline Spark KV calculation; never dispatch")
    resource.add_argument("--tokens", type=int, default=32768)
    resource.add_argument("--batch", type=int, default=1)
    resource.add_argument("--config", type=Path)
    resource.add_argument("--config-sha256")
    resource.add_argument("--model-revision")
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
    t.add_argument("--archive-stage-manifest", type=Path,
                   help="Local staged-data manifest; requires its independently recorded SHA-256")
    t.add_argument("--archive-stage-sha256")
    stage = s.add_parser("stage-archive", help="Copy pinned local archive JSONL to a new local stage")
    stage.add_argument("--request", type=Path, required=True)
    stage.add_argument("--output", type=Path, required=True)
    stage.add_argument("--materialized-input-root", type=Path,
                       help="Plain local copy of archive input when a cloud reparse placeholder is unsupported")
    copied = s.add_parser("archive-candidate", help="Copy a completed candidate into the local archive")
    copied.add_argument("--candidate", type=Path, required=True)
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
        elif args.command == "storage-plan":
            from .storage import SparkShape, memory_plan, reference_plan
            from .safeio import read_regular
            supplied = (args.config is not None, args.config_sha256 is not None, args.model_revision is not None)
            if any(supplied) and not all(supplied):
                raise ValueError("config_digest_and_revision_required_together")
            if args.config is None:
                result = reference_plan(args.tokens, args.batch)
            else:
                raw = read_regular(args.config, 65536)
                shape = SparkShape.from_config(raw, args.config_sha256)
                result = memory_plan(shape, tokens=args.tokens, batch=args.batch,
                                     model_revision=args.model_revision, config_sha256=args.config_sha256)
        elif args.command == "validate":
            from .data import Dataset
            result = Dataset.read(args.data, args.track).summary()
        elif args.command == "stage-archive":
            from .archive_stage import (_archive_root, _local_root,
                                        require_local_workspace, stage_archive)
            require_local_workspace(args.output, _local_root(), _archive_root())
            result = stage_archive(args.materialized_input_root or _archive_root(), args.request,
                                   args.output, _local_root())
        elif args.command == "archive-candidate":
            from .archive_stage import _archive_root, _local_root, archive_candidate
            result = archive_candidate(args.candidate, _archive_root(), _local_root())
        elif args.command == "train":
            if (args.archive_stage_manifest is None) != (args.archive_stage_sha256 is None):
                raise ValueError("archive_stage_manifest_and_digest_required_together")
            if args.archive_stage_manifest is not None:
                from .archive_stage import (_archive_root, _local_root, require_local_workspace,
                                            require_stage_output_disjoint, verify_stage)
                require_local_workspace(args.data, _local_root(), _archive_root())
                require_local_workspace(args.output, _local_root(), _archive_root())
                require_stage_output_disjoint(args.output, args.archive_stage_manifest)
                verify_stage(args.archive_stage_manifest, args.archive_stage_sha256,
                             args.data, args.track)
            import torch
            from .training import train_candidate
            torch.set_num_threads(1)
            result = train_candidate(args.data, args.track, args.output, args.source_revision,
                                     args.epochs, args.seed, args.batch_size,
                                     archive_stage_manifest=args.archive_stage_manifest,
                                     archive_stage_sha256=args.archive_stage_sha256)
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
