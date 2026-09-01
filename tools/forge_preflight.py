#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""SZL Forge preflight — prove the environment BEFORE a training run starts.

Every check is labeled honestly: MEASURED for facts observed on this machine,
UNAVAILABLE for anything not installed or not detectable. A missing GPU is
reported as CUDA UNAVAILABLE, never guessed. Stdlib only; safe on any host.

  python tools/forge_preflight.py --dataset szl_dataset.jsonl
  python tools/forge_preflight.py --dataset szl_dataset.jsonl --no-require-cuda
  python tools/forge_preflight.py --dataset szl_dataset.jsonl --out preflight.json

Exit codes: 0 = READY, 1 = NOT_READY, 2 = usage error.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)  # so `import validate_sft_dataset` works as script or module

SCHEMA = "szl.forge-preflight/v1"
TRAINING_PACKAGES = (
    "torch",
    "unsloth",
    "trl",
    "peft",
    "datasets",
    "accelerate",
    "bitsandbytes",
)
DEFAULT_MIN_DISK_GIB = 20.0  # 4-bit base + merged_16bit output headroom


def _package_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for name in TRAINING_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "UNAVAILABLE"
    return versions


def _cuda_state() -> Dict[str, Any]:
    """Import torch only if installed; never fail, never guess."""
    try:
        import torch  # noqa: PLC0415 - intentional lazy optional import
    except Exception:
        return {
            "status": "UNAVAILABLE",
            "detail": "torch is not importable in this environment",
        }
    try:
        available = bool(torch.cuda.is_available())
    except Exception as exc:  # pragma: no cover - driver-specific
        return {"status": "UNAVAILABLE", "detail": f"cuda probe failed: {exc}"}
    if not available:
        return {
            "status": "UNAVAILABLE",
            "detail": "torch imports but torch.cuda.is_available() is False",
        }
    return {
        "status": "MEASURED",
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "vram_gib": round(
            torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
        ),
    }


def run_preflight(
    dataset: str,
    workdir: str = ".",
    require_cuda: bool = True,
    min_disk_gib: float = DEFAULT_MIN_DISK_GIB,
    min_examples: int = 1,
) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []
    blocking: List[str] = []

    version = sys.version_info
    python_ok = version >= (3, 9)
    checks.append(
        {
            "name": "python",
            "label": "MEASURED",
            "value": f"{version.major}.{version.minor}.{version.micro}",
            "ok": python_ok,
        }
    )
    if not python_ok:
        blocking.append("python >= 3.9 required")

    usage = shutil.disk_usage(os.path.abspath(workdir))
    free_gib = round(usage.free / (1024**3), 2)
    disk_ok = free_gib >= min_disk_gib
    checks.append(
        {
            "name": "disk_free_gib",
            "label": "MEASURED",
            "value": free_gib,
            "required_gib": min_disk_gib,
            "ok": disk_ok,
        }
    )
    if not disk_ok:
        blocking.append(f"free disk {free_gib} GiB < required {min_disk_gib} GiB")

    versions = _package_versions()
    for name, value in versions.items():
        checks.append(
            {
                "name": f"package:{name}",
                "label": "MEASURED" if value != "UNAVAILABLE" else "UNAVAILABLE",
                "value": value,
                "ok": value != "UNAVAILABLE",
            }
        )
        if value == "UNAVAILABLE":
            blocking.append(f"package {name} is not installed")

    cuda = _cuda_state()
    cuda_ok = cuda.get("status") == "MEASURED"
    checks.append(
        {
            "name": "cuda",
            "label": cuda.get("status", "UNAVAILABLE"),
            "value": cuda,
            "ok": cuda_ok,
            "required": require_cuda,
        }
    )
    if require_cuda and not cuda_ok:
        blocking.append("CUDA is required for QLoRA training and is UNAVAILABLE")

    dataset_report: Dict[str, Any]
    if os.path.isfile(dataset):
        from validate_sft_dataset import validate_dataset

        dataset_report = validate_dataset(dataset, min_examples=min_examples)
    else:
        dataset_report = {
            "schema": "szl.forge-sft-dataset-report/v1",
            "path": dataset,
            "status": "INVALID",
            "records": 0,
            "errors": [f"dataset not found: {dataset}"],
        }
    dataset_ok = dataset_report["status"] == "VALID"
    checks.append(
        {
            "name": "dataset",
            "label": "MEASURED",
            "value": {
                "path": dataset,
                "status": dataset_report["status"],
                "records": dataset_report["records"],
                "dataset_sha256": dataset_report.get("dataset_sha256"),
            },
            "ok": dataset_ok,
        }
    )
    if not dataset_ok:
        blocking.append("dataset validation failed — refusing to start training")

    status = "READY" if not blocking else "NOT_READY"
    payload: Dict[str, Any] = {
        "schema": SCHEMA,
        "status": status,
        "require_cuda": require_cuda,
        "checks": checks,
        "blocking_reasons": blocking,
        "dataset_report": dataset_report,
        "honesty": (
            "READY means only that this environment passed preflight. It is "
            "not a training receipt: a run still needs its own artifact, "
            "evaluation, and receipt evidence before any promotion claim."
        ),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    payload["preflight_sha256"] = "sha256:" + hashlib.sha256(canonical).hexdigest()
    return payload


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="szl_dataset.jsonl")
    parser.add_argument("--workdir", default=".")
    parser.add_argument(
        "--no-require-cuda",
        action="store_true",
        help="report CUDA state honestly without blocking (CPU audits only)",
    )
    parser.add_argument("--min-disk-gib", type=float, default=DEFAULT_MIN_DISK_GIB)
    parser.add_argument("--min-examples", type=int, default=1)
    parser.add_argument("--out", help="write the preflight receipt JSON here")
    args = parser.parse_args(argv)

    report = run_preflight(
        dataset=args.dataset,
        workdir=args.workdir,
        require_cuda=not args.no_require_cuda,
        min_disk_gib=args.min_disk_gib,
        min_examples=args.min_examples,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print(f"preflight: {report['status']}")
    for reason in report["blocking_reasons"]:
        print(f"blocking: {reason}")
    print(f"preflight_sha256: {report['preflight_sha256']}")
    return 0 if report["status"] == "READY" else 1


if __name__ == "__main__":
    sys.exit(main())
