#!/usr/bin/env python3
"""Qualify retained compatibility kernels and publish immutable bindings."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Iterable

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "publishing" / "compatibility-kernels.json"
DEFAULT_REPORT = ROOT / "reports" / "compatibility-kernels.json"


class QualificationError(RuntimeError):
    pass


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "szl.compatibility-kernel-binding/v1":
        raise QualificationError("unsupported compatibility-kernel schema")
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise QualificationError("artifacts must be a non-empty list")
    for item in artifacts:
        if item.get("artifact_class") != "RETAINED_COMPATIBILITY_KERNEL":
            raise QualificationError("compatibility artifact must not be classified as a model")
        for field in ("source_revision", "replacement_revision", "expected_hub_revision"):
            revision = item.get(field, "")
            if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
                raise QualificationError(f"{field} must be an exact lowercase revision")
        files = item.get("expected_files")
        if not isinstance(files, dict) or not files:
            raise QualificationError("expected_files must be a non-empty object")
    return value


def download_exact(
    repo_id: str,
    filename: str,
    revision: str,
    token: str | None,
    download_fn: Callable[..., str],
) -> bytes:
    return Path(download_fn(repo_id, filename, repo_type="model", revision=revision, token=token)).read_bytes()


def verify_files(
    item: dict[str, Any],
    *,
    token: str | None,
    download_fn: Callable[..., str],
) -> tuple[dict[str, bytes], list[dict[str, Any]]]:
    blobs: dict[str, bytes] = {}
    evidence: list[dict[str, Any]] = []
    for filename, expected in sorted(item["expected_files"].items()):
        data = download_exact(item["repo_id"], filename, item["expected_hub_revision"], token, download_fn)
        observed = sha256(data)
        if observed != expected:
            raise QualificationError(f"{filename} hash mismatch: expected {expected}, observed {observed}")
        blobs[filename] = data
        evidence.append({"path": filename, "bytes": len(data), "sha256": observed})
    license_evidence = item["license"]
    if license_evidence["evidence_sha256"] != dict(item["expected_files"])[license_evidence["evidence_path"]]:
        raise QualificationError("license evidence does not match the pinned file set")
    return blobs, evidence


def run_selfcheck(item: dict[str, Any], blobs: dict[str, bytes]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for filename, data in blobs.items():
            path = root / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        module_root = root / item["runtime"]["module_path"]
        script = (
            "import importlib,json,sys,time,tracemalloc;"
            f"sys.path.insert(0,{str(module_root)!r});"
            "tracemalloc.start();start=time.perf_counter();"
            f"m=importlib.import_module({item['runtime']['module']!r});"
            f"r=getattr(m,{item['runtime']['selfcheck']!r})();"
            "elapsed=(time.perf_counter()-start)*1000;cur,peak=tracemalloc.get_traced_memory();"
            "print(json.dumps({'result':r,'latency_ms':elapsed,'python_peak_bytes':peak,'module_file':m.__file__},sort_keys=True))"
        )
        proc = subprocess.run(
            [sys.executable, "-I", "-c", script],
            text=True,
            capture_output=True,
            timeout=90,
            check=False,
        )
        if proc.returncode != 0:
            raise QualificationError(f"clean self-check failed: {proc.stderr[-2000:]}")
        line = proc.stdout.strip().splitlines()[-1]
        measured = json.loads(line)
        if measured.get("result", {}).get("ok") is not True:
            raise QualificationError(
                "kernel self-check did not return ok=true: "
                + json.dumps(measured.get("result"), sort_keys=True)
            )
        return measured


def publication(item: dict[str, Any], files: list[dict[str, Any]], runtime: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "szl.hf-compatibility-kernel-publication/v1",
        "artifact": {
            "repo_id": item["repo_id"],
            "repo_type": "model",
            "artifact_class": item["artifact_class"],
            "promotion_state": item["promotion_state"],
            "technical_note": "The Hugging Face repository type is model for historical compatibility; the artifact is a kernel and optional surrogate, not a language model.",
        },
        "source_repository": item["source_repository"],
        "source_revision": item["source_revision"],
        "replacement_repository": item["replacement_repository"],
        "replacement_revision": item["replacement_revision"],
        "hub_revision_qualified": item["expected_hub_revision"],
        "license": item["license"],
        "files": files,
        "runtime_receipt": {
            "status": "MEASURED_CLEAN_IMPORT_AND_SELFCHECK",
            "selfcheck": runtime["result"],
            "latency_ms": runtime["latency_ms"],
            "python_peak_bytes": runtime["python_peak_bytes"],
            "energy": "UNAVAILABLE_NO_MEASUREMENT_DEVICE",
            "restart_reproducibility": "MEASURED_FRESH_ISOLATED_PROCESS",
        },
        "autonomy_boundary": item["runtime"]["autonomy_boundary"],
        "performance_boundary": item["runtime"]["performance_boundary"],
        "release_receipt": {
            "kind": "UNSIGNED_IMMUTABLE_PUBLICATION_RECORD",
            "owner_signature": "UNAVAILABLE_APPROVED_PRIVATE_KEY_NOT_PRESENT",
        },
    }


def run(
    *,
    manifest_path: Path,
    report_path: Path,
    publish: bool,
    token: str | None,
    api: HfApi | None = None,
    download_fn: Callable[..., str] = hf_hub_download,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    api = api or HfApi(token=token)
    records: list[dict[str, Any]] = []
    for item in manifest["artifacts"]:
        info = api.model_info(
            item["repo_id"],
            revision=item["expected_hub_revision"],
            token=token,
        )
        if info.sha != item["expected_hub_revision"]:
            raise QualificationError(f"Hub revision drifted for {item['repo_id']}: {info.sha}")
        blobs, file_evidence = verify_files(item, token=token, download_fn=download_fn)
        runtime = run_selfcheck(item, blobs)
        payload = publication(item, file_evidence, runtime)
        payload_bytes = canonical_json(payload)
        record: dict[str, Any] = {
            "repo_id": item["repo_id"],
            "status": "VERIFIED_DRY_RUN",
            "hub_revision_before": info.sha,
            "publication_sha256": sha256(payload_bytes),
            "runtime_receipt": payload["runtime_receipt"],
        }
        if publish:
            if not token:
                raise QualificationError("HF_TOKEN is required for publication")
            commit = api.create_commit(
                repo_id=item["repo_id"],
                repo_type="model",
                operations=[CommitOperationAdd(path_in_repo="publication.json", path_or_fileobj=io.BytesIO(payload_bytes))],
                commit_message="Bind retained compatibility kernel evidence",
                token=token,
            )
            revision = commit.oid
            readback = download_exact(item["repo_id"], "publication.json", revision, token, download_fn)
            if readback != payload_bytes:
                raise QualificationError("immutable publication readback mismatch")
            record.update({"status": "PUBLISHED_AND_EXACT_READBACK_VERIFIED", "hub_revision_after": revision})
        records.append(record)
    result = {"schema": "szl.compatibility-kernel-binding-report/v1", "mode": "PUBLISH" if publish else "DRY_RUN", "records": records}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_bytes(canonical_json(result))
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run(manifest_path=args.manifest, report_path=args.report, publish=args.publish, token=os.getenv("HF_TOKEN"))
    except QualificationError as error:
        result = {
            "schema": "szl.compatibility-kernel-binding-report/v1",
            "mode": "PUBLISH" if args.publish else "DRY_RUN",
            "status": "REFUSED",
            "error": str(error),
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_bytes(canonical_json(result))
        print(canonical_json(result).decode(), end="")
        return 1
    print(canonical_json(result).decode(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
