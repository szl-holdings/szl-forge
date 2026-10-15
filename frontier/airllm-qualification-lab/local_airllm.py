"""Opt-in, local-only AirLLM smoke test with a fail-closed preflight.

This is base-model inference research, not SZL adapter evaluation or release.
This script does no model acquisition, package installation or file deletion;
it refuses custom-code metadata for the exact byte-pinned local checkpoint.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
import zipfile


SCHEMA = "szl.airllm.local-smoke/v1"
AIRLLM_VERSION = "4.0.0"
AIRLLM_WHEEL_SHA256 = "155e7c286acbfb5f82a358970d980efcefff1e3fcd8a4ebf3c5850137a31d841"
AIRLLM_SOURCE_COMMIT = "ff35db207a0c559af9aa95d686057c3fe84f1d40"
GIB = 1 << 30
SUPPORTED_CONFIGS = {("llama", "LlamaForCausalLM"), ("qwen2", "Qwen2ForCausalLM")}
FORBIDDEN_SUFFIXES = {".bin", ".pt", ".pth", ".pkl", ".pickle", ".gguf", ".py"}
MANIFEST_KEYS = {"schema", "model_id", "revision", "snapshot_dir", "scratch_root", "split_dir", "gpu_uuid", "files", "airllm_wheel"}
SOURCE_EVIDENCE = Path(__file__).resolve().parents[1] / "minicpm5" / "evidence" / "2026-09-07-a10g-smoke.json"
SCRATCH_PARENT = Path(__file__).resolve().parents[3]
SCRATCH_MARKER = ".szl-airllm-task-owned.json"


class PreflightError(ValueError):
    """The local experiment request is malformed or unsafe."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def _check_manifest(manifest: dict) -> tuple[Path, Path, Path, dict]:
    if type(manifest) is not dict or set(manifest) != MANIFEST_KEYS:
        raise PreflightError("manifest fields must exactly match the local-smoke/v1 schema")
    if manifest["schema"] != SCHEMA:
        raise PreflightError("unsupported manifest schema")
    if type(manifest["model_id"]) is not str or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", manifest["model_id"]):
        raise PreflightError("model_id must be a Hub repository ID")
    if type(manifest["revision"]) is not str or not re.fullmatch(r"[0-9a-f]{40}", manifest["revision"]):
        raise PreflightError("revision must be an immutable 40-hex commit")
    if type(manifest["gpu_uuid"]) is not str or not re.fullmatch(r"GPU-[A-Za-z0-9-]+", manifest["gpu_uuid"]):
        raise PreflightError("gpu_uuid must explicitly bind one GPU")
    for key in ("snapshot_dir", "scratch_root", "split_dir", "airllm_wheel"):
        if type(manifest[key]) is not str or not Path(manifest[key]).is_absolute():
            raise PreflightError(f"{key} must be an absolute path")
    snapshot = Path(manifest["snapshot_dir"]).resolve()
    scratch = Path(manifest["scratch_root"]).resolve()
    split = Path(manifest["split_dir"]).resolve()
    wheel = Path(manifest["airllm_wheel"]).resolve()
    if not snapshot.is_dir():
        raise PreflightError("local checkpoint snapshot is unavailable")
    if _inside(split, snapshot) or _inside(snapshot, split):
        raise PreflightError("split directory must be disjoint from the source checkpoint")
    if not scratch.is_dir() or split.parent != scratch or _inside(scratch, snapshot) or _inside(snapshot, scratch):
        raise PreflightError("split directory must be a direct child of a disjoint task-owned scratch root")
    if not _inside(scratch, SCRATCH_PARENT) or _inside(scratch, Path(__file__).resolve().parents[2]):
        raise PreflightError("scratch root must be a sibling of this Forge checkout under its task work parent")
    marker_path = scratch / SCRATCH_MARKER
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PreflightError("task-owned scratch marker is missing or malformed") from exc
    if marker != {"schema": "szl.airllm.scratch-owner/v1", "model_id": manifest["model_id"], "revision": manifest["revision"]}:
        raise PreflightError("task-owned scratch marker does not bind this checkpoint")
    if split.exists():
        raise PreflightError("split directory must be fresh and nonexistent")
    if _inside(wheel, snapshot) or _inside(wheel, split):
        raise PreflightError("wheel must be outside checkpoint and split directories")
    files = manifest["files"]
    if type(files) is not dict or not files:
        raise PreflightError("files must enumerate every snapshot file with SHA-256")
    for name, digest in files.items():
        if type(name) is not str or not name or "\\" in name or name.startswith("/") or any(part in ("", ".", "..") for part in name.split("/")):
            raise PreflightError("unsafe file path in manifest")
        if type(digest) is not str or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise PreflightError("file SHA-256 must be lowercase 64-hex")
    evidence = json.loads(SOURCE_EVIDENCE.read_text(encoding="utf-8"))
    if manifest["model_id"] != evidence["plan"]["modelId"] or manifest["revision"] != evidence["plan"]["modelRevision"]:
        raise PreflightError("model identity differs from canonical Forge MiniCPM evidence")
    if files != evidence["artifactSha256"]:
        raise PreflightError("file SHA-256 manifest differs from canonical Forge MiniCPM evidence")
    return snapshot, split, wheel, evidence


def _check_snapshot(snapshot: Path, expected: dict[str, str]) -> tuple[int, int, dict]:
    actual = {}
    for path in snapshot.rglob("*"):
        if path.is_symlink() and path.is_dir():
            raise PreflightError("symlinked snapshot directories are refused")
        if not path.is_file():
            continue
        name = path.relative_to(snapshot).as_posix()
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            raise PreflightError(f"unsafe checkpoint format or executable source: {name}")
        actual[name] = path
    if set(actual) != set(expected):
        raise PreflightError("manifest does not cover exactly the local snapshot files")
    if "config.json" not in actual or "tokenizer.json" not in actual:
        raise PreflightError("config.json and tokenizer.json are required")
    weights = [name for name in actual if name.endswith(".safetensors")]
    if not weights:
        raise PreflightError("a Transformers safetensors checkpoint is required; GGUF is incompatible")
    for name, path in sorted(actual.items()):
        if _sha256(path) != expected[name]:
            raise PreflightError(f"local file SHA-256 mismatch: {name}")
    for name, path in actual.items():
        if name.endswith(".json"):
            document = json.loads(path.read_text(encoding="utf-8"))
            if type(document) is dict and "auto_map" in document:
                raise PreflightError(f"custom/remote code mapping is refused: {name}")
    config = json.loads(actual["config.json"].read_text(encoding="utf-8"))
    if type(config) is not dict or "auto_map" in config:
        raise PreflightError("custom/remote model code is refused")
    architectures = config.get("architectures")
    if type(architectures) is not list or len(architectures) != 1 or (config.get("model_type"), architectures[0]) not in SUPPORTED_CONFIGS:
        raise PreflightError("only reviewed built-in Llama or Qwen2 causal-LM configs are accepted")
    tokenizer_config_path = actual.get("tokenizer_config.json")
    if tokenizer_config_path is not None:
        tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
        if type(tokenizer_config) is not dict or tokenizer_config.get("tokenizer_class") not in (None, "PreTrainedTokenizerFast", "LlamaTokenizer", "LlamaTokenizerFast", "Qwen2Tokenizer", "Qwen2TokenizerFast"):
            raise PreflightError("unreviewed tokenizer class is refused")
    index = actual.get("model.safetensors.index.json")
    if index is not None:
        weight_map = json.loads(index.read_text(encoding="utf-8")).get("weight_map")
        if type(weight_map) is not dict or set(weight_map.values()) != set(weights):
            raise PreflightError("sharded safetensors index does not cover exactly the weight files")
    sizes = [actual[name].stat().st_size for name in weights]
    return sum(sizes), max(sizes), {"model_type": config["model_type"], "architecture": architectures[0],
                                    "declared_transformers_version": config.get("transformers_version"),
                                    "files_sha256": expected}


def _version_tuple(value: str) -> tuple[int, int, int] | None:
    match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", value)
    return tuple(int(part or 0) for part in match.groups()) if match else None


def _dependency_versions(config: dict) -> tuple[dict, list[str]]:
    versions = {}
    blockers = []
    for package in ("torch", "transformers", "accelerate", "safetensors", "huggingface_hub", "scipy", "sentencepiece", "tqdm"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
            blockers.append(f"MISSING_DEPENDENCY_{package.upper()}")
    minimums = {"torch": (2, 4, 0), "transformers": (4, 49, 0), "accelerate": (1, 0, 0)}
    for package, minimum in minimums.items():
        value = versions[package]
        if value is not None and (_version_tuple(value) is None or _version_tuple(value) < minimum):
            blockers.append(f"UNSUPPORTED_{package.upper()}_VERSION")
    transformers = versions["transformers"]
    if transformers is not None and (_version_tuple(transformers) is None or _version_tuple(transformers) >= (6, 0, 0)):
        blockers.append("UNSUPPORTED_TRANSFORMERS_VERSION")
    declared = config.get("declared_transformers_version")
    if type(declared) is str and transformers is not None and _version_tuple(declared) is not None and _version_tuple(transformers) is not None and _version_tuple(transformers) < _version_tuple(declared):
        blockers.append("TRANSFORMERS_BELOW_CHECKPOINT_DECLARATION")
    return versions, blockers


def _check_airllm_wheel(wheel: Path) -> dict:
    if not wheel.is_file() or wheel.name != "airllm-4.0.0-py3-none-any.whl" or _sha256(wheel) != AIRLLM_WHEEL_SHA256:
        raise PreflightError("AirLLM wheel is absent or differs from the pinned PyPI SHA-256")
    try:
        installed = importlib.metadata.distribution("airllm")
    except importlib.metadata.PackageNotFoundError as exc:
        raise PreflightError("AirLLM 4.0.0 is not installed in this Python environment") from exc
    if installed.version != AIRLLM_VERSION:
        raise PreflightError("installed AirLLM version differs from the pinned wheel")
    with zipfile.ZipFile(wheel) as archive:
        package_files = [name for name in archive.namelist() if name.startswith("airllm/") and not name.endswith("/")]
        if not package_files:
            raise PreflightError("pinned wheel has no importable package files")
        for name in package_files:
            installed_path = Path(installed.locate_file(name))
            if not installed_path.is_file() or hashlib.sha256(archive.read(name)).digest() != bytes.fromhex(_sha256(installed_path)):
                raise PreflightError(f"installed AirLLM package differs from pinned wheel: {name}")
    return {"version": installed.version, "wheel_sha256": AIRLLM_WHEEL_SHA256, "source_commit": AIRLLM_SOURCE_COMMIT, "installed_package_files_matched": len(package_files)}


def _available_ram() -> int | None:
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        return status.ullAvailPhys if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)) else None
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return None


def _gpu_free_bytes(uuid: str) -> int | None:
    try:
        probe = subprocess.run(["nvidia-smi", "--query-gpu=uuid,memory.free", "--format=csv,noheader,nounits"],
                               check=True, capture_output=True, text=True, timeout=10)
        rows = [row.split(",") for row in probe.stdout.splitlines()]
        matching = [int(row[1].strip()) * (1 << 20) for row in rows if len(row) == 2 and row[0].strip() == uuid]
        return matching[0] if len(matching) == 1 else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def preflight(manifest: dict) -> dict:
    snapshot, split, wheel, evidence = _check_manifest(manifest)
    weight_bytes, largest_shard_bytes, checkpoint = _check_snapshot(snapshot, manifest["files"])
    blockers = []
    try:
        package = _check_airllm_wheel(wheel)
    except PreflightError as exc:
        package = {"status": "UNAVAILABLE", "reason": str(exc)}
        blockers.append("PINNED_AIRLLM_RUNTIME_UNAVAILABLE")
    dependencies, dependency_blockers = _dependency_versions(checkpoint)
    blockers += dependency_blockers
    ram = _available_ram()
    gpu = _gpu_free_bytes(manifest["gpu_uuid"])
    disk = shutil.disk_usage(split.parent).free
    # Conservative planning bounds, never measured runtime requirements.
    required = {"disk": 2 * weight_bytes + 4 * GIB,
                "ram": max(2 * weight_bytes + 2 * GIB, largest_shard_bytes + 4 * GIB),
                "gpu": 3 * GIB}
    observed = {"disk": disk, "ram": ram, "gpu": gpu}
    blockers += [f"{tier.upper()}_CAPACITY_UNAVAILABLE" if observed[tier] is None else f"{tier.upper()}_CAPACITY_BELOW_PLANNING_BOUND"
                 for tier in ("disk", "ram", "gpu") if observed[tier] is None or observed[tier] < required[tier]]
    return {"schema": SCHEMA, "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "evidence_class": "MEASURED", "disposition": "BLOCKED" if blockers else "READY_FOR_LOCAL_EXPERIMENT_ONLY",
            "blockers": blockers, "model_id": manifest["model_id"], "revision_declared": manifest["revision"],
            "model_origin_verification": "FORGE_REPORTED_HASH_BINDING_LOCAL_BYTES_RECHECKED",
            "forge_source_evidence_sha256": _sha256(SOURCE_EVIDENCE),
            "forge_source_evidence_status": evidence.get("status"),
            "forge_source_evidence_production_disposition": evidence.get("productionDisposition"),
            "checkpoint": checkpoint,
            "checkpoint_weight_bytes": weight_bytes, "largest_safetensors_shard_bytes": largest_shard_bytes,
            "airllm": package, "dependency_versions": dependencies,
            "dependency_integrity": "VERSION_OBSERVED_HASH_CLOSURE_NOT_VERIFIED",
            "gpu_uuid": manifest["gpu_uuid"], "observed_free_bytes": observed,
            "planning_required_bytes": required, "planning_bound_class": "MODELED_CONSERVATIVE_NOT_MEASURED_RUNTIME",
            "split_dir": str(split), "delete_original": False, "network": "OFFLINE",
            "scratch_ownership": "DECLARED_MARKER_AND_PATH_POLICY",
            "shared_cache_mutation": "UNKNOWN",
            "adapter_loaded": False, "training_eligible": False, "publication_eligible": False,
            "execution_authority": False, "inference_run": "NOT_RUN"}


def _worker(manifest_path: Path) -> int:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = preflight(manifest)
    if result["disposition"] != "READY_FOR_LOCAL_EXPERIMENT_ONLY":
        print(json.dumps({"status": "BLOCKED", "blockers": result["blockers"]}))
        return 2
    cache = Path(manifest["scratch_root"]) / "runtime-cache"
    cache_paths = {"HF_HOME": cache / "hf", "HF_HUB_CACHE": cache / "hf" / "hub",
                   "HUGGINGFACE_HUB_CACHE": cache / "hf" / "hub",
                   "TORCH_HOME": cache / "torch", "TORCH_EXTENSIONS_DIR": cache / "torch-extensions",
                   "TRITON_CACHE_DIR": cache / "triton", "CUDA_CACHE_PATH": cache / "cuda",
                   "XDG_CACHE_HOME": cache / "xdg", "TMP": cache / "tmp", "TEMP": cache / "tmp"}
    for path in set(cache_paths.values()):
        path.mkdir(parents=True, exist_ok=True)
    os.environ.update({name: str(path) for name, path in cache_paths.items()})
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = manifest["gpu_uuid"]
    from airllm import AutoModel  # noqa: PLC0415 - import only after the local preflight
    import torch  # noqa: PLC0415
    started = time.monotonic()
    model = AutoModel.from_pretrained(manifest["snapshot_dir"], device="cuda:0",
                                      layer_shards_saving_path=manifest["split_dir"],
                                      delete_original=False, prefetching=False)
    tokens = model.tokenizer(["Two plus two equals"], return_tensors="pt", return_attention_mask=False)
    output = model.generate(tokens["input_ids"].to("cuda:0"), max_new_tokens=1,
                            use_cache=True, return_dict_in_generate=True)
    ids = output.sequences[0].tolist()
    print(json.dumps({"status": "MEASURED_LOCAL_SMOKE_ONLY", "generated_token_count": max(0, len(ids) - tokens["input_ids"].shape[-1]),
                      "token_ids_sha256": hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
                      "elapsed_seconds": round(time.monotonic() - started, 3),
                      "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                      "task_cache_root": str(cache), "shared_cache_mutation": "UNKNOWN",
                      "model_quality": "UNKNOWN", "numerical_parity": "NOT_RUN", "execution_authority": False}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--run", action="store_true", help="execute one synthetic token only after preflight passes")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return _worker(args.manifest)
    if args.receipt.exists():
        print(json.dumps({"disposition": "BLOCKED", "reason": "receipt path already exists; use a fresh path"}))
        return 2
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(json.dumps({"disposition": "BLOCKED", "reason": "manifest unavailable or malformed", "error_type": type(exc).__name__}))
        return 2
    if type(manifest) is not dict:
        print(json.dumps({"disposition": "BLOCKED", "reason": "manifest must be an object"}))
        return 2
    receipt = args.receipt.resolve()
    for key in ("snapshot_dir", "scratch_root"):
        value = manifest.get(key)
        if type(value) is str and Path(value).is_absolute() and _inside(receipt, Path(value).resolve()):
            print(json.dumps({"disposition": "BLOCKED", "reason": "receipt must be outside checkpoint and task-owned scratch"}))
            return 2
    result = None
    worker_started = False
    try:
        result = preflight(manifest)
        if args.run and result["disposition"] == "READY_FOR_LOCAL_EXPERIMENT_ONLY":
            env = os.environ.copy()
            env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
                        "CUDA_VISIBLE_DEVICES": manifest["gpu_uuid"]})
            worker_started = True
            result["worker_started"] = True
            completed = subprocess.run([sys.executable, "-I", "-B", str(Path(__file__).resolve()),
                                        "--manifest", str(args.manifest.resolve()), "--receipt", str(args.receipt.resolve()), "--worker"],
                                       capture_output=True, text=True, timeout=300, env=env)
            result["inference_run"] = "MEASURED_LOCAL_SMOKE_ONLY" if completed.returncode == 0 else "FAILED_OR_BLOCKED"
            result["worker_exit_code"] = completed.returncode
            if completed.returncode == 0:
                result["worker_result"] = json.loads(completed.stdout.splitlines()[-1])
            else:
                result["disposition"] = "BLOCKED"
                result["blockers"].append("AIRLLM_WORKER_FAILED")
                result["worker_error_tail"] = (completed.stderr or completed.stdout)[-2000:]
    except (OSError, ValueError, KeyError, json.JSONDecodeError, zipfile.BadZipFile, subprocess.TimeoutExpired) as exc:
        if result is None:
            result = {"schema": SCHEMA, "captured_at_utc": datetime.now(timezone.utc).isoformat(),
                      "evidence_class": "BLOCKED", "disposition": "BLOCKED", "blockers": [],
                      "inference_run": "NOT_RUN", "training_eligible": False, "publication_eligible": False,
                      "execution_authority": False}
        result["disposition"] = "BLOCKED"
        result["blockers"].append(type(exc).__name__ + ": " + str(exc))
        result["inference_run"] = "UNKNOWN_AFTER_ATTEMPT" if worker_started else "NOT_RUN"
        result["worker_started"] = worker_started
    if worker_started:
        try:
            result["post_attempt_free_bytes"] = {"disk": shutil.disk_usage(Path(manifest["split_dir"]).parent).free,
                                                 "ram": _available_ram(), "gpu": _gpu_free_bytes(manifest["gpu_uuid"])}
        except (OSError, KeyError) as exc:
            result["post_attempt_free_bytes"] = {"status": "UNAVAILABLE", "error_type": type(exc).__name__}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"disposition": result["disposition"], "blockers": result.get("blockers", []), "inference_run": result["inference_run"], "receipt": str(args.receipt)}))
    return 0 if result["disposition"] == "READY_FOR_LOCAL_EXPERIMENT_ONLY" and (not args.run or result["inference_run"] == "MEASURED_LOCAL_SMOKE_ONLY") else 2


if __name__ == "__main__":
    raise SystemExit(main())
