"""Manual, source-bound AirLLM CPU plumbing smoke for one public base model.

This never evaluates or publishes an SZL model. It downloads only the exact
allowlisted checkpoint bytes and executes one synthetic token without network.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time
from urllib.request import Request, urlopen
import zipfile

try:
    import resource  # available on the hosted Linux runner
except ImportError:  # local Windows software controls do not execute the worker
    resource = None


SCHEMA = "szl.airllm.hosted-cpu-smoke/v2"
MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
MODEL_REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
MODEL_LICENSE_CARD = "apache-2.0"
HARNESS_PATH = "frontier/airllm-qualification-lab/hosted_cpu_smoke.py"
AIRLLM_VERSION = "4.0.0"
AIRLLM_SOURCE_COMMIT = "ff35db207a0c559af9aa95d686057c3fe84f1d40"
WHEEL_NAME = "airllm-4.0.0-py3-none-any.whl"
WHEEL_URL = "https://files.pythonhosted.org/packages/6f/11/b64bd345075adad6e26cbeec8258cdbcd90b619867488a5f39ea1b6727f1/airllm-4.0.0-py3-none-any.whl"
WHEEL_SHA256 = "155e7c286acbfb5f82a358970d980efcefff1e3fcd8a4ebf3c5850137a31d841"
WHEEL_BYTES = 83338
GIB = 1 << 30
FILES = {
    "config.json": (861, "8eb740e8bbe4cff95ea7b4588d17a2432deb16e8075bc5828ff7ba9be94d982a"),
    "generation_config.json": (132, "87b916edaaab66b3899b9d0dd0752727dff6666686da0504d89ae0a6e055a013"),
    "merges.txt": (466391, "0b54e8aa4e53d5383e2e4bc635a56b43f9647f7b13832d5d9ecd8f82dac4f510"),
    "model.safetensors": (269060552, "5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c"),
    "special_tokens_map.json": (655, "2b7379f3ae813529281a5c602bc5a11c1d4e0a99107aaa597fe936c1e813ca52"),
    "tokenizer.json": (2104556, "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c"),
    "tokenizer_config.json": (3764, "4ec77d44f62efeb38d7e044a1db318f6a939438425312dfa333b8382dbad98df"),
    "vocab.json": (800662, "82b84012e3add4d01d12ba14442026e49b8cbbaead1f79ecf3d919784f82dc79"),
}
MARKER = ".szl-airllm-hosted-task.json"
PROMPT = "Two plus two equals"
INPUT_DATASET_BYTES = (json.dumps({"prompt": PROMPT}, sort_keys=True,
                                  separators=(",", ":")) + "\n").encode("utf-8")
INPUT_DATASET_SHA256 = hashlib.sha256(INPUT_DATASET_BYTES).hexdigest()
DATASET_EVIDENCE_CLASS = "SAMPLE"
DATASET_SCOPE = "SAMPLE_ONE_RECORD_SYNTHETIC_INFERENCE_INPUT_ONLY"
DO_SAMPLE = False
GENERATION_SEED = 0
MAX_NEW_TOKENS = 1
DEPENDENCIES = {"torch": "2.6.0", "transformers": "4.57.3", "accelerate": "1.12.0",
                "safetensors": "0.7.0", "huggingface-hub": "0.36.0", "scipy": "1.15.3",
                "sentencepiece": "0.2.1", "tqdm": "4.67.1"}


class SmokeError(ValueError):
    """A source, path, capacity, package, or execution gate failed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _harness_sha256() -> str:
    """Bind executed bytes to the tracked Git blob, not merely to HEAD."""
    executed = Path(__file__).resolve().read_bytes()
    tracked = subprocess.run(["git", "show", "HEAD:" + HARNESS_PATH], check=True,
                             capture_output=True, timeout=10).stdout
    if executed.replace(b"\r\n", b"\n") != tracked:
        raise SmokeError("executed harness differs from checked-out Git blob")
    return hashlib.sha256(executed).hexdigest()


def _loss_record() -> dict:
    return {"evidence_class": "UNAVAILABLE", "status": "NOT_RUN", "reason": "INFERENCE_ONLY"}


def _empty_metrics() -> dict:
    return {"evidence_class": "UNAVAILABLE", "generated_token_count": None,
            "elapsed_seconds": None, "peak_rss_kib": None, "token_ids_sha256": None}


def _generation_binding(*, seed_applied: bool = False) -> dict:
    if DO_SAMPLE or not isinstance(GENERATION_SEED, int) or MAX_NEW_TOKENS != 1:
        raise SmokeError("hosted smoke is greedy, seeded, and limited to one token")
    return {"do_sample": False, "max_new_tokens": 1, "seed": GENERATION_SEED, "seed_applied": seed_applied,
            "input_sha256": hashlib.sha256(PROMPT.encode("utf-8")).hexdigest(),
            "dataset_sha256": INPUT_DATASET_SHA256,
            "harness_sha256": _harness_sha256()}


def _generation_record(*, seed_applied: bool = False) -> dict:
    binding = _generation_binding(seed_applied=seed_applied)
    return {key: binding[key] for key in ("do_sample", "max_new_tokens", "seed", "seed_applied")}


def _bind_prepare_receipt(path: Path, source_sha: str) -> str:
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise SmokeError("prepare receipt must be an absolute file")
    raw = path.read_bytes()
    document = json.loads(raw.decode("utf-8"))
    if not isinstance(document, dict):
        raise SmokeError("prepare receipt must be a JSON object")
    binding = _generation_binding()
    source = document.get("source") if isinstance(document.get("source"), dict) else {}
    if (document.get("schema") != SCHEMA or document.get("stage") != "PREPARE"
            or document.get("disposition") != "MEASURED" or document.get("evidence_class") != "MEASURED"):
        raise SmokeError("prepare receipt is not a measured prepare stage")
    if (document.get("model_id") != MODEL_ID or document.get("model_revision") != MODEL_REVISION
            or document.get("base_revision") != MODEL_REVISION):
        raise SmokeError("prepare receipt model binding differs")
    if document.get("airllm_source_commit") != AIRLLM_SOURCE_COMMIT or document.get("delete_original") is not False:
        raise SmokeError("prepare receipt AirLLM pin differs")
    if source.get("repository") != "szl-holdings/szl-forge" or source.get("source_sha") != source_sha:
        raise SmokeError("prepare receipt is not bound to this source")
    if (document.get("harness_sha256") != binding["harness_sha256"]
            or document.get("input_sha256") != binding["input_sha256"]
            or document.get("dataset_sha256") != INPUT_DATASET_SHA256
            or document.get("dataset_evidence_class") != DATASET_EVIDENCE_CLASS
            or document.get("dataset_scope") != DATASET_SCOPE
            or document.get("seed") != GENERATION_SEED
            or document.get("loss") != _loss_record()
            or document.get("metrics") != _empty_metrics()):
        raise SmokeError("prepare receipt harness or input hash differs")
    if document.get("generation") != _generation_record():
        raise SmokeError("prepare receipt generation binding differs")
    return hashlib.sha256(raw).hexdigest()


def _token_result(sequence: list, prompt_tokens: int, elapsed: float, peak) -> dict:
    generated = len(sequence) - prompt_tokens
    if generated != 1:
        raise SmokeError("AirLLM did not generate exactly one token")
    return {"status": "MEASURED", "scope": "HOSTED_CPU_ONE_TOKEN_SMOKE_ONLY",
            "generated_token_count": generated,
            "token_ids_sha256": hashlib.sha256(json.dumps(sequence, separators=(",", ":")).encode()).hexdigest(),
            "elapsed_seconds": round(elapsed, 3), "peak_rss_kib": peak,
            "model_quality": "UNKNOWN", "numerical_parity": "NOT_RUN", "execution_authority": False,
            **_generation_binding(seed_applied=True)}


def _runner_paths(root: Path, receipt: Path, *, prepare: bool) -> None:
    runner_temp = os.environ.get("RUNNER_TEMP")
    if not runner_temp or not Path(runner_temp).is_absolute():
        raise SmokeError("RUNNER_TEMP is required")
    task_parent = Path(runner_temp).resolve()
    if not root.is_absolute() or root.parent.resolve() != task_parent or root.is_symlink():
        raise SmokeError("root must be a direct, non-symlink child of RUNNER_TEMP")
    if not receipt.is_absolute() or root == receipt or root in receipt.parents:
        raise SmokeError("receipt must be absolute and outside task root")
    if prepare and root.exists():
        raise SmokeError("task root must be fresh")
    if not prepare and (not root.is_dir() or root.is_symlink()):
        raise SmokeError("task root is missing or symlinked")


def _source_binding() -> dict:
    if os.environ.get("GITHUB_REPOSITORY") != "szl-holdings/szl-forge":
        raise SmokeError("hosted smoke requires the canonical Forge repository")
    if os.environ.get("GITHUB_REF") != "refs/heads/main":
        raise SmokeError("manual smoke must run from main")
    source_sha = os.environ.get("GITHUB_SHA", "")
    head = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True,
                          text=True, timeout=10).stdout.strip()
    if len(source_sha) != 40 or source_sha != head:
        raise SmokeError("checked-out source differs from workflow event SHA")
    return {"repository": os.environ.get("GITHUB_REPOSITORY"), "source_sha": head,
            "run_id": os.environ.get("GITHUB_RUN_ID"), "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT")}


def _available_ram() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise SmokeError("host available RAM is unavailable")


def _hardware() -> dict:
    cpu_model = "UNAVAILABLE"
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {"runner_os": os.environ.get("RUNNER_OS", "UNAVAILABLE"),
            "runner_arch": os.environ.get("RUNNER_ARCH", "UNAVAILABLE"),
            "cpu_model": cpu_model, "logical_cpu_count": os.cpu_count(),
            "accelerator": "CPU_ONLY_NO_GPU_REQUESTED"}


def _capacity(root: Path, *, download: bool) -> dict:
    disk = shutil.disk_usage(root.parent).free
    ram = _available_ram()
    required = {"disk": (8 if download else 6) * GIB, "ram": 4 * GIB}
    if disk < required["disk"] or ram < required["ram"]:
        raise SmokeError("fresh free disk or RAM is below the conservative smoke bound")
    return {"observed_free_bytes": {"disk": disk, "ram": ram},
            "required_free_bytes": required, "bound_class": "MODELED",
            "bound_scope": "CONSERVATIVE_ADMISSION_NOT_MEASURED_CONSUMPTION"}


def _download(url: str, path: Path, size: int, expected: str) -> None:
    partial = path.with_name(path.name + ".partial")
    if path.exists() or partial.exists():
        raise SmokeError("download target already exists")
    digest = hashlib.sha256()
    count = 0
    request = Request(url, headers={"User-Agent": "szl-forge-airllm-hosted-smoke/1"})
    with urlopen(request, timeout=45) as response, partial.open("xb") as output:
        if response.url.split(":", 1)[0] != "https":
            raise SmokeError("download redirected away from HTTPS")
        while block := response.read(1 << 20):
            count += len(block)
            if count > size:
                raise SmokeError("download exceeds immutable declared file size")
            digest.update(block)
            output.write(block)
    if count != size or digest.hexdigest() != expected:
        raise SmokeError("download differs from immutable size/SHA-256 binding")
    partial.rename(path)


def _no_auto_map(value) -> bool:
    if isinstance(value, dict):
        return "auto_map" not in value and all(_no_auto_map(child) for child in value.values())
    if isinstance(value, list):
        return all(_no_auto_map(child) for child in value)
    return True


def _verify_snapshot(snapshot: Path) -> dict:
    if not snapshot.is_dir() or snapshot.is_symlink():
        raise SmokeError("snapshot directory missing or symlinked")
    entries = list(snapshot.iterdir())
    if any(not path.is_file() or path.is_symlink() for path in entries):
        raise SmokeError("snapshot contains directory, symlink, or non-file")
    if {path.name for path in entries} != set(FILES):
        raise SmokeError("snapshot differs from eight-file allowlist")
    for name, (size, expected) in FILES.items():
        path = snapshot / name
        if path.stat().st_size != size or _sha256(path) != expected:
            raise SmokeError("snapshot byte/hash mismatch: " + name)
        if name.endswith(".json") and not _no_auto_map(json.loads(path.read_text(encoding="utf-8"))):
            raise SmokeError("remote-code auto_map is forbidden: " + name)
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    tokenizer = json.loads((snapshot / "tokenizer_config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "llama" or config.get("architectures") != ["LlamaForCausalLM"]:
        raise SmokeError("model is not the reviewed built-in Llama architecture")
    if "quantization_config" in config or tokenizer.get("tokenizer_class") != "GPT2Tokenizer":
        raise SmokeError("model/tokenizer config differs from reviewed built-in path")
    return {"model_type": "llama", "architecture": "LlamaForCausalLM",
            "tokenizer_class": "GPT2Tokenizer", "file_bytes": {name: size for name, (size, _) in FILES.items()},
            "file_sha256": {name: digest for name, (_, digest) in FILES.items()}}


def _verify_wheel(wheel: Path) -> dict:
    if wheel.name != WHEEL_NAME or not wheel.is_file() or wheel.stat().st_size != WHEEL_BYTES or _sha256(wheel) != WHEEL_SHA256:
        raise SmokeError("AirLLM wheel differs from the pinned PyPI bytes")
    dist = importlib.metadata.distribution("airllm")
    if dist.version != AIRLLM_VERSION:
        raise SmokeError("installed AirLLM version differs from pinned wheel")
    with zipfile.ZipFile(wheel) as archive:
        names = [name for name in archive.namelist() if name.startswith("airllm/") and not name.endswith("/")]
        if not names:
            raise SmokeError("pinned wheel contains no AirLLM package")
        for name in names:
            installed = Path(dist.locate_file(name))
            if not installed.is_file() or hashlib.sha256(archive.read(name)).hexdigest() != _sha256(installed):
                raise SmokeError("installed AirLLM file differs from pinned wheel: " + name)
    return {"version": dist.version, "wheel_sha256": WHEEL_SHA256,
            "source_commit": AIRLLM_SOURCE_COMMIT, "installed_package_files_matched": len(names)}


def _dependency_versions() -> dict:
    versions = {name: importlib.metadata.version(name) for name in DEPENDENCIES}
    for name, expected in DEPENDENCIES.items():
        if versions[name] != expected and not (name == "torch" and versions[name] == expected + "+cpu"):
            raise SmokeError("dependency version differs from hosted workflow pin: " + name)
    return versions


def _marker(root: Path) -> dict:
    value = json.loads((root / MARKER).read_text(encoding="utf-8"))
    if value != {"schema": "szl.airllm.hosted-task/v1", "model_id": MODEL_ID,
                 "revision": MODEL_REVISION, "source_sha": os.environ.get("GITHUB_SHA")}:
        raise SmokeError("task root marker differs from source/model binding")
    return value


def _task_bytes(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        if path.is_symlink():
            raise SmokeError("task root contains a symlink")
        if path.is_file():
            total += path.stat().st_size
    return total


def _run_worker(root: Path, receipt: Path, env: dict) -> tuple[int, str]:
    command = [sys.executable, "-I", "-B", str(Path(__file__).resolve()),
               "--worker", "--root", str(root), "--receipt", str(receipt)]
    log = root / "worker.log"
    with log.open("x", encoding="utf-8") as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                                   text=True, env=env, start_new_session=True)
        started = time.monotonic()
        try:
            while process.poll() is None:
                if _task_bytes(root) > 4 * GIB or shutil.disk_usage(root).free < 4 * GIB:
                    raise SmokeError("task-owned bytes exceeded 4 GiB or free disk fell below 4 GiB")
                if time.monotonic() - started > 240:
                    raise subprocess.TimeoutExpired(command, 240)
                time.sleep(2)
        except (OSError, SmokeError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=10)
            raise
    if _task_bytes(root) > 4 * GIB:
        raise SmokeError("task-owned bytes exceeded 4 GiB after worker exit")
    with log.open("rb") as stream:
        stream.seek(max(0, log.stat().st_size - 4096))
        tail = stream.read().decode("utf-8", errors="replace")
    return process.returncode, tail


def _worker(root: Path) -> int:
    # Rehash after the parent preflight. No Hub or custom model code may load.
    _verify_snapshot(root / "snapshot")
    _verify_wheel(root / "wheel" / WHEEL_NAME)
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                       "HF_DATASETS_OFFLINE": "1", "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
                       "HF_HUB_DISABLE_TELEMETRY": "1", "HF_HOME": str(root / "cache" / "hf"),
                       "XDG_CACHE_HOME": str(root / "cache" / "xdg"),
                       "TORCH_HOME": str(root / "cache" / "torch"),
                       "TMPDIR": str(root / "cache" / "tmp")})
    for path in (root / "cache" / "hf", root / "cache" / "xdg", root / "cache" / "torch", root / "cache" / "tmp"):
        path.mkdir(parents=True, exist_ok=True)
    from airllm import AutoModel  # import only after offline and byte checks
    import torch
    random.seed(GENERATION_SEED)
    torch.manual_seed(GENERATION_SEED)
    started = time.monotonic()
    model = AutoModel.from_pretrained(str(root / "snapshot"), device="cpu", dtype=torch.float32,
                                      layer_shards_saving_path=str(root / "split"),
                                      delete_original=False, prefetching=False)
    tokenized = model.tokenizer([PROMPT], return_tensors="pt", return_attention_mask=False)
    prompt_tokens = tokenized["input_ids"].shape[-1]
    binding = _generation_binding()
    with torch.inference_mode():
        result = model.generate(tokenized["input_ids"], max_new_tokens=binding["max_new_tokens"],
                                do_sample=binding["do_sample"], use_cache=True, return_dict_in_generate=True)
    sequence = result.sequences[0].tolist()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if resource else None
    print(json.dumps(_token_result(sequence, prompt_tokens, time.monotonic() - started, peak)))
    return 0


def _write_receipt(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--prepare-receipt", type=Path)
    args = parser.parse_args()
    root, receipt = args.root.absolute(), args.receipt.absolute()
    if args.prepare and args.prepare_receipt is not None:
        raise SmokeError("prepare does not accept a prepare receipt")
    if args.worker and args.prepare_receipt is not None:
        raise SmokeError("worker does not accept a prepare receipt")
    if args.worker:
        _runner_paths(root, receipt, prepare=False)
        _source_binding()
        _marker(root)
        return _worker(root)
    result = {"schema": SCHEMA, "captured_at_utc": datetime.now(timezone.utc).isoformat(),
              "evidence_class": "BLOCKED", "disposition": "BLOCKED", "stage": "PREPARE" if args.prepare else "RUN",
              "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "base_revision": MODEL_REVISION,
              "license_card_metadata": MODEL_LICENSE_CARD,
              "dataset_sha256": INPUT_DATASET_SHA256, "dataset_evidence_class": DATASET_EVIDENCE_CLASS,
              "dataset_scope": DATASET_SCOPE, "seed": GENERATION_SEED,
              "seed_scope": "PYTHON_AND_TORCH_GREEDY_WORKER_ONLY", "loss": _loss_record(),
              "metrics": _empty_metrics(), "prepare_receipt_sha256": "UNAVAILABLE",
              "previous_receipt_sha256": "UNAVAILABLE",
              "airllm_source_commit": AIRLLM_SOURCE_COMMIT, "delete_original": False,
              "training_eligible": False, "publication_eligible": False, "execution_authority": False,
              "held_out_gate": "NOT_RUN", "numerical_parity": "NOT_RUN", "signing_status": "UNSIGNED",
              "model_quality": "UNKNOWN", "inference_run": "UNAVAILABLE", "inference_attempted": False}
    exit_code = 2
    started_worker = False
    try:
        _runner_paths(root, receipt, prepare=args.prepare)
        if receipt.exists():
            raise SmokeError("receipt path must be fresh")
        result["source"] = _source_binding()
        result["hardware"] = _hardware()
        result["capacity_before"] = _capacity(root, download=args.prepare)
        if args.prepare:
            root.mkdir(mode=0o700)
            (root / "snapshot").mkdir()
            (root / "wheel").mkdir()
            (root / "split").mkdir()
            (root / MARKER).write_text(json.dumps({"schema": "szl.airllm.hosted-task/v1", "model_id": MODEL_ID,
                                                    "revision": MODEL_REVISION, "source_sha": result["source"]["source_sha"]}) + "\n", encoding="utf-8")
            for name, (size, digest) in FILES.items():
                _download(f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/{name}",
                          root / "snapshot" / name, size, digest)
            _download(WHEEL_URL, root / "wheel" / WHEEL_NAME, WHEEL_BYTES, WHEEL_SHA256)
            result["snapshot"] = _verify_snapshot(root / "snapshot")
            result["wheel_sha256"] = WHEEL_SHA256
            result["task_cache_root"] = str(root / "cache")
            result.update(_generation_binding())
            result["generation"] = _generation_record(seed_applied=True)
            result["disposition"] = "MEASURED"
            result["evidence_class"] = "MEASURED"
            result["measurement_scope"] = "PINNED_SOURCE_BYTES_PREPARED_ONLY"
            exit_code = 0
        else:
            if args.prepare_receipt is None:
                raise SmokeError("run requires the prepare receipt")
            _marker(root)
            binding = _generation_binding()
            result["prepare_receipt_sha256"] = _bind_prepare_receipt(args.prepare_receipt.absolute(), result["source"]["source_sha"])
            result["previous_receipt_sha256"] = result["prepare_receipt_sha256"]
            result["input_sha256"] = binding["input_sha256"]
            result["harness_sha256"] = binding["harness_sha256"]
            result["generation"] = _generation_record()
            result["snapshot"] = _verify_snapshot(root / "snapshot")
            result["airllm"] = _verify_wheel(root / "wheel" / WHEEL_NAME)
            result["dependency_versions"] = _dependency_versions()
            result["dependency_integrity"] = "VERSION_OBSERVED_HASH_CLOSURE_NOT_VERIFIED"
            for path in (root / "cache" / "hf", root / "cache" / "xdg",
                         root / "cache" / "torch", root / "cache" / "tmp"):
                path.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
                env.pop(name, None)
            env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                        "HF_DATASETS_OFFLINE": "1", "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
                        "HF_HUB_DISABLE_TELEMETRY": "1", "HF_HOME": str(root / "cache" / "hf"),
                        "XDG_CACHE_HOME": str(root / "cache" / "xdg"),
                        "TORCH_HOME": str(root / "cache" / "torch"),
                        "TMPDIR": str(root / "cache" / "tmp")})
            started_worker = True
            result["inference_attempted"] = True
            worker_exit_code, worker_tail = _run_worker(root, receipt, env)
            result["worker_exit_code"] = worker_exit_code
            if worker_exit_code:
                raise SmokeError("AirLLM worker failed: " + worker_tail[-1200:])
            result["worker_result"] = json.loads(worker_tail.splitlines()[-1])
            worker_result = result["worker_result"]
            if worker_result.get("generated_token_count") != 1:
                raise SmokeError("worker returned no one-token result")
            worker_binding = _generation_binding(seed_applied=True)
            for key, expected in worker_binding.items():
                if worker_result.get(key) != expected:
                    raise SmokeError("worker generation binding differs from harness")
            result["metrics"] = {"evidence_class": "MEASURED",
                                 **{name: worker_result[name] for name in
                                    ("generated_token_count", "elapsed_seconds", "peak_rss_kib", "token_ids_sha256")}}
            result["inference_run"] = "MEASURED"
            result["disposition"] = "MEASURED"
            result["evidence_class"] = "MEASURED"
            result["measurement_scope"] = "HOSTED_CPU_ONE_TOKEN_SMOKE_ONLY"
            exit_code = 0
    except (OSError, ValueError, KeyError, IndexError, zipfile.BadZipFile,
            subprocess.SubprocessError, importlib.metadata.PackageNotFoundError) as exc:
        result["reason"] = f"{type(exc).__name__}: {str(exc)[-1200:]}"
        result["inference_run"] = "UNKNOWN" if started_worker else "UNAVAILABLE"
        result["inference_attempted"] = started_worker
    if started_worker:
        try:
            result["capacity_after"] = {"disk_free_bytes": shutil.disk_usage(root).free,
                                        "ram_available_bytes": _available_ram()}
        except OSError:
            result["capacity_after"] = "UNAVAILABLE"
    try:
        _write_receipt(receipt, result)
    except OSError as exc:
        print(json.dumps({"disposition": "BLOCKED", "receipt": "UNAVAILABLE", "error_type": type(exc).__name__}))
        return 2
    print(json.dumps({"disposition": result["disposition"], "receipt": str(receipt),
                      "inference_run": result["inference_run"]}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
