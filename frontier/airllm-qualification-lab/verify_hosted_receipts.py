"""Offline integrity checks for profiled schema-v2 hosted CPU receipts.

Expected identities must come from an independent source. This checks recorded
bindings, not signatures, downloaded model bytes, runtime, or model quality.
The sibling harness must match the supplied SHA-256 byte for byte (including
line endings); only its literal constants are read, and no harness code runs.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import stat


MAX_RECEIPT_BYTES = 256 * 1024
MAX_JSON_DEPTH = 32
SCHEMA = "szl.airllm.offline-receipt-verification/v1"
SCOPE = "OFFLINE_RECEIPT_INTEGRITY_AND_BINDING_ONLY"
DEPENDENCY_INTEGRITY = "VERSION_OBSERVED_HASH_CLOSURE_NOT_VERIFIED"
_MISSING = object()
_PIN_NAMES = {
    "SCHEMA", "DEFAULT_PROFILE", "PROFILE_360M", "MODEL_ID", "MODEL_REVISION",
    "MODEL_LICENSE_CARD", "MODEL_360M_ID", "MODEL_360M_REVISION", "FILES", "FILES_360M",
    "AIRLLM_VERSION", "AIRLLM_SOURCE_COMMIT", "WHEEL_SHA256", "PROMPT",
    "DATASET_EVIDENCE_CLASS", "DATASET_SCOPE", "DO_SAMPLE", "GENERATION_SEED",
    "MAX_NEW_TOKENS", "DEPENDENCIES",
}


class ReceiptError(ValueError):
    """A receipt or independently supplied expectation failed validation."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReceiptError(message)


def _same(actual, expected, label: str) -> None:
    """Compare JSON values without Python's bool/int or int/float coercion."""
    _require(type(actual) is type(expected), label + " has an invalid type")
    if isinstance(expected, dict):
        _require(actual.keys() == expected.keys(), label + " has unexpected fields")
        for key, value in expected.items():
            _same(actual[key], value, label + "." + key)
    else:
        _require(actual == expected, label + " differs")


def _fields(document, expected: dict, label: str) -> None:
    _require(type(document) is dict, label + " must be an object")
    for key, value in expected.items():
        _same(document.get(key, _MISSING), value, label + "." + key)


def _unique_object(pairs: list) -> dict:
    result = {}
    for key, value in pairs:
        _require(key not in result, "receipt has duplicate JSON keys")
        result[key] = value
    return result


def _finite_float(value: str) -> float:
    result = float(value)
    _require(math.isfinite(result), "receipt contains a nonfinite number")
    return result


def _bounded_int(value: str) -> int:
    _require(len(value.lstrip("-")) <= 20, "receipt integer exceeds the digit limit")
    return int(value)


def _reject_constant(_value: str):
    raise ReceiptError("receipt contains a nonfinite number")


def _parse(raw: bytes, label: str) -> dict:
    _require(type(raw) is bytes and 0 < len(raw) <= MAX_RECEIPT_BYTES,
             label + " must contain bounded JSON bytes")
    try:
        document = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object,
                              parse_int=_bounded_int, parse_float=_finite_float, parse_constant=_reject_constant)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ReceiptError(label + " is not strict JSON") from exc
    _require(type(document) is dict, label + " must be a JSON object")
    pending = [(document, 0)]
    while pending:
        value, depth = pending.pop()
        _require(depth <= MAX_JSON_DEPTH, label + " exceeds the JSON depth limit")
        if isinstance(value, dict):
            pending.extend((child, depth + 1) for child in value.values())
        elif isinstance(value, list):
            pending.extend((child, depth + 1) for child in value)
    return document


def _read_file(path: Path) -> bytes:
    _require(not path.is_symlink() and stat.S_ISREG(path.stat().st_mode),
             "input must be a regular, non-symlink file")
    with path.open("rb") as stream:
        raw = stream.read(MAX_RECEIPT_BYTES + 1)
    _require(0 < len(raw) <= MAX_RECEIPT_BYTES, "input exceeds the byte limit or is empty")
    return raw


def _load_pins(expected_hash: str) -> dict:
    try:
        raw = _read_file(Path(__file__).resolve().with_name("hosted_cpu_smoke.py"))
    except OSError as exc:
        raise ReceiptError("cannot read local trusted harness") from exc
    _require(hashlib.sha256(raw).hexdigest() == expected_hash,
             "local harness bytes do not match the expected harness SHA-256")
    try:
        pins = {}
        for node in ast.parse(raw).body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id in _PIN_NAMES:
                    _require(target.id not in pins, "harness has duplicate pin assignments")
                    pins[target.id] = ast.literal_eval(node.value)
        _require(pins.keys() == _PIN_NAMES, "harness literal pins are incomplete")
    except (ValueError, SyntaxError, RecursionError) as exc:
        raise ReceiptError("cannot read trusted harness literal pins") from exc
    return pins


def _hex(value, length: int, label: str) -> None:
    _require(type(value) is str and re.fullmatch("[0-9a-f]{" + str(length) + "}", value) is not None,
             label + " must be a lowercase hexadecimal digest")


def _metrics(worker: dict) -> dict:
    _same(worker.get("generated_token_count"), 1, "worker.generated_token_count")
    elapsed, peak = worker.get("elapsed_seconds"), worker.get("peak_rss_kib")
    _require(type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed >= 0,
             "worker elapsed_seconds must be finite and nonnegative")
    _require(type(peak) is int and peak > 0, "worker peak_rss_kib must be a positive integer")
    _hex(worker.get("token_ids_sha256"), 64, "worker.token_ids_sha256")
    return {"evidence_class": "MEASURED", **{key: worker[key] for key in
            ("generated_token_count", "elapsed_seconds", "peak_rss_kib", "token_ids_sha256")}}


def _report(disposition: str) -> dict:
    return {"schema": SCHEMA, "scope": SCOPE, "disposition": disposition,
            "evidence_class": disposition, "signature_verified": False,
            "inference_replayed": False, "model_qualification": "UNKNOWN",
            "execution_authority": False, "publication_eligible": False, "training_eligible": False}


def verify_receipts(prepare_bytes: bytes, smoke_bytes: bytes, *, expected_source_sha: str,
                    expected_profile: str, expected_run_id: str, expected_run_attempt: str,
                    expected_harness_sha256: str) -> dict:
    """Return an offline readback report; raise ReceiptError on any mismatch."""
    _hex(expected_source_sha, 40, "expected source SHA")
    _hex(expected_harness_sha256, 64, "expected harness SHA-256")
    for value in (expected_run_id, expected_run_attempt):
        _require(type(value) is str and re.fullmatch("[1-9][0-9]{0,19}", value) is not None,
                 "expected run ID and attempt must be positive decimal strings")
    pins = _load_pins(expected_harness_sha256)
    profiles = {
        pins["DEFAULT_PROFILE"]: (pins["MODEL_ID"], pins["MODEL_REVISION"], pins["FILES"]),
        pins["PROFILE_360M"]: (pins["MODEL_360M_ID"], pins["MODEL_360M_REVISION"], pins["FILES_360M"]),
    }
    _require(type(expected_profile) is str and expected_profile in profiles, "expected profile is not pinned")
    model_id, revision, files = profiles[expected_profile]
    _same(pins["DO_SAMPLE"], False, "harness.DO_SAMPLE")
    _same(pins["GENERATION_SEED"], 0, "harness.GENERATION_SEED")
    _same(pins["MAX_NEW_TOKENS"], 1, "harness.MAX_NEW_TOKENS")
    dataset = (json.dumps({"prompt": pins["PROMPT"]}, sort_keys=True, separators=(",", ":")) + "\n").encode()
    dataset_hash = hashlib.sha256(dataset).hexdigest()
    generation = {"model_profile": expected_profile, "do_sample": False, "max_new_tokens": 1,
                  "seed": 0, "seed_applied": False}
    binding = {**generation, "input_sha256": hashlib.sha256(pins["PROMPT"].encode()).hexdigest(),
               "dataset_sha256": dataset_hash, "harness_sha256": expected_harness_sha256}
    source = {"repository": "szl-holdings/szl-forge", "source_sha": expected_source_sha,
              "run_id": expected_run_id, "run_attempt": expected_run_attempt}
    snapshot = {"model_profile": expected_profile, "model_type": "llama", "architecture": "LlamaForCausalLM",
                "tokenizer_class": "GPT2Tokenizer", "file_bytes": {name: size for name, (size, _) in files.items()},
                "file_sha256": {name: digest for name, (_, digest) in files.items()}}
    common = {"schema": pins["SCHEMA"], "source": source, "model_profile": expected_profile,
              "model_id": model_id, "model_revision": revision, "base_revision": revision,
              "license_card_metadata": pins["MODEL_LICENSE_CARD"], "snapshot": snapshot,
              "dataset_sha256": dataset_hash, "dataset_evidence_class": pins["DATASET_EVIDENCE_CLASS"],
              "dataset_scope": pins["DATASET_SCOPE"], "seed": 0,
              "seed_scope": "PYTHON_AND_TORCH_GREEDY_WORKER_ONLY",
              "input_sha256": binding["input_sha256"], "harness_sha256": expected_harness_sha256,
              "loss": {"evidence_class": "UNAVAILABLE", "status": "NOT_RUN", "reason": "INFERENCE_ONLY"},
              "airllm_source_commit": pins["AIRLLM_SOURCE_COMMIT"], "delete_original": False,
              "disposition": "MEASURED", "evidence_class": "MEASURED", "signing_status": "UNSIGNED",
              "execution_authority": False, "publication_eligible": False, "training_eligible": False,
              "held_out_gate": "NOT_RUN", "numerical_parity": "NOT_RUN", "model_quality": "UNKNOWN"}
    prepare, smoke = _parse(prepare_bytes, "PREPARE"), _parse(smoke_bytes, "RUN")
    for document, stage in ((prepare, "PREPARE"), (smoke, "RUN")):
        _fields(document, {**common, "stage": stage}, stage)
        hardware = document.get("hardware")
        _fields(hardware, {"runner_os": "Linux", "accelerator": "CPU_ONLY_NO_GPU_REQUESTED"}, stage + ".hardware")
        _require(hardware.keys() == {"runner_os", "runner_arch", "cpu_model", "logical_cpu_count", "accelerator"},
                 stage + ".hardware fields differ")
        for name in ("runner_arch", "cpu_model"):
            _require(type(hardware[name]) is str and bool(hardware[name].strip()),
                     stage + ".hardware." + name + " must be a nonempty string")
        count = hardware["logical_cpu_count"]
        _require(type(count) is int and count > 0, stage + ".hardware.logical_cpu_count must be a positive integer")
        for name in ("signature_verified", "inference_replayed"):
            if name in document:
                _same(document[name], False, stage + "." + name)
    _same(smoke["hardware"], prepare["hardware"], "RUN.hardware")
    _fields(prepare, {**binding, "generation": generation, "inference_attempted": False,
                     "inference_run": "UNAVAILABLE", "wheel_sha256": pins["WHEEL_SHA256"],
                     "measurement_scope": "PINNED_SOURCE_BYTES_PREPARED_ONLY",
                     "prepare_receipt_sha256": "UNAVAILABLE", "previous_receipt_sha256": "UNAVAILABLE",
                     "metrics": {"evidence_class": "UNAVAILABLE", "generated_token_count": None,
                                 "elapsed_seconds": None, "peak_rss_kib": None, "token_ids_sha256": None}}, "PREPARE")
    prepare_hash = hashlib.sha256(prepare_bytes).hexdigest()
    _fields(smoke, {"generation": {**generation, "seed_applied": True}, "inference_attempted": True,
                   "inference_run": "MEASURED", "worker_exit_code": 0,
                   "measurement_scope": "HOSTED_CPU_ONE_TOKEN_SMOKE_ONLY",
                   "prepare_receipt_sha256": prepare_hash, "previous_receipt_sha256": prepare_hash,
                   "dependency_integrity": DEPENDENCY_INTEGRITY}, "RUN")
    airllm = smoke.get("airllm")
    _fields(airllm, {"version": pins["AIRLLM_VERSION"], "wheel_sha256": pins["WHEEL_SHA256"],
                     "source_commit": pins["AIRLLM_SOURCE_COMMIT"]}, "RUN.airllm")
    matched = airllm.get("installed_package_files_matched")
    _require(type(matched) is int and matched > 0, "RUN.airllm package file count must be a positive integer")
    dependencies = smoke.get("dependency_versions")
    _require(type(dependencies) is dict and dependencies.keys() == pins["DEPENDENCIES"].keys(),
             "RUN dependency version fields differ")
    for name, version in pins["DEPENDENCIES"].items():
        allowed = (version, version + "+cpu") if name == "torch" else (version,)
        _require(type(dependencies[name]) is str and dependencies[name] in allowed, "RUN dependency version differs")
    worker = smoke.get("worker_result")
    _fields(worker, {**binding, "seed_applied": True, "status": "MEASURED",
                     "scope": "HOSTED_CPU_ONE_TOKEN_SMOKE_ONLY", "model_quality": "UNKNOWN",
                     "numerical_parity": "NOT_RUN", "execution_authority": False}, "worker")
    for name in ("publication_eligible", "training_eligible", "signature_verified", "inference_replayed"):
        if name in worker:
            _same(worker[name], False, "worker." + name)
    _same(smoke.get("metrics"), _metrics(worker), "RUN.metrics")
    return {**_report("MEASURED"), "source_sha": expected_source_sha, "model_profile": expected_profile,
            "run_id": expected_run_id, "run_attempt": expected_run_attempt,
            "harness_sha256": expected_harness_sha256, "prepare_sha256": prepare_hash,
            "smoke_sha256": hashlib.sha256(smoke_bytes).hexdigest(), "dependency_integrity": DEPENDENCY_INTEGRITY}


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ReceiptError("invalid or missing command-line arguments")


def main(argv: list[str] | None = None) -> int:
    parser = _Parser(description=__doc__)
    parser.add_argument("--prepare", type=Path, required=True)
    parser.add_argument("--smoke", type=Path, required=True)
    for name in ("source-sha", "profile", "run-id", "run-attempt", "harness-sha256"):
        parser.add_argument("--expected-" + name, required=True)
    try:
        args = parser.parse_args(argv)
        result = verify_receipts(_read_file(args.prepare), _read_file(args.smoke),
                                 expected_source_sha=args.expected_source_sha, expected_profile=args.expected_profile,
                                 expected_run_id=args.expected_run_id, expected_run_attempt=args.expected_run_attempt,
                                 expected_harness_sha256=args.expected_harness_sha256)
        exit_code = 0
    except (ReceiptError, OSError) as exc:
        result = {**_report("BLOCKED"), "reason": str(exc)[:240] if isinstance(exc, ReceiptError) else "input read failed"}
        exit_code = 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
