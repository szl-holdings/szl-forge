# SPDX-License-Identifier: Apache-2.0
"""Closed native-smoke context checks, not a release or Python capability.

The owner policy digest must come from outside this request. A file describing
its own key trust does not establish provenance. The committed candidate denies
training, independently of signatures. The v1 admission's container digest is
NOT a native-rootfs digest: no native identity adapter exists in this version,
so even a future eligible candidate remains blocked before model imports or
dispatch. Do not remove that hold by changing a qualification label.

All filesystem checks assume trusted source in a single-writer process. They
reject aliases and changes during reads, but are not a hostile same-UID sandbox.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import time

HERE = Path(__file__).absolute().parent
SOURCE_ROOT = HERE.parent.parent
RELATIVE = "frontier/qwen35-receiptagent-v4-json/"
REQUEST_SCHEMA = "szl.receiptagent.v4-native-run-request/v1"
POLICY_SCHEMA = "szl.receiptagent.v4-native-owner-policy/v1"
QUALIFICATION_SCHEMA = "szl.receiptagent.v4-native-qualification/v1"
PATH_NAMES = frozenset({
    "source_root", "model_dir", "tokenizer_dir", "nemo_source_root", "run_dir",
    "output_dir", "runtime_lock", "wheel_lock", "qualification", "expected_payload",
    "owner_policy", "admission", "python_executable",
})
QUANTIZATION_RECIPE = {
    "load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": "bfloat16", "bnb_4bit_use_double_quant": True,
}
PEFT_RECIPE = {
    "r": 16, "lora_alpha": 32, "lora_dropout": 0, "bias": "none",
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "task_type": "CAUSAL_LM",
}
OPTIMIZER_RECIPE = {
    "name": "AdamW", "lr": 0.0001, "betas": [0.9, 0.999], "eps": 1e-8,
    "weight_decay": 0, "max_grad_norm": 1.0,
}
NATIVE_RECIPE = {"quantization": QUANTIZATION_RECIPE, "peft": PEFT_RECIPE,
                 "optimizer": OPTIMIZER_RECIPE}
MAX_BOUNDS = {
    "sequence_tokens": 2048, "total_tokens": 400000, "wall_seconds": 1200,
    "kill_seconds": 30, "drain_seconds": 30, "telemetry_max_gap_seconds": 5,
    "gpu_memory_bytes": 40 * 1024**3, "host_memory_bytes": 64 * 1024**3,
    "temperature_limit_millicelsius": 85000, "log_cap_bytes": 8 * 1024**2,
    "output_cap_bytes": 1024**3, "artifact_cap_bytes": 512 * 1024**2,
    "cost_cap_minor_units": 200,
}
MIN_BOUNDS = {"disk_reserve_bytes": 10 * 1024**3, "evidence_reserve_bytes": 1024**3}
# This closed mapping prevents closure IDs being treated as attacker paths.
CODE_PATHS = {name.replace(".py", ""): RELATIVE + name for name in (
    "native_runtime.py", "native_worker.py", "native_supervisor.py", "signed_admission.py",
    "curriculum_admission.py", "training_tokens.py", "training_data.py", "json_contract.py",
    "probe_train_token_tensors.py", "probe_base_tokenizer.py", "probe_base_config.py",
    "nemo_source_binding.py",
)}
CODE_PATHS.update({"generic_sft_validator": "tools/validate_sft_dataset.py",
                   "nemo_wrapper": "tools/nemo_doctrine_gate.py"})
POLICY_KEYS = frozenset({
    "schema", "key_trust", "public_keys", "revoked_key_ids", "max_admission_age_seconds",
    "expected_payload_sha256", "paths_sha256", "qualification_sha256", "source_revision",
    "source_tree", "smoke_authorized", "recipe_sha256", "limits",
})
BARRIER_OBSERVATIONS = (
    "kernel_barrier_observed", "worker_cgroup_escape_denied", "worker_environment_cleared",
    "worker_home_credentials_hidden", "worker_network_denied",
)
QUALIFICATION_KEYS = frozenset({
    "schema", "qualified_at_epoch", "device_uuid", "python_prefix", "python_sha256",
    "packages", "lora_target_names", "loader_missing_keys", "loader_unexpected_keys",
    "loader_mismatched_keys", "loader_error_msgs", "systemd_properties", "containment_boundary",
    "containment_policy_sha256", "container_kind", "control_executable_sha256",
    "runner_rate_minor_units_per_hour", "cost_rate_source", *BARRIER_OBSERVATIONS,
})
# Published RFC8032 test vector: it must never become an owner production pin.
PUBLIC_TEST_KEY = "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a"


class RuntimeBindingError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def require(condition: bool, code: str) -> None:
    if condition is not True:
        raise RuntimeBindingError(code)


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _hash(value) -> bool:
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "DUPLICATE_JSON_FIELD")
        result[key] = value
    return result


def _parse(raw: bytes, *, allow_finite_floats: bool = False) -> dict:
    try:
        result = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs,
                            parse_constant=lambda _: (_ for _ in ()).throw(
                                RuntimeBindingError("NONFINITE_JSON_FORBIDDEN")))
    except (UnicodeError, ValueError, RecursionError) as exc:
        if isinstance(exc, RuntimeBindingError):
            raise
        raise RuntimeBindingError("JSON_INVALID") from exc
    require(type(result) is dict, "JSON_OBJECT_REQUIRED")
    pending = [(result, 0)]
    count = 0
    while pending:
        item, depth = pending.pop()
        count += 1
        require(depth <= 16 and count <= 20000, "JSON_STRUCTURE_CAP")
        if type(item) in (dict, list):
            require(len(item) <= 4096, "JSON_COLLECTION_CAP")
            pending.extend((child, depth + 1) for child in
                           (item.values() if type(item) is dict else item))
        elif type(item) is str:
            require(len(item) <= 8192, "JSON_STRING_CAP")
        elif type(item) is float and allow_finite_floats:
            require(math.isfinite(item), "NONFINITE_JSON_FORBIDDEN")
        else:
            require(type(item) in (int, bool, type(None)), "JSON_SCALAR_FORBIDDEN")
            if type(item) is int:
                require(abs(item) <= 2**63 - 1, "JSON_INTEGER_CAP")
    return result


def check_path(path: Path, *, may_not_exist: bool = False) -> Path:
    require(type(path) in (Path, type(Path())) and path.is_absolute(), "ABSOLUTE_PATH_REQUIRED")
    require(".." not in path.parts, "PATH_TRAVERSAL_FORBIDDEN")
    for part in reversed((path, *path.parents)):
        try:
            meta = part.lstat()
        except FileNotFoundError:
            require(may_not_exist, "PATH_UNAVAILABLE")
            continue
        require(not stat.S_ISLNK(meta.st_mode)
                and not getattr(meta, "st_file_attributes", 0) & 0x400,
                "PATH_ALIAS_FORBIDDEN")
    return path


def _read(path: Path, maximum: int) -> bytes:
    check_path(path)
    require(type(maximum) is int and 0 < maximum <= 64 * 1024**2, "READ_CAP_INVALID")
    path_before = path.lstat()
    require(stat.S_ISREG(path_before.st_mode) and path_before.st_nlink == 1,
            "REGULAR_UNALIASED_FILE_REQUIRED")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                         | getattr(os, "O_BINARY", 0))
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1,
                "REGULAR_UNALIASED_FILE_REQUIRED")
        require(before.st_size <= maximum, "FILE_BYTE_CAP")
        pieces = []
        remaining = maximum + 1
        while remaining:
            piece = os.read(descriptor, min(1024**2, remaining))
            if not piece:
                break
            pieces.append(piece)
            remaining -= len(piece)
        result = b"".join(pieces)
        after = os.fstat(descriptor)
        current = path.lstat()
        identity = lambda meta: (meta.st_dev, meta.st_ino, meta.st_size,
                                 meta.st_mtime_ns, meta.st_ctime_ns, meta.st_nlink)
        # Windows fstat ctime is change-time whereas pathlib lstat ctime can
        # report creation-time. Compare each API's before/after values, plus
        # cross-API file identity/size/mtime, rather than conflating those clocks.
        stable = lambda meta: (meta.st_dev, meta.st_ino, meta.st_size, meta.st_mtime_ns, meta.st_nlink)
        require(identity(before) == identity(after)
                and identity(path_before) == identity(current)
                and stable(before) == stable(current), "FILE_CHANGED_DURING_READ")
        require(len(result) == before.st_size and len(result) <= maximum, "FILE_BYTE_CAP")
        return result
    finally:
        os.close(descriptor)


def read_json(path: Path, maximum: int = 262144) -> dict:
    return _parse(_read(path, maximum))


def read_worker_result(path: Path, maximum: int = 65536) -> dict:
    """Metrics alone may contain finite floats; never an admission reader."""
    require(type(maximum) is int and 1 <= maximum <= 65536, "WORKER_RESULT_CAP_INVALID")
    return _parse(_read(path, maximum), allow_finite_floats=True)


def sha256_file(path: Path, max_bytes: int = 16 * 1024**2) -> str:
    """Stream bounded artifacts without the control-reader allocation limit."""
    check_path(path)
    require(type(max_bytes) is int and 1 <= max_bytes <= 8 * 1024**3, "HASH_CAP_INVALID")
    path_before = path.lstat()
    require(stat.S_ISREG(path_before.st_mode) and path_before.st_nlink == 1,
            "REGULAR_UNALIASED_FILE_REQUIRED")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                         | getattr(os, "O_BINARY", 0))
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1,
                "REGULAR_UNALIASED_FILE_REQUIRED")
        require(before.st_size <= max_bytes, "FILE_BYTE_CAP")
        hasher, size = hashlib.sha256(), 0
        while True:
            block = os.read(descriptor, min(1024**2, max_bytes - size + 1))
            if not block:
                break
            size += len(block)
            require(size <= max_bytes, "FILE_BYTE_CAP")
            hasher.update(block)
        after = os.fstat(descriptor)
        path_after = path.lstat()
        stable = lambda meta: (meta.st_dev, meta.st_ino, meta.st_size, meta.st_mtime_ns, meta.st_nlink)
        require(stable(before) == stable(after) == stable(path_before) == stable(path_after)
                and before.st_ctime_ns == after.st_ctime_ns
                and path_before.st_ctime_ns == path_after.st_ctime_ns
                and size == before.st_size, "FILE_CHANGED_DURING_READ")
        return hasher.hexdigest()
    finally:
        os.close(descriptor)


def check_source_eligibility(source_root: Path) -> None:
    """No signature/qualification can override these committed false flags."""
    check_path(source_root)
    require(source_root == SOURCE_ROOT, "EXECUTING_SOURCE_ROOT_MISMATCH")
    candidate = read_json(source_root / RELATIVE / "candidate.json")
    prereg = read_json(source_root / RELATIVE / "preregistration.json")
    require(candidate.get("candidate_id") == "SZL-ReceiptAgent-Qwen3.5-0.8B-v4-JSON",
            "CANDIDATE_ID_MISMATCH")
    require(candidate.get("training_eligible") is True
            and prereg.get("training_eligible") is True, "SOURCE_TRAINING_INELIGIBLE")
    require(candidate.get("publication_eligible") is False
            and candidate.get("execution_authority") is False
            and prereg.get("publication_eligible") is False
            and prereg.get("execution_authority") is False, "SMOKE_ONLY_AUTHORITY_REQUIRED")
    require(prereg.get("key_trust") == "INDEPENDENTLY_PINNED"
            and prereg.get("signature") not in (None, "UNAVAILABLE")
            and prereg.get("trainer_supervisor_binding") != "UNAVAILABLE",
            "SOURCE_ADMISSION_UNAVAILABLE")


def validate_policy(policy: dict, config: dict, expected_raw: bytes) -> None:
    require(set(policy) == POLICY_KEYS and policy["schema"] == POLICY_SCHEMA,
            "OWNER_POLICY_FIELDS_INVALID")
    require(policy["smoke_authorized"] is True
            and policy["key_trust"] == "INDEPENDENTLY_PINNED", "OWNER_AUTHORITY_UNAVAILABLE")
    require(policy["paths_sha256"] == _digest(canonical(config["paths"]))
            and policy["expected_payload_sha256"] == _digest(expected_raw)
            and policy["recipe_sha256"] == _digest(canonical(NATIVE_RECIPE)),
            "OWNER_CONTEXT_MISMATCH")
    keys = policy["public_keys"]
    require(type(keys) is dict and 1 <= len(keys) <= 16, "OWNER_PUBLIC_PINS_REQUIRED")
    for name, key in keys.items():
        require(type(name) is str and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", name)
                is not None and _hash(key) and key != PUBLIC_TEST_KEY,
                "OWNER_PUBLIC_PIN_INVALID_OR_TEST_KEY")
    revoked = policy["revoked_key_ids"]
    require(type(revoked) is list and len(revoked) <= 128
            and all(type(key) is str for key in revoked) and len(set(revoked)) == len(revoked),
            "REVOCATION_STATE_INVALID")
    require(type(policy["max_admission_age_seconds"]) is int
            and 1 <= policy["max_admission_age_seconds"] <= 3600, "ADMISSION_MAX_AGE_INVALID")
    require(_hash(policy["qualification_sha256"]), "QUALIFICATION_PIN_INVALID")
    for field in ("source_revision", "source_tree"):
        require(type(policy[field]) is str and re.fullmatch(r"[0-9a-f]{40}", policy[field]) is not None,
                "SOURCE_REVISION_PIN_INVALID")
    limits = policy["limits"]
    require(type(limits) is dict and set(limits) == set(MAX_BOUNDS) | set(MIN_BOUNDS),
            "OWNER_LIMIT_FIELDS_INVALID")
    for field, maximum in MAX_BOUNDS.items():
        require(type(limits[field]) is int and 0 <= limits[field] <= maximum,
                "OWNER_LIMIT_UNBOUNDED")
    for field, minimum in MIN_BOUNDS.items():
        require(type(limits[field]) is int and minimum <= limits[field] <= 1024**4,
                "OWNER_RESERVE_INVALID")


def validate_bounds(bounds: dict, limits: dict) -> None:
    for field in MAX_BOUNDS:
        minimum = 0 if field == "cost_cap_minor_units" else 1
        require(type(bounds.get(field)) is int and minimum <= bounds[field] <= limits[field],
                "ADMISSION_RESOURCE_LIMIT_EXCEEDED")
    for field in MIN_BOUNDS:
        require(type(bounds.get(field)) is int and limits[field] <= bounds[field] <= 1024**4,
                "ADMISSION_RESERVE_TOO_SMALL")
    require(bounds["artifact_cap_bytes"] <= bounds["output_cap_bytes"]
            and bounds["log_cap_bytes"] < bounds["evidence_reserve_bytes"],
            "OUTPUT_RESERVE_ORDER_INVALID")


def check_closure(entries: list, paths: dict[str, Path]) -> None:
    require(type(entries) is list and len(entries) == len(paths), "CLOSURE_COVERAGE_MISMATCH")
    seen = set()
    for entry in entries:
        require(type(entry) is dict and set(entry) == {"id", "sha256"}, "CLOSURE_FIELDS_INVALID")
        key = entry["id"]
        require(type(key) is str and key in paths and key not in seen,
                "CLOSURE_ID_UNKNOWN_OR_DUPLICATE")
        require(_hash(entry["sha256"]) and sha256_file(paths[key]) == entry["sha256"],
                "CLOSURE_BYTES_MISMATCH")
        seen.add(key)


def validate_qualification(qualification: dict, payload: dict, digest: str, now: int) -> None:
    require(set(qualification) == QUALIFICATION_KEYS
            and qualification["schema"] == QUALIFICATION_SCHEMA, "QUALIFICATION_FIELDS_INVALID")
    runner = payload["runner"]
    require(runner["backend"] == "LOCAL_ISOLATED", "NATIVE_BACKEND_REQUIRED")
    require(type(qualification["qualified_at_epoch"]) is int
            and qualification["qualified_at_epoch"] == runner["qualified_at_epoch"]
            and 0 <= now - runner["qualified_at_epoch"] <= min(3600, runner["max_qualification_age_seconds"]),
            "QUALIFICATION_STALE_OR_FUTURE")
    require(qualification["device_uuid"] == runner["device_id"]
            and re.fullmatch(r"GPU-[A-Za-z0-9-]{16,96}", runner["device_id"]) is not None,
            "QUALIFIED_DEVICE_MISMATCH")
    for field in ("device_qualification_sha256", "telemetry_qualification_sha256",
                  "containment_qualification_sha256"):
        require(runner[field] == digest, "QUALIFICATION_RECORD_BINDING_MISMATCH")
    require(payload["tokens"]["cpu_qualification_sha256"] == digest,
            "TOKEN_QUALIFICATION_BINDING_MISMATCH")
    require(all(qualification[field] is True for field in BARRIER_OBSERVATIONS),
            "QUALIFIED_BARRIER_UNAVAILABLE")
    require(qualification["containment_boundary"]
            == "TRUSTED_SOURCE_BOUND_PROCESS_NOT_HOSTILE_SAME_UID_SANDBOX",
            "CONTAINMENT_BOUNDARY_MISMATCH")
    require(qualification["containment_policy_sha256"] == runner["containment_policy_sha256"],
            "CONTAINMENT_POLICY_MISMATCH")
    require(all(qualification[field] == [] for field in (
        "loader_missing_keys", "loader_unexpected_keys", "loader_mismatched_keys", "loader_error_msgs")),
        "LOADER_KEY_COVERAGE_NOT_EXACT")
    names = qualification["lora_target_names"]
    require(type(names) is list and names and all(type(name) is str for name in names)
            and names == sorted(set(names)), "LORA_QUALIFIED_TARGETS_INVALID")
    require(_hash(qualification["python_sha256"]), "PYTHON_PIN_INVALID")
    prefix = qualification["python_prefix"]
    require(type(prefix) is str and Path(prefix).is_absolute(), "PYTHON_PREFIX_REQUIRED")
    packages = qualification["packages"]
    require(type(packages) is dict and set(packages) == {
        "torch", "transformers", "peft", "bitsandbytes", "safetensors", "tokenizers",
        "accelerate", "jsonschema", "cryptography"}, "DEPENDENCY_CLOSURE_INCOMPLETE")
    for package in packages.values():
        require(type(package) is dict and set(package) == {"version", "tree_sha256", "root"}
                and type(package["version"]) is str and 0 < len(package["version"]) <= 64
                and _hash(package["tree_sha256"]) and type(package["root"]) is str
                and Path(package["root"]).is_absolute(), "DEPENDENCY_QUALIFICATION_INVALID")
    controls = qualification["control_executable_sha256"]
    require(type(controls) is dict and set(controls) == {
        "/usr/bin/systemd-run", "/usr/bin/systemctl", "/usr/bin/nvidia-smi", "/usr/bin/env"}
        and all(_hash(value) for value in controls.values()), "CONTROL_EXECUTABLE_PINS_INCOMPLETE")
    rate = qualification["runner_rate_minor_units_per_hour"]
    require(type(rate) is int and 0 <= rate <= 100000
            and type(qualification["cost_rate_source"]) is str
            and 0 < len(qualification["cost_rate_source"]) <= 256, "COST_QUALIFICATION_UNAVAILABLE")
    bounds = payload["bounds"]
    worst_seconds = sum(bounds[field] for field in ("wall_seconds", "kill_seconds", "drain_seconds")) + 10
    require((rate * worst_seconds + 3599) // 3600 <= bounds["cost_cap_minor_units"],
            "COST_CAP_INSUFFICIENT")


@dataclass(frozen=True)
class BoundRun:
    """Captured data for trusted helpers, never an unforgeable Python permit."""
    request_path: Path
    owner_policy_sha256: str
    payload_raw: bytes
    config_raw: bytes
    qualification_raw: bytes

    @property
    def payload(self):
        return _parse(self.payload_raw)

    @property
    def config(self):
        return _parse(self.config_raw)

    @property
    def qualification(self):
        return _parse(self.qualification_raw)

    def path(self, name):
        require(name in PATH_NAMES, "UNKNOWN_PATH_NAME")
        return Path(self.config["paths"][name])

    @property
    def recipe(self):
        return json.loads(canonical(NATIVE_RECIPE))

    candidate_root = property(lambda self: self.path("source_root") / RELATIVE)
    train_path = property(lambda self: self.candidate_root / "train.jsonl")
    manifest_path = property(lambda self: self.candidate_root / "curriculum-manifest.json")
    nemo_source_root = property(lambda self: self.path("nemo_source_root"))
    model_dir = property(lambda self: self.path("model_dir"))
    tokenizer_dir = property(lambda self: self.path("tokenizer_dir"))
    run_dir = property(lambda self: self.path("run_dir"))
    output_dir = property(lambda self: self.path("output_dir"))
    worker_python = property(lambda self: self.path("python_executable"))
    worker_path = property(lambda self: self.candidate_root / "native_worker.py")
    nonce = property(lambda self: self.payload["identity"]["attempt_nonce"])
    seed = property(lambda self: self.payload["curriculum"]["seed"])
    python_prefix = property(lambda self: Path(self.qualification["python_prefix"]))

    @property
    def readonly_paths(self):
        files = [self.path("source_root") / relative for relative in CODE_PATHS.values()]
        files.extend((self.train_path, self.manifest_path, self.candidate_root / "candidate.json",
                      self.candidate_root / "preregistration.json"))
        files.extend(self.path(name) for name in (
            "runtime_lock", "wheel_lock", "qualification", "expected_payload", "owner_policy", "admission"))
        files.extend((self.request_path, self.model_dir, self.tokenizer_dir,
                      self.python_prefix))
        files.extend(self.nemo_source_root / ("szl_nemo/" + name) for name in (
            "__init__.py", "engine.py", "envelope.py", "receipt.py", "rules.py", "schema.py"))
        return tuple(sorted(set(files), key=str))


def load_bound_run(request_path: Path, *, owner_policy_sha256: str, role="supervisor") -> BoundRun:
    require(role in ("worker", "supervisor"), "RUNTIME_ROLE_INVALID")
    require(_hash(owner_policy_sha256), "OUT_OF_BAND_OWNER_POLICY_PIN_REQUIRED")
    request_raw = _read(request_path, 262144)
    config = _parse(request_raw)
    require(set(config) == {"schema", "paths"} and config["schema"] == REQUEST_SCHEMA
            and type(config["paths"]) is dict and set(config["paths"]) == PATH_NAMES,
            "RUN_REQUEST_FIELDS_INVALID")
    paths = config["paths"]
    require(all(type(value) is str and 0 < len(value) <= 4096 for value in paths.values()),
            "RUN_REQUEST_PATH_INVALID")
    # Mandatory current-source denial precedes signatures, nonce, telemetry,
    # subprocesses, model assets and every heavy framework import.
    check_source_eligibility(Path(paths["source_root"]))
    for name, value in paths.items():
        check_path(Path(value), may_not_exist=name in ("run_dir", "output_dir"))
    policy_raw = _read(Path(paths["owner_policy"]), 262144)
    require(_digest(policy_raw) == owner_policy_sha256, "OWNER_POLICY_PIN_MISMATCH")
    policy = _parse(policy_raw)
    expected_raw = _read(Path(paths["expected_payload"]), 65536)
    validate_policy(policy, config, expected_raw)
    from signed_admission import observe_admission, parse_canonical_bytes
    payload = parse_canonical_bytes(expected_raw)
    now = int(time.time())
    observation = observe_admission(
        _read(Path(paths["admission"]), 65536), expected_payload=expected_raw,
        public_keys={name: bytes.fromhex(key) for name, key in policy["public_keys"].items()},
        key_trust=policy["key_trust"], now_epoch=now,
        revoked_key_ids=frozenset(policy["revoked_key_ids"]),
        max_age_seconds=policy["max_admission_age_seconds"])
    require(all(getattr(observation, field) is True for field in (
        "document_matches_profile", "context_matches", "public_trust_observed",
        "time_window_matches", "revocation_observed", "signature_matches"))
        and observation.cryptographic_readiness == "AVAILABLE"
        and observation.cryptographic_verification == "RUN"
        and not observation.reasons and not observation.mismatch_fields, "ADMISSION_OBSERVATIONS_INCOMPLETE")
    require(payload["source"]["revision"] == policy["source_revision"]
            and payload["source"]["tree"] == policy["source_tree"], "SOURCE_PIN_MISMATCH")
    validate_bounds(payload["bounds"], policy["limits"])
    check_closure(payload["source"]["code_closure"],
                  {key: Path(paths["source_root"]) / relative for key, relative in CODE_PATHS.items()})
    require(payload["runtime"]["quantization_recipe_sha256"] == _digest(canonical(QUANTIZATION_RECIPE))
            and payload["runtime"]["peft_recipe_sha256"] == _digest(canonical(PEFT_RECIPE)),
            "RECIPE_BINDING_MISMATCH")
    require(sha256_file(Path(paths["runtime_lock"])) == payload["runtime"]["dependency_lock_sha256"]
            and sha256_file(Path(paths["wheel_lock"])) == payload["runtime"]["wheel_lock_sha256"],
            "RUNTIME_LOCK_BINDING_MISMATCH")
    qualification_raw = _read(Path(paths["qualification"]), 262144)
    digest = _digest(qualification_raw)
    require(digest == policy["qualification_sha256"], "QUALIFICATION_PIN_MISMATCH")
    validate_qualification(_parse(qualification_raw), payload, digest, now)
    # Deliberate unavailable boundary: v1 explicitly binds a container artifact.
    # No label, owner JSON, fixture, or caller-selected callback can make native
    # systemd a measured OCI identity. A versioned profile/adapter is required.
    raise RuntimeBindingError("CONTAINER_IDENTITY_UNAVAILABLE")


def revalidate_bound_run(run: BoundRun, *, role="worker") -> BoundRun:
    require(type(run) is BoundRun, "BOUND_RUN_TYPE_INVALID")
    fresh = load_bound_run(run.request_path, owner_policy_sha256=run.owner_policy_sha256, role=role)
    require(fresh == run, "BOUND_RUN_CHANGED")
    return fresh
