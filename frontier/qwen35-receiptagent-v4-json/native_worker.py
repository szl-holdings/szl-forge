# SPDX-License-Identifier: Apache-2.0
"""Native, train-only v4 supervised-smoke worker; never a launch authority.

The command line must first pass the separately reviewed native runtime and
supervisor gates. Heavy runtime imports occur only after those gates. The pure
helpers below are trusted-process implementation details, not capabilities or a
sandbox. CPU fixtures exercise math and fault paths, not Qwen/GPU qualification.
No decode, evaluation-set input, Hub client, signing, or secret-key interface is
provided. The current committed candidate remains ineligible and must be denied.
"""
from __future__ import annotations

from dataclasses import replace
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys

# ``-I`` deliberately removes the script directory. Add only this exact reviewed
# source directory, never cwd or an environment-supplied search path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import curriculum_admission as admission
import training_tokens as tokens

RELEASE_MARKER = b"SZL-RA4-CONTAINMENT-VERIFIED\n"
RESULT_SCHEMA = "szl.receiptagent.v4-native-worker-result/v1"


class WorkerError(ValueError):
    """Value-free failure code; never propagate training text or token IDs."""


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise WorkerError(code)


def _sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def _stream_asset_hash(path: Path, maximum: int = 8 * 1024**3) -> str:
    """Hash local immutable qualification inputs without allocating their bytes."""
    import native_runtime as runtime
    runtime.check_path(path)
    path_before = path.lstat()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    with os.fdopen(os.open(path, flags), "rb") as stream:
        before = os.fstat(stream.fileno())
        _require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
                 and 0 < before.st_size <= maximum, "LOCAL_MODEL_ASSET_INVALID")
        digest = hashlib.sha256()
        length = 0
        while True:
            block = stream.read(1024**2)
            if not block:
                break
            length += len(block)
            _require(length <= maximum, "LOCAL_MODEL_ASSET_TOO_LARGE")
            digest.update(block)
        after = os.fstat(stream.fileno())
    path_after = path.lstat()
    # Windows fstat and pathname stat can report different ctime semantics.
    # Compare each API's own before/after ctime and cross-compare stable fields.
    identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_nlink, value.st_mtime_ns)
    _require(identity(before) == identity(after) == identity(path_before) == identity(path_after)
             and before.st_ctime_ns == after.st_ctime_ns
             and path_before.st_ctime_ns == path_after.st_ctime_ns
             and length == before.st_size, "LOCAL_MODEL_ASSET_CHANGED")
    return digest.hexdigest()


def verify_local_model_assets(run) -> None:
    """Closed safetensors-only base closure, streamed before framework imports.

    Hashes bind owner-qualified bytes, not their origin. Mutable same-UID source
    roots are not a sandbox. The actual production CLI is separately blocked
    until the container/qualification identity adapter exists and is reviewed.
    """
    closure = run.payload["base"]["assets_closure"]
    _require(type(closure) is list and 2 <= len(closure) <= 128, "MODEL_ASSET_CLOSURE_REQUIRED")
    allowed = {"config.json", "generation_config.json", "model.safetensors", "model.safetensors.index.json"}
    committed = {}
    for item in closure:
        _require(type(item) is dict and set(item) == {"id", "sha256"}, "MODEL_ASSET_DESCRIPTOR_INVALID")
        name = item["id"]
        _require(type(name) is str and (name in allowed or re.fullmatch(
            r"model-[0-9]{5}-of-[0-9]{5}\.safetensors", name) is not None)
            and name not in committed and type(item["sha256"]) is str
            and re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is not None,
            "MODEL_ASSET_NAME_OR_DIGEST_FORBIDDEN")
        committed[name] = item["sha256"]
    _require("config.json" in committed, "MODEL_CONFIG_ASSET_REQUIRED")
    names = {entry.name for entry in run.model_dir.iterdir()}
    _require(names == set(committed), "LOCAL_MODEL_ASSET_SET_MISMATCH")
    for name, digest in committed.items():
        _require(_stream_asset_hash(run.model_dir / name) == digest, "LOCAL_MODEL_ASSET_HASH_MISMATCH")
    shards = {name for name in committed if name.endswith(".safetensors")}
    if "model.safetensors" in shards:
        _require(shards == {"model.safetensors"} and "model.safetensors.index.json" not in committed,
                 "MIXED_MODEL_WEIGHT_LAYOUT_FORBIDDEN")
    else:
        _require(shards and "model.safetensors.index.json" in committed, "SHARDED_WEIGHT_INDEX_REQUIRED")
        index_path = run.model_dir / "model.safetensors.index.json"
        _require(index_path.stat().st_size <= 4 * 1024**2, "SHARDED_WEIGHT_INDEX_TOO_LARGE")

        def unique_pairs(pairs):
            result = {}
            for key, value in pairs:
                _require(key not in result, "SHARDED_WEIGHT_INDEX_DUPLICATE_KEY")
                result[key] = value
            return result

        index = json.loads(index_path.read_bytes(), object_pairs_hook=unique_pairs)
        _require(type(index) is dict and set(index) <= {"metadata", "weight_map"}
                 and type(index.get("weight_map")) is dict and index["weight_map"]
                 and all(type(key) is str and type(value) is str
                         for key, value in index["weight_map"].items())
                 and set(index["weight_map"].values()) == shards, "SHARDED_WEIGHT_INDEX_CLOSURE_MISMATCH")


def verify_token_asset_closure(run) -> None:
    import probe_base_tokenizer as tokenizer_probe
    closure = run.payload["tokens"]["assets_closure"]
    _require(type(closure) is list and len(closure) == len(tokenizer_probe.ASSET_SHA256),
             "TOKEN_ASSET_CLOSURE_REQUIRED")
    observed = {}
    for item in closure:
        _require(type(item) is dict and set(item) == {"id", "sha256"}
                 and type(item["id"]) is str and item["id"] not in observed,
                 "TOKEN_ASSET_DESCRIPTOR_INVALID")
        observed[item["id"]] = item["sha256"]
    _require(observed == tokenizer_probe.ASSET_SHA256, "SIGNED_TOKEN_ASSET_CLOSURE_MISMATCH")
    tokenizer_probe._read_assets(run.tokenizer_dir)


def verify_dependency_identity(run) -> None:
    """No version/report equality is promoted to executable package identity.

    This v1 source has no reviewed byte-tree/installed-wheel identity adapter.
    The separate shared runtime also refuses the absent OCI identity adapter.
    Both remain hard denials; a future versioned qualification must implement
    and test those adapters, not change labels or supply a callback override.
    """
    raise WorkerError("DEPENDENCY_IDENTITY_UNAVAILABLE")


def checked_rows(retained: tokens.RetainedTrainingTokens) -> tuple[int, ...]:
    """Recheck the exact immutable rows and their ledger before tensor creation.

    This does not rerender, retokenize, drop, reorder, truncate, shift stored
    labels, or accept a serialized tensor report in place of freshly retained
    data. Causal shifting is applied only by the loss function below.
    """
    _require(type(retained) is tokens.RetainedTrainingTokens, "RETAINED_TOKENS_REQUIRED")
    _require(type(retained.rows) is tuple and len(retained.rows) == admission.MAX_ROWS,
             "COMPLETE_360_ROWS_REQUIRED")
    report = retained.report
    _require(type(report) is tokens.TokenReport and type(report.rows) is tuple
             and len(report.rows) == admission.MAX_ROWS, "TOKEN_LEDGER_REQUIRED")
    counts = []
    total_tokens = 0
    for index, (row, ledger) in enumerate(zip(retained.rows, report.rows)):
        _require(type(row) is tokens.TrainingTensorRow and type(ledger) is tokens.RowLedger
                 and type(ledger.index) is int and ledger.index == index,
                 "TOKEN_ROW_ORDER_MISMATCH")
        n = len(row.input_ids)
        _require(2 <= n <= report.sequence_limit, "TOKEN_ROW_CAPACITY_MISMATCH")
        _require(all(type(values) is tuple and len(values) == n
                     for values in (row.input_ids, row.attention_mask, row.labels)),
                 "TOKEN_ROW_SHAPE_MISMATCH")
        _require(all(type(value) is int and value >= 0 for value in row.input_ids)
                 and all(type(value) is int and value in (0, 1) for value in row.attention_mask)
                 and all(type(value) is int and value >= tokens.IGNORE_INDEX
                         and (value == tokens.IGNORE_INDEX or value >= 0) for value in row.labels),
                 "TOKEN_ROW_TYPE_MISMATCH")
        _require(_sha((row.input_ids, row.attention_mask, row.labels)) == ledger.tensors_sha256
                 and ledger.sequence_tokens == n, "TOKEN_ROW_HASH_MISMATCH")
        _require(row.labels[0] == tokens.IGNORE_INDEX, "FIRST_CAUSAL_LABEL_MUST_BE_MASKED")
        count = sum(label != tokens.IGNORE_INDEX for label in row.labels[1:])
        _require(count > 0 and count == ledger.assistant_tokens + ledger.termination_tokens,
                 "SUPERVISED_TOKEN_COVERAGE_MISMATCH")
        _require(all(attention == 1 or label == tokens.IGNORE_INDEX
                     for attention, label in zip(row.attention_mask, row.labels)),
                 "PADDING_SUPERVISION_FORBIDDEN")
        counts.append(count)
        total_tokens += n
    _require(total_tokens == report.total_tokens, "AGGREGATE_TOKEN_LEDGER_MISMATCH")
    _require(_sha(tuple((row.index, row.tensors_sha256) for row in report.rows))
             == report.ordered_tensors_sha256, "ORDERED_TENSOR_LEDGER_MISMATCH")
    return tuple(counts)


def collate_exact(rows: tuple[tokens.TrainingTensorRow, ...], *, pad_id: int,
                  torch, device: str) -> dict:
    """Create tensors directly from retained rows; only add masked right padding."""
    _require(type(rows) is tuple and rows and type(pad_id) is int and pad_id >= 0,
             "COLLATION_INPUT_INVALID")
    width = max(len(row.input_ids) for row in rows)
    values = {"input_ids": [], "attention_mask": [], "labels": []}
    for row in rows:
        padding = width - len(row.input_ids)
        values["input_ids"].append(row.input_ids + (pad_id,) * padding)
        values["attention_mask"].append(row.attention_mask + (0,) * padding)
        values["labels"].append(row.labels + (tokens.IGNORE_INDEX,) * padding)
    return {name: torch.tensor(value, dtype=torch.long, device=device)
            for name, value in values.items()}


def verify_optimizer_coverage(model, optimizer, expected_names: tuple[str, ...]) -> tuple:
    """Require exactly the declared LoRA parameters, no base or duplicate entries."""
    _require(type(expected_names) is tuple and expected_names
             and len(set(expected_names)) == len(expected_names), "LORA_PARAMETER_SET_INVALID")
    named = tuple(model.named_parameters())
    trainable = tuple((name, parameter) for name, parameter in named if parameter.requires_grad)
    _require(tuple(sorted(name for name, _ in trainable)) == tuple(sorted(expected_names)),
             "TRAINABLE_PARAMETER_COVERAGE_MISMATCH")
    actual = tuple(parameter for group in optimizer.param_groups for parameter in group["params"])
    expected = tuple(parameter for _, parameter in trainable)
    _require(len(actual) == len(expected) and len({id(p) for p in actual}) == len(actual)
             and {id(p) for p in actual} == {id(p) for p in expected},
             "OPTIMIZER_PARAMETER_COVERAGE_MISMATCH")
    _require(all(parameter.requires_grad is False for name, parameter in named
                 if name not in expected_names), "BASE_PARAMETERS_NOT_FROZEN")
    return expected


def one_update(model, optimizer, retained: tokens.RetainedTrainingTokens, *,
               microbatch_rows: int, pad_id: int, max_grad_norm: float,
               expected_names: tuple[str, ...], torch, device: str,
               before_update, after_microbatch=None) -> dict:
    """One token-weighted Adam update after each of all 360 rows exactly once.

    Loss is explicitly the sum of next-token cross entropies over nonignored
    labels divided by the single global supervised-token denominator. It is not
    an unweighted mean of unequal row or microbatch means. The caller supplies
    the final admission/lease revalidation callback; a helper is not authority.
    """
    counts = checked_rows(retained)
    _require(type(microbatch_rows) is int and 1 <= microbatch_rows <= admission.MAX_ROWS
             and admission.MAX_ROWS % microbatch_rows == 0, "MICROBATCH_COVERAGE_INVALID")
    _require(type(max_grad_norm) in (int, float) and not isinstance(max_grad_norm, bool)
             and math.isfinite(max_grad_norm) and max_grad_norm > 0, "GRADIENT_NORM_INVALID")
    _require(callable(before_update), "FINAL_BOUND_REVALIDATION_REQUIRED")
    if after_microbatch is not None:
        _require(callable(after_microbatch), "MICROBATCH_OBSERVER_INVALID")
    parameters = verify_optimizer_coverage(model, optimizer, expected_names)
    total_supervised = sum(counts)
    optimizer.zero_grad(set_to_none=True)
    model.train()
    accumulated_loss = 0.0
    seen = 0
    for start in range(0, admission.MAX_ROWS, microbatch_rows):
        selected = retained.rows[start:start + microbatch_rows]
        batch = collate_exact(selected, pad_id=pad_id, torch=torch, device=device)
        labels = batch.pop("labels")
        outputs = model(**batch, use_cache=False)
        logits = outputs.logits
        _require(logits.ndim == 3 and tuple(logits.shape[:2]) == tuple(labels.shape)
                 and logits.shape[-1] >= 2, "MODEL_LOGIT_SHAPE_MISMATCH")
        expected_count = sum(counts[start:start + microbatch_rows])
        _require(int(labels[:, 1:].ne(tokens.IGNORE_INDEX).sum().item()) == expected_count,
                 "CAUSAL_SUPERVISION_COUNT_CHANGED")
        loss_sum = torch.nn.functional.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.shape[-1]).float(),
            labels[:, 1:].reshape(-1), ignore_index=tokens.IGNORE_INDEX, reduction="sum")
        _require(bool(torch.isfinite(loss_sum).item()), "NONFINITE_LOSS")
        weighted_loss = loss_sum / total_supervised
        weighted_loss.backward()
        accumulated_loss += float(loss_sum.detach().item()) / total_supervised
        seen += len(selected)
        # Do not hold an entire batch's logits or graph until the next batch.
        del outputs, logits, loss_sum, weighted_loss, batch, labels
        if after_microbatch is not None:
            after_microbatch(seen)
    _require(seen == admission.MAX_ROWS and math.isfinite(accumulated_loss),
             "INCOMPLETE_OR_NONFINITE_ACCUMULATION")
    verify_optimizer_coverage(model, optimizer, expected_names)
    for parameter in parameters:
        _require(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all().item()),
                 "MISSING_OR_NONFINITE_GRADIENT")
    norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm, error_if_nonfinite=True)
    _require(bool(torch.isfinite(norm).item()), "NONFINITE_GRADIENT_NORM")
    before_update()
    optimizer.step()
    for parameter in parameters:
        _require(bool(torch.isfinite(parameter).all().item()), "NONFINITE_UPDATED_PARAMETER")
    optimizer.zero_grad(set_to_none=True)
    return {"rows": seen, "optimizer_updates": 1, "supervised_tokens": total_supervised,
            "token_weighted_loss": accumulated_loss, "gradient_norm_before_clip": float(norm.item()),
            "ordered_tensors_sha256": retained.report.ordered_tensors_sha256,
            "row_order_sha256": _sha(tuple(range(admission.MAX_ROWS)))}


def read_supervisor_release(stream) -> None:
    """Containment sequencing only; a marker is deliberately not authorization."""
    _require(stream.read(len(RELEASE_MARKER)) == RELEASE_MARKER,
             "SUPERVISOR_CONTAINMENT_RELEASE_REQUIRED")


def verify_linux_containment(run, *, proc: Path = Path("/proc"),
                             cgroup_root: Path = Path("/sys/fs/cgroup")) -> None:
    """Read back the worker's actual cgroup-v2 placement and resource barriers."""
    _require(sys.platform == "linux", "LINUX_CONTAINMENT_REQUIRED")
    raw = (proc / "self" / "cgroup").read_bytes()
    _require(len(raw) <= 4096, "CGROUP_MEMBERSHIP_TOO_LARGE")
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeError:
        raise WorkerError("CGROUP_MEMBERSHIP_INVALID") from None
    _require(len(lines) == 1 and lines[0].startswith("0::/"), "CGROUP_V2_MEMBERSHIP_REQUIRED")
    location = lines[0][3:]
    parts = location.split("/")[1:]
    _require(parts and all(part and part not in (".", "..") for part in parts)
             and parts[-1] == "szl-ra4-" + run.nonce + ".service",
             "OWNED_CGROUP_MEMBERSHIP_REQUIRED")
    current = cgroup_root.joinpath(*parts)
    for name, expected in (("memory.max", run.payload["bounds"]["host_memory_bytes"]),
                           ("memory.swap.max", 0), ("pids.max", 128)):
        value = (current / name).read_bytes()
        _require(len(value) < 64 and value.strip() == str(expected).encode("ascii"),
                 "CGROUP_RESOURCE_BARRIER_MISMATCH")


def _target_inventory(model, qualified: list[str], suffixes: list[str], linear4bit) -> tuple[str, ...]:
    """No automatic target expansion into hybrid linear-attention projections."""
    _require(type(qualified) is list and qualified == sorted(set(qualified)) and qualified,
             "QUALIFIED_LORA_INVENTORY_REQUIRED")
    targets = []
    for name, module in model.named_modules():
        if name.startswith("model.language_model.layers.") and name.rsplit(".", 1)[-1] in suffixes:
            _require(isinstance(module, linear4bit), "LORA_TARGET_NOT_NF4_LINEAR")
            targets.append(name)
    _require(sorted(targets) == qualified, "ACTUAL_LORA_TARGET_INVENTORY_MISMATCH")
    return tuple(sorted(targets))


def _lora_parameters(model, targets: tuple[str, ...], rank: int) -> tuple[str, ...]:
    expected = []
    for target in targets:
        prefix = "base_model.model." + target
        module = model.get_submodule(prefix)
        _require(set(module.lora_A) == {"default"} and set(module.lora_B) == {"default"},
                 "LORA_ADAPTER_SET_MISMATCH")
        a, b = module.lora_A["default"].weight, module.lora_B["default"].weight
        _require(tuple(a.shape) == (rank, module.in_features)
                 and tuple(b.shape) == (module.out_features, rank), "LORA_PARAMETER_SHAPE_MISMATCH")
        expected.extend((prefix + ".lora_A.default.weight", prefix + ".lora_B.default.weight"))
    _require(set(model.peft_config) == {"default"}, "PEFT_CONFIGURATION_SET_MISMATCH")
    return tuple(sorted(expected))


def load_native_model(run, torch, transformers, peft, bitsandbytes):
    """Load only the bound local explicit class and exactly inventoried QLoRA."""
    quantization = run.recipe["quantization"]
    peft_recipe = run.recipe["peft"]
    quantization_config = transformers.BitsAndBytesConfig(
        load_in_4bit=quantization["load_in_4bit"],
        bnb_4bit_quant_type=quantization["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=quantization["bnb_4bit_use_double_quant"])
    loaded = transformers.Qwen3_5ForConditionalGeneration.from_pretrained(
        str(run.model_dir), local_files_only=True, trust_remote_code=False,
        use_safetensors=True, output_loading_info=True, device_map={"": 0},
        dtype=torch.bfloat16, quantization_config=quantization_config,
        attn_implementation="eager")
    _require(type(loaded) is tuple and len(loaded) == 2, "EXPLICIT_LOADER_INFO_REQUIRED")
    model, info = loaded
    _require(type(model) is transformers.Qwen3_5ForConditionalGeneration
             and model.config.model_type == "qwen3_5"
             and model.config.architectures == ["Qwen3_5ForConditionalGeneration"]
             and model.is_loaded_in_4bit is True, "EXPLICIT_NF4_MODEL_CLASS_MISMATCH")
    _require(type(info) is dict and all(type(info.get(name)) is list and info[name] == []
                 and run.qualification["loader_" + name] == []
                 for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")),
             "MODEL_LOADING_KEY_COVERAGE_MISMATCH")
    targets = _target_inventory(model, run.qualification["lora_target_names"],
                                peft_recipe["target_modules"], bitsandbytes.nn.Linear4bit)
    model.config.use_cache = False
    model = peft.prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False})
    _require(all(parameter.requires_grad is False for parameter in model.parameters()),
             "PREPARED_BASE_NOT_FROZEN")
    configuration = peft.LoraConfig(r=peft_recipe["r"], lora_alpha=peft_recipe["lora_alpha"],
                                    lora_dropout=peft_recipe["lora_dropout"], bias=peft_recipe["bias"],
                                    target_modules=list(targets),
                                    task_type=peft_recipe["task_type"],
                                    modules_to_save=None, use_rslora=False, use_dora=False)
    model = peft.get_peft_model(model, configuration, adapter_name="default")
    expected = _lora_parameters(model, targets, peft_recipe["r"])
    _require(tuple(sorted(name for name, parameter in model.named_parameters()
                          if parameter.requires_grad)) == expected,
             "ACTUAL_LORA_TRAINABLE_COVERAGE_MISMATCH")
    parameters = [parameter for name, parameter in model.named_parameters() if name in expected]
    recipe = run.recipe["optimizer"]
    _require(recipe["name"] == "AdamW", "EXPLICIT_OPTIMIZER_REQUIRED")
    optimizer = torch.optim.AdamW(parameters, lr=recipe["lr"], betas=tuple(recipe["betas"]),
                                  eps=recipe["eps"], weight_decay=recipe["weight_decay"],
                                  foreach=False, fused=False)
    verify_optimizer_coverage(model, optimizer, expected)
    return model, optimizer, expected


def verify_token_policy(run, binding: tokens.TokenBinding) -> None:
    """A tensor ledger alone does not bind the claimed masking/stopping policies."""
    actual = {"template_sha256": binding.template_sha256,
              "special_tokens_sha256": _sha(binding.special_tokens),
              "mask_policy_sha256": binding.masking_sha256,
              "eos_policy_sha256": binding.eos_sha256,
              "thinking_policy_sha256": binding.thinking_sha256}
    _require(all(run.payload["tokens"][name] == value for name, value in actual.items()),
             "SIGNED_FRESH_TOKEN_POLICY_MISMATCH")


def _jsonable(value):
    """Serialize only the safe primitive PEFT configuration, not Python objects."""
    if value is None or type(value) in (str, int, float, bool):
        return value
    # PEFT string-valued enums are safe strings, not executable serialization.
    if isinstance(value, str):
        return str(value)
    if type(value) is dict and all(type(key) is str for key in value):
        return {key: _jsonable(child) for key, child in value.items()}
    if type(value) in (list, tuple):
        return [_jsonable(child) for child in value]
    if type(value) is set and all(type(child) is str for child in value):
        return sorted(value)
    raise WorkerError("ADAPTER_CONFIGURATION_NOT_PRIMITIVE")


def _write_exclusive(path: Path, raw: bytes, limit: int) -> str:
    _require(type(raw) is bytes and 0 < len(raw) <= limit, "OUTPUT_BYTE_CAP_EXCEEDED")
    parent_info = os.lstat(path.parent)
    _require(stat.S_ISDIR(parent_info.st_mode) and not stat.S_ISLNK(parent_info.st_mode),
             "OUTPUT_PARENT_NOT_PLAIN_DIRECTORY")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "wb") as output:
        output.write(raw)
        output.flush()
        os.fsync(output.fileno())
    if sys.platform == "linux":
        directory_descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    return hashlib.sha256(raw).hexdigest()


def save_adapter(run, model, expected_names: tuple[str, ...], peft, torch) -> tuple[dict, int]:
    """Export only checked LoRA matrices as safetensors; never a full base model."""
    from safetensors.torch import save as save_safetensors

    state = peft.get_peft_model_state_dict(model, adapter_name="default", save_embedding_layers=False)
    expected = {name.replace(".default.weight", ".weight") for name in expected_names}
    _require(type(state) is dict and set(state) == expected, "ADAPTER_EXPORT_PARAMETER_COVERAGE_MISMATCH")
    cpu_state = {}
    for name, tensor in state.items():
        _require(bool(torch.isfinite(tensor).all().item()), "ADAPTER_EXPORT_NONFINITE")
        cpu_state[name] = tensor.detach().to(device="cpu").contiguous()
    # Safetensors emits data-only bytes. Neither pickle nor arbitrary objects are
    # accepted. Framework versions and these output methods are separately bound.
    tensor_bytes = save_safetensors(cpu_state, metadata={"format": "pt"})
    configuration = _jsonable(model.peft_config["default"].to_dict())
    configuration.update(base_model_name_or_path=tokens.BASE_ID,
                         revision=tokens.BASE_REVISION, inference_mode=True)
    config_bytes = json.dumps(configuration, sort_keys=True, separators=(",", ":"),
                              ensure_ascii=True, allow_nan=False).encode("ascii")
    artifact_cap = run.payload["bounds"]["artifact_cap_bytes"]
    size = len(tensor_bytes) + len(config_bytes)
    _require(size <= artifact_cap and size < run.payload["bounds"]["output_cap_bytes"],
             "ADAPTER_ARTIFACT_CAP_EXCEEDED")
    directory = run.output_dir / "adapter"
    directory.mkdir(mode=0o700)  # Existing or redirected output is never overwritten.
    artifacts = {}
    for name, raw in (("adapter_model.safetensors", tensor_bytes), ("adapter_config.json", config_bytes)):
        artifacts["adapter/" + name] = _write_exclusive(directory / name, raw, artifact_cap)
    return artifacts, size


def _run(request_path: Path, owner_policy_sha256: str) -> dict:
    # Local reviewed module only. The owner pin, false source flags, signature,
    # artifacts and runtime qualifications are checked before framework imports.
    import native_runtime as runtime
    import probe_base_config as config_probe
    import probe_base_tokenizer as tokenizer_probe
    import probe_train_token_tensors as reference

    run = runtime.load_bound_run(request_path, owner_policy_sha256=owner_policy_sha256, role="worker")
    verify_linux_containment(run)
    snapshot = admission.capture_train_only(run.train_path, run.manifest_path,
                                            nemo_source_root=run.nemo_source_root)
    verify_token_asset_closure(run)
    config_probe._read_config(run.model_dir / "config.json")
    verify_local_model_assets(run)
    verify_dependency_identity(run)
    runtime.revalidate_bound_run(run, role="worker")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == run.qualification["device_uuid"],
             "EXACT_PHYSICAL_GPU_BINDING_REQUIRED")
    import torch
    import transformers
    import peft
    import bitsandbytes

    _require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
             "ONE_QUALIFIED_CUDA_DEVICE_REQUIRED")
    actual_uuid = str(torch.cuda.get_device_properties(0).uuid)
    _require(actual_uuid == run.qualification["device_uuid"], "ACTUAL_GPU_UUID_MISMATCH")
    _require(torch.cuda.is_bf16_supported(), "QUALIFIED_BFLOAT16_DEVICE_REQUIRED")
    torch.manual_seed(run.seed)
    torch.cuda.manual_seed_all(run.seed)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(run.tokenizer_dir), local_files_only=True, trust_remote_code=False, use_fast=True)
    config_probe._qualify_tokenizer(tokenizer)
    binding = replace(reference._binding(tokenizer),
                       admitted_limit=run.payload["bounds"]["sequence_tokens"],
                       total_token_limit=run.payload["bounds"]["total_tokens"])
    verify_token_policy(run, binding)
    retained = tokens.retain_training_tokens(
        snapshot, binding, lambda messages: reference._encode(tokenizer, messages),
        lambda ids: reference._decode(tokenizer, ids))
    checked_rows(retained)
    _require(retained.report.ordered_tensors_sha256 == run.payload["tokens"]["tensor_ledger_sha256"],
             "SIGNED_TENSOR_LEDGER_MISMATCH")
    _require(_sha(tuple(range(admission.MAX_ROWS))) == run.payload["bounds"]["row_order_sha256"],
             "SIGNED_ROW_ORDER_MISMATCH")
    runtime.revalidate_bound_run(run, role="worker")
    model, optimizer, expected = load_native_model(run, torch, transformers, peft, bitsandbytes)

    def recheck(_seen=None):
        runtime.revalidate_bound_run(run, role="worker")
        verify_linux_containment(run)

    recheck()
    measured = one_update(model, optimizer, retained,
                          microbatch_rows=run.payload["bounds"]["microbatch_rows"],
                          pad_id=binding.pad_token_id,
                          max_grad_norm=run.recipe["optimizer"]["max_grad_norm"],
                          expected_names=expected, torch=torch, device="cuda:0",
                          before_update=recheck, after_microbatch=recheck)
    recheck()
    artifacts, artifact_size = save_adapter(run, model, expected, peft, torch)
    record = {
        "schema": RESULT_SCHEMA, "evidence_class": "REPORTED", "state": "SMOKE_COMPLETED_NOT_RELEASED",
        "admission_nonce": run.nonce, "base_id": tokens.BASE_ID, "base_revision": tokens.BASE_REVISION,
        "source_revision": run.payload["source"]["revision"],
        "train_sha256": retained.report.train_sha256, "manifest_sha256": retained.report.manifest_sha256,
        "token_ledger_sha256": measured.pop("ordered_tensors_sha256"), **measured,
        "seed": run.seed, "device_uuid": actual_uuid, "artifact_sha256": artifacts,
        "energy_joules": "UNAVAILABLE", "model_evaluation": "NOT_RUN", "independent_replay": "UNAVAILABLE",
        "training_eligible": False, "publication_eligible": False, "execution_authority": False,
    }
    raw = runtime.canonical(record)
    _require(artifact_size + len(raw) <= run.payload["bounds"]["output_cap_bytes"], "OUTPUT_BYTE_CAP_EXCEEDED")
    recheck()
    _write_exclusive(run.output_dir / "worker-result.json", raw, run.payload["bounds"]["output_cap_bytes"])
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--bound-run", type=Path, required=True)
    parser.add_argument("--owner-policy-sha256", required=True)
    parser.add_argument("--release-stdin", action="store_true", required=True)
    args = parser.parse_args(argv)
    try:
        read_supervisor_release(sys.stdin.buffer)
        record = _run(args.bound_run, args.owner_policy_sha256)
    except Exception:
        # Unknown framework/filesystem exceptions can contain tensor or path
        # content. Only this constant redacted rejection crosses stdout.
        print(json.dumps({"state": "BLOCKED", "code": "NATIVE_WORKER_DENIED_OR_FAILED",
                          "training_eligible": False, "publication_eligible": False,
                          "execution_authority": False}, sort_keys=True))
        return 1
    print(json.dumps({"state": record["state"], "worker_result": "worker-result.json",
                      "training_eligible": False, "publication_eligible": False,
                      "execution_authority": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
