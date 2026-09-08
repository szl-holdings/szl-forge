#!/usr/bin/env python3
"""Bounded, native bf16 ReceiptAgent continuation; local candidate only.

No downloads, cloud tracking, model publication, signing, or promotion. Source,
base, adapter and owned synthetic curriculum are pinned before a GPU load.
The deadline is cooperative; an external supervisor must bound a hung GPU call.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import re
import shutil
import subprocess
import time
import warnings

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASE_REPO = "unsloth/Qwen3.5-0.8B"
BASE_REVISION = "23c69c53358a07516b5827588b3fdb12ae78fd65"
ADAPTER_REPO = "SZLHOLDINGS/szl-receiptagent-qwen35-0.8b-v2"
ADAPTER_REVISION = "46f54373c6bf8f17a288b4c8799e9fbe4b82ecc1"
ADAPTER_WEIGHT_SHA256 = "885fc29fcb4cf55c280dc085fdb0a40f40d6b946fee400dd5e4ed3459fe6334f"
# Independent pinned Hub metadata + local Git/LFS byte verification.
# The adjacent manifest is untrusted, not its own authority. Changing these
# digests requires reviewed source and renewed provider verification.
TRUSTED_MANIFESTS = {
    (BASE_REPO, BASE_REVISION): "38838f6e620416c7d29b360de7d6b674cb85fd4e99ece5c9144f9e8034bbc0e6",
    (ADAPTER_REPO, ADAPTER_REVISION): "c0068e21b3a686dd9a9136520a9c5c4a01693e7e5616683b41d60155420cdaa6",
}
DATA = {
    "receiptagent/train.jsonl": ("775e25b526a96d1486e80aae048f731bdad02a0d94ea76144593e115802fa24f", 15, "train", 1),
    "receiptagent/train.refusals.jsonl": ("c5136b612951c209d1041839d1ac19a8fa378c31015d61dd7565994b4f3e2b47", 8, "train", 2),
    "frontier/qwen35-receiptagent-v2/train.status-refusals.jsonl": ("96a5518400293913620f0c9bb9aa9a4fa180202b3d5c1bd1ab66a38eb272067d", 3, "train", 2),
    "receiptagent/eval.jsonl": ("1a92bea9b163882ae7c7aa00be6c1fda3e9c98683f0214af985bfb184e27ff65", 5, "heldout", 1),
    "receiptagent/adversarial.jsonl": ("81d2a494ee41256c7909a54435c4e7971fea65de81fffbf3f2b8a008aab5a572", 6, "heldout", 1),
    "receiptagent/receiptagent.schema.json": ("1db07ef9e1581cc8cc471dd42e79a21b728c4e3bf1222fd0f06f2b7257bb288e", 0, "schema", 1),
}
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


class GateError(ValueError):
    """A stable non-sensitive admission/failure code."""


def require(condition, code):
    if not condition:
        raise GateError(code)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "DUPLICATE_JSON_KEY")
            result[key] = value
        return result

    def reject_constant(_):
        raise GateError("NONFINITE_JSON")

    require(len(raw) <= 8 * 1024 * 1024, "JSON_TOO_LARGE")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=reject_constant)


def artifact_path(root, name):
    require(isinstance(name, str) and name and "\\" not in name and ":" not in name, "UNSAFE_ARTIFACT_PATH")
    parts = PurePosixPath(name)
    require(not parts.is_absolute() and all(p not in (".", "..", "") for p in name.split("/")), "UNSAFE_ARTIFACT_PATH")
    path = root.joinpath(*parts.parts)
    require(not path.is_symlink() and not any(p.is_symlink() for p in path.parents if p != root.parent), "ARTIFACT_SYMLINK")
    require(path.resolve().is_relative_to(root.resolve()), "ARTIFACT_ESCAPE")
    return path


def verify_artifact(directory, manifest_path, repo, revision):
    directory = Path(directory).resolve(strict=True)
    raw = Path(manifest_path).read_bytes()
    manifest = strict_json(raw)
    require(manifest.get("schema") == "szl.local-artifact/v1", "ARTIFACT_SCHEMA")
    require(manifest.get("repo_id") == repo and manifest.get("revision") == revision, "ARTIFACT_IDENTITY")
    require(sha(raw) == TRUSTED_MANIFESTS.get((repo, revision)), "ARTIFACT_MANIFEST_NOT_AUTHENTICATED")
    require(manifest.get("verified_from") == "PINNED_HUB_METADATA_AND_LOCAL_BYTES", "ARTIFACT_PROVENANCE_REQUIRED")
    entries = manifest.get("files")
    require(isinstance(entries, list) and 1 <= len(entries) <= 256, "ARTIFACT_FILE_COUNT")
    declared = {}
    for item in entries:
        require(isinstance(item, dict), "ARTIFACT_ENTRY")
        name = item.get("path")
        path = artifact_path(directory, name)
        require(name not in declared and path.is_file(), "ARTIFACT_MISSING_OR_DUPLICATE")
        size, expected = item.get("bytes"), item.get("sha256")
        require(type(size) is int and 0 < size <= 12 * 1024**3, "ARTIFACT_SIZE")
        require(isinstance(expected, str) and HEX64.fullmatch(expected), "ARTIFACT_HASH_FORMAT")
        require(path.stat().st_size == size and file_sha(path) == expected, "ARTIFACT_BYTES_MISMATCH")
        declared[name] = {"bytes": size, "sha256": expected}
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file() and ".cache" not in p.relative_to(directory).parts}
    require(actual == set(declared), "ARTIFACT_UNDECLARED_FILES")
    require(any(name.endswith(".safetensors") for name in declared), "SAFETENSORS_REQUIRED")
    require(not any(name.endswith((".bin", ".pt", ".pth", ".pkl", ".pickle", ".joblib", ".py")) for name in declared), "EXECUTABLE_SERIALIZATION_NOT_ADMITTED")
    return {"repo_id": repo, "revision": revision, "manifest_sha256": sha(raw), "files": declared}


def validate_base(config):
    require(config.get("architectures") == ["Qwen3_5ForConditionalGeneration"] and config.get("model_type") == "qwen3_5", "BASE_ARCHITECTURE")
    require(not config.get("quantization_config"), "QUANTIZED_BASE_NOT_ADMITTED")
    text = config.get("text_config", {})
    require(text.get("hidden_size") == 1024 and text.get("num_hidden_layers") == 24 and text.get("vocab_size") == 248320, "BASE_DIMENSIONS")


def validate_adapter_config(config, supported_fields=None):
    require(config.get("base_model_name_or_path") == BASE_REPO, "ADAPTER_BASE_MISMATCH")
    require(config.get("peft_type") == "LORA" and config.get("task_type") == "CAUSAL_LM", "ADAPTER_TYPE")
    require(config.get("r") == 16 and config.get("lora_alpha") == 32 and config.get("lora_dropout") == 0, "ADAPTER_HYPERPARAMETERS")
    require(config.get("auto_mapping", {}).get("base_model_class") == "Qwen3_5ForConditionalGeneration", "ADAPTER_ARCHITECTURE")
    require(config.get("revision") in (None, BASE_REVISION), "ADAPTER_BASE_REVISION")
    require(isinstance(config.get("target_modules"), str) and config["target_modules"], "ADAPTER_TARGETS")
    require(not config.get("modules_to_save") and not config.get("use_dora") and not config.get("use_qalora"), "ADAPTER_UNSUPPORTED_VARIANT")
    if supported_fields is not None:
        require(set(config).issubset(set(supported_fields)), "PEFT_UNKNOWN_CONFIG_FIELDS")


def committed(root, revision, path):
    require(isinstance(revision, str) and HEX40.fullmatch(revision), "SOURCE_REVISION_REQUIRED")
    require(path in DATA, "CURRICULUM_PATH_NOT_ADMITTED")
    return subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=root, timeout=20)


def validate_row(row):
    require(isinstance(row, dict) and set(row) == {"messages"}, "CURRICULUM_ROW_SCHEMA")
    messages = row["messages"]
    require(isinstance(messages, list) and [m.get("role") for m in messages] == ["system", "user", "assistant"], "CURRICULUM_ROLES")
    require(all(set(m) == {"role", "content"} and isinstance(m["content"], str) and 0 < len(m["content"]) <= 20000 for m in messages), "CURRICULUM_TEXT_ONLY")


def load_curriculum(root, revision, reader=committed):
    train, heldout, hashes = [], [], {}
    for path, (expected, count, split, repeats) in DATA.items():
        raw = reader(root, revision, path)
        require(sha(raw) == expected, "CURRICULUM_HASH_MISMATCH")
        hashes[path] = expected
        if split == "schema":
            strict_json(raw)
            continue
        rows = [strict_json(line) for line in raw.splitlines() if line.strip()]
        require(len(rows) == count, "CURRICULUM_ROW_COUNT")
        for row in rows:
            validate_row(row)
        (train if split == "train" else heldout).extend(rows * repeats)
    train_prompts = {sha(canonical(r["messages"][:-1])) for r in train}
    heldout_prompts = {sha(canonical(r["messages"][:-1])) for r in heldout}
    require(not train_prompts.intersection(heldout_prompts), "TRAIN_HELDOUT_OVERLAP")
    require(len(train) == 37 and len(train_prompts) == 26 and len(heldout) == 11, "CURRICULUM_TOTALS")
    return train, heldout, hashes


def label_mask(full_ids, prompt_ids, mode, max_length):
    require(isinstance(full_ids, list) and all(type(v) is int and v >= 0 for v in full_ids), "TOKEN_IDS_INVALID")
    require(2 <= len(full_ids) <= max_length, "SEQUENCE_LENGTH_NOT_ADMITTED")
    require(mode in ("assistant", "full-sequence"), "LOSS_MODE")
    if mode == "full-sequence":
        return list(full_ids)
    require(prompt_ids and full_ids[:len(prompt_ids)] == prompt_ids, "ASSISTANT_MASK_PREFIX_MISMATCH")
    require(len(prompt_ids) < len(full_ids), "NO_ASSISTANT_TOKENS")
    return [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]


def encode_rows(processor, rows, mode, max_length):
    encoded = []
    for row in rows:
        messages = row["messages"]
        full = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
        prompt = processor.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = processor.tokenizer(full, add_special_tokens=False)["input_ids"]
        prefix = processor.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        labels = label_mask(ids, prefix, mode, max_length)
        encoded.append({"input_ids": ids, "labels": labels, "supervised_tokens": sum(v != -100 for v in labels[1:])})
    return encoded


def finite_number(value, low, high, code):
    require(type(value) in (int, float) and math.isfinite(value) and low <= value <= high, code)


def validate_options(args):
    require(type(args.steps) is int and args.steps in (1, 16, 32), "STEPS_MUST_BE_1_16_OR_32")
    finite_number(args.learning_rate, 1e-6, 1e-4, "LEARNING_RATE_BOUNDS")
    require(type(args.max_length) is int and 64 <= args.max_length <= 2048, "MAX_LENGTH_BOUNDS")
    finite_number(args.max_runtime_seconds, 10, 7200, "RUNTIME_BOUNDS")
    finite_number(args.max_temperature_c, 40, 85, "TEMPERATURE_BOUNDS")
    finite_number(args.min_free_gib, 1, 8, "MEMORY_BOUNDS")
    require(HEX40.fullmatch(args.source_commit or ""), "SOURCE_REVISION_REQUIRED")


def configure_local_environment(output):
    remote_keys = ("TRACKIO_SPACE_ID", "TRACKIO_SERVER_URL", "TRACKIO_DATASET_ID", "TRACKIO_BUCKET_ID", "TRACKIO_WEBHOOK_URL", "PERSISTANT_STORAGE_ENABLED", "SPACE_ID")
    require(not any(os.environ.get(key) for key in remote_keys), "REMOTE_TRACKING_ENVIRONMENT_NOT_ADMITTED")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", WANDB_DISABLED="true", CUBLAS_WORKSPACE_CONFIG=":4096:8", TOKENIZERS_PARALLELISM="false", TRACKIO_DIR=str(output / "trackio"))


def runtime_guard(torch, args, started, *, before_load=False):
    require(time.monotonic() - started < args.max_runtime_seconds, "COOPERATIVE_DEADLINE_EXCEEDED")
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "BF16_CUDA_REQUIRED")
    measured = subprocess.check_output(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"], text=True, timeout=10)
    temperature = int(measured.strip().splitlines()[0])
    require(temperature <= args.max_temperature_c, "THERMAL_LIMIT")
    free, total = torch.cuda.mem_get_info()
    minimum = (args.min_free_gib if before_load else 0.15) * 1024**3
    cached_reclaimed = False
    if not before_load and free < minimum:
        # Only unused allocator cache can be returned. Do not change the floor,
        # discard live tensors, or infer availability from reserved-byte counts.
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        cached_reclaimed = True
    require(free >= minimum, "GPU_MEMORY_LIMIT")
    require(shutil.disk_usage(args.output).free >= 1024**3, "DISK_LIMIT")
    return {"temperature_c": temperature, "free_bytes": free, "total_bytes": total, "reserved_bytes": torch.cuda.memory_reserved(), "unused_cache_reclaim_attempted": cached_reclaimed}


def compute_loss(model, item, torch):
    inputs = torch.tensor([item["input_ids"]], device="cuda", dtype=torch.long)
    labels = torch.tensor([item["labels"]], device="cuda", dtype=torch.long)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(input_ids=inputs, attention_mask=torch.ones_like(inputs), labels=labels, use_cache=False).loss
    require(bool(torch.isfinite(loss).item()), "NONFINITE_LOSS")
    return loss


def evaluate_loss(model, items, torch, args, started):
    model.eval()
    total_loss, tokens = 0.0, 0
    with torch.inference_mode():
        for item in items:
            runtime_guard(torch, args, started)
            value = float(compute_loss(model, item, torch).item())
            count = item["supervised_tokens"]
            total_loss += value * count
            tokens += count
    require(tokens > 0 and math.isfinite(total_loss), "HELDOUT_LOSS_INVALID")
    return {"token_weighted_cross_entropy": total_loss / tokens, "rows": len(items), "supervised_tokens": tokens, "is_generation_benchmark": False}


def train(args, report):
    configure_local_environment(args.output)
    base = verify_artifact(args.base_dir, args.base_manifest, BASE_REPO, BASE_REVISION)
    adapter = verify_artifact(args.adapter_dir, args.adapter_manifest, ADAPTER_REPO, ADAPTER_REVISION)
    require(adapter["files"].get("adapter_model.safetensors", {}).get("sha256") == ADAPTER_WEIGHT_SHA256, "ADAPTER_WEIGHT_IDENTITY")
    validate_base(strict_json((args.base_dir / "config.json").read_bytes()))
    config = strict_json((args.adapter_dir / "adapter_config.json").read_bytes())
    validate_adapter_config(config)
    train_rows, heldout_rows, hashes = load_curriculum(ROOT, args.source_commit)
    report.update(base=base, adapter_parent=adapter, curriculum_hashes=hashes, unique_training_rows=26, training_rows_oversampled=37, heldout_rows=11, exact_prompt_overlap=0)
    print(json.dumps({"phase": "artifacts_and_curriculum_verified"}), flush=True)

    import torch
    from peft import LoraConfig, PeftModel
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    validate_adapter_config(config, {f.name for f in fields(LoraConfig)})
    declared_version = str(config.get("peft_version", ""))
    installed = importlib.metadata.version("peft")
    version = lambda value: tuple(int(v) for v in value.split(".")[:2])
    require(version(installed) >= version(declared_version), "PEFT_VERSION_BEHIND_ARTIFACT")
    # Constructor is intentional: no forward-compatibility helper may drop fields.
    LoraConfig(**config)
    report["versions"] = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft")}
    report["gpu"] = {"name": torch.cuda.get_device_name(0), "initial": runtime_guard(torch, args, report["_started"], before_load=True)}
    print(json.dumps({"phase": "native_runtime_verified", "versions": report["versions"]}), flush=True)
    random.seed(11)
    torch.manual_seed(11)
    torch.cuda.manual_seed_all(11)
    torch.use_deterministic_algorithms(True)
    processor = AutoProcessor.from_pretrained(str(args.base_dir), local_files_only=True, trust_remote_code=False)
    train_items = encode_rows(processor, train_rows, args.loss_mode, args.max_length)
    heldout_items = encode_rows(processor, heldout_rows, args.loss_mode, args.max_length)
    report["tokenization"] = {"max_observed_length": max(len(v["input_ids"]) for v in train_items + heldout_items), "loss_mode": args.loss_mode, "prompt_padding_masked": args.loss_mode == "assistant", "thinking_enabled": False, "truncation": False}
    tracker = None
    if not args.no_trackio:
        import trackio
        tracker = trackio
        tracker.init(project="szl-native-local-continuation", name=args.output.name, space_id=None, config={"steps": args.steps, "learning_rate": args.learning_rate, "base_revision": BASE_REVISION, "adapter_revision": ADAPTER_REVISION, "loss_mode": args.loss_mode, "provider_cost_usd": 0})
    report["tracking"] = "LOCAL_TRACKIO" if tracker else "EXPLICITLY_DISABLED"
    try:
        model, loading = Qwen3_5ForConditionalGeneration.from_pretrained(str(args.base_dir), local_files_only=True, trust_remote_code=False, use_safetensors=True, dtype=torch.bfloat16, attn_implementation="sdpa", output_loading_info=True)
        require(not any(loading.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")), "BASE_LOADING_KEYS_MISMATCH")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = PeftModel.from_pretrained(model, str(args.adapter_dir), is_trainable=True, local_files_only=True)
        require(not any(re.search(r"missing|unexpected|ignored|mismatch", str(w.message), re.I) for w in caught), "ADAPTER_LOADING_WARNING")
        model.to("cuda")
        print(json.dumps({"phase": "parent_adapter_loaded"}), flush=True)
        model.config.use_cache = False
        if hasattr(model.config, "text_config"):
            model.config.text_config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        require(parameters and all("lora_" in name and "visual" not in name and "vision" not in name for name, _ in parameters), "TRAINABLE_PARAMETERS_NOT_LORA_ONLY")
        report["trainable_parameters"] = sum(p.numel() for _, p in parameters)
        original = {name: p.detach().cpu().clone() for name, p in parameters}
        runtime_guard(torch, args, report["_started"])
        if args.include_base_loss:
            with model.disable_adapter():
                report["base_heldout_loss"] = evaluate_loss(model, heldout_items, torch, args, report["_started"])
        report["baseline_heldout_loss"] = evaluate_loss(model, heldout_items, torch, args, report["_started"])
        print(json.dumps({"phase": "baseline_measured", **report["baseline_heldout_loss"]}), flush=True)
        optimizer = torch.optim.AdamW([p for _, p in parameters], lr=args.learning_rate, weight_decay=0.01)
        schedule = list(range(len(train_items)))
        random.Random(11).shuffle(schedule)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for step in range(args.steps):
            guard = runtime_guard(torch, args, report["_started"])
            losses = []
            for accumulation in range(2):
                item = train_items[schedule[(step * 2 + accumulation) % len(schedule)]]
                loss = compute_loss(model, item, torch)
                losses.append(float(loss.detach().item()))
                (loss / 2).backward()
            require(all(p.grad is None or bool(torch.isfinite(p.grad).all().item()) for _, p in parameters), "NONFINITE_GRADIENT")
            grad_norm = torch.nn.utils.clip_grad_norm_([p for _, p in parameters], 1.0, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            metric = {"step": step + 1, "training_loss": sum(losses) / len(losses), "gradient_norm": float(grad_norm), "temperature_c": guard["temperature_c"]}
            require(all(math.isfinite(v) for v in metric.values()), "NONFINITE_METRIC")
            report["steps"].append(metric)
            if tracker:
                tracker.log(metric)
            print(json.dumps(metric, allow_nan=False), flush=True)
        require(any(not torch.equal(original[name], p.detach().cpu()) for name, p in parameters), "ADAPTER_UNCHANGED_AFTER_TRAINING")
        report["post_heldout_loss"] = evaluate_loss(model, heldout_items, torch, args, report["_started"])
        candidate_dir = args.output / "adapter"
        require(not candidate_dir.exists(), "CANDIDATE_OUTPUT_EXISTS")
        for adapter_config in model.peft_config.values():
            adapter_config.base_model_name_or_path = BASE_REPO
            adapter_config.revision = BASE_REVISION
        model.save_pretrained(candidate_dir, safe_serialization=True)
        processor.save_pretrained(candidate_dir)
        report["candidate_files"] = [{"path": p.relative_to(candidate_dir).as_posix(), "bytes": p.stat().st_size, "sha256": file_sha(p)} for p in sorted(candidate_dir.rglob("*")) if p.is_file()]
        require(any(item["path"] == "adapter_model.safetensors" for item in report["candidate_files"]), "CANDIDATE_WEIGHTS_MISSING")
        verify_artifact(args.base_dir, args.base_manifest, BASE_REPO, BASE_REVISION)
        verify_artifact(args.adapter_dir, args.adapter_manifest, ADAPTER_REPO, ADAPTER_REVISION)
        report["parent_artifacts_unchanged"] = True
        report["state"] = "MEASURED_LOCAL_CONTINUATION_COMPLETED"
        report["candidate_reloaded_and_generation_tested"] = False
        report["heldout_loss_delta"] = report["post_heldout_loss"]["token_weighted_cross_entropy"] - report["baseline_heldout_loss"]["token_weighted_cross_entropy"]
        report["gpu"]["final"] = runtime_guard(torch, args, report["_started"])
    finally:
        if tracker:
            tracker.finish()


def run(args, train_function=train):
    validate_options(args)
    output = args.output.resolve()
    require(output.is_relative_to((HERE / "results").resolve()) and output != (HERE / "results").resolve(), "OUTPUT_MUST_BE_NEW_RESULTS_CHILD")
    require(not output.exists() and not any(p.is_symlink() for p in output.parents), "OUTPUT_EXISTS_OR_SYMLINK")
    output.mkdir(parents=True, exist_ok=False)
    args.output = output
    report = {"schema": "szl.native-bf16-continuation/v1", "state": "UNAVAILABLE", "started_at": datetime.now(timezone.utc).isoformat(), "_started": time.monotonic(), "source_revision": args.source_commit, "runner_sha256": file_sha(Path(__file__)), "candidate_id": output.name, "provider_cost_usd": 0, "electricity_cost": "NOT_MEASURED", "publication_eligible": False, "autonomy_eligible": False, "steps": [], "recipe": {"optimizer_steps": args.steps, "gradient_accumulation": 2, "learning_rate": args.learning_rate, "seed": 11, "bf16": True, "quantized": False, "loss_mode": args.loss_mode, "max_length": args.max_length}, "claim_boundary": "Local continuation and cross-entropy only. No generation acceptance, broad benchmark, safety certification, clinical use, publication or deployment. Lower loss does not by itself prove improvement."}
    lock = HERE / ".native-training.lock"
    owned_lock = False
    try:
        with lock.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps({"pid": os.getpid(), "candidate": output.name}))
        owned_lock = True
        train_function(args, report)
        code = 0 if report["state"] == "MEASURED_LOCAL_CONTINUATION_COMPLETED" else 1
    except BaseException as error:
        report["state"] = "FAILED_CLOSED"
        report["error_type"] = type(error).__name__
        report["error_code"] = str(error) if isinstance(error, GateError) else "LOCAL_EXECUTION_FAILED_NO_PROMOTION"
        report["candidate_may_be_partial"] = (output / "adapter").exists()
        code = 1
    finally:
        if owned_lock:
            lock.unlink()
        report["elapsed_seconds"] = round(time.monotonic() - report.pop("_started"), 6)
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        report["report_sha256"] = sha(canonical(report))
        with (output / "training-report.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
    return code, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--adapter-manifest", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--loss-mode", choices=("assistant", "full-sequence"), default="assistant", help="full-sequence explicitly includes prompt loss; no automatic fallback")
    parser.add_argument("--max-runtime-seconds", type=float, default=1800)
    parser.add_argument("--max-temperature-c", type=int, default=80)
    parser.add_argument("--min-free-gib", type=float, default=4)
    parser.add_argument("--include-base-loss", action="store_true")
    parser.add_argument("--no-trackio", action="store_true", help="explicitly disable default local-only tracking")
    args = parser.parse_args()
    try:
        code, report = run(args)
    except Exception as error:
        report = {"state": "FAILED_CLOSED", "error_type": type(error).__name__, "error_code": str(error) if isinstance(error, GateError) else "OUTPUT_OR_ARGUMENT_ADMISSION_FAILED", "publication_eligible": False, "autonomy_eligible": False}
        code = 1
    print(json.dumps({key: report[key] for key in ("state", "publication_eligible", "autonomy_eligible", "error_code") if key in report}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
