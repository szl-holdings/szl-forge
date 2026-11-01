#!/usr/bin/env python3
"""Local, bounded, zero-update Nemotron development-loss comparison.

check: schema only; verify: local bytes only; run: explicit local inference.
This is an evaluation candidate, not a qualified Linux launcher or a promotion gate.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import signal
import socket
import subprocess
import sys
import time

BASE = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
REVISION = "dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f"
SCHEMA = "szl.yachay-development-eval/v1"
SHA = re.compile(r"^[0-9a-f]{64}$")
ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,100}$")


class Blocked(ValueError):
    pass


def require(ok, message):
    if not ok:
        raise Blocked(message)


def pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def parse(raw):
    def invalid(value):
        raise Blocked("nonfinite JSON: " + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def read_json(path):
    require(Path(path).stat().st_size <= 1024 * 1024, "JSON file exceeds 1 MiB")
    return parse(Path(path).read_bytes())


def unknowns(value, prefix="plan"):
    if value is None or value == "UNKNOWN":
        return [prefix]
    if isinstance(value, dict):
        return [p for k, v in value.items() for p in unknowns(v, prefix + "." + k)]
    if isinstance(value, list):
        return [p for i, v in enumerate(value) for p in unknowns(v, f"{prefix}[{i}]")]
    return []


def integer(value, low, high, name):
    require(type(value) is int and low <= value <= high, "invalid " + name)


def file_spec(spec):
    require(isinstance(spec, dict), "file binding must be an object")
    require(isinstance(spec.get("path"), str) and Path(spec["path"]).is_absolute(), "file path must be absolute")
    require(bool(SHA.fullmatch(str(spec.get("sha256")))), "file SHA-256 missing")


def snapshot_spec(spec):
    require(isinstance(spec, dict), "snapshot binding must be an object")
    require(isinstance(spec.get("root"), str) and Path(spec["root"]).is_absolute(), "snapshot root must be absolute")
    files = spec.get("files")
    require(isinstance(files, dict) and 0 < len(files) <= 512, "complete snapshot file/hash map required")
    for name, sha in files.items():
        p = PurePosixPath(name)
        require(not p.is_absolute() and ".." not in p.parts and "\\" not in name and str(p) == name, "unsafe snapshot relative path")
        require(bool(SHA.fullmatch(str(sha))), "invalid snapshot hash: " + name)


def validate(plan):
    require(isinstance(plan, dict), "plan must be an object")
    missing = unknowns(plan)
    require(not missing, "unbound fields: " + ", ".join(missing))
    require(plan.get("schema") == SCHEMA, "unsupported schema")
    require(bool(SHA.fullmatch(str(plan.get("evaluator_sha256")))), "evaluator SHA-256 required")
    require(bool(ID.fullmatch(str(plan.get("run_id")))), "invalid run ID")
    require(plan.get("optimizer_updates") == 0 and type(plan.get("optimizer_updates")) is int, "optimizer updates must be zero")
    require(plan.get("production_eligible") is False, "production eligibility must stay false")
    require(plan.get("base", {}).get("id") == BASE, "wrong base model")
    require(plan["base"].get("revision") == REVISION, "wrong historical base revision")
    snapshot_spec(plan["base"])
    require("config.json" in plan["base"]["files"], "base config missing")
    require(any(n.endswith(".safetensors") for n in plan["base"]["files"]), "base safetensors missing")
    snapshot_spec(plan["tokenizer"])
    require("tokenizer_config.json" in plan["tokenizer"]["files"], "tokenizer config missing")
    require(plan["tokenizer"].get("revision") == REVISION, "tokenizer revision mismatch")
    require(plan.get("data_role") == "development_diagnostic", "this protocol cannot label validation as held-out")
    file_spec(plan["data"])
    file_spec(plan["loader_review"])
    adapters = plan.get("adapters")
    require(isinstance(adapters, list) and 1 <= len(adapters) <= 3, "need 1 to 3 pinned adapters")
    identities = {"base"}
    for adapter in adapters:
        snapshot_spec(adapter)
        name = adapter.get("id")
        require(isinstance(name, str) and bool(ID.fullmatch(name)) and name not in identities, "invalid/duplicate candidate ID")
        identities.add(name)
        require(isinstance(adapter.get("campaign_id"), str) and bool(ID.fullmatch(adapter["campaign_id"])), "campaign identity required")
        integer(adapter.get("step"), 0, 10**9, "step")
        require(set(adapter["files"]) == {"adapter_config.json", "adapter_model.safetensors"}, "adapter snapshot must contain only the bound config and safetensors")
    runtime = plan["runtime"]
    require(runtime.get("python_minor") == "3.12", "this candidate targets Python 3.12")
    require(set(runtime.get("packages", {})) == {"torch", "transformers", "peft", "safetensors", "accelerate"}, "all five runtime packages must be pinned")
    for version in runtime["packages"].values():
        require(isinstance(version, str) and bool(re.fullmatch(r"[0-9][A-Za-z0-9.+_-]{0,60}", version)), "exact package version required")
    require(bool(runtime.get("hostname")) and bool(runtime.get("boot_id")), "live host and boot bindings required")
    require(str(runtime.get("gpu_uuid", "")).startswith("GPU-"), "GPU UUID required")
    require(runtime.get("compute_capability") == [8, 6], "A6000 compute capability must be 8.6")
    limits = plan["limits"]
    integer(limits.get("wall_seconds"), 1, 900, "wall_seconds")
    integer(limits.get("rows"), 1, 256, "rows")
    integer(limits.get("sequence_tokens"), 2, 2048, "sequence_tokens")
    integer(limits.get("cpu_cores"), 1, 4, "cpu_cores")
    integer(limits.get("ram_bytes"), 1024**3, 64 * 1024**3, "ram_bytes")
    fraction = limits.get("gpu_allocator_fraction")
    require(type(fraction) in (int, float) and math.isfinite(fraction) and 0 < fraction <= 0.5, "GPU allocator fraction must be (0, 0.5]")
    integer(plan.get("seed"), 0, 2**32 - 1, "seed")
    return plan


def verify_file(spec):
    path = Path(spec["path"])
    require(path.is_file(), "missing file: " + str(path))
    require(digest(path) == spec["sha256"], "file hash mismatch: " + str(path))


def verify_snapshot(spec):
    root = Path(spec["root"])
    require(root.is_dir() and not root.is_symlink(), "snapshot root absent or symlink")
    actual = set()
    for directory, dirs, names in os.walk(root, followlinks=False):
        for name in dirs:
            require(not (Path(directory) / name).is_symlink(), "snapshot directory symlink rejected")
        for name in names:
            path = Path(directory) / name
            require(path.is_file() and not path.is_symlink(), "snapshot must be materialized regular files")
            actual.add(path.relative_to(root).as_posix())
            require(len(actual) <= 512, "too many snapshot files")
    require(actual == set(spec["files"]), "snapshot inventory differs from complete binding")
    for name, sha in spec["files"].items():
        require(digest(root / name) == sha, "snapshot hash mismatch: " + name)


def load_rows(plan):
    path = Path(plan["data"]["path"])
    require(path.stat().st_size <= 16 * 1024 * 1024, "evaluation data exceeds 16 MiB")
    rows, ids = [], set()
    for line in path.read_text(encoding="utf-8").splitlines():
        require(bool(line.strip()), "blank JSONL row")
        row = parse(line)
        require(isinstance(row, dict) and set(row) == {"id", "messages"}, "rows require exactly id and messages; normalize under a separate pinned transform")
        require(isinstance(row["id"], str) and bool(ID.fullmatch(row["id"])) and row["id"] not in ids, "invalid/duplicate row ID")
        ids.add(row["id"])
        messages = row["messages"]
        require(isinstance(messages, list) and 2 <= len(messages) <= 32, "invalid message count")
        for msg in messages:
            require(isinstance(msg, dict) and set(msg) == {"role", "content"}, "invalid message schema")
            require(msg["role"] in {"system", "user", "assistant"} and isinstance(msg["content"], str) and bool(msg["content"].strip()), "invalid message content")
        require(messages[-1]["role"] == "assistant" and messages[-2]["role"] == "user", "last turn must be user then assistant")
        rows.append(row)
        require(len(rows) <= plan["limits"]["rows"], "data exceeds frozen row count")
    require(len(rows) == plan["limits"]["rows"], "data row count mismatch")
    return rows


def verify(plan):
    validate(plan)
    for spec in [plan["base"], plan["tokenizer"], *plan["adapters"]]:
        verify_snapshot(spec)
    for spec in [plan["data"], plan["loader_review"]]:
        verify_file(spec)
    config = read_json(Path(plan["base"]["root"]) / "config.json")
    require(config.get("model_type") == "nemotron_h", "base architecture mismatch")
    for adapter in plan["adapters"]:
        config = read_json(Path(adapter["root"]) / "adapter_config.json")
        require(config.get("peft_type") == "LORA" and config.get("task_type") == "CAUSAL_LM", "only causal LoRA supported")
        require(config.get("base_model_name_or_path") == BASE, "adapter base binding mismatch")
        require(config.get("revision") in (None, REVISION), "adapter revision mismatch")
    review = read_json(plan["loader_review"]["path"])
    binding = hashlib.sha256(canonical({"base": plan["base"]["files"], "tokenizer": plan["tokenizer"]["files"], "packages": plan["runtime"]["packages"]})).hexdigest()
    require(review.get("binding_sha256") == binding and review.get("local_custom_code_reviewed") is True, "loader review does not bind these files and packages")
    require(isinstance(review.get("reviewer"), str) and bool(review["reviewer"].strip()), "loader reviewer required")
    return load_rows(plan)


def encode(tokenizer, row, maximum):
    messages = row["messages"]
    prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    require(isinstance(prefix, list) and isinstance(full, list) and bool(prefix), "tokenizer must return flat token lists")
    require(all(type(x) is int and x >= 0 for x in full + prefix), "invalid token IDs")
    require(len(prefix) < len(full) <= maximum, "empty answer or sequence exceeds bound; no truncation")
    require(full[:len(prefix)] == prefix, "chat template prefix mismatch; masking requires separate qualification")
    return full, [-100] * len(prefix) + full[len(prefix):]


def aggregate(records):
    require(bool(records), "no measured rows")
    total, count = 0.0, 0
    for item in records:
        integer(item["target_tokens"], 1, 2048, "target token count")
        nll = item["nll_sum"]
        require(type(nll) in (float, int) and math.isfinite(nll) and nll >= 0, "nonfinite/negative loss")
        total += nll
        count += item["target_tokens"]
    mean = total / count
    return {"rows": len(records), "target_tokens": count, "nll_sum": total, "token_weighted_nll": mean,
            "perplexity": math.exp(mean) if mean < 700 else None}


def runtime_guard(plan):
    require(sys.platform == "linux", "execution requires Linux")
    runtime, limits = plan["runtime"], plan["limits"]
    require(f"{sys.version_info.major}.{sys.version_info.minor}" == runtime["python_minor"], "Python mismatch")
    require(socket.gethostname() == runtime["hostname"], "hostname mismatch")
    require(Path("/proc/sys/kernel/random/boot_id").read_text().strip() == runtime["boot_id"], "boot mismatch; obtain a fresh plan")
    for package, expected in runtime["packages"].items():
        require(importlib.metadata.version(package) == expected, "package version mismatch: " + package)
    # Enforce an existing cgroup; never create, move or reconfigure a process.
    entries = Path("/proc/self/cgroup").read_text().splitlines()
    require(len(entries) == 1 and entries[0].startswith("0::/"), "cgroup v2 required")
    rel = entries[0][3:]
    require(".." not in PurePosixPath(rel).parts, "invalid cgroup path")
    cg = Path("/sys/fs/cgroup") / rel.lstrip("/")
    mem = (cg / "memory.max").read_text().strip()
    require(mem.isdigit() and 0 < int(mem) <= limits["ram_bytes"], "qualified cgroup memory limit required")
    quota, period = (cg / "cpu.max").read_text().split()
    require(quota.isdigit() and period.isdigit() and 0 < int(quota) <= int(period) * limits["cpu_cores"], "qualified cgroup CPU quota required")
    pids = (cg / "pids.max").read_text().strip()
    require(pids.isdigit() and 1 <= int(pids) <= 128, "bounded cgroup process count required")
    paths = [Path(p["root"]) for p in [plan["base"], plan["tokenizer"], *plan["adapters"]]]
    paths += [Path(plan[key]["path"]) for key in ("data", "loader_review")]
    for path in paths:
        require(bool(os.statvfs(path).f_flag & os.ST_RDONLY), "artifacts and data must be mounted read-only")
    def smi(arguments):
        return subprocess.run(["nvidia-smi", *arguments], check=True, capture_output=True, text=True, timeout=10).stdout.strip()
    devices = smi(["--query-gpu=uuid,name", "--format=csv,noheader"]).splitlines()
    require(any(line.split(",")[0].strip() == runtime["gpu_uuid"] and "RTX A6000" in line for line in devices), "expected A6000 UUID absent")
    processes = smi(["-i", runtime["gpu_uuid"], "--query-compute-apps=pid", "--format=csv,noheader,nounits"])
    require(not processes, "selected GPU is occupied; coordinate a fresh exclusive window")


def append_record(path, record):
    with Path(path).open("ab", buffering=0) as target:
        target.write(canonical(record) + b"\n")
        os.fsync(target.fileno())


def worker(plan, output):
    # The supervising parent bounds all imports, hashing and inference together.
    require(os.environ.get("YACHAY_EVALUATOR_PARENT") == str(os.getppid()), "worker requires its supervising parent")
    require(digest(__file__) == plan["evaluator_sha256"], "evaluator source mismatch")
    runtime_guard(plan)
    import gc
    import random
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    rows = verify(plan)
    require(torch.cuda.device_count() == 1, "only the selected GPU may be visible")
    require(list(torch.cuda.get_device_capability(0)) == plan["runtime"]["compute_capability"], "CUDA capability mismatch")
    torch.cuda.set_per_process_memory_fraction(plan["limits"]["gpu_allocator_fraction"], 0)
    torch.set_num_threads(plan["limits"]["cpu_cores"])
    tokenizer = AutoTokenizer.from_pretrained(plan["tokenizer"]["root"], local_files_only=True, trust_remote_code=True)
    encoded = [encode(tokenizer, row, plan["limits"]["sequence_tokens"]) for row in rows]
    for candidate in [{"id": "base"}, *plan["adapters"]]:
        name = candidate["id"]
        append_record(output / "events.jsonl", {"candidate": name, "state": "STARTED"})
        random.seed(plan["seed"])
        torch.manual_seed(plan["seed"])
        torch.cuda.manual_seed_all(plan["seed"])
        # Reload for every candidate: never carry adapter mutations into baseline.
        model = AutoModelForCausalLM.from_pretrained(plan["base"]["root"], local_files_only=True,
            trust_remote_code=True, use_safetensors=True, torch_dtype=torch.bfloat16, device_map={"": 0})
        if name != "base":
            model = PeftModel.from_pretrained(model, candidate["root"], is_trainable=False, local_files_only=True)
        model.requires_grad_(False)
        model.eval()
        records = []
        with torch.inference_mode():
            for row, (ids, labels) in zip(rows, encoded):
                inputs = torch.tensor([ids], device="cuda:0", dtype=torch.long)
                targets = torch.tensor(labels[1:], device="cuda:0", dtype=torch.long)
                logits = model(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False).logits
                nll = torch.nn.functional.cross_entropy(logits[0, :-1].float(), targets, ignore_index=-100, reduction="sum")
                record = {"candidate": name, "row_id": row["id"], "target_tokens": sum(x != -100 for x in labels[1:]), "nll_sum": float(nll.item())}
                aggregate([record])
                append_record(output / "raw.jsonl", record)
                records.append(record)
                del inputs, targets, logits, nll
        append_record(output / "events.jsonl", {"candidate": name, "state": "COMPLETED", "metrics": aggregate(records)})
        del model
        gc.collect()
        torch.cuda.empty_cache()


def write_new(path, value):
    with Path(path).open("xb") as target:
        target.write(canonical(value) + b"\n")
        target.flush()
        os.fsync(target.fileno())


def read_events(path, candidates):
    events, errors = [], []
    if not path.exists():
        return events, errors
    for index, line in enumerate(path.read_bytes().splitlines(), 1):
        try:
            event = parse(line)
            require(isinstance(event, dict) and event.get("candidate") in candidates, "unplanned candidate")
            require(event.get("state") in {"STARTED", "COMPLETED"}, "invalid candidate state")
            if event["state"] == "COMPLETED":
                require(isinstance(event.get("metrics"), dict), "missing completed metrics")
            events.append(event)
        except (Blocked, json.JSONDecodeError, UnicodeDecodeError) as exc:
            errors.append({"line": index, "reason": str(exc)})
    return events, errors


def run(plan, plan_path, output, expected_sha):
    require(SHA.fullmatch(expected_sha or "") is not None and digest(plan_path) == expected_sha, "externally retained plan hash required")
    validate(plan)
    require(canonical(read_json(plan_path)) == canonical(plan), "plan changed during admission")
    require(digest(__file__) == plan["evaluator_sha256"], "evaluator source mismatch")
    require(not output.exists(), "output already exists; never overwrite or resume this evaluation")
    runtime_guard(plan)
    output.mkdir(parents=False)
    started = time.monotonic()
    candidates = ["base", *(x["id"] for x in plan["adapters"])]
    write_new(output / "plan.json", plan)
    write_new(output / "started.json", {"plan_sha256": expected_sha, "evaluator_sha256": digest(__file__), "planned_candidates": candidates,
              "optimizer_updates": 0, "production_eligible": False})
    env = dict(os.environ)
    for key in ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "PYTHONPATH", "PYTHONHOME"]:
        env.pop(key, None)
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
               YACHAY_EVALUATOR_PARENT=str(os.getpid()),
               CUDA_VISIBLE_DEVICES=plan["runtime"]["gpu_uuid"], TOKENIZERS_PARALLELISM="false",
               HF_HOME=str(output / "hf-cache"), HF_MODULES_CACHE=str(output / "hf-modules"))
    state, code, failure = "FAILED", None, None
    child = None
    # flock is cooperative exclusion, not a substitute for the owner's resource lease.
    import fcntl
    with (output.parent / ".yachay-evaluation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with (output / "worker.log").open("xb") as log:
                child = subprocess.Popen([sys.executable, "-I", "-B", str(Path(__file__).resolve()), "_worker", str(output / "plan.json"), "--output", str(output)],
                                         env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    code = child.wait(timeout=max(0.001, plan["limits"]["wall_seconds"] - (time.monotonic() - started)))
                    state = "COMPLETED" if code == 0 else "FAILED"
                except subprocess.TimeoutExpired:
                    state = "TIMED_OUT"
        except OSError as exc:
            failure = str(exc)
        finally:
            # Only this newly created worker session is signalled, never an observed trainer PID.
            if child is not None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)
    events_path = output / "events.jsonl"
    events, event_errors = read_events(events_path, candidates)
    completed = {x["candidate"]: x["metrics"] for x in events if x.get("state") == "COMPLETED"}
    started_ids = {x["candidate"] for x in events if x.get("state") == "STARTED"}
    if set(completed) != set(candidates):
        state = "INCOMPLETE" if state == "COMPLETED" else state
    if event_errors:
        state = "FAILED"
    result = {"schema": SCHEMA, "state": state, "exit_code": code, "elapsed_seconds": time.monotonic() - started,
              "failure": failure, "event_errors": event_errors,
              "optimizer_updates": 0, "data_role": "development_diagnostic", "held_out_quality": "NOT_MEASURED",
              "production_eligible": False, "exact_training_resume": False, "plan_sha256": expected_sha,
              "evaluator_sha256": digest(__file__), "candidates": [
                  {"id": name, "state": "COMPLETED" if name in completed else state if name in started_ids else "NOT_RUN", "metrics": completed.get(name)} for name in candidates],
              "files": {p.name: digest(p) for p in output.iterdir() if p.is_file()}}
    write_new(output / "receipt.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["check", "verify", "run", "_worker"])
    parser.add_argument("plan", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plan-sha256")
    args = parser.parse_args(argv)
    try:
        plan = read_json(args.plan)
        validate(plan)
        if args.mode == "check":
            result = {"status": "PLAN_SCHEMA_VALID", "artifacts_verified": False, "runtime_qualified": False, "production_eligible": False}
        elif args.mode == "verify":
            rows = verify(plan)
            result = {"status": "LOCAL_BYTES_VERIFIED", "rows": len(rows), "runtime_qualified": False, "production_eligible": False}
        else:
            require(args.output is not None and args.output.is_absolute(), "absolute new output path required")
            if args.mode == "_worker":
                worker(plan, args.output)
                return 0
            result = run(plan, args.plan.resolve(), args.output, args.plan_sha256)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0 if result.get("state", "COMPLETED") == "COMPLETED" else 1
    except (Blocked, OSError, KeyError, TypeError, json.JSONDecodeError, subprocess.SubprocessError, importlib.metadata.PackageNotFoundError) as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc), "production_eligible": False}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
