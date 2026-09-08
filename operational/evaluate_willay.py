#!/usr/bin/env python3
"""Compare an immutable WILLAY adapter with its base using a frozen local suite.

Exit 0: thresholds met. Exit 1: measured failure. Exit 2: evidence unavailable.
Optional ML dependencies load after input validation. This script never uploads.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def strict_json(raw):
    def pairs(items):
        obj = {}
        for key, value in items:
            if key in obj:
                raise ValueError("duplicate JSON key")
            obj[key] = value
        return obj

    def constant(value):
        raise ValueError("nonfinite JSON number")

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def revision(value):
    if not re.fullmatch(r"[0-9a-f]{40}", value or "") or value == "0" * 40:
        raise ValueError("a nonzero immutable 40-character revision is required")
    return value


def threshold(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("threshold must be numeric")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("threshold must be finite and within [0, 1]")


def load_suite(path, expected_sha256):
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256 or ""):
        raise ValueError("suite SHA-256 is required")
    if path.stat().st_size > 1_048_576:
        raise ValueError("suite exceeds 1 MiB")
    raw = path.read_bytes()
    if digest(raw) != expected_sha256:
        raise ValueError("suite digest mismatch")
    suite = strict_json(raw)
    if not isinstance(suite, dict) or suite.get("schema") != "szl.willay-suite/v1":
        raise ValueError("unsupported suite schema")
    cases = suite.get("cases")
    if not isinstance(cases, list) or not 1 <= len(cases) <= 128:
        raise ValueError("suite must contain 1..128 cases")
    ids = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("case must be an object")
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id or case_id in ids:
            raise ValueError("case IDs must be nonempty and unique")
        ids.add(case_id)
        messages = case.get("messages")
        if not isinstance(messages, list) or not 1 <= len(messages) <= 16:
            raise ValueError("invalid message count")
        for message in messages:
            if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant"}:
                raise ValueError("invalid message role")
            content = message.get("content")
            if not isinstance(content, str) or not content or len(content) > 8192:
                raise ValueError("invalid message content")
        if not isinstance(case.get("expected"), dict) or not case["expected"]:
            raise ValueError("case requires expected JSON fields")
        canonical(case["expected"])
    return suite


def score_output(raw, expected):
    checks = {"json_object": False}
    try:
        parsed = strict_json(raw)
        canonical(parsed)
        checks["json_object"] = isinstance(parsed, dict)
    except (ValueError, TypeError, RecursionError):
        parsed = None
    for field, value in expected.items():
        checks["field:" + field] = (
            isinstance(parsed, dict) and field in parsed
            and canonical(parsed[field]) == canonical(value)
        )
    return {"checks": checks, "score": sum(checks.values()) / len(checks),
            "passed": all(checks.values())}


def compare_outputs(cases, base_outputs, candidate_outputs, minimum_pass_rate=1.0,
                    minimum_improvement=0.0):
    threshold(minimum_pass_rate)
    threshold(minimum_improvement)
    if not cases or len(base_outputs) != len(cases) or len(candidate_outputs) != len(cases):
        raise ValueError("each lane must produce exactly one output per case")
    lanes = {}
    for name, outputs in (("base", base_outputs), ("candidate", candidate_outputs)):
        rows = []
        for case, raw in zip(cases, outputs):
            if not isinstance(raw, str):
                raise ValueError("generation output must be text")
            rows.append({"case_id": case["id"], "raw_output": raw,
                         "output_sha256": digest(raw.encode("utf-8")),
                         **score_output(raw, case["expected"])})
        lanes[name] = {"cases": rows, "mean_score": sum(r["score"] for r in rows) / len(rows),
                       "pass_rate": sum(r["passed"] for r in rows) / len(rows)}
    delta = lanes["candidate"]["mean_score"] - lanes["base"]["mean_score"]
    pass_delta = lanes["candidate"]["pass_rate"] - lanes["base"]["pass_rate"]
    passed = (lanes["candidate"]["pass_rate"] >= minimum_pass_rate
              and delta >= minimum_improvement and pass_delta >= 0)
    return {"lanes": lanes, "delta_mean_score": delta, "delta_pass_rate": pass_delta,
            "thresholds": {"minimum_pass_rate": minimum_pass_rate,
                           "minimum_improvement": minimum_improvement},
            "result": "PASS" if passed else "FAIL"}


def generate_comparison(args, cases):
    import torch
    from huggingface_hub import hf_hub_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config_path = Path(hf_hub_download(args.model, "adapter_config.json", revision=args.model_revision))
    config_bytes = config_path.read_bytes()
    config = strict_json(config_bytes)
    if config.get("base_model_name_or_path") != args.base:
        raise ValueError("adapter base identity mismatch")
    if config.get("revision") not in (None, args.base_revision):
        raise ValueError("adapter base revision mismatch")
    torch.manual_seed(0)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    dtype = torch.float32 if device == "cpu" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.base, revision=args.base_revision, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, revision=args.base_revision, trust_remote_code=False,
        use_safetensors=True, dtype=dtype, attn_implementation="eager",
    ).to(device).eval()

    def run_lane(active):
        outputs = []
        for case in cases:
            prompt = tokenizer.apply_chat_template(case["messages"], tokenize=False, add_generation_prompt=True)
            tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
            if tokens["input_ids"].shape[-1] > args.max_input_tokens:
                raise ValueError("prompt exceeds input token budget")
            with torch.inference_mode():
                result = active.generate(**tokens, do_sample=False, max_new_tokens=args.max_new_tokens,
                                         pad_token_id=tokenizer.eos_token_id)
            outputs.append(tokenizer.decode(result[0, tokens["input_ids"].shape[-1]:], skip_special_tokens=True))
        return outputs

    base_outputs = run_lane(model)
    candidate = PeftModel.from_pretrained(model, args.model, revision=args.model_revision,
                                         is_trainable=False).eval()
    candidate_outputs = run_lane(candidate)
    return base_outputs, candidate_outputs, {
        "device": device, "dtype": str(dtype), "seed": 0, "do_sample": False,
        "adapter_config_sha256": digest(config_bytes),
        "max_new_tokens": args.max_new_tokens, "max_input_tokens": args.max_input_tokens,
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name)
                     for name in ("torch", "transformers", "peft", "huggingface-hub")},
    }


def write_receipt(path, receipt):
    path.parent.mkdir(parents=True, exist_ok=True)
    receipt["receipt_sha256"] = digest(canonical(receipt))
    fd, temporary = tempfile.mkstemp(prefix=".willay-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv=None, generator=generate_comparison):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=HERE / "willay-suite.json")
    parser.add_argument("--suite-sha256", default=os.getenv("WILLAY_SUITE_SHA256", ""))
    parser.add_argument("--model", default="SZLHOLDINGS/WILLAY")
    parser.add_argument("--model-revision", default=os.getenv("WILLAY_REVISION", ""))
    parser.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--base-revision", default=os.getenv("WILLAY_BASE_REVISION", ""))
    parser.add_argument("--output", type=Path, default=HERE / "out" / "willay-eval-receipt.json")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--minimum-pass-rate", type=float, default=1.0)
    parser.add_argument("--minimum-improvement", type=float, default=0.0)
    args = parser.parse_args(argv)
    receipt = {
        "schema": "szl.willay-evaluation/v2", "kind": "szl-ops-willay-eval",
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "status": "UNKNOWN", "result": "UNAVAILABLE", "publication_eligible": False,
        "limits": "Bounded local fixture evidence; no general benchmark or safety certification.",
        "source": {"evaluator_sha256": digest(Path(__file__).read_bytes())},
        "model": {"id": args.model, "revision": args.model_revision},
        "base": {"id": args.base, "revision": args.base_revision},
        "suite_sha256": args.suite_sha256,
    }
    phase = "validate_inputs"
    started = time.monotonic()
    exit_code = 2
    try:
        revision(args.model_revision)
        revision(args.base_revision)
        if not 1 <= args.max_new_tokens <= 512 or not 1 <= args.max_input_tokens <= 4096:
            raise ValueError("token budget out of range")
        threshold(args.minimum_pass_rate)
        threshold(args.minimum_improvement)
        suite = load_suite(args.suite, args.suite_sha256)
        receipt["suite"] = suite
        phase = "local_generation"
        base, candidate, environment = generator(args, suite["cases"])
        phase = "score"
        comparison = compare_outputs(suite["cases"], base, candidate,
                                     args.minimum_pass_rate, args.minimum_improvement)
        receipt.update(comparison)
        receipt.update(status="MEASURED", environment=environment)
        exit_code = 0 if comparison["result"] == "PASS" else 1
    except (Exception, KeyboardInterrupt) as exc:
        # Provider exception messages can contain credentials or request headers.
        receipt["failure"] = {"phase": phase, "type": type(exc).__name__}
    receipt["elapsed_seconds"] = round(time.monotonic() - started, 3)
    receipt["exit_code"] = exit_code
    write_receipt(args.output, receipt)
    print(json.dumps({"receipt": str(args.output), "status": receipt["status"],
                      "result": receipt["result"], "exit_code": exit_code}))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
