#!/usr/bin/env python3
"""Preregistered, paired local diagnostic of Khipu's selective abstention.

This never trains, uploads, calls tools, executes a proposed plan, or rewrites
the canonical evaluation receipt. The baseline and policy-assisted arms use
the same pinned merged checkpoint. A result is exploratory local evidence,
not a model promotion or an independent replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PREREG = HERE / "preregister.json"
POLICY = HERE / "policy_prompt.txt"
SCHEMA = ROOT / "khipu" / "khipu.schema.json"
TARGET_MODULES = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
MIN_POST_DOWNLOAD_FREE_BYTES = 8 * (1024**3)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_bytes(commit: str, path: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{commit}:{path}"],
        check=True,
        capture_output=True,
    ).stdout


def load_grading_schema(frozen: dict) -> dict:
    """Read the exact preregistered schema that will grade this run.

    Historical Git-blob verification alone does not bind a later checkout's
    working-tree schema. Hash and parse the same bytes before model effects.
    """
    raw = SCHEMA.read_bytes()
    if hashlib.sha256(raw).hexdigest() != frozen.get("plan_schema_sha256"):
        raise ValueError("working-tree grading schema differs from preregistration")
    schema = strict_json(raw.decode("utf-8"))
    if not isinstance(schema, dict):
        raise ValueError("grading schema must be a JSON object")
    return schema


def signed_anchor() -> str:
    head = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(
        ["git", "-C", str(ROOT), "verify-commit", head],
        check=True,
        capture_output=True,
    )
    dirty = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain=v1", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if dirty:
        raise ValueError("source checkout has tracked modifications after its signed anchor")
    for relative, path in (
        ("experiments/khipu-selective-holdout-v2/run.py", Path(__file__)),
        ("experiments/khipu-selective-holdout-v2/policy_prompt.txt", POLICY),
        ("experiments/khipu-selective-holdout-v2/preregister.json", PREREG),
    ):
        if hashlib.sha256(git_bytes(head, relative)).hexdigest() != sha256_file(path):
            raise ValueError(f"signed anchor does not contain current source bytes: {relative}")
    return head


def check_frozen_inputs(cases_path: Path) -> tuple[dict, list[dict], str]:
    frozen = load_json(PREREG)
    if frozen.get("schema") != "szl.khipu.selective-holdout.preregister/v2":
        raise ValueError("unknown preregistration schema")
    if cases_path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("sealed cases must be outside the source checkout")
    pinned = {
        "runner_sha256": Path(__file__),
        "dataset_sha256": cases_path,
        "policy_prompt_sha256": POLICY,
    }
    for field, path in pinned.items():
        actual = sha256_file(path)
        if frozen.get(field) != actual:
            raise ValueError(f"frozen {field} mismatch: {actual}")
    base = frozen["source_base_commit"]
    source_files = {
        "plan_schema_sha256": "khipu/khipu.schema.json",
        "training_receipt_file_sha256": "khipu/training_receipt.signed.json",
        "source_binding_sha256": "publishing/model-source-bindings.json",
        "curriculum_manifest_sha256": "khipu/manifest.json",
    }
    for field, relative in source_files.items():
        actual = hashlib.sha256(git_bytes(base, relative)).hexdigest()
        if frozen.get(field) != actual:
            raise ValueError(f"source-base {field} mismatch: {actual}")
    if frozen.get("training_eligible") is not False or frozen.get("promotion_eligible") is not False:
        raise ValueError("the diagnostic fixture must not be a curriculum or promotion gate")

    binding = json.loads(git_bytes(base, "publishing/model-source-bindings.json"))
    artifact = next(
        (item for item in binding["artifacts"] if item.get("repo_id") == frozen["hub_repo_id"]),
        None,
    )
    if artifact is None or artifact.get("promotion_state") != "NOT_PROMOTED_RESEARCH_ONLY":
        raise ValueError("Khipu source binding or promotion state changed")
    if artifact.get("expected_weight_sha256") != frozen.get("expected_weight_sha256"):
        raise ValueError("checkpoint hashes differ from canonical source binding")
    lineage = next(
        (entry for entry in artifact.get("lineage", []) if entry.get("repo_id") == "Qwen/Qwen2.5-1.5B-Instruct"),
        None,
    )
    if lineage is None or lineage.get("revision") != frozen.get("declared_lineage_base_revision"):
        raise ValueError("declared base lineage revision differs from canonical source binding")
    if frozen.get("hub_revision") != "8a5c666c4fc1ec75a51b8bd9784b4504e807757a":
        raise ValueError("this replay requires the preregistered immutable Hub revision")

    receipt = json.loads(git_bytes(base, "khipu/training_receipt.signed.json"))
    canonical = receipt.get("canonical")
    if not isinstance(canonical, str) or hashlib.sha256(canonical.encode("utf-8")).hexdigest() != frozen.get("training_receipt_canonical_sha256"):
        raise ValueError("owner training receipt canonical chain changed")
    for name, record in json.loads(git_bytes(base, "khipu/manifest.json"))["files"].items():
        actual = hashlib.sha256(git_bytes(base, f"khipu/{name}")).hexdigest()
        if actual != record["sha256"]:
            raise ValueError(f"canonical curriculum file drift: {name}")

    fixture = load_json(cases_path)
    if fixture.get("schema") != "szl.khipu.selective-holdout/v2" or fixture.get("training_eligible") is not False:
        raise ValueError("fixture contract changed")
    cases = fixture["cases"]
    if len(cases) != 16 or {c["id"] for c in cases} != {
        *(f"NAV-{i:02d}" for i in range(1, 9)),
        *(f"ABS-{i:02d}" for i in range(1, 9)),
    }:
        raise ValueError("expected eight NAVIGATE and eight ABSTAIN cases")
    all_ids: set[str] = set()
    all_queries: set[str] = set()
    for case in cases:
        ids = [c["nodeId"] for c in case["candidates"]]
        if len(ids) != len(set(ids)) or any(not x.startswith("node://khipu-synthetic/selective-v2-") for x in ids):
            raise ValueError(f"non-unique or non-synthetic candidate ids in {case['id']}")
        if all_ids.intersection(ids):
            raise ValueError(f"candidate id reused across cases: {case['id']}")
        all_ids.update(ids)
        if any(c["label"] != "SIMULATED" for c in case["candidates"]):
            raise ValueError(f"fixture evidence label drift: {case['id']}")
        query = case["query"].casefold()
        if query in all_queries:
            raise ValueError(f"duplicate query: {case['id']}")
        all_queries.add(query)
        expected = case["expected_cited_node_ids"]
        if not set(expected).issubset(ids):
            raise ValueError(f"reference cites a missing handle: {case['id']}")
        if case["expected_decision"] == "NAVIGATE" and len(expected) != 1:
            raise ValueError(f"NAVIGATE reference requires one handle: {case['id']}")
        if case["expected_decision"] == "ABSTAIN" and expected:
            raise ValueError(f"ABSTAIN reference cannot cite: {case['id']}")

    # Exact-overlap check only. It cannot establish absence of semantic leakage.
    old_queries: set[str] = set()
    old_ids: set[str] = set()
    for name in ("train.jsonl", "train.abstain.jsonl", "eval.jsonl", "adversarial.jsonl"):
        for line in git_bytes(base, f"khipu/{name}").decode("utf-8").splitlines():
            row = json.loads(line)
            user = next(m for m in row["messages"] if m["role"] == "user")
            payload = json.loads(user["content"])
            old_queries.add(payload["query"].casefold())
            old_ids.update(c["nodeId"] for c in payload["candidates"])
    if old_queries.intersection(all_queries) or old_ids.intersection(all_ids):
        raise ValueError("constructed holdout overlaps the canonical curriculum")
    system = json.loads(git_bytes(base, "khipu/train.jsonl").decode("utf-8").splitlines()[0])["messages"][0]["content"]
    if hashlib.sha256(system.encode("utf-8")).hexdigest() != frozen.get("system_prompt_sha256"):
        raise ValueError("canonical system prompt changed")
    if frozen.get("generation", {}).get("tool_access") is not False:
        raise ValueError("tool access must remain disabled")
    if frozen.get("arms") != ["baseline", "policy"]:
        raise ValueError("paired arm contract changed")
    load_grading_schema(frozen)
    signed_anchor()
    return frozen, cases, system


def check_local_artifacts(model_dir: Path, frozen: dict) -> dict:
    """Hash both public artifacts and verify adapter key/shape coverage.

    The adapter is inspected, not applied: model.safetensors is the merged
    checkpoint. Structural coverage is not a numerical-equivalence claim.
    """
    from safetensors import safe_open

    if shutil.disk_usage(model_dir).free < MIN_POST_DOWNLOAD_FREE_BYTES:
        raise RuntimeError("less than 8 GiB free disk remains after asset preparation")
    for name, expected in frozen["expected_weight_sha256"].items():
        path = model_dir / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"missing or byte-mismatched immutable Hub artifact: {name}")
    for name, expected in frozen["expected_support_sha256"].items():
        path = model_dir / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"missing or byte-mismatched immutable Hub support file: {name}")
    config = load_json(model_dir / "config.json")
    adapter_config = load_json(model_dir / "adapter" / "adapter_config.json")
    if config.get("architectures") != ["Qwen2ForCausalLM"] or config.get("num_hidden_layers") != 28:
        raise ValueError("unexpected original checkpoint architecture")
    if adapter_config.get("peft_type") != "LORA" or adapter_config.get("r") != 32:
        raise ValueError("unexpected adapter configuration")
    if set(adapter_config.get("target_modules", [])) != TARGET_MODULES:
        raise ValueError("adapter target modules differ from the original training contract")

    with safe_open(str(model_dir / "model.safetensors"), framework="pt", device="cpu") as merged:
        merged_shapes = {key: tuple(merged.get_slice(key).get_shape()) for key in merged.keys()}
    with safe_open(str(model_dir / "adapter" / "adapter_model.safetensors"), framework="pt", device="cpu") as adapter:
        adapter_shapes = {key: tuple(adapter.get_slice(key).get_shape()) for key in adapter.keys()}
    pattern = re.compile(r"layers\.(\d+)\.(self_attn|mlp)\.([a-z_]+)\.lora_([AB])(?:\.default)?\.weight$")
    found: dict[tuple[int, str, str], dict[str, tuple[int, ...]]] = {}
    for key, shape in adapter_shapes.items():
        match = pattern.search(key)
        if not match:
            raise ValueError(f"unrecognized adapter key: {key}")
        layer, block, module, side = match.groups()
        if module not in TARGET_MODULES or (module in {"q_proj", "k_proj", "v_proj", "o_proj"}) != (block == "self_attn"):
            raise ValueError(f"adapter module mismatch: {key}")
        slot = found.setdefault((int(layer), block, module), {})
        if side in slot:
            raise ValueError(f"duplicate adapter matrix: {key}")
        slot[side] = shape
    expected_slots = {
        (layer, "self_attn" if module in {"q_proj", "k_proj", "v_proj", "o_proj"} else "mlp", module)
        for layer in range(28) for module in TARGET_MODULES
    }
    if set(found) != expected_slots or len(adapter_shapes) != 392:
        raise ValueError(f"adapter coverage incomplete: {len(found)}/196 module pairs, {len(adapter_shapes)}/392 keys")
    for (layer, block, module), matrices in found.items():
        base_key = f"model.layers.{layer}.{block}.{module}.weight"
        if base_key not in merged_shapes:
            raise ValueError(f"merged checkpoint lacks adapter target: {base_key}")
        out_features, in_features = merged_shapes[base_key]
        if matrices.get("A") != (32, in_features) or matrices.get("B") != (out_features, 32):
            raise ValueError(f"adapter tensor shape mismatch: {base_key}")
    return {
        "merged_tensor_count": len(merged_shapes),
        "adapter_tensor_count": len(adapter_shapes),
        "adapter_matched_module_pairs": len(found),
        "adapter_applied_to_merged_inference": False,
        "numerical_equivalence_of_adapter_and_merged": "UNKNOWN",
        "training_base_immutable_revision": "UNKNOWN",
    }


def strict_json(raw: str) -> object:
    def unique_pairs(pairs: list[tuple[str, object]]) -> dict:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-JSON constant: {value}")

    return json.loads(raw, object_pairs_hook=unique_pairs, parse_constant=reject_constant)


def grade(raw: str, case: dict, schema: dict) -> dict:
    from jsonschema import Draft202012Validator

    result = {
        "strict_json_valid": False,
        "schema_valid": False,
        "contract_valid": False,
        "proposed_decision": None,
        "decision_correct": False,
        "citation_correct": False,
        "case_pass": False,
    }
    try:
        plan = strict_json(raw)
    except (ValueError, TypeError):
        return result
    result["strict_json_valid"] = isinstance(plan, dict)
    if isinstance(plan, dict):
        result["proposed_decision"] = plan.get("decision")
    if not isinstance(plan, dict) or not Draft202012Validator(schema).is_valid(plan):
        return result
    result["schema_valid"] = True
    offered = {c["nodeId"] for c in case["candidates"]}
    candidates = plan.get("candidates") or []
    steps = plan.get("steps") or []
    cited = plan.get("citedNodeIds") or []
    cited_set = set(cited)
    cite_steps = {step["nodeId"] for step in steps if step["action"] == "CITE"}
    contract = (
        plan.get("provenance") == "MODEL_PROPOSED"
        and plan.get("query") == case["query"]
        and candidates == case["candidates"]
        and len(cited) == len(cited_set)
        and cited_set.issubset(offered)
        and all(step["nodeId"] in offered for step in steps)
        and cited_set == cite_steps
        and (
            (plan.get("decision") == "ABSTAIN" and not steps and not cited and bool(plan.get("abstainReason")))
            or (plan.get("decision") == "NAVIGATE" and bool(cited) and plan.get("abstainReason") is None)
        )
    )
    result["contract_valid"] = contract
    result["decision_correct"] = plan.get("decision") == case["expected_decision"]
    result["citation_correct"] = cited_set == set(case["expected_cited_node_ids"])
    result["case_pass"] = contract and result["decision_correct"] and result["citation_correct"]
    return result


def run(cases_path: Path, model_dir: Path, output_dir: Path) -> None:
    frozen, cases, system = check_frozen_inputs(cases_path)
    schema = load_grading_schema(frozen)
    anchor = signed_anchor()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    started = utc_now()
    partial = output_dir / "outputs.partial.jsonl"
    completed = 0
    artifact_check: dict = {}
    try:
        artifact_check = check_local_artifacts(model_dir, frozen)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("local bf16 CUDA device unavailable")
        free_vram, total_vram = torch.cuda.mem_get_info()
        if free_vram < 5 * (1024**3):
            raise RuntimeError("less than 5 GiB free VRAM before original checkpoint load")
        torch.manual_seed(frozen["generation"]["seed"])
        torch.cuda.manual_seed_all(frozen["generation"]["seed"])
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            dtype=torch.bfloat16,
            device_map="cuda:0",
            low_cpu_mem_usage=True,
            local_files_only=True,
            trust_remote_code=False,
            attn_implementation="eager",
        )
        if type(model).__name__ != "Qwen2ForCausalLM":
            raise RuntimeError(f"unexpected loaded model class: {type(model).__name__}")
        model.eval()
        policy = POLICY.read_text(encoding="utf-8").strip()
        with partial.open("x", encoding="utf-8", newline="\n") as stream:
            for index, case in enumerate(cases):
                # Alternating within-pair order limits a simple thermal/order confound.
                arms = ("baseline", "policy") if index % 2 == 0 else ("policy", "baseline")
                for arm in arms:
                    payload = {"query": case["query"], "candidates": case["candidates"]}
                    messages = [
                        {"role": "system", "content": system if arm == "baseline" else system + "\n\n" + policy},
                        {"role": "user", "content": json.dumps(payload, separators=(",", ":"), ensure_ascii=False)},
                    ]
                    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                    case_started = time.monotonic()
                    with torch.inference_mode():
                        generated = model.generate(
                            **inputs,
                            max_new_tokens=frozen["generation"]["max_new_tokens"],
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                    suffix = generated[0][inputs["input_ids"].shape[1]:]
                    raw = tokenizer.decode(suffix, skip_special_tokens=True)
                    verdict = grade(raw, case, schema)
                    record = {
                        "case_id": case["id"],
                        "arm": arm,
                        "case_index": index,
                        "expected_decision": case["expected_decision"],
                        "expected_cited_node_ids": case["expected_cited_node_ids"],
                        "input_token_count": int(inputs["input_ids"].shape[1]),
                        "generated_token_count": int(suffix.shape[0]),
                        "elapsed_seconds": round(time.monotonic() - case_started, 3),
                        "raw_output": raw,
                        "grade": verdict,
                    }
                    stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                    completed += 1
                    print(f"{case['id']} {arm}: {verdict['case_pass']}", flush=True)
        if completed != 2 * len(cases):
            raise RuntimeError("incomplete output set")
        output = output_dir / "outputs.jsonl"
        partial.replace(output)
        rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        if {(row["case_id"], row["arm"]) for row in rows} != {
            (case["id"], arm) for case in cases for arm in ("baseline", "policy")
        }:
            raise RuntimeError("paired output coverage mismatch")
        for field, path in (
            ("runner_sha256", Path(__file__)),
            ("dataset_sha256", cases_path),
            ("policy_prompt_sha256", POLICY),
        ):
            if sha256_file(path) != frozen[field]:
                raise RuntimeError(f"frozen {field} changed during inference")
        if signed_anchor() != anchor:
            raise RuntimeError("source anchor changed during inference")
        metrics = {}
        for arm in ("baseline", "policy"):
            selected = [row for row in rows if row["arm"] == arm]
            metrics[arm] = {
                "total": len(selected),
                "strict_json_valid": sum(row["grade"]["strict_json_valid"] for row in selected),
                "schema_valid": sum(row["grade"]["schema_valid"] for row in selected),
                "contract_valid": sum(row["grade"]["contract_valid"] for row in selected),
                "navigation_correct": sum(row["grade"]["case_pass"] for row in selected if row["expected_decision"] == "NAVIGATE"),
                "navigation_total": 8,
                "abstention_correct": sum(row["grade"]["case_pass"] for row in selected if row["expected_decision"] == "ABSTAIN"),
                "abstention_total": 8,
            }
        metrics["exploratory_paired_difference"] = {
            "abstention_correct_policy_minus_baseline": metrics["policy"]["abstention_correct"] - metrics["baseline"]["abstention_correct"],
            "navigation_correct_policy_minus_baseline": metrics["policy"]["navigation_correct"] - metrics["baseline"]["navigation_correct"],
        }
        receipt = {
            "schema": "szl.khipu.selective-holdout.local-receipt/v2",
            "status": "MEASURED_LOCAL_EXPLORATORY",
            "started_at_utc": started,
            "finished_at_utc": utc_now(),
            "source_anchor_commit": anchor,
            "source_base_commit": frozen["source_base_commit"],
            "hub_repo_id": frozen["hub_repo_id"],
            "hub_revision": frozen["hub_revision"],
            "preregister_sha256": sha256_file(PREREG),
            "dataset_sha256": frozen["dataset_sha256"],
            "dataset_training_eligible": False,
            "runner_sha256": frozen["runner_sha256"],
            "harness_sha256": frozen["runner_sha256"],
            "policy_prompt_sha256": frozen["policy_prompt_sha256"],
            "plan_schema_sha256": frozen["plan_schema_sha256"],
            "source_binding_sha256": frozen["source_binding_sha256"],
            "training_receipt_file_sha256": frozen["training_receipt_file_sha256"],
            "training_receipt_canonical_sha256": frozen["training_receipt_canonical_sha256"],
            "declared_lineage_base_revision": frozen["declared_lineage_base_revision"],
            "actual_training_implementation_revision": "UNKNOWN",
            "loss": "NOT_APPLICABLE_INFERENCE_ONLY",
            "expected_weight_sha256": frozen["expected_weight_sha256"],
            "adapter_coverage": artifact_check,
            "generation": frozen["generation"],
            "hardware": {"gpu": torch.cuda.get_device_name(0), "vram_total_bytes": total_vram, "vram_free_before_load_bytes": free_vram},
            "software": {"python": platform.python_version(), "torch": torch.__version__, "transformers": __import__("transformers").__version__},
            "outputs_sha256": sha256_file(output),
            "metrics": metrics,
            "energy_joules": "UNAVAILABLE",
            "carbon": "ROADMAP",
            "independent_replay": "NOT_RUN",
            "signed": False,
            "exploratory_comparison": True,
            "promotion_eligible": False,
        }
        (output_dir / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(metrics, sort_keys=True), flush=True)
    except Exception as error:
        blocked = {
            "schema": "szl.khipu.selective-holdout.local-receipt/v2",
            "status": "BLOCKED_NO_SCORE",
            "started_at_utc": started,
            "finished_at_utc": utc_now(),
            "completed_cases": completed,
            "expected_cases": 2 * len(cases),
            "error_type": type(error).__name__,
            "error": str(error),
            "source_anchor_commit": anchor,
            "preregister_sha256": sha256_file(PREREG),
            "partial_outputs_sha256": sha256_file(partial) if partial.exists() else None,
            "artifact_check": artifact_check,
            "metrics": "NOT_RUN",
            "independent_replay": "NOT_RUN",
            "promotion_eligible": False,
        }
        (output_dir / "blocked-receipt.json").write_text(json.dumps(blocked, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true", help="verify sealed fixture and signed source anchor without loading a model")
    parser.add_argument("--cases", type=Path, required=True, help="sealed 16-case fixture outside the source checkout")
    parser.add_argument("--model-dir", type=Path, help="local snapshot of immutable original Hub revision")
    parser.add_argument("--output-dir", type=Path, help="new output directory outside the source checkout")
    args = parser.parse_args()
    cases_path = args.cases.resolve(strict=True)
    if args.preflight_only:
        frozen, cases, _ = check_frozen_inputs(cases_path)
        print(json.dumps({"status": "PREFLIGHT_PASS", "cases": len(cases), "arms": 2, "source_anchor_commit": signed_anchor(), "dataset_sha256": frozen["dataset_sha256"], "preregister_sha256": sha256_file(PREREG), "hub_revision": frozen["hub_revision"]}, sort_keys=True))
        return 0
    if args.model_dir is None or args.output_dir is None:
        parser.error("--model-dir and --output-dir are required for inference")
    run(cases_path, args.model_dir.resolve(strict=True), args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
