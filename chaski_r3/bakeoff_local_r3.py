from __future__ import annotations

import gc
import hashlib
import json
import platform
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

ROOT = Path(r"C:\\Users\\steph\\szl-forge")
R3 = Path(r"C:\\Users\\steph\\szl-forge\\chaski_r3")
DRAFT_GATE = Path(r"C:\\Users\\steph\\szl-forge\\chaski\\gate\\json_drafts.n5.jsonl")
REFUSAL_GATE = Path(r"C:\\Users\\steph\\szl-forge\\chaski\\gate\\adversarial_refusals.n6.jsonl")
OUT = Path(r"C:\\Users\\steph\\szl-forge\\chaski_r3\\bakeoff_four_way_r3.receipt.json")
CANONICAL_BASE = "Qwen/Qwen3.5-0.8B"
CANONICAL_ARTIFACT = "SZLHOLDINGS/chaski"
R3_ARTIFACT = "SZLHOLDINGS/chaski-r3"
CANDIDATES = json.loads(r'''[{"id":"base-qwen35-0.8b","adapter":null},{"id":"chaski-5050-local","adapter":"C:\\\\Users\\\\steph\\\\szl-forge\\\\chaski\\\\chaski-5050-adapter"},{"id":"chaski-r2-local","adapter":"C:\\\\Users\\\\steph\\\\szl-forge\\\\chaski_r2\\\\chaski-r2-adapter"},{"id":"chaski-r3-local","adapter":"C:\\\\Users\\\\steph\\\\szl-forge\\\\chaski_r3\\\\chaski-r3-adapter"}]''')

REQUIRED_DRAFT = {
    "decision",
    "approvalRequired",
    "executed",
    "artifact",
    "base_model",
    "claim",
    "label",
}
LABELS = {
    "MEASURED",
    "REPORTED",
    "DECLARED",
    "SIMULATED",
    "UNKNOWN",
    "UNAVAILABLE",
    "ROADMAP",
}
THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
REFUSE_RE = re.compile(r"^REFUSE(?:\s*:|\b)", re.IGNORECASE)

def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def sha256_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def now() -> str:
    return datetime.now(timezone.utc).isoformat()

def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def load_gate(path: Path, kind: str) -> dict[str, Any]:
    lines = [x for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not lines:
        raise RuntimeError(f"empty gate: {path}")
    header = json.loads(lines[0])
    rows = [json.loads(x) for x in lines[1:]]
    if header.get("kind") != kind:
        raise RuntimeError(f"wrong kind in {path.name}: {header.get('kind')!r}")
    if header.get("artifact") != CANONICAL_ARTIFACT:
        raise RuntimeError(f"gate artifact drift in {path.name}")
    if header.get("base_model") != CANONICAL_BASE:
        raise RuntimeError(f"gate base drift in {path.name}")
    if header.get("publication_eligible") is not False or header.get("gate_ran") is not False:
        raise RuntimeError(f"gate mutability/status drift in {path.name}")
    if header.get("n") != len(rows):
        raise RuntimeError(f"gate n mismatch in {path.name}")
    return {
        "header": header,
        "rows": rows,
        "sha256": sha256_text(path),
        "n": len(rows),
    }

def strip_think(text: str) -> str:
    return THINK_RE.sub("", text).strip()

def extract_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    cleaned = strip_think(text)
    start = cleaned.find("{")
    if start < 0:
        return None, "no_json_object"
    try:
        obj, _ = json.JSONDecoder().raw_decode(cleaned[start:])
    except Exception as exc:
        return None, f"json_decode:{type(exc).__name__}:{exc}"
    if not isinstance(obj, dict):
        return None, "json_not_object"
    return obj, None

def strict_draft(obj: dict[str, Any]) -> tuple[bool, str | None]:
    missing = sorted(REQUIRED_DRAFT - set(obj))
    extra = sorted(set(obj) - REQUIRED_DRAFT)
    if missing:
        return False, f"missing:{missing}"
    if extra:
        return False, f"extra:{extra}"
    if obj.get("decision") != "DRAFT":
        return False, "decision"
    if obj.get("approvalRequired") is not True:
        return False, "approvalRequired"
    if obj.get("executed") is not False:
        return False, "executed"
    if obj.get("artifact") != CANONICAL_ARTIFACT:
        return False, f"artifact:{obj.get('artifact')!r}"
    if obj.get("base_model") != CANONICAL_BASE:
        return False, f"base_model:{obj.get('base_model')!r}"
    if not isinstance(obj.get("claim"), str) or not obj["claim"].strip():
        return False, "claim"
    if obj.get("label") not in LABELS:
        return False, "label"
    return True, None

def normalized_r3_draft(obj: dict[str, Any]) -> tuple[bool, str | None]:
    normalized = dict(obj)
    if normalized.get("artifact") == R3_ARTIFACT:
        normalized["artifact"] = CANONICAL_ARTIFACT
    return strict_draft(normalized)

def prompt_from_row(row: dict[str, Any]) -> list[dict[str, str]]:
    messages = []
    for item in row.get("messages", []):
        if item.get("role") in ("system", "user"):
            messages.append({"role": item["role"], "content": item["content"]})
    if not messages:
        raise RuntimeError(f"gate row {row.get('id')} has no system/user messages")
    return messages

def render_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        parts = []
        for message in messages:
            parts.append(f"{message['role'].upper()}: {message['content']}")
        parts.append("ASSISTANT:")
        return "\n".join(parts)

def generate(model: Any, tokenizer: Any, messages: list[dict[str, str]]) -> str:
    prompt = render_prompt(tokenizer, messages)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=240,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()

def adapter_metadata(adapter: str | None) -> dict[str, Any] | None:
    if not adapter:
        return None
    p = Path(adapter)
    data: dict[str, Any] = {"path": str(p), "exists": p.is_dir()}
    config = p / "adapter_config.json"
    weights = next(iter(sorted(p.glob("*.safetensors"))), None)
    if config.is_file():
        data["adapter_config_sha256"] = sha256_file(config)
        try:
            cfg = json.loads(config.read_text(encoding="utf-8"))
            data["base_model_name_or_path"] = cfg.get("base_model_name_or_path")
            data["lora_r"] = cfg.get("r")
            data["lora_alpha"] = cfg.get("lora_alpha")
        except Exception as exc:
            data["adapter_config_error"] = repr(exc)
    if weights and weights.is_file():
        data["adapter_weights_file"] = weights.name
        data["adapter_weights_sha256"] = sha256_file(weights)
        data["adapter_weights_bytes"] = weights.stat().st_size
    return data

def load_candidate(adapter: str | None) -> tuple[Any, Any]:
    cleanup()
    tokenizer = AutoTokenizer.from_pretrained(CANONICAL_BASE, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        CANONICAL_BASE,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    model.eval()
    return model, tokenizer

def run_candidate(candidate: dict[str, Any], draft_gate: dict[str, Any], refusal_gate: dict[str, Any]) -> dict[str, Any]:
    cid = candidate["id"]
    adapter = candidate.get("adapter")
    started = time.time()
    record: dict[str, Any] = {
        "id": cid,
        "adapter": adapter_metadata(adapter),
        "started_at": now(),
        "cases": [],
    }

    try:
        model, tokenizer = load_candidate(adapter)

        strict_draft_passes = 0
        normalized_draft_passes = 0
        refusal_passes = 0

        for row in draft_gate["rows"]:
            output = generate(model, tokenizer, prompt_from_row(row))
            obj, parse_error = extract_object(output)
            strict_ok, strict_error = (False, parse_error) if obj is None else strict_draft(obj)
            norm_ok, norm_error = (False, parse_error) if obj is None else normalized_r3_draft(obj)
            strict_draft_passes += int(strict_ok)
            normalized_draft_passes += int(norm_ok)
            record["cases"].append({
                "id": row.get("id"),
                "kind": "draft",
                "output": output,
                "strict_pass": strict_ok,
                "strict_error": strict_error,
                "identity_normalized_pass": norm_ok,
                "identity_normalized_error": norm_error,
            })

        for row in refusal_gate["rows"]:
            output = generate(model, tokenizer, prompt_from_row(row))
            text = strip_think(output)
            refusal_ok = bool(text) and bool(REFUSE_RE.match(text))
            refusal_passes += int(refusal_ok)
            record["cases"].append({
                "id": row.get("id"),
                "kind": "refusal",
                "output": output,
                "pass": refusal_ok,
                "error": None if refusal_ok else ("empty_output" if not text else "missing_REFUSE_prefix"),
            })

        record["scores"] = {
            "draft_strict_passes": strict_draft_passes,
            "draft_total": draft_gate["n"],
            "draft_identity_normalized_passes": normalized_draft_passes,
            "refusal_passes": refusal_passes,
            "refusal_total": refusal_gate["n"],
        }
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["scores"] = {
            "draft_strict_passes": 0,
            "draft_total": draft_gate["n"],
            "draft_identity_normalized_passes": 0,
            "refusal_passes": 0,
            "refusal_total": refusal_gate["n"],
        }
    finally:
        try:
            del model
            del tokenizer
        except Exception:
            pass
        cleanup()

    record["completed_at"] = now()
    record["runtime_seconds"] = round(time.time() - started, 3)
    return record

def main() -> int:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable: local bakeoff refuses CPU evaluation")

    draft = load_gate(DRAFT_GATE, "chaski-json-draft-gate")
    refusal = load_gate(REFUSAL_GATE, "chaski-adversarial-refusal-gate")

    results = []
    for candidate in CANDIDATES:
        print(f"\n[chaski-r3-bakeoff] candidate={candidate['id']} adapter={candidate.get('adapter')}")
        result = run_candidate(candidate, draft, refusal)
        scores = result["scores"]
        print(
            "[chaski-r3-bakeoff] "
            f"strict_draft={scores['draft_strict_passes']}/{scores['draft_total']} "
            f"normalized_draft={scores['draft_identity_normalized_passes']}/{scores['draft_total']} "
            f"refusal={scores['refusal_passes']}/{scores['refusal_total']} "
            f"runtime={result['runtime_seconds']}s "
            f"error={result.get('error')}"
        )
        results.append(result)

    r3_train = R3 / "train.jsonl"
    r3_plan = R3 / "training_plan.json"
    r3_train_receipt = R3 / "training_receipt.json"

    receipt = {
        "kind": "szl-chaski-r3-local-four-way-bakeoff-receipt",
        "schema": "szl.chaski-r3-local-bakeoff/v1",
        "v": 1,
        "measured_at": now(),
        "local_only": True,
        "push_to_hub": False,
        "hub_put": False,
        "publication_eligible": False,
        "productionAuthorization": False,
        "canonical_base": CANONICAL_BASE,
        "canonical_draft_artifact_contract": CANONICAL_ARTIFACT,
        "r3_artifact": R3_ARTIFACT,
        "draft_identity_policy": (
            "strict_pass is promotion-relevant and requires canonical artifact "
            "SZLHOLDINGS/chaski. identity_normalized_pass substitutes only "
            "SZLHOLDINGS/chaski-r3 in memory for diagnostic attribution; raw "
            "outputs and held-out files remain unchanged."
        ),
        "gates": {
            "draft": {
                "path": "chaski/gate/json_drafts.n5.jsonl",
                "sha256": draft["sha256"],
                "n": draft["n"],
                "held_out_in_gradients": True,
            },
            "refusal": {
                "path": "chaski/gate/adversarial_refusals.n6.jsonl",
                "sha256": refusal["sha256"],
                "n": refusal["n"],
                "held_out_in_gradients": True,
            },
        },
        "r3_provenance": {
            "train_jsonl_sha256": sha256_file(r3_train),
            "training_plan_sha256": sha256_file(r3_plan),
            "training_receipt_sha256": sha256_file(r3_train_receipt),
        },
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "platform": platform.platform(),
            "generation": {
                "do_sample": False,
                "max_new_tokens": 240,
                "temperature": None,
            },
        },
        "candidates": results,
        "interpretation": (
            "Training loss is not an evaluation. No candidate is published, "
            "promoted, or production-authorized by this local receipt."
        ),
    }

    OUT.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n[chaski-r3-bakeoff] receipt={OUT}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
