#!/usr/bin/env python3
"""Run a real, bounded CUDA load and generation for the frontier candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_CANDIDATE = HERE / "candidate.json"
GIB = 1024**3


class QualificationError(RuntimeError):
    """The candidate did not satisfy a mandatory runtime gate."""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def load_candidate(path: Path) -> dict[str, Any]:
    candidate = json.loads(path.read_text(encoding="utf-8"))
    if candidate.get("schema") != "szl.frontier-model-candidate/v1":
        raise QualificationError("unsupported candidate schema")
    implementation = candidate.get("training_implementation", {})
    if not implementation.get("repo_id") or not implementation.get("revision"):
        raise QualificationError("candidate does not pin its implementation")
    if candidate.get("autonomy_eligible") is not False:
        raise QualificationError("unqualified candidate cannot be autonomous")
    if candidate.get("publication_eligible") is not False:
        raise QualificationError("unqualified candidate cannot be publishable")
    return candidate


def gpu_gate(torch: Any, *, min_free_gib: float, max_temp_c: int) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise QualificationError("CUDA is unavailable")
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    device = torch.cuda.get_device_properties(0)
    temperature = None
    try:
        import subprocess

        measured = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        temperature = int(measured.stdout.strip().splitlines()[0])
    except Exception:
        temperature = None
    if free_bytes < int(min_free_gib * GIB):
        raise QualificationError(
            f"GPU free memory {free_bytes / GIB:.2f} GiB is below "
            f"{min_free_gib:.2f} GiB"
        )
    if temperature is None:
        raise QualificationError("GPU temperature is UNKNOWN")
    if temperature > max_temp_c:
        raise QualificationError(
            f"GPU temperature {temperature} C exceeds {max_temp_c} C"
        )
    return {
        "name": device.name,
        "compute_capability": f"{device.major}.{device.minor}",
        "total_bytes": total_bytes,
        "free_bytes_before_load": free_bytes,
        "temperature_c_before_load": temperature,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }


def qualify(
    candidate: dict[str, Any],
    *,
    min_free_gib: float,
    max_temp_c: int,
) -> dict[str, Any]:
    import torch

    gpu = gpu_gate(torch, min_free_gib=min_free_gib, max_temp_c=max_temp_c)
    tensor = torch.arange(1024, device="cuda", dtype=torch.float32)
    tensor_sum = tensor.sum().item()
    if tensor_sum != 523776:
        raise QualificationError("CUDA tensor result did not match the golden value")
    del tensor
    torch.cuda.empty_cache()

    from unsloth import FastVisionModel

    implementation = candidate["training_implementation"]
    started = time.perf_counter()
    model, processor = FastVisionModel.from_pretrained(
        model_name=implementation["repo_id"],
        revision=implementation["revision"],
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )
    FastVisionModel.for_inference(model)
    load_seconds = time.perf_counter() - started

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "State in one short sentence that this is a bounded "
                        "local CUDA runtime qualification, not a benchmark."
                    ),
                }
            ],
        }
    ]
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=prompt,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")
    torch.cuda.reset_peak_memory_stats()
    generation_started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=48,
            do_sample=False,
            use_cache=True,
        )
    generation_seconds = time.perf_counter() - generation_started
    new_tokens = generated[:, inputs["input_ids"].shape[1] :]
    output = processor.batch_decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    if not output:
        raise QualificationError("model returned an empty generation")

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    report = {
        "schema": "szl.frontier-runtime-qualification/v1",
        "candidate_id": candidate["candidate_id"],
        "state": "MEASURED_RUNTIME_QUALIFIED",
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "host": platform.node() or "unknown-host",
        "python": sys.version.split()[0],
        "implementation": implementation,
        "gpu": {
            **gpu,
            "cuda_tensor_sum": int(tensor_sum),
            "peak_reserved_bytes_generation": torch.cuda.max_memory_reserved(),
        },
        "model": {
            "parameter_count": parameter_count,
            "load_in_4bit_requested": True,
            "load_seconds": round(load_seconds, 6),
        },
        "generation": {
            "seconds": round(generation_seconds, 6),
            "new_tokens": int(new_tokens.shape[-1]),
            "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
            "output": output,
        },
        "claim_boundary": (
            "This proves one local CUDA load and generation only. It is not "
            "training, evaluation, safety, quality, or publication evidence."
        ),
        "publication_eligible": False,
        "autonomy_eligible": False,
    }
    report["report_sha256"] = sha256_json(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--min-free-gib", type=float, default=4.0)
    parser.add_argument("--max-temp-c", type=int, default=80)
    args = parser.parse_args()
    try:
        candidate = load_candidate(args.candidate)
        report = qualify(
            candidate,
            min_free_gib=args.min_free_gib,
            max_temp_c=args.max_temp_c,
        )
        exit_code = 0
    except Exception as exc:  # noqa: BLE001 - always emit terminal evidence
        report = {
            "schema": "szl.frontier-runtime-qualification/v1",
            "state": "UNAVAILABLE",
            "measured_at": datetime.now(timezone.utc).isoformat(),
            "fatal": f"{type(exc).__name__}: {exc}",
            "publication_eligible": False,
            "autonomy_eligible": False,
        }
        report["report_sha256"] = sha256_json(report)
        exit_code = 1
    args.report.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
