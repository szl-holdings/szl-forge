"""Pilot manifests. They name the pool fields and stay unlaunchable.

The swap-pool runner (szl-gpu-bridge ``tools/gpu/pool_runner.py``) admits a
run only when ``reviewed_jobspec`` is a ``jobspecs/*-reviewed.json`` file and
``reviewed_binding`` hashes match a signed envelope. These documents leave
both unset, so that gate refuses them. Caps, providers, and GPU-hour boxes
are filled in so a later review can attach a binding without inventing them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from training.chaski_0p8b_pilot.constants import (
    ALLOWED_PROVIDERS,
    CAPS_USD,
    GPU_H_BUDGET,
    OUTPUT_DATASET_REPO,
    OUTPUT_MODEL_REPO,
    PHASE1_GPU_H_EXPECTED,
    PHASE1_GPU_H_HARD,
    PHASE2_GPU_H_EXPECTED,
    PHASE2_GPU_H_HARD,
    STUDENT_ID,
    STUDENT_LICENSE,
    STUDENT_REVISION,
    TEACHER_ID,
    TEACHER_LICENSE_ID,
    TEACHER_REVISION,
    TRAIN_IMAGE,
)
from training.chaski_0p8b_pilot.prompts import select_prompts

ENTRYPOINT = "laptop/dispatcher.py"
QUEUE_SCHEMA = "szl.gpu-pool.queue/v1"


def _phase_common() -> dict[str, Any]:
    return {
        "launch_authorized": False,
        "hub_write": False,
        "train_image": TRAIN_IMAGE,
        "providers": list(ALLOWED_PROVIDERS),
        "caps": dict(CAPS_USD),
        "gpu_h_budget": GPU_H_BUDGET,
        "entrypoint": ENTRYPOINT,
        "reviewed_jobspec": None,
        "signed_envelope": None,
        "reviewed_binding": None,
    }


def phase1(prompt_report: dict[str, Any] | None = None) -> dict[str, Any]:
    report = prompt_report if prompt_report is not None else select_prompts()
    inventory = [dict(row) for row in report["inventory"]]
    return {
        "schema": "szl.chaski-0p8b-pilot.phase/v1",
        "phase": 1,
        "name": "teacher-generation",
        "run_id": "chaski-0p8b-pilot-teacher",
        "model": TEACHER_ID,
        "teacher": {"id": TEACHER_ID, "revision": TEACHER_REVISION, "license_id": TEACHER_LICENSE_ID},
        "student": {"id": STUDENT_ID, "revision": STUDENT_REVISION, "license": STUDENT_LICENSE},
        "dataset_repo": OUTPUT_DATASET_REPO,
        "enable_thinking": False,
        "guardrails": {
            "nvidia_guardrails": "kept",
            "license_id": TEACHER_LICENSE_ID,
            "system_prompt": "preserve",
        },
        "admitted_prompts": report["admitted_prompts"],
        "heldout_user_prompts": report["heldout_user_prompts"],
        "prompt_inventory": inventory,
        "max_new_tokens": 256,
        "vram_gb": 24,
        "gpu_h_expected": PHASE1_GPU_H_EXPECTED,
        "max_hours": PHASE1_GPU_H_HARD,
        "plateau": {"k": 4, "threshold": 0.01, "min_evals": 5},
        "wave": {
            "schema": "szl.forge.wave-config/v1",
            "kind": "distill",
            "teacher": {"id": TEACHER_ID, "revision": TEACHER_REVISION},
            "student": {"id": STUDENT_ID, "revision": STUDENT_REVISION},
            "launch_authorized": False,
        },
        **_phase_common(),
    }


def phase2() -> dict[str, Any]:
    return {
        "schema": "szl.chaski-0p8b-pilot.phase/v1",
        "phase": 2,
        "name": "student-lora-sft",
        "run_id": "chaski-0p8b-pilot-student",
        "model": STUDENT_ID,
        "student": {"id": STUDENT_ID, "revision": STUDENT_REVISION, "license": STUDENT_LICENSE},
        "teacher": {"id": TEACHER_ID, "revision": TEACHER_REVISION, "license_id": TEACHER_LICENSE_ID},
        "output_repo": OUTPUT_MODEL_REPO,
        "dataset_repo": OUTPUT_DATASET_REPO,
        "dataset_revision": None,
        "method": "lora16",
        "lora_r": 16,
        "lora_alpha": 32,
        "epochs": 3,
        "lr": 0.0002,
        "seqlen": 1024,
        "batch_size": 1,
        "grad_accum": 4,
        "seed": 11,
        "assistant_only_loss": True,
        "vram_gb": 16,
        "gpu_h_expected": PHASE2_GPU_H_EXPECTED,
        "max_hours": PHASE2_GPU_H_HARD,
        "plateau": {"k": 3, "threshold": 0.01, "min_evals": 4},
        "wave": {
            "schema": "szl.forge.wave-config/v1",
            "kind": "sft",
            "method": "lora16",
            "student": {"id": STUDENT_ID, "revision": STUDENT_REVISION},
            "teacher": {"id": TEACHER_ID, "revision": TEACHER_REVISION},
            "dataset": {"id": OUTPUT_DATASET_REPO, "revision": None, "split": "train"},
            "target_repo": OUTPUT_MODEL_REPO,
            "local_only": True,
            "launch_authorized": False,
        },
        **_phase_common(),
    }


def _run(phase: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": phase["run_id"],
        "model": phase["model"],
        "entrypoint": ENTRYPOINT,
        "reviewed_jobspec": None,
        "signed_envelope": None,
        "reviewed_binding": None,
        "vram_gb": phase["vram_gb"],
        "max_hours": phase["max_hours"],
        "plateau": phase["plateau"],
        "caps": dict(CAPS_USD),
        "allowed_providers": list(ALLOWED_PROVIDERS),
        "launch_authorized": False,
        "gpu_h_expected": phase["gpu_h_expected"],
    }


def pool_queue(phase_one: dict[str, Any], phase_two: dict[str, Any]) -> dict[str, Any]:
    hard = phase_one["max_hours"] + phase_two["max_hours"]
    if hard > GPU_H_BUDGET:
        raise RuntimeError(f"hard GPU-h {hard} exceeds the {GPU_H_BUDGET} GPU-h approval")
    return {
        "schema": QUEUE_SCHEMA,
        "launch_authorized": False,
        "note": (
            "Field names match szl.gpu-pool.queue/v1. reviewed_jobspec and "
            "reviewed_binding are unset, so pool_runner.validate_run_entry "
            "refuses the run before a child starts."
        ),
        "allowed_providers": list(ALLOWED_PROVIDERS),
        "caps": dict(CAPS_USD),
        "gpu_h_hard_total": hard,
        "gpu_h_budget": GPU_H_BUDGET,
        "runs": [_run(phase_one), _run(phase_two)],
    }


def pool_blockers(run: dict[str, Any]) -> list[str]:
    """Refusal codes the swap-pool runner emits for this unreviewed run."""

    codes: list[str] = []
    if run.get("entrypoint") != ENTRYPOINT:
        codes.append("RUN_ENTRYPOINT_NOT_DISPATCHER")
    jobspec = run.get("reviewed_jobspec")
    if not isinstance(jobspec, str) or not jobspec:
        codes.append("RUN_JOBSPEC_MISSING")
    elif not (jobspec.startswith("jobspecs/") and jobspec.endswith("-reviewed.json")):
        codes.append("RUN_JOBSPEC_NOT_REVIEWED")
    if not isinstance(run.get("reviewed_binding"), dict):
        codes.append("RUN_BINDING_MISSING")
    if run.get("launch_authorized") is not False:
        codes.append("LAUNCH_FLAG_NOT_CLOSED")
    if tuple(run.get("allowed_providers") or ()) != ALLOWED_PROVIDERS:
        codes.append("PROVIDER_NOT_ALLOWED")
    if run.get("caps") != CAPS_USD:
        codes.append("CAPS_MISMATCH")
    return codes


def build_all(root: Path | None = None) -> dict[str, dict[str, Any]]:
    report = select_prompts(root)
    one = phase1(report)
    two = phase2()
    return {"phase1": one, "phase2": two, "queue": pool_queue(one, two)}


def write_manifests(directory: Path, root: Path | None = None) -> dict[str, dict[str, Any]]:
    documents = build_all(root)
    directory.mkdir(parents=True, exist_ok=True)
    names = {
        "phase1": "phase1-teacher.json",
        "phase2": "phase2-student.json",
        "queue": "pool-queue.json",
    }
    for key, name in names.items():
        text = json.dumps(documents[key], indent=2, sort_keys=True) + "\n"
        (directory / name).write_text(text, encoding="utf-8")
    return documents


def main() -> int:
    directory = Path(__file__).resolve().parent / "manifests"
    write_manifests(directory)
    print(f"wrote unlaunchable manifests under {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
