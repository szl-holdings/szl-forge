"""Pins for the unlaunched Chaski 0.8B pilot.

Student and teacher revisions were read from the Hugging Face model API
(``sha`` plus ``cardData.license`` / ``license_name``) on 2026-11-27. The
student LICENSE file at that revision is the Apache License 2.0. The teacher
card names ``nvidia-nemotron-open-model-license``. The NOTICE sentence is the
one the Nemotron trainer lane already pins; this module does not restate a
second copy of the Nemotron base table.
"""

from __future__ import annotations

STUDENT_ID = "Qwen/Qwen3.5-0.8B"
STUDENT_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
STUDENT_LICENSE = "apache-2.0"
STUDENT_ARCHITECTURE = "Qwen3_5ForConditionalGeneration"

TEACHER_ID = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
TEACHER_REVISION = "dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f"
TEACHER_LICENSE_ID = "nvidia-nemotron-open-model-license"
TEACHER_LICENSE_URL = (
    "https://www.nvidia.com/en-us/agreements/enterprise-software/"
    "nvidia-nemotron-open-model-license/"
)

# New repos. Both were absent on the Hub on 2026-11-27. They are not
# SZLHOLDINGS/chaski and not SZLHOLDINGS/chaski-r4.
OUTPUT_MODEL_REPO = "SZLHOLDINGS/chaski-0p8b-pilot"
OUTPUT_DATASET_REPO = "SZLHOLDINGS/chaski-0p8b-pilot-sft"
FORBIDDEN_OUTPUT_REPOS = frozenset(
    {
        "szlholdings/chaski",
        "szlholdings/chaski-r4",
    }
)

TRAIN_IMAGE = (
    "ghcr.io/szl-holdings/szl-train-pinned"
    "@sha256:8eb919654e0da406d41a2e307e7705cce8f6b79ad0be9556d74d4c4640e63235"
)
IMAGE_PACKAGES = {
    "transformers": "5.22.2",
    "trl": "1.17.1",
    "peft": "0.21.2",
}

# Stephen's Q3 approval is 16 GPU-h for the pilot. Hard boxes are the pool
# time limits. Expected hours are the planning estimate, not a measurement.
GPU_H_BUDGET = 16
PHASE1_GPU_H_EXPECTED = 0.5
PHASE1_GPU_H_HARD = 2
PHASE2_GPU_H_EXPECTED = 1.0
PHASE2_GPU_H_HARD = 4

CAPS_USD = {"per_run_usd": 100, "per_day_usd": 500, "per_month_usd": 5000}
ALLOWED_PROVIDERS = ("salad", "massed")

# Live Chaski card named-N, revision 1c55df8652e9d0f7b84356b1e2d54849165ae884,
# dated 2026-08-28. publication_eligible stays false.
BASELINE_REPO = "SZLHOLDINGS/chaski"
BASELINE_REVISION = "1c55df8652e9d0f7b84356b1e2d54849165ae884"
BASELINE_JSON_DRAFT = (0, 5)
BASELINE_ADVERSARIAL = (2, 6)
BASELINE_REPORT_SHA256 = (
    "4d057eb9867285e69b00222be110bbb660330a96fe7b284a4d7f488268a13e05"
)

# Go bar versus that live card: JSON drafts must clear the failed 0/5, and
# refusals must not fall below 2/6. Five-of-five on the family leaderboard
# is a research comparison, not this pilot's go threshold.
GO_JSON_DRAFT_MIN = 3
GO_ADVERSARIAL_MIN = 2

HOLDOUT_FILES = (
    "chaski/gate/json_drafts.n5.jsonl",
    "chaski/gate/adversarial_refusals.n6.jsonl",
)
HOLDOUT_SHA256 = {
    "chaski/gate/json_drafts.n5.jsonl": (
        "dd093403e4aeeb906c169d8b6d8229115b1d170a0226afa939c1beaa6fa5e9be"
    ),
    "chaski/gate/adversarial_refusals.n6.jsonl": (
        "1e58dda9cf07e22406f0f953fddc2c26397fb5e784c27516578d133035cb006d"
    ),
}


def refuse_forbidden_repo(repo_id: object) -> str:
    """Return ``repo_id`` when it is a new repo. Refuse the live Chaski ids."""

    if not isinstance(repo_id, str) or repo_id.count("/") != 1:
        raise RuntimeError("output repo id must be org/name")
    folded = repo_id.casefold()
    if folded in FORBIDDEN_OUTPUT_REPOS:
        raise RuntimeError(f"{repo_id} is an existing Chaski repo and is not a pilot output")
    return repo_id
