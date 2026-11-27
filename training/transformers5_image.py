# SPDX-License-Identifier: Apache-2.0
"""The one transformers 5 train-image pin.

``kallpa/nemotron-h-trainer`` (pull request 804) owns the Nemotron-H trainer
that runs on this image. Q5–Q8 reuse the digest. This module does not copy
that trainer, its LoRA targets, or a second image.
"""

from __future__ import annotations

from pathlib import Path

TRAIN_IMAGE = (
    "ghcr.io/szl-holdings/szl-train-pinned"
    "@sha256:8eb919654e0da406d41a2e307e7705cce8f6b79ad0be9556d74d4c4640e63235"
)
TRANSFORMERS = "5.22.2"
PEFT = "0.21.2"
TRL = "1.17.1"
REJECTED_TRAINER = "trl-0.23.1-unsloth"
NEMOTRON_LANE = "kallpa/nemotron-h-trainer"


def nemotron_trainer_present(root: Path | None = None) -> bool:
    """True only when the parallel lane's module is actually in the tree."""

    if root is None:
        root = Path(__file__).resolve().parents[1]
    return (root / "training" / "nemotron_wave.py").is_file()
