"""Verify immutable artifacts from the mounted model revision or local cache."""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download


REPO = "SZLHOLDINGS/SZL-Khipu-1.5B-GGUF"
REVISION = "67d60ec577730747055491640cfb91fc4a4b5d25"
ARTIFACTS = {
    "SZL-Khipu-1.5B-Q4_K_M.gguf": (
        986_047_904,
        "13c1a1993063e1dff92f7413ccf48eaca6d48efc8801ae9af35961ae3396623a",
    ),
    "training_receipt.signed.json": (
        None,
        "7af76dd4f26dcd122012bfd1e47a0f55481a952b86aee28956cf7cfaaf59bd04",
    ),
    "eval_receipt.signed.json": (
        None,
        "32edd2d862fd5abac390bee3d30950f4718afedc41f4da4e24f3d0dfe67f8450",
    ),
    "owner_pubkey.json": (
        None,
        "843d0958392b4ee11ad8e36519261bebf841ee20caec479cbbc4bb9e8c991031",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


for filename, (expected_size, expected_sha) in ARTIFACTS.items():
    override = os.getenv("MODEL_DIR_OVERRIDE")
    if override:
        path = Path(override) / filename
    else:
        path = Path(
            hf_hub_download(
                repo_id=REPO,
                filename=filename,
                revision=REVISION,
                local_files_only=True,
                token=False,
            )
        )
    if expected_size is not None and path.stat().st_size != expected_size:
        raise SystemExit(f"size mismatch: {filename}")
    if sha256_file(path) != expected_sha:
        raise SystemExit(f"sha256 mismatch: {filename}")
    print(f"verified {filename}")

shutil.rmtree(Path.home() / ".cache" / "huggingface" / "xet", ignore_errors=True)
