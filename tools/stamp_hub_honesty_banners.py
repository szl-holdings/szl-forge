#!/usr/bin/env python3
"""Stamp Hub honesty banners using HF_TOKEN already acquired."""
from __future__ import annotations

import os
import sys
from pathlib import Path

BANNERS = {
    "SZLHOLDINGS/szl-receiptagent-qwen35-0.8b-v3": (
        "> **NON-RELEASE / PLACEHOLDER.** Not a ReceiptAgent tournament winner.\n"
        "> `publication_eligible=false`. `autonomy_eligible=false`.\n"
        "> Do not treat this ID as flagship.\n\n"
    ),
    "SZLHOLDINGS/SZL-Khipu-1.5B-abstain": (
        "> **EXPERIMENT. Adapter bytes missing or unverified.**\n"
        "> Evaluators use `SZLHOLDINGS/SZL-Khipu-1.5B` until a receipted adapter exists.\n\n"
    ),
    "SZLHOLDINGS/chaski": (
        "> **RESEARCH / NEGATIVE EVIDENCE.** Failed qualification. Not flagship.\n"
        "> Later SKU `A11OY-MINI` inherits this failed parent. Not a product claim.\n\n"
    ),
    "SZLHOLDINGS/chaski-5050": (
        "> **QUARANTINE.** Research residue. Strip owner-machine absolute paths.\n"
        "> Not flagship. Not a production checkpoint.\n\n"
    ),
    "SZLHOLDINGS/A11OY-MINI": (
        "> **GGUF of a failed parent (chaski).** Not flagship. Not a11oy production.\n\n"
    ),
    "SZLHOLDINGS/governed-inference-meter": (
        "> **DEPRECATED.** Successor is GitHub `szl-holdings/szl-energy-attest`.\n"
        "> This Hub card is not operational energy evidence.\n\n"
    ),
}


def _prepend(readme: str, banner: str) -> str:
    if banner.strip() in readme:
        return readme
    if readme.startswith("---"):
        end = readme.find("\n---", 3)
        if end != -1:
            split = end + 4
            return readme[:split] + "\n\n" + banner + readme[split:].lstrip("\n")
    return banner + readme


def main() -> int:
    token = str(os.environ.get("HF_TOKEN") or "").strip()
    if not token:
        print("UNAVAILABLE: HF_TOKEN missing after credential acquire")
        return 2
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=token)
    for repo_id, banner in BANNERS.items():
        try:
            path = hf_hub_download(
                repo_id=repo_id,
                filename="README.md",
                repo_type="model",
                token=token,
            )
            readme = Path(path).read_text(encoding="utf-8")
        except Exception as exc:
            print(f"SKIP_READ {repo_id} {type(exc).__name__}")
            continue
        updated = _prepend(readme, banner)
        if updated == readme:
            print(f"ALREADY {repo_id}")
            continue
        api.upload_file(
            path_or_fileobj=updated.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="honesty: stamp NON-RELEASE / RESEARCH banner",
        )
        print(f"STAMPED {repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
