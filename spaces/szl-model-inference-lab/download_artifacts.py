"""Fetch or verify the exact immutable runtime artifacts.

The Docker build is the only caller allowed to use ``--fetch``. It copies
verified regular bytes out of the temporary Hugging Face cache so the final
runtime can stay offline and cache-independent.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from pathlib import Path
from typing import Sequence

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


def artifact_path(filename: str, *, fetch: bool) -> Path:
    override = os.getenv("MODEL_DIR_OVERRIDE")
    if override:
        if fetch:
            raise RuntimeError(
                "MODEL_DIR_OVERRIDE is not permitted for the immutable build fetch"
            )
        return Path(override) / filename
    return Path(
        hf_hub_download(
            repo_id=REPO,
            filename=filename,
            revision=REVISION,
            local_files_only=not fetch,
            token=False,
        )
    )


def verify_artifacts(
    *, fetch: bool = False, output_dir: Path | None = None
) -> tuple[Path, ...]:
    if output_dir is not None:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise RuntimeError("output directory must be empty")
        output_dir.mkdir(parents=True, exist_ok=True)
    verified: list[Path] = []
    for filename, (expected_size, expected_sha) in ARTIFACTS.items():
        path = artifact_path(filename, fetch=fetch)
        if expected_size is not None and path.stat().st_size != expected_size:
            raise RuntimeError(f"size mismatch: {filename}")
        if sha256_file(path) != expected_sha:
            raise RuntimeError(f"sha256 mismatch: {filename}")
        if output_dir is not None:
            target = output_dir / filename
            staging = output_dir / f".{filename}.partial"
            try:
                staging.unlink(missing_ok=True)
                shutil.copyfile(path, staging)
                if (
                    expected_size is not None
                    and staging.stat().st_size != expected_size
                ):
                    raise RuntimeError(f"copied size mismatch: {filename}")
                if sha256_file(staging) != expected_sha:
                    raise RuntimeError(f"copied sha256 mismatch: {filename}")
                staging.chmod(0o444)
                staging.replace(target)
            finally:
                staging.unlink(missing_ok=True)
            path = target
        print(f"verified {filename}")
        verified.append(path)
    if output_dir is not None:
        output_names = {entry.name for entry in output_dir.iterdir()}
        if output_names != set(ARTIFACTS):
            raise RuntimeError("output directory does not match the artifact lock")
        if any(entry.is_symlink() or not entry.is_file() for entry in output_dir.iterdir()):
            raise RuntimeError("output directory contains a non-regular artifact")
        output_dir.chmod(0o555)
    return tuple(verified)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="allow exact-revision public Hub downloads during the image build",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="copy verified regular bytes into this immutable image directory",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.fetch and args.output_dir is None:
        raise SystemExit("--fetch requires --output-dir")
    if args.output_dir is not None and not args.fetch:
        raise SystemExit("--output-dir requires --fetch")
    verify_artifacts(fetch=args.fetch, output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
