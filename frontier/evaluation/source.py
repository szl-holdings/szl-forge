# SPDX-License-Identifier: Apache-2.0
"""Source identity verification for bounded frontier evaluation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


class SourceVerificationError(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SourceVerificationError(f"{path} must contain a JSON object")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def verify_source(
    attestation: Mapping[str, Any], token: str, cache: Path
) -> dict[str, Any]:
    """Re-fetch only attested metadata files; never download weight shards."""
    from huggingface_hub import HfApi, hf_hub_download

    model_id = str(attestation["model_id"])
    revision = str(attestation["requested_revision"])
    api = HfApi(token=token)
    info = api.model_info(model_id, revision=revision, files_metadata=True)
    python_files = sorted(
        sibling.rfilename
        for sibling in info.siblings
        if sibling.rfilename.endswith(".py")
    )
    expected = dict(attestation["metadata_files"])
    expected[str(attestation["license"]["file"])] = str(
        attestation["license"]["sha256"]
    )
    observed: dict[str, dict[str, Any]] = {}
    cache.mkdir(parents=True, exist_ok=True)
    for name, expected_sha in sorted(expected.items()):
        path = Path(
            hf_hub_download(
                model_id,
                filename=name,
                revision=revision,
                token=token,
                cache_dir=cache,
            )
        )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        observed[name] = {
            "sha256": digest,
            "bytes": path.stat().st_size,
            "expected_sha256": expected_sha,
            "pass": digest == expected_sha,
        }
    passed = (
        str(info.sha) == revision
        and not python_files
        and all(item["pass"] for item in observed.values())
        and attestation["license"]["compatibility_for_bounded_evaluation"]
        == "PASS"
    )
    return {
        "pass": passed,
        "model_id": model_id,
        "requested_revision": revision,
        "resolved_revision": str(info.sha),
        "python_files": python_files,
        "files": observed,
        "candidate_weights_downloaded": False,
        "repository_code_executed": False,
    }
