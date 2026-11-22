#!/usr/bin/env python3
"""Anonymous read-only live revision + immutable source parity for one SZL Space.

A positive result proves only that declared source bytes match Hub repository
bytes. It is not deployment, image, signer, model execution or release proof.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

import check_public_space_release_parity as parity

SPACE = "SZLHOLDINGS/szl-model-inference-lab"
METADATA_URL = "https://huggingface.co/api/spaces/" + SPACE + "/runtime"
METADATA_LIMIT = 65536


def _reject_nonfinite(value: Any) -> None:
    """Reject ambiguous JSON numbers, including overflowed exponent values."""
    if isinstance(value, float) and not math.isfinite(value):
        raise parity.ValidationError("NONFINITE_METADATA_NUMBER")
    if isinstance(value, dict):
        for child in value.values():
            _reject_nonfinite(child)
    elif isinstance(value, list):
        for child in value:
            _reject_nonfinite(child)


def observe(
    source_dir: Path,
    fetch: Callable[[str, int], bytes] = parity.fetch_public_bytes,
    compare: Callable[..., dict[str, Any]] = parity.verify,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": "szl.model-inference-lab-public-parity/v1",
        "space_id": SPACE,
        "observed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "provider_stage": None,
        "hf_repository_revision": None,
        "state": "UNAVAILABLE",
        "reason": "METADATA_UNAVAILABLE",
        "parity": None,
        "source_match_only": False,
        "model_inference_verified": False,
        "served_image_verified": False,
        "signer_verified": False,
        "release_admitted": False,
        "provider_writes": False,
    }
    try:
        metadata = parity.unique_json(fetch(METADATA_URL, METADATA_LIMIT))
        _reject_nonfinite(metadata)
    except parity.RemoteUnavailable as exc:
        result["reason"] = "RUNTIME_METADATA_" + exc.code
        return result
    except (parity.ValidationError, TypeError, ValueError, RecursionError):
        result["reason"] = "RUNTIME_METADATA_INVALID"
        return result
    if not isinstance(metadata, dict) or type(metadata.get("stage")) is not str:
        result["reason"] = "RUNTIME_METADATA_INVALID"
        return result
    result["provider_stage"] = metadata["stage"]
    if metadata["stage"] != "RUNNING":
        result["state"] = "HOLD"
        result["reason"] = "PROVIDER_NOT_RUNNING"
        return result
    revision = metadata.get("sha")
    if type(revision) is not str or not parity.SHA40.fullmatch(revision):
        result["state"] = "HOLD"
        result["reason"] = "IMMUTABLE_HF_REVISION_UNAVAILABLE"
        return result
    result["hf_repository_revision"] = revision
    # The existing comparer bounds all reads and examines the exact 26-file
    # source-release manifest. Its returned booleans cannot authorize release.
    try:
        witness = compare(source_dir, SPACE, revision)
    except (parity.ValidationError, parity.RemoteUnavailable, OSError):
        result["reason"] = "PARITY_OBSERVER_UNAVAILABLE"
        return result
    result["parity"] = witness
    if witness.get("space_repository_parity") is True and witness.get("state") == "REPOSITORY_BYTES_MATCH_ONLY":
        result["state"] = "REPOSITORY_BYTES_MATCH_ONLY"
        result["reason"] = "HUB_REPOSITORY_BYTES_MATCH_NOT_RUNTIME_PROOF"
        result["source_match_only"] = True
    elif witness.get("state") == "DRIFT":
        result["state"] = "DRIFT"
        result["reason"] = witness.get("reason", "SOURCE_DRIFT")
    else:
        result["state"] = "HOLD"
        result["reason"] = witness.get("reason", "PARITY_UNAVAILABLE")
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", required=True, type=Path)
    a = p.parse_args(argv)
    report = observe(a.source_dir)
    sys.stdout.write(json.dumps(report, sort_keys=True, indent=2) + "\n")
    return 0 if report["source_match_only"] else 1 if report["state"] == "DRIFT" else 2


if __name__ == "__main__":
    raise SystemExit(main())
