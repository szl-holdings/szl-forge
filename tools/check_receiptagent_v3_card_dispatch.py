#!/usr/bin/env python3
"""Verify a first-attempt manual public-card intent before reading credentials."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import publish_receiptagent_v3_public_card as writer  # noqa: E402

WORKFLOW_PATH = ".github/workflows/publish-receiptagent-v3-public-card.yml"


class DispatchError(ValueError):
    """The manual release intent is absent or does not match this source."""


def validate_dispatch(environment: Mapping[str, str]) -> dict:
    required = {
        "GITHUB_EVENT_NAME": "workflow_dispatch",
        "GITHUB_REPOSITORY": writer.SOURCE_REPOSITORY,
        "GITHUB_REF": "refs/heads/main",
        "GITHUB_RUN_ATTEMPT": "1",
        "CARD_EXPECTED_HUB_PARENT": writer.card.HUB_PARENT,
        "CARD_CONFIRMATION": "README_ONLY_LOADER_WITHDRAWAL",
    }
    for name, expected in required.items():
        if environment.get(name) != expected:
            raise DispatchError(f"manual card intent rejected: {name}")
    source = environment.get("GITHUB_SHA", "")
    if writer.FULL_SHA.fullmatch(source) is None:
        raise DispatchError("manual card intent requires an exact source SHA")
    if environment.get("CARD_SOURCE_REVISION") != source:
        raise DispatchError("manual card intent does not match the dispatched source")
    writer.assert_current_main(source)
    writer.assert_writer_files_committed(source)
    return {
        "schema": "szl.receiptagent-v3-public-card-dispatch/v1",
        "state": "MANUAL_INTENT_AND_SOURCE_VERIFIED",
        "source_repository": writer.SOURCE_REPOSITORY,
        "source_revision": source,
        "target_repository": writer.TARGET_REPOSITORY,
        "expected_hub_parent": writer.card.HUB_PARENT,
        "changed_paths": ["README.md"],
        "candidate_kind": writer.card.CANDIDATE_KIND,
        "commit_attempted": False,
        "release_status": "UNQUALIFIED",
    }


def main() -> int:
    try:
        result = validate_dispatch(os.environ)
    except (DispatchError, writer.PublicationError) as error:
        print(json.dumps({
            "state": "BLOCKED_NO_WRITE",
            "failure_type": type(error).__name__,
            "reason": str(error),
            "commit_attempted": False,
        }, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
