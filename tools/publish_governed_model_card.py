#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Publish the closed set of governed SZL model cards through one source-bound lane."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

if __package__:
    from tools import publish_khipu_card as base
else:
    import publish_khipu_card as base

CHASKI_5050 = base.CardProfile(
    key="chaski-5050",
    repo_id="SZLHOLDINGS/chaski-5050",
    source_directory=base.ROOT / "chaski-5050" / "card",
    display_name="Chaski-5050",
    report_schema="szl.hf.chaski-5050-card-publication/v1",
    abstention_result="none-this-run",
    required_card_boundaries=(
        "publication_eligible: false",
        "autonomy_eligible: false",
        "artifact_class: ADAPTER",
        "jobs: local-5050",
        "evals: none-this-run",
        "copied_live_chaski_weights: false",
        "train_loss: 2.228136855544466",
        "620b3488fac2ebc6518090424de5b3c6a182293cf52dfd5bd9f886f54aef0df5",
        "**QUARANTINE.** Research residue.",
        "Adapters are on this repo. Evals none-this-run. Not MEASURED.",
        "**Does NOT overwrite**",
        "**Status: none-this-run.**",
        "Not live Chaski",
        "Lab load forbidden.",
        "Conjecture 1",
    ),
    forbidden_card_claims=(
        "publication_eligible: true",
        "autonomy_eligible: true",
        "copied_live_chaski_weights: true",
        "lifecycle: production",
        "evals: passed",
        "STATUS: PRODUCTION READY",
    ),
    required_svg_boundaries=(
        'viewBox="0 0 1200 360"',
        'id="c1"',
        'id="c2"',
        'id="glow"',
        'id="fade"',
        'stop-color="#22d3ee"',
        'stop-color="#e879a9"',
    ),
    commit_description=(
        "Exact-source Chaski-5050 card publication. The adapter remains "
        "QUARANTINE research residue with no evaluation this run; "
        "publication_eligible and autonomy_eligible remain false. No weights, "
        "adapter, configs, evals, visibility, hardware, or runtime state changed."
    ),
)

PROFILES: Mapping[str, base.CardProfile] = MappingProxyType(
    {**dict(base.PROFILES), CHASKI_5050.key: CHASKI_5050}
)
DEFAULT_PROFILE = base.DEFAULT_PROFILE
PublicationError = base.PublicationError


def resolve_profile(
    profile: str | base.CardProfile | None = None,
) -> base.CardProfile:
    """Resolve only the immutable governed profile registry."""
    if isinstance(profile, base.CardProfile):
        return profile
    key = str(profile or DEFAULT_PROFILE).strip().lower()
    try:
        return PROFILES[key]
    except KeyError as error:
        raise PublicationError(f"unknown card profile: {key}") from error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), default=DEFAULT_PROFILE)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args(argv)

    selected = resolve_profile(args.profile)
    source_revision = args.source_revision.strip().lower()
    if not base.FULL_SHA.fullmatch(source_revision):
        raise SystemExit("--source-revision must be a full lowercase commit SHA")

    assets = base.load_assets(selected)
    source_evidence = base.validate_assets(assets, selected)
    target_assets = base.build_target_assets(
        profile=selected,
        source_revision=source_revision,
        source_assets=assets,
        source_evidence=source_evidence,
    )
    target_evidence = base.evidence_for(target_assets)

    if not args.publish:
        base.write_report(
            args.report,
            base.build_report(
                state="DRY_RUN_VALIDATED",
                source_revision=source_revision,
                source_assets=source_evidence,
                target_assets=target_evidence,
                profile=selected,
                commit_created=None,
            ),
        )
        return 0

    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise SystemExit("HF_TOKEN is required for --publish")

    try:
        hub_revision, readback, publisher, commit_created = base.publish(
            token=token,
            source_revision=source_revision,
            assets=assets,
            profile=selected,
        )
        report = base.build_report(
            state="SOURCE_BOUND_READBACK_VERIFIED",
            source_revision=source_revision,
            source_assets=source_evidence,
            target_assets=readback,
            profile=selected,
            hub_revision=hub_revision,
            publisher=publisher,
            commit_created=commit_created,
        )
        base.write_report(args.report, report)
        print(
            json.dumps(
                {
                    "state": report["state"],
                    "profile": selected.key,
                    "source_revision": source_revision,
                    "hub_revision": hub_revision,
                    "commit_created": commit_created,
                    "files": sorted(readback),
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception as error:
        failure = base.build_report(
            state="FAILED",
            source_revision=source_revision,
            source_assets=source_evidence,
            target_assets=target_evidence,
            profile=selected,
        )
        failure.update(
            {
                "error_type": type(error).__name__,
                "error_sha256": base.sha256_bytes(str(error).encode("utf-8")),
            }
        )
        base.write_report(args.report, failure)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
