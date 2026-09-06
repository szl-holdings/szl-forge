#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Publish reviewed Khipu-family model cards as exact-source Hub commits.

A closed profile registry binds each public target to one source directory and
one qualification contract.  Only ``README.md``, ``holo-banner.svg``, and a
deterministic ``szl-source-binding.json`` receipt are written.  Model weights,
adapters, configs, evaluation evidence, visibility, hardware, collections, and
runtime state are outside this publisher's authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

ROOT = Path(__file__).resolve().parents[1]
SOURCE_REPOSITORY = "szl-holdings/szl-forge"
FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
SOURCE_BINDING_PATH = "szl-source-binding.json"


@dataclass(frozen=True)
class CardProfile:
    """Immutable publication contract for one model-card target."""

    key: str
    repo_id: str
    source_directory: Path
    display_name: str
    report_schema: str
    abstention_result: str
    required_card_boundaries: tuple[str, ...]
    forbidden_card_claims: tuple[str, ...]
    required_svg_boundaries: tuple[str, ...]
    commit_description: str

    @property
    def source_files(self) -> dict[str, Path]:
        return {
            "README.md": self.source_directory / "README.md",
            "holo-banner.svg": self.source_directory / "holo-banner.svg",
        }

    @property
    def qualification(self) -> dict[str, Any]:
        return {
            "publication_eligible": False,
            "abstention_result": self.abstention_result,
            "release_blocker_preserved": True,
        }


PROFILES: Mapping[str, CardProfile] = {
    "khipu": CardProfile(
        key="khipu",
        repo_id="SZLHOLDINGS/SZL-Khipu-1.5B",
        source_directory=ROOT / "khipu" / "card",
        display_name="Khipu",
        report_schema="szl.hf.khipu-card-publication/v2",
        abstention_result="2/6",
        required_card_boundaries=(
            "publication_eligible: false",
            "STATUS: TRAINED + OWNER-EVALUATED on a small synthetic harness.",
            "keyId `89540347a69b789e`",
            "| plan-valid | 11 / 11 |",
            "| grounding | 4 / 5 |",
            "| abstain | 2 / 6 |",
            "| hallucinated citations | 0 |",
            "6f9f5b9df2a877c999e33faf542dc6e62ce63f4a2bf6b358fc48a4b6b113c3c9",
            "0a71b3a28b9f77ca3651f38c8caa1e34121934f5584dae24454d4c6eea823a66",
            "The 2/6 abstention result is a visible release blocker",
            "No deployed Alloy endpoint status is asserted by this card.",
            "contentAccess=HANDLES_ONLY",
            "Conjecture 1",
        ),
        forbidden_card_claims=(
            "publication_eligible: true",
            "STATUS: PRODUCTION READY",
            "abstain-6%2F6%20PASS",
            "| abstain | 6 / 6 |",
        ),
        required_svg_boundaries=(
            'viewBox="0 0 1200 360"',
            'id="khipu"',
            'id="glow"',
            'id="fade"',
            'stop-color="#3AF4C8"',
            'stop-color="#C9B787"',
        ),
        commit_description=(
            "Exact-source Khipu card publication. The owner-evaluated 2/6 "
            "abstention result remains a visible release blocker and "
            "publication_eligible remains false. No weights, adapter, configs, "
            "evals, visibility, hardware, or runtime state changed."
        ),
    ),
    "khipu-r2": CardProfile(
        key="khipu-r2",
        repo_id="SZLHOLDINGS/KHIPU-R2",
        source_directory=ROOT / "khipu-r2" / "card",
        display_name="KHIPU-R2",
        report_schema="szl.hf.khipu-r2-card-publication/v1",
        abstention_result="3/6",
        required_card_boundaries=(
            "publication_eligible: false",
            "autonomy_eligible: false",
            "research-only",
            "Adapters are on this repo. Abstain is MEASURED 3/6, not a pass.",
            "No signed R2 eval receipt in this atelier.",
            "| plan-valid | **11 / 11** |",
            "| grounding (`eval.jsonl` navigate) | **5 / 5** |",
            "| abstain (`adversarial.jsonl`) | **3 / 6** |",
            "| hallucinated citations | **0** |",
            "held_out_in_gradients: false",
            "e44d53f29f2d443598e06d6c0441557fd3a5010888c7aa97b56ec3c0e050d349",
            "Not a replacement for `SZL-Khipu-1.5B`",
            "GPU **UNAVAILABLE**",
        ),
        forbidden_card_claims=(
            "publication_eligible: true",
            "autonomy_eligible: true",
            "lifecycle: production",
            "abstain-6%2F6%20PASS",
            "| abstain (`adversarial.jsonl`) | **6 / 6** |",
        ),
        required_svg_boundaries=(
            'viewBox="0 0 1200 360"',
            'id="kr2"',
            'id="glow"',
            'id="fade"',
            'stop-color="#0d9488"',
            'stop-color="#3AF4C8"',
            'stop-color="#a7f3d0"',
        ),
        commit_description=(
            "Exact-source KHIPU-R2 card publication. The measured 3/6 "
            "abstention result remains explicitly not a pass; publication_eligible "
            "and autonomy_eligible remain false. No weights, adapter, configs, "
            "evals, visibility, hardware, or runtime state changed."
        ),
    ),
}
DEFAULT_PROFILE = "khipu"

# Backward-compatible aliases for callers that import the original constants.
REPO_ID = PROFILES[DEFAULT_PROFILE].repo_id
SOURCE_FILES = PROFILES[DEFAULT_PROFILE].source_files
REQUIRED_CARD_BOUNDARIES = PROFILES[DEFAULT_PROFILE].required_card_boundaries
FORBIDDEN_CARD_CLAIMS = PROFILES[DEFAULT_PROFILE].forbidden_card_claims


class PublicationError(RuntimeError):
    """Exact-source card publication could not be completed safely."""


def resolve_profile(profile: str | CardProfile | None = None) -> CardProfile:
    if isinstance(profile, CardProfile):
        return profile
    key = str(profile or DEFAULT_PROFILE).strip().lower()
    try:
        return PROFILES[key]
    except KeyError as error:
        raise PublicationError(f"unknown card profile: {key}") from error


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def evidence_for(assets: Mapping[str, bytes]) -> dict[str, dict[str, Any]]:
    return {
        target: {"bytes": len(value), "sha256": sha256_bytes(value)}
        for target, value in sorted(assets.items())
    }


def load_assets(
    profile: str | CardProfile | None = None,
) -> dict[str, bytes]:
    selected = resolve_profile(profile)
    assets: dict[str, bytes] = {}
    for target, source in selected.source_files.items():
        if not source.is_file() or source.is_symlink():
            raise PublicationError(f"source asset missing or unsafe: {source}")
        assets[target] = source.read_bytes()
    return assets


def validate_assets(
    assets: Mapping[str, bytes],
    profile: str | CardProfile | None = None,
) -> dict[str, dict[str, Any]]:
    selected = resolve_profile(profile)
    if set(assets) != set(selected.source_files):
        raise PublicationError("source asset set drifted")
    try:
        card = assets["README.md"].decode("utf-8")
        banner = assets["holo-banner.svg"].decode("utf-8")
    except UnicodeDecodeError as error:
        raise PublicationError("card assets must be UTF-8") from error

    if not card.startswith("---\n") or card.count("\n---\n") < 1:
        raise PublicationError("model card front matter is missing")
    for boundary in selected.required_card_boundaries:
        if boundary not in card:
            raise PublicationError(f"required model-card boundary missing: {boundary}")
    for forbidden in selected.forbidden_card_claims:
        if forbidden in card:
            raise PublicationError(f"forbidden qualification claim present: {forbidden}")
    if len(assets["README.md"]) > 100_000:
        raise PublicationError("model card exceeded the bounded publication size")

    normalized_banner = banner.lstrip()
    if not normalized_banner.startswith("<svg"):
        raise PublicationError("holographic banner is not an SVG document")
    for required in selected.required_svg_boundaries:
        if required not in banner:
            raise PublicationError(f"required SVG contract missing: {required}")
    lowered = banner.lower()
    namespace = 'xmlns="http://www.w3.org/2000/svg"'
    if lowered.count(namespace) != 1:
        raise PublicationError("canonical SVG namespace must appear exactly once")
    remote_scan = lowered.replace(namespace, "", 1)
    for forbidden in ("<script", "javascript:", "http://", "https://"):
        if forbidden in remote_scan:
            raise PublicationError(f"unsafe or remote SVG content present: {forbidden}")
    if len(assets["holo-banner.svg"]) > 10_000:
        raise PublicationError("holographic banner exceeded the bounded publication size")

    return evidence_for(assets)


def build_source_binding(
    *,
    profile: str | CardProfile,
    source_revision: str,
    source_assets: Mapping[str, Mapping[str, Any]],
) -> bytes:
    selected = resolve_profile(profile)
    if not FULL_SHA.fullmatch(source_revision):
        raise PublicationError("source binding requires a full lowercase commit SHA")
    payload = {
        "schema": "szl.hf-card-source-binding/v1",
        "profile": selected.key,
        "source": {
            "repository": SOURCE_REPOSITORY,
            "revision": source_revision,
            "assets": {
                path: dict(metadata)
                for path, metadata in sorted(source_assets.items())
            },
        },
        "target": {
            "repo_id": selected.repo_id,
            "repo_type": "model",
            "controlled_paths": [
                "README.md",
                "holo-banner.svg",
                SOURCE_BINDING_PATH,
            ],
        },
        "qualification": selected.qualification,
        "authority": {
            "weights": False,
            "adapter": False,
            "configs": False,
            "evals": False,
            "visibility": False,
            "hardware": False,
            "runtime": False,
        },
    }
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def build_target_assets(
    *,
    profile: str | CardProfile,
    source_revision: str,
    source_assets: Mapping[str, bytes],
    source_evidence: Mapping[str, Mapping[str, Any]],
) -> dict[str, bytes]:
    target_assets = dict(source_assets)
    target_assets[SOURCE_BINDING_PATH] = build_source_binding(
        profile=profile,
        source_revision=source_revision,
        source_assets=source_evidence,
    )
    return target_assets


def _retry_after_seconds(error: BaseException, default: int) -> int:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", {}) or {}
    raw = str(headers.get("Retry-After") or "").strip()
    try:
        seconds = int(raw)
    except ValueError:
        seconds = default
    return max(1, min(seconds, 180))


def publish_with_bounded_retry(
    operation: Callable[[], Any],
    *,
    attempts: int = 3,
    sleeper: Callable[[float], None] = time.sleep,
) -> Any:
    if attempts < 1 or attempts > 5:
        raise ValueError("attempts must be between one and five")
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as error:
            response = getattr(error, "response", None)
            status = getattr(response, "status_code", None)
            if status != 429 or attempt == attempts:
                raise
            sleeper(_retry_after_seconds(error, default=30 * attempt))
    raise AssertionError("bounded retry loop exhausted without a result")


def _readback_bytes(
    *,
    repo_id: str,
    revision: str,
    token: str,
    paths: Mapping[str, bytes],
) -> dict[str, bytes]:
    from huggingface_hub import hf_hub_download

    observed: dict[str, bytes] = {}
    for target in sorted(paths):
        local = hf_hub_download(
            repo_id=repo_id,
            repo_type="model",
            filename=target,
            revision=revision,
            token=token,
            force_download=True,
        )
        observed[target] = Path(local).read_bytes()
    return observed


def _current_target_matches(
    *,
    api: Any,
    token: str,
    profile: CardProfile,
    target_assets: Mapping[str, bytes],
) -> tuple[str, dict[str, bytes]] | None:
    info = api.repo_info(repo_id=profile.repo_id, repo_type="model")
    revision = str(getattr(info, "sha", "") or "").strip().lower()
    if not FULL_SHA.fullmatch(revision):
        raise PublicationError("Hub repository did not expose an exact revision")
    files = set(
        api.list_repo_files(
            repo_id=profile.repo_id,
            repo_type="model",
            revision=revision,
        )
    )
    if not set(target_assets).issubset(files):
        return None
    observed = _readback_bytes(
        repo_id=profile.repo_id,
        revision=revision,
        token=token,
        paths=target_assets,
    )
    if observed != dict(target_assets):
        return None
    return revision, observed


def publish(
    *,
    token: str,
    source_revision: str,
    assets: Mapping[str, bytes],
    profile: str | CardProfile | None = None,
) -> tuple[str, dict[str, dict[str, Any]], str, bool]:
    from huggingface_hub import CommitOperationAdd, HfApi

    selected = resolve_profile(profile)
    source_evidence = validate_assets(assets, selected)
    target_assets = build_target_assets(
        profile=selected,
        source_revision=source_revision,
        source_assets=assets,
        source_evidence=source_evidence,
    )

    api = HfApi(token=token)
    api.auth_check(repo_id=selected.repo_id, repo_type="model", write=True)
    identity = api.whoami()
    publisher = str((identity or {}).get("name") or "").strip()
    if not publisher:
        raise PublicationError("publisher identity is unavailable")

    current = _current_target_matches(
        api=api,
        token=token,
        profile=selected,
        target_assets=target_assets,
    )
    if current is not None:
        revision, observed = current
        return revision, evidence_for(observed), publisher, False

    operations = [
        CommitOperationAdd(path_in_repo=target, path_or_fileobj=value)
        for target, value in sorted(target_assets.items())
    ]
    commit = publish_with_bounded_retry(
        lambda: api.create_commit(
            repo_id=selected.repo_id,
            repo_type="model",
            operations=operations,
            commit_message=(
                f"docs: publish {selected.display_name} card from "
                f"szl-forge@{source_revision}"
            ),
            commit_description=selected.commit_description,
        )
    )
    revision = str(getattr(commit, "oid", "") or "").strip().lower()
    if not FULL_SHA.fullmatch(revision):
        raise PublicationError("Hub commit did not return an exact revision")

    observed = _readback_bytes(
        repo_id=selected.repo_id,
        revision=revision,
        token=token,
        paths=target_assets,
    )
    if observed != target_assets:
        mismatches = sorted(
            target
            for target in target_assets
            if observed.get(target) != target_assets[target]
        )
        raise PublicationError(
            "Hub byte readback mismatch: " + ", ".join(mismatches)
        )
    return revision, evidence_for(observed), publisher, True


def build_report(
    *,
    state: str,
    source_revision: str,
    source_assets: Mapping[str, Mapping[str, Any]],
    profile: str | CardProfile | None = None,
    target_assets: Mapping[str, Mapping[str, Any]] | None = None,
    hub_revision: str | None = None,
    publisher: str | None = None,
    commit_created: bool | None = None,
) -> dict[str, Any]:
    selected = resolve_profile(profile)
    controlled_files = ["README.md", "holo-banner.svg", SOURCE_BINDING_PATH]
    return {
        "schema": selected.report_schema,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "state": state,
        "profile": selected.key,
        "source": {
            "repository": SOURCE_REPOSITORY,
            "revision": source_revision,
            "assets": dict(source_assets),
        },
        "target": {
            "repo_id": selected.repo_id,
            "repo_type": "model",
            "revision": hub_revision,
            "assets": dict(target_assets) if target_assets is not None else None,
            "commit_created": commit_created,
        },
        "publisher_identity": publisher,
        "authority": {
            "files": controlled_files,
            "weights_changed": False,
            "adapter_changed": False,
            "configs_changed": False,
            "evals_changed": False,
            "visibility_changed": False,
            "hardware_changed": False,
            "runtime_changed": False,
        },
        "qualification": selected.qualification,
        "secret_values_recorded": False,
    }


def write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), default=DEFAULT_PROFILE)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args(argv)

    selected = resolve_profile(args.profile)
    source_revision = args.source_revision.strip().lower()
    if not FULL_SHA.fullmatch(source_revision):
        raise SystemExit("--source-revision must be a full lowercase commit SHA")
    assets = load_assets(selected)
    source_evidence = validate_assets(assets, selected)
    target_assets = build_target_assets(
        profile=selected,
        source_revision=source_revision,
        source_assets=assets,
        source_evidence=source_evidence,
    )
    target_evidence = evidence_for(target_assets)

    if not args.publish:
        write_report(
            args.report,
            build_report(
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
        hub_revision, readback, publisher, commit_created = publish(
            token=token,
            source_revision=source_revision,
            assets=assets,
            profile=selected,
        )
        report = build_report(
            state="SOURCE_BOUND_READBACK_VERIFIED",
            source_revision=source_revision,
            source_assets=source_evidence,
            target_assets=readback,
            profile=selected,
            hub_revision=hub_revision,
            publisher=publisher,
            commit_created=commit_created,
        )
        write_report(args.report, report)
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
        failure = build_report(
            state="FAILED",
            source_revision=source_revision,
            source_assets=source_evidence,
            target_assets=target_evidence,
            profile=selected,
        )
        failure.update(
            {
                "error_type": type(error).__name__,
                "error_sha256": sha256_bytes(str(error).encode("utf-8")),
            }
        )
        write_report(args.report, failure)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
