#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Publish reviewed Chaski-family model cards as exact-source Hub commits.

A closed profile registry binds each public target to one source directory and
one qualification contract. Only ``README.md``, ``holo-banner.svg``, and a
deterministic ``szl-source-binding.json`` receipt are written. Model weights,
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

import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE_REPOSITORY = "szl-holdings/szl-forge"
FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
SOURCE_BINDING_PATH = "szl-source-binding.json"


@dataclass(frozen=True)
class CardProfile:
    """Immutable publication contract for one Chaski-family model target."""

    key: str
    repo_id: str
    source_directory: Path
    display_name: str
    report_schema: str
    evaluation_state: str
    release_blocker: str
    autonomy_eligible: bool
    required_card_boundaries: tuple[str, ...]
    forbidden_card_claims: tuple[str, ...]
    required_svg_boundaries: tuple[str, ...]
    commit_description: str
    required_card_tags: tuple[str, ...] = ()

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
            "autonomy_eligible": self.autonomy_eligible,
            "evaluation_state": self.evaluation_state,
            "release_blocker": self.release_blocker,
            "release_blocker_preserved": True,
        }


PROFILES: Mapping[str, CardProfile] = {
    "chaski": CardProfile(
        key="chaski",
        repo_id="SZLHOLDINGS/chaski",
        source_directory=ROOT / "chaski" / "card",
        display_name="Chaski",
        report_schema="szl.hf.chaski-card-publication/v2",
        evaluation_state="MEASURED_FAIL",
        release_blocker="json_draft=0/5;adversarial_refusal=2/6",
        autonomy_eligible=False,
        required_card_boundaries=(
            "base_model_relation: finetune",
            "artifact_class: MERGED_FINETUNE",
            "originality: FINETUNE_DISCLOSED_BASE",
            "publication_eligible: false",
            "json_draft: 0/5",
            "adversarial_refusal: 2/6",
            "Conjecture 1",
            "Failed qualification. Not flagship.",
        ),
        forbidden_card_claims=(
            "publication_eligible: true",
            "autonomy_eligible: true",
            "named-n-measured%20pass",
            "json_draft: 5/5",
            "adversarial_refusal: 6/6",
            "status: production ready",
        ),
        required_svg_boundaries=(
            'viewBox="0 0 1200 360"',
            'id="holo"',
            'id="glow"',
            'id="fade"',
            'stop-color="#22d3ee"',
            'stop-color="#818cf8"',
            'stop-color="#c084fc"',
            'stop-color="#f472b6"',
        ),
        commit_description=(
            "Exact-source Chaski card publication. Named-N remains a measured "
            "failed gate, publication_eligible and autonomy_eligible remain "
            "false, and no model or provider state is promoted. No weights, "
            "adapter, configs, evals, visibility, hardware, collection, or "
            "runtime state changed."
        ),
    ),
    "chaski-5050": CardProfile(
        key="chaski-5050",
        repo_id="SZLHOLDINGS/chaski-5050",
        source_directory=ROOT / "chaski-5050" / "card",
        display_name="Chaski-5050",
        report_schema="szl.hf.chaski-5050-card-publication/v1",
        evaluation_state="NONE_THIS_RUN",
        release_blocker="no_json_or_refusal_gate",
        autonomy_eligible=False,
        required_card_boundaries=(
            "base_model: Qwen/Qwen3.5-0.8B",
            "base_model_relation: adapter",
            "artifact_class: ADAPTER",
            "originality: FINETUNE_DISCLOSED_BASE",
            "job_id: local-5050",
            "weights: AVAILABLE",
            "evals: none-this-run",
            "publication_eligible: false",
            "autonomy_eligible: false",
            "never_overwrite: SZLHOLDINGS/chaski",
            "copied_live_chaski_weights: false",
            "train_loss: 2.228136855544466",
            "adapter_sha256: 620b3488fac2ebc6518090424de5b3c6a182293cf52dfd5bd9f886f54aef0df5",
            "> **QUARANTINE.** Research residue.",
            "Evaluation state: none-this-run; no evaluation score is claimed by this card.",
            "No signed held-out evaluation receipt for this 5050 adapter is present in this source tree.",
            "**Status: none-this-run.** No JSON/refusal gate ran. Not 5/5. Not 6/6.",
            "Card, banner, and source binding only; weights, adapter, configs, evals, visibility, hardware, collection, and runtime state unchanged",
            "Canonical GitHub: [`chaski/README_5050.md`]",
        ),
        forbidden_card_claims=(
            "publication_eligible: true",
            "autonomy_eligible: true",
            "nobody else ships this combination",
            "one-of-one",
            "world-class",
            "best in the world",
            "status: production ready",
            "named-n-measured%20pass",
            "evals: measured pass",
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
            "Exact-source Chaski-5050 card publication. Evaluation remains "
            "none-this-run; no JSON/refusal gate is claimed; publication_eligible "
            "and autonomy_eligible remain false. No weights, adapter, configs, "
            "evals, visibility, hardware, collection, or runtime state changed."
        ),
    ),
    "chaski-r2": CardProfile(
        key="chaski-r2",
        repo_id="SZLHOLDINGS/chaski-r2",
        source_directory=ROOT / "chaski-r2" / "card",
        display_name="Chaski-R2",
        required_card_tags=("proposal-only",),
        report_schema="szl.hf.chaski-r2-card-publication/v1",
        evaluation_state="NONE_THIS_RUN",
        release_blocker="no_json_or_refusal_gate",
        autonomy_eligible=False,
        required_card_boundaries=(
            "base_model: Qwen/Qwen3.5-0.8B",
            "base_model_relation: adapter",
            "artifact_class: ADAPTER",
            "sku: CHASKI-R2",
            "quant: bf16-lora",
            "qlora: false",
            "- proposal-only\n",
            "evals: none-this-run",
            "publication_eligible: false",
            "autonomy_eligible: false",
            "never_overwrite: SZLHOLDINGS/chaski",
            "Train loss is not a JSON-draft or refusal gate. Not 5/5 or 6/6.",
        ),
        forbidden_card_claims=(
            "publication_eligible: true",
            "autonomy_eligible: true",
            "nobody else ships this combination",
            "one-of-one",
            "status: production ready",
            "named-n-measured%20pass",
            "evals: measured pass",
        ),
        required_svg_boundaries=(
            'viewBox="0 0 1200 360"',
            'id="r2"',
            'id="glow"',
            'id="fade"',
            'stop-color="#8b5cf6"',
            'stop-color="#f472b6"',
        ),
        commit_description=(
            "Exact-source Chaski-R2 card publication with proposal-only search "
            "metadata. Evaluation remains none-this-run; publication_eligible "
            "and autonomy_eligible remain false. No weights, adapter, configs, "
            "evals, visibility, hardware, collection, or runtime state changed."
        ),
    ),
}
DEFAULT_PROFILE = "chaski"

# Backward-compatible aliases for existing callers.
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
    if selected.required_card_tags:
        try:
            metadata = yaml.safe_load(card.split("\n---\n", 1)[0][4:])
        except yaml.YAMLError as error:
            raise PublicationError("model card front matter is invalid") from error
        tags = metadata.get("tags") if isinstance(metadata, dict) else None
        if not isinstance(tags, list) or any(
            tag not in tags for tag in selected.required_card_tags
        ):
            raise PublicationError("required model-card search tags missing")
    for boundary in selected.required_card_boundaries:
        if boundary not in card:
            raise PublicationError(f"required model-card boundary missing: {boundary}")
    folded_card = card.casefold()
    for forbidden in selected.forbidden_card_claims:
        if forbidden.casefold() in folded_card:
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
            "collection": False,
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
            "collection_changed": False,
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
