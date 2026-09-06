#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Publish reviewed Chaski-family cards as exact-source Hub commits.

A closed profile registry binds each target to one source directory and one
qualification contract. Only ``README.md``, ``holo-banner.svg``, and a
deterministic ``szl-source-binding.json`` receipt are writable. Model weights,
adapters, configs, evaluations, visibility, hardware, collections, and runtime
state remain outside this publisher's authority.
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
SOURCE_BINDING_PATH = "szl-source-binding.json"
FULL_SHA = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class CardProfile:
    """Immutable publication contract for one Chaski-family target."""

    key: str
    repo_id: str
    source_directory: Path
    display_name: str
    report_schema: str
    evaluation_state: str
    required_card_boundaries: tuple[str, ...]
    forbidden_card_claims: tuple[str, ...]
    required_svg_boundaries: tuple[str, ...]
    commit_description: str
    evidence_path: Path | None = None
    required_evidence_boundaries: tuple[str, ...] = ()

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
            "autonomy_eligible": False,
            "evaluation_state": self.evaluation_state,
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
        required_card_boundaries=(
            "base_model_relation: finetune",
            "artifact_class: MERGED_FINETUNE",
            "originality: FINETUNE_DISCLOSED_BASE",
            "publication_eligible: false",
            "json_draft: 0/5",
            "adversarial_refusal: 2/6",
            "Named-N: MEASURED FAIL",
            "Failed qualification. Not flagship.",
            "Not an autonomous agent",
            "Conjecture 1",
        ),
        forbidden_card_claims=(
            "publication_eligible: true",
            "Named-N-MEASURED%20PASS",
            "json_draft: 5/5",
            "adversarial_refusal: 6/6",
            "STATUS: PRODUCTION READY",
        ),
        required_svg_boundaries=(
            'viewBox="0 0 1200 360"',
            'id="holo"',
            'id="glow"',
            'id="fade"',
        ),
        commit_description=(
            "Exact-source Chaski card publication. Named-N remains a measured "
            "failed gate and publication_eligible remains false. No weights, "
            "adapter, configs, evals, visibility, hardware, or runtime changed."
        ),
    ),
    "chaski-5050": CardProfile(
        key="chaski-5050",
        repo_id="SZLHOLDINGS/chaski-5050",
        source_directory=ROOT / "chaski-5050" / "card",
        display_name="Chaski-5050",
        report_schema="szl.hf.chaski-5050-card-publication/v1",
        evaluation_state="NONE_THIS_RUN",
        required_card_boundaries=(
            "base_model_relation: adapter",
            "artifact_class: ADAPTER",
            "publication_eligible: false",
            "autonomy_eligible: false",
            "evals: none-this-run",
            "never_overwrite: SZLHOLDINGS/chaski",
            "copied_live_chaski_weights: false",
            "QUARANTINE",
            "Status: none-this-run.",
            "No signed mix-ablation in this atelier.",
            "Publishing this card is a documentation update, not model promotion",
            "620b3488fac2ebc6518090424de5b3c6a182293cf52dfd5bd9f886f54aef0df5",
            "Not production. Lab load forbidden.",
            "Conjecture 1",
        ),
        forbidden_card_claims=(
            "publication_eligible: true",
            "autonomy_eligible: true",
            "evals: measured",
            "STATUS: PRODUCTION READY",
            "Nobody else ships this combination",
            "one-of-one",
            "Not 5/5 unless that receipt measures it. PASS",
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
            "none-this-run; publication_eligible and autonomy_eligible remain "
            "false. No weights, adapter, configs, evals, visibility, hardware, "
            "or runtime changed."
        ),
        evidence_path=ROOT / "chaski" / "README_5050.md",
        required_evidence_boundaries=(
            "Separate SKU.",
            "Not live `SZLHOLDINGS/chaski`.",
            "Not an HF Job.",
            "`publication_eligible: false`",
            "SKU eval none-this-run.",
            "No Hub PUT from this checkout.",
        ),
    ),
}
DEFAULT_PROFILE = "chaski"

# Compatibility aliases for existing imports and callers.
REPO_ID = PROFILES[DEFAULT_PROFILE].repo_id
SOURCE_FILES = PROFILES[DEFAULT_PROFILE].source_files
REQUIRED_CARD_BOUNDARIES = PROFILES[DEFAULT_PROFILE].required_card_boundaries


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
        path: {"bytes": len(value), "sha256": sha256_bytes(value)}
        for path, value in sorted(assets.items())
    }


def load_assets(profile: str | CardProfile | None = None) -> dict[str, bytes]:
    selected = resolve_profile(profile)
    assets: dict[str, bytes] = {}
    for target, source in selected.source_files.items():
        if not source.is_file() or source.is_symlink():
            raise PublicationError(f"source asset missing or unsafe: {source}")
        assets[target] = source.read_bytes()
    return assets


def load_evidence(profile: str | CardProfile | None = None) -> bytes | None:
    selected = resolve_profile(profile)
    if selected.evidence_path is None:
        return None
    if not selected.evidence_path.is_file() or selected.evidence_path.is_symlink():
        raise PublicationError(f"canonical evidence missing or unsafe: {selected.evidence_path}")
    return selected.evidence_path.read_bytes()


def validate_assets(
    assets: Mapping[str, bytes],
    profile: str | CardProfile | None = None,
    evidence: bytes | None = None,
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
        if forbidden.casefold() in card.casefold():
            raise PublicationError(f"forbidden qualification claim present: {forbidden}")
    if len(assets["README.md"]) > 100_000:
        raise PublicationError("model card exceeded the bounded publication size")

    if selected.evidence_path is not None:
        evidence = load_evidence(selected) if evidence is None else evidence
        try:
            evidence_text = evidence.decode("utf-8")
        except UnicodeDecodeError as error:
            raise PublicationError("canonical evidence must be UTF-8") from error
        for boundary in selected.required_evidence_boundaries:
            if boundary not in evidence_text:
                raise PublicationError(f"canonical evidence boundary missing: {boundary}")

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
    evidence_sha256: str | None,
) -> bytes:
    selected = resolve_profile(profile)
    if not FULL_SHA.fullmatch(source_revision):
        raise PublicationError("source binding requires a full lowercase commit SHA")
    payload: dict[str, Any] = {
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
    if selected.evidence_path is not None:
        payload["source"]["canonical_evidence"] = {
            "path": str(selected.evidence_path.relative_to(ROOT)),
            "sha256": evidence_sha256,
        }
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def build_target_assets(
    *,
    profile: str | CardProfile,
    source_revision: str,
    source_assets: Mapping[str, bytes],
    source_evidence: Mapping[str, Mapping[str, Any]],
    evidence_sha256: str | None,
) -> dict[str, bytes]:
    target_assets = dict(source_assets)
    target_assets[SOURCE_BINDING_PATH] = build_source_binding(
        profile=profile,
        source_revision=source_revision,
        source_assets=source_evidence,
        evidence_sha256=evidence_sha256,
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
    canonical_evidence = load_evidence(selected)
    source_evidence = validate_assets(assets, selected, canonical_evidence)
    evidence_sha256 = (
        sha256_bytes(canonical_evidence) if canonical_evidence is not None else None
    )
    target_assets = build_target_assets(
        profile=selected,
        source_revision=source_revision,
        source_assets=assets,
        source_evidence=source_evidence,
        evidence_sha256=evidence_sha256,
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
        CommitOperationAdd(path_in_repo=path, path_or_fileobj=value)
        for path, value in sorted(target_assets.items())
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
        raise PublicationError("Hub exact-revision byte readback failed")
    return revision, evidence_for(observed), publisher, True


def build_report(
    *,
    state: str,
    profile: str | CardProfile,
    source_revision: str,
    source_assets: Mapping[str, Mapping[str, Any]],
    evidence_sha256: str | None,
    hub_revision: str | None = None,
    hub_assets: Mapping[str, Mapping[str, Any]] | None = None,
    publisher: str | None = None,
    commit_created: bool | None = None,
) -> dict[str, Any]:
    selected = resolve_profile(profile)
    source: dict[str, Any] = {
        "repository": SOURCE_REPOSITORY,
        "revision": source_revision,
        "assets": dict(source_assets),
    }
    if selected.evidence_path is not None:
        source["canonical_evidence"] = {
            "path": str(selected.evidence_path.relative_to(ROOT)),
            "sha256": evidence_sha256,
        }
    return {
        "schema": selected.report_schema,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "state": state,
        "profile": selected.key,
        "source": source,
        "target": {
            "repo_id": selected.repo_id,
            "repo_type": "model",
            "revision": hub_revision,
            "assets": dict(hub_assets) if hub_assets is not None else None,
            "commit_created": commit_created,
        },
        "publisher_identity": publisher,
        "qualification": selected.qualification,
        "authority": {
            "files": ["README.md", "holo-banner.svg", SOURCE_BINDING_PATH],
            "weights_changed": False,
            "adapter_changed": False,
            "configs_changed": False,
            "evals_changed": False,
            "visibility_changed": False,
            "hardware_changed": False,
            "runtime_changed": False,
        },
        "secret_values_recorded": False,
    }


def write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    canonical_evidence = load_evidence(selected)
    source_evidence = validate_assets(assets, selected, canonical_evidence)
    evidence_sha256 = (
        sha256_bytes(canonical_evidence) if canonical_evidence is not None else None
    )

    if not args.publish:
        write_report(
            args.report,
            build_report(
                state="DRY_RUN_VALIDATED",
                profile=selected,
                source_revision=source_revision,
                source_assets=source_evidence,
                evidence_sha256=evidence_sha256,
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
            profile=selected,
            source_revision=source_revision,
            source_assets=source_evidence,
            evidence_sha256=evidence_sha256,
            hub_revision=hub_revision,
            hub_assets=readback,
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
        report = build_report(
            state="FAILED",
            profile=selected,
            source_revision=source_revision,
            source_assets=source_evidence,
            evidence_sha256=evidence_sha256,
        )
        report.update(
            {
                "error_type": type(error).__name__,
                "error_sha256": sha256_bytes(str(error).encode("utf-8")),
            }
        )
        write_report(args.report, report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
