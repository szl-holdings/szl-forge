#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Publish the six governed kernel-mirror cards with exact Hub readback.

The publisher has a closed target registry and a closed two-file write set.
It never deletes Hub files, changes model weights, alters runtime settings, or
promotes any artifact. Publication is fail-closed on malformed source assets,
target drift, concurrent Hub updates, missing credentials, or byte mismatch.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

SOURCE_REPOSITORY = "szl-holdings/szl-forge"
TARGET_REPO_TYPE = "model"
FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
MAX_README_BYTES = 512 * 1024
MAX_SVG_BYTES = 2 * 1024 * 1024
WRITE_SET: tuple[tuple[str, str], ...] = (
    ("README.md", "README.md"),
    ("card/holo-banner.svg", "card/holo-banner.svg"),
)


class PublicationError(RuntimeError):
    """Raised when publication cannot be proven safe and exact."""


@dataclass(frozen=True)
class Profile:
    name: str
    source_dir: str
    repo_id: str


@dataclass(frozen=True)
class Asset:
    source_path: Path
    relative_source_path: str
    path_in_repo: str
    data: bytes

    def evidence(self) -> dict[str, Any]:
        return {
            "source_path": self.relative_source_path,
            "path_in_repo": self.path_in_repo,
            "bytes": len(self.data),
            "sha256": hashlib.sha256(self.data).hexdigest(),
        }


PROFILES: dict[str, Profile] = {
    "szl-blocked": Profile(
        name="szl-blocked",
        source_dir="szl-blocked",
        repo_id="SZLHOLDINGS/szl-blocked",
    ),
    "szl-formulas": Profile(
        name="szl-formulas",
        source_dir="szl-formulas",
        repo_id="SZLHOLDINGS/szl-formulas",
    ),
    "szl-govsign": Profile(
        name="szl-govsign",
        source_dir="szl-govsign",
        repo_id="SZLHOLDINGS/szl-govsign",
    ),
    "szl-invariants": Profile(
        name="szl-invariants",
        source_dir="szl-invariants",
        repo_id="SZLHOLDINGS/szl-invariants",
    ),
    "szl-ouroboros": Profile(
        name="szl-ouroboros",
        source_dir="szl-ouroboros",
        repo_id="SZLHOLDINGS/szl-ouroboros",
    ),
    "szl-provctl": Profile(
        name="szl-provctl",
        source_dir="szl-provctl",
        repo_id="SZLHOLDINGS/szl-provctl",
    ),
}


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def resolve_profile(name: str) -> Profile:
    try:
        return PROFILES[name]
    except KeyError as error:
        raise PublicationError(f"unknown closed profile: {name}") from error


def validate_source_revision(revision: str) -> str:
    normalized = str(revision or "").strip()
    if FULL_SHA_RE.fullmatch(normalized) is None:
        raise PublicationError(
            "source revision must be an exact lowercase 40-character Git SHA"
        )
    return normalized


def safe_file(root: Path, relative: str) -> Path:
    resolved_root = root.resolve()
    path = (resolved_root / relative).resolve()
    if path == resolved_root or resolved_root not in path.parents or not path.is_file():
        raise PublicationError(
            f"source asset is missing or outside the repository root: {relative}"
        )
    return path


def _frontmatter(text: str) -> str:
    if not text.startswith("---\n"):
        raise PublicationError("README must begin with YAML frontmatter")
    end = text.find("\n---\n", 4)
    if end < 0:
        raise PublicationError("README YAML frontmatter is not terminated")
    return text[4:end]


def validate_readme(profile: Profile, payload: bytes) -> None:
    if not payload or len(payload) > MAX_README_BYTES:
        raise PublicationError("README size is outside the closed publication bound")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise PublicationError("README is not valid UTF-8") from error

    frontmatter = _frontmatter(text)
    if re.search(r"(?m)^library_name:\s*kernels\s*$", frontmatter) is None:
        raise PublicationError("README must declare library_name: kernels")
    if re.search(r"(?m)^license:\s*apache-2\.0\s*$", frontmatter) is None:
        raise PublicationError("README must declare license: apache-2.0")

    expected_banner = (
        f"https://raw.githubusercontent.com/{SOURCE_REPOSITORY}/main/"
        f"{profile.source_dir}/card/holo-banner.svg"
    )
    if expected_banner not in text:
        raise PublicationError("README is not bound to its exact Forge banner path")

    for other in PROFILES.values():
        if other.name == profile.name:
            continue
        foreign_banner = (
            f"https://raw.githubusercontent.com/{SOURCE_REPOSITORY}/main/"
            f"{other.source_dir}/card/holo-banner.svg"
        )
        if foreign_banner in text:
            raise PublicationError("README contains another mirror's banner path")

    if "\x00" in text:
        raise PublicationError("README contains a NUL byte")
    if re.search(r"(?i)<script\b|javascript\s*:", text):
        raise PublicationError("README contains active script content")


def _local_name(value: str) -> str:
    return value.rsplit("}", 1)[-1].lower()


def validate_svg(payload: bytes) -> None:
    if not payload or len(payload) > MAX_SVG_BYTES:
        raise PublicationError("SVG size is outside the closed publication bound")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise PublicationError("SVG is not valid UTF-8") from error

    lowered = text.lower()
    forbidden_fragments = (
        "<script",
        "<foreignobject",
        "<!doctype",
        "<!entity",
        "<?xml-stylesheet",
        "javascript:",
        "url(http:",
        "url(https:",
        "@import",
        "expression(",
    )
    if any(fragment in lowered for fragment in forbidden_fragments):
        raise PublicationError("SVG contains forbidden active or external content")

    try:
        root = ET.fromstring(text)
    except ET.ParseError as error:
        raise PublicationError("SVG is not well-formed XML") from error
    if _local_name(root.tag) != "svg":
        raise PublicationError("SVG root element is not <svg>")

    for node in root.iter():
        for raw_name, raw_value in node.attrib.items():
            name = _local_name(raw_name)
            value = str(raw_value).strip()
            if name.startswith("on"):
                raise PublicationError("SVG event-handler attributes are forbidden")
            if name in {"href", "src"} and value and not value.startswith("#"):
                raise PublicationError(
                    "SVG may reference only local fragment identifiers"
                )


def collect_assets(root: Path, profile: Profile) -> list[Asset]:
    assets: list[Asset] = []
    for local_path, path_in_repo in WRITE_SET:
        relative = f"{profile.source_dir}/{local_path}"
        source_path = safe_file(root, relative)
        payload = source_path.read_bytes()
        if local_path == "README.md":
            validate_readme(profile, payload)
        elif local_path.endswith(".svg"):
            validate_svg(payload)
        else:  # pragma: no cover - protected by the closed constant
            raise PublicationError("closed write set contains an unsupported asset")
        assets.append(
            Asset(
                source_path=source_path,
                relative_source_path=relative,
                path_in_repo=path_in_repo,
                data=payload,
            )
        )

    if {asset.path_in_repo for asset in assets} != {
        target for _, target in WRITE_SET
    }:
        raise PublicationError(
            "resolved asset destinations drifted from the closed write set"
        )
    return assets


def _download_bytes(
    download_fn: Callable[..., str],
    *,
    repo_id: str,
    path_in_repo: str,
    revision: str,
    token: str,
) -> bytes:
    downloaded = Path(
        download_fn(
            repo_id=repo_id,
            filename=path_in_repo,
            repo_type=TARGET_REPO_TYPE,
            revision=revision,
            token=token,
        )
    )
    return downloaded.read_bytes()


def current_matches(
    *,
    repo_id: str,
    revision: str,
    sibling_paths: Iterable[str],
    assets: list[Asset],
    token: str,
    download_fn: Callable[..., str],
) -> bool:
    observed_paths = set(sibling_paths)
    for asset in assets:
        if asset.path_in_repo not in observed_paths:
            return False
        observed = _download_bytes(
            download_fn,
            repo_id=repo_id,
            path_in_repo=asset.path_in_repo,
            revision=revision,
            token=token,
        )
        if observed != asset.data:
            return False
    return True


def publication_report(
    *,
    profile: Profile,
    source_revision: str,
    assets: list[Asset],
    status: str,
    before_revision: str | None = None,
    after_revision: str | None = None,
    changed: bool | None = None,
    credential_source: str | None = None,
    error: BaseException | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema": "szl.kernel-mirror-card-publication/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "source": {
            "repository": SOURCE_REPOSITORY,
            "revision": source_revision,
        },
        "target": {
            "profile": profile.name,
            "repo_id": profile.repo_id,
            "repo_type": TARGET_REPO_TYPE,
        },
        "scope": {
            "authority": "CARD_ONLY",
            "write_set": [asset.path_in_repo for asset in assets],
            "deletes_allowed": False,
            "model_weights_changed": False,
            "runtime_authority_changed": False,
            "promotion_effect": "NONE",
        },
        "files": [asset.evidence() for asset in assets],
        "publication": {
            "before_revision": before_revision,
            "after_revision": after_revision,
            "changed": changed,
            "exact_readback": status
            in {"PUBLISHED_AND_VERIFIED", "NO_CHANGE_VERIFIED"},
        },
        "credential": {
            "source": credential_source,
            "token_persisted": False,
            "token_logged": False,
        },
    }
    if error is not None:
        message = str(error)
        report["failure"] = {
            "type": type(error).__name__,
            "message_sha256": sha256_bytes(message.encode("utf-8")),
        }
    return report


def publish(
    *,
    profile: Profile,
    source_revision: str,
    assets: list[Asset],
    token: str,
    credential_source: str | None,
) -> dict[str, Any]:
    try:
        from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
    except ImportError as error:  # pragma: no cover - publisher workflow owns it
        raise PublicationError(
            "huggingface_hub is required for publication"
        ) from error

    if not token.strip():
        raise PublicationError("HF_TOKEN is required for publication")

    api = HfApi(token=token)
    before = api.model_info(
        profile.repo_id,
        files_metadata=True,
        token=token,
    )
    before_revision = str(before.sha or "")
    if FULL_SHA_RE.fullmatch(before_revision) is None:
        raise PublicationError("Hub did not return an exact target revision")
    sibling_paths = {
        str(getattr(sibling, "rfilename", ""))
        for sibling in (before.siblings or [])
        if getattr(sibling, "rfilename", None)
    }

    unchanged = current_matches(
        repo_id=profile.repo_id,
        revision=before_revision,
        sibling_paths=sibling_paths,
        assets=assets,
        token=token,
        download_fn=hf_hub_download,
    )
    if unchanged:
        after_revision = before_revision
        status = "NO_CHANGE_VERIFIED"
        changed = False
    else:
        operations = [
            CommitOperationAdd(
                path_in_repo=asset.path_in_repo,
                path_or_fileobj=io.BytesIO(asset.data),
            )
            for asset in assets
        ]
        commit = api.create_commit(
            repo_id=profile.repo_id,
            repo_type=TARGET_REPO_TYPE,
            operations=operations,
            commit_message=(
                f"cards: mirror {profile.name} from "
                f"{SOURCE_REPOSITORY}@{source_revision}"
            ),
            parent_commit=before_revision,
            token=token,
        )
        after_revision = str(getattr(commit, "oid", "") or "")
        if FULL_SHA_RE.fullmatch(after_revision) is None:
            refreshed = api.model_info(profile.repo_id, token=token)
            after_revision = str(refreshed.sha or "")
        if FULL_SHA_RE.fullmatch(after_revision) is None:
            raise PublicationError(
                "Hub publication did not return an exact revision"
            )
        status = "PUBLISHED_AND_VERIFIED"
        changed = True

    for asset in assets:
        observed = _download_bytes(
            hf_hub_download,
            repo_id=profile.repo_id,
            path_in_repo=asset.path_in_repo,
            revision=after_revision,
            token=token,
        )
        if observed != asset.data:
            raise PublicationError(
                f"exact Hub readback failed for {asset.path_in_repo}"
            )

    return publication_report(
        profile=profile,
        source_revision=source_revision,
        assets=assets,
        status=status,
        before_revision=before_revision,
        after_revision=after_revision,
        changed=changed,
        credential_source=credential_source,
    )


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(payload), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, choices=sorted(PROFILES))
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args(argv)

    profile = resolve_profile(args.profile)
    assets: list[Asset] = []
    try:
        revision = validate_source_revision(args.source_revision)
        assets = collect_assets(args.root, profile)
        if args.publish:
            report = publish(
                profile=profile,
                source_revision=revision,
                assets=assets,
                token=os.environ.get("HF_TOKEN", ""),
                credential_source=os.environ.get("HF_TOKEN_SOURCE"),
            )
        else:
            report = publication_report(
                profile=profile,
                source_revision=revision,
                assets=assets,
                status="VALIDATED_NOT_PUBLISHED",
            )
        write_report(args.report, report)
        print(
            f"{profile.name}: {report['status']} "
            f"source={revision} target={profile.repo_id}"
        )
        return 0
    except Exception as error:
        revision = str(args.source_revision or "").strip()
        failure = publication_report(
            profile=profile,
            source_revision=revision,
            assets=assets,
            status="FAILED_CLOSED",
            credential_source=os.environ.get("HF_TOKEN_SOURCE"),
            error=error,
        )
        write_report(args.report, failure)
        print(
            f"::error::{profile.name} publication failed closed: "
            f"{type(error).__name__}"
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
