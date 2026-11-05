#!/usr/bin/env python3
"""Check the committed Hub/source alignment record. No network and no Hub write.

The scheduled Omen card job fails because three public READMEs lost
``<!-- szl:artifact-identity-reconciled -->`` on 2026-11-04. This tool only
proves the Git record of that divergence still matches the files in this
checkout, and that the drift gate is still fail-closed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "hub-source-alignment-2026-11-05.json"
WORKFLOW = ROOT / ".github" / "workflows" / "omen-cards-identity-note.yml"
SYNC_TOOL = ROOT / "tools" / "prepare_receiptagent_v3_card_sync.py"
SHA256 = __import__("re").compile(r"^[0-9a-f]{64}$")
SHA40 = __import__("re").compile(r"^[0-9a-f]{40}$")
FORBIDDEN = ("continue-on-error", "skip the drift", "disable the gate", "exit 0")


class AlignmentError(ValueError):
    """The alignment record does not match this checkout."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_manifest(path: Path = MANIFEST) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != "szl.hub-source-alignment/v1":
        raise AlignmentError("unexpected alignment schema")
    if document.get("hf_writes") is not False:
        raise AlignmentError("alignment record must not authorize a Hub write")
    items = document.get("items")
    if not isinstance(items, list) or not items:
        raise AlignmentError("alignment record has no items")
    return document


def verify(root: Path = ROOT) -> dict:
    """Return the checked record or raise AlignmentError."""
    document = load_manifest(root / MANIFEST.relative_to(ROOT))
    marker = document.get("identity_marker")
    if marker != "<!-- szl:artifact-identity-reconciled -->":
        raise AlignmentError("identity marker changed")
    workflow = (root / WORKFLOW.relative_to(ROOT)).read_text(encoding="utf-8")
    if "sys.exit(1 if drift else 0)" not in workflow:
        raise AlignmentError("Omen card drift gate is no longer fail-closed")
    if "continue-on-error" in workflow:
        raise AlignmentError("Omen card workflow gained continue-on-error")
    sync = (root / SYNC_TOOL.relative_to(ROOT)).read_text(encoding="utf-8")
    seen = set()
    for item in document["items"]:
        repo = item.get("hub_repo")
        if repo in seen or not isinstance(repo, str) or not repo.startswith("SZLHOLDINGS/"):
            raise AlignmentError(f"bad hub repo row: {repo}")
        seen.add(repo)
        if item.get("alignment") != "HUB_DIVERGED":
            raise AlignmentError(f"{repo} is not an open divergence")
        for field in ("observed_hub_revision", "last_aligned_hub_revision"):
            if SHA40.fullmatch(str(item.get(field, ""))) is None:
                raise AlignmentError(f"{repo} {field} is not a commit sha")
        for field in ("observed_readme_sha256", "last_aligned_readme_sha256"):
            if SHA256.fullmatch(str(item.get(field, ""))) is None:
                raise AlignmentError(f"{repo} {field} is not a sha256")
        if item.get("observed_readme_has_marker") is not False:
            raise AlignmentError(f"{repo} observation no longer records a missing marker")
        checks = item.get("acceptance")
        if not isinstance(checks, list) or not checks or any(not isinstance(check, str) or not check.strip() for check in checks):
            raise AlignmentError(f"{repo} has no acceptance check")
        blob = "\n".join(checks).lower()
        if any(phrase in blob for phrase in FORBIDDEN):
            raise AlignmentError(f"{repo} acceptance weakens a gate")
        if marker not in blob:
            raise AlignmentError(f"{repo} acceptance does not name the identity marker")
        source = item.get("canonical_source")
        if source is not None:
            path = root / source["path"]
            if source.get("sha256") is not None:
                digest = _sha256(path.read_bytes())
                if digest != source["sha256"]:
                    raise AlignmentError(f"{source['path']} hash drifted from the alignment record")
                if repo.endswith("/WILLAY") and marker.encode() not in path.read_bytes():
                    raise AlignmentError("WILLAY source card lost the identity marker")
        pins = item.get("card_sync_pins")
        if pins:
            for name in ("hub_readme_sha256", "candidate_readme_sha256"):
                if pins[name] not in sync:
                    raise AlignmentError(f"card-sync pin {name} moved")
            if item["observed_readme_sha256"] in sync:
                raise AlignmentError("card-sync tool now accepts the marker-less README")
    if "SZLHOLDINGS/WILLAY" not in seen:
        raise AlignmentError("WILLAY divergence is missing")
    return document


def main() -> int:
    try:
        document = verify()
    except (AlignmentError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"state": "REFUSED", "error": str(exc)}))
        return 1
    print(json.dumps({
        "state": "RECORDED_DIVERGENCE",
        "items": len(document["items"]),
        "hf_writes": False,
        "gate": "omen-cards drift remains fail-closed",
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
