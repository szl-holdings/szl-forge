#!/usr/bin/env python3
"""Reconcile only Chaski-R4's qualification sidecar from protected Forge source.

This is not a model, weight, card, promotion, or autonomy publisher. The
historic bytes-publication receipt anchors the frozen artifacts. A separately
governed card publisher may update its three controlled files; those must match
the current immutable Forge card blobs and the Hub source binding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

if __package__:
    from .publish_chaski_card import build_source_binding
else:
    from publish_chaski_card import build_source_binding


ROOT = Path(__file__).resolve().parents[1]
REPO = "SZLHOLDINGS/chaski-r4"
SIDECAR = "evidence/QUALIFICATION_STATE.md"
SOURCE_SIDECAR = "chaski_r4/QUALIFICATION_STATE.md"
SOURCE_RECEIPT = "chaski_r4/evidence/publication_receipt_20261001_171009.json"
SOURCE_CARD = "chaski_r4/card/README.md"
SOURCE_BANNER = "chaski_r4/card/holo-banner.svg"
PINNED_RECEIPT_SHA256 = "d8eea67be2062b0741c4d52dedb511ac1e87e99c43d54b6e1f34809b1de49af2"
PINNED_BYTES_REVISION = "f662e24aa9e878dc6c2df50151d4fd500ea8a16c"
FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
FILE_SHA = re.compile(r"^[0-9a-f]{64}$")
RECEIPT_FILES = frozenset(
    {
        "LICENSE",
        "adapter_config.json",
        "adapter_model.safetensors",
        "evidence/canonical_rerun_20261001_140615.receipt.json",
        SIDECAR,
        "training_receipt.json",
    }
)
CARD_FILES = frozenset({"README.md", "holo-banner.svg", "szl-source-binding.json"})
HUB_FILES = frozenset({".gitattributes"}) | RECEIPT_FILES | CARD_FILES


class PublicationError(RuntimeError):
    """A controlled, secret-free contract failure."""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_sha(value: Any, pattern: re.Pattern[str]) -> bool:
    return isinstance(value, str) and bool(pattern.fullmatch(value))


@dataclass(frozen=True)
class SourceContract:
    source_revision: str
    sidecar: bytes
    card: bytes
    banner: bytes
    receipt: dict[str, Any]


def build_contract(
    read_blob: Callable[[str], bytes],
    source_revision: str,
    *,
    expected_receipt_sha256: str = PINNED_RECEIPT_SHA256,
    expected_bytes_revision: str = PINNED_BYTES_REVISION,
) -> SourceContract:
    """Validate the fixed publication decision before any provider access."""
    if not _is_sha(source_revision, FULL_SHA):
        raise PublicationError("source revision must be a full lowercase commit SHA")
    sidecar = read_blob(SOURCE_SIDECAR)
    lines = sidecar.splitlines()
    publication_lines = [line for line in lines if line.startswith(b"Publication:")]
    promotion_lines = [line for line in lines if line.startswith(b"Promotion:")]
    if (
        len(publication_lines) != 1
        or not publication_lines[0].startswith(b"Publication: PUBLIC_EXPERIMENTAL_ARTIFACT ")
        or len(promotion_lines) != 1
        or not promotion_lines[0].startswith(b"Promotion: NOT_PROMOTABLE ")
        or b"publication_eligible=false" not in sidecar
        or b"receipts A/B provenance stays unresolved" not in sidecar
        or b"publication_eligible=true" in sidecar.lower()
    ):
        raise PublicationError("source qualification claim is not the bounded public experimental state")
    receipt_bytes = read_blob(SOURCE_RECEIPT)
    # This frozen Git-blob digest pins every historical weight/evidence hash in
    # the receipt. A later protected-main edit cannot silently re-baseline it.
    if sha256(receipt_bytes) != expected_receipt_sha256:
        raise PublicationError("historic bytes-publication receipt changed from its pinned source digest")
    try:
        receipt = json.loads(receipt_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise PublicationError("source publication receipt is invalid JSON") from exc
    if not isinstance(receipt, dict) or any(
        receipt.get(key) != value
        for key, value in {
            "schema": "szl.hf-artifact-publication/v1",
            "repo_id": REPO,
            "artifact_state": "PUBLIC_EXPERIMENTAL_ARTIFACT",
            "promotion": "NOT_PROMOTABLE",
            "publication_eligible": False,
            "autonomy_eligible": False,
        }.items()
    ):
        raise PublicationError("source publication receipt changed its target or qualification boundary")
    if not all(_is_sha(receipt.get(key), FULL_SHA) for key in ("card_revision", "bytes_revision")):
        raise PublicationError("source receipt has no immutable card and bytes revisions")
    if receipt["bytes_revision"] != expected_bytes_revision:
        raise PublicationError("historic bytes revision changed from its pinned Hub commit")
    files = receipt.get("files")
    if not isinstance(files, dict) or set(files) != RECEIPT_FILES:
        raise PublicationError("source receipt has an unexpected file set")
    for name in RECEIPT_FILES:
        item = files[name]
        if not isinstance(item, dict) or not _is_sha(item.get("sha256"), FILE_SHA):
            raise PublicationError("source receipt has an invalid file digest")
        if item.get("readback_sha256") != item["sha256"]:
            raise PublicationError("source receipt lacks exact byte readback")
    if receipt.get("receipt_c_sha256") != files[
        "evidence/canonical_rerun_20261001_140615.receipt.json"
    ]["sha256"]:
        raise PublicationError("receipt C identity is inconsistent")
    return SourceContract(
        source_revision=source_revision,
        sidecar=sidecar,
        card=read_blob(SOURCE_CARD),
        banner=read_blob(SOURCE_BANNER),
        receipt=receipt,
    )


def load_source_contract(root: Path, source_revision: str) -> SourceContract:
    """Read Git blob bytes, not Windows checkout bytes converted to CRLF."""
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, check=False
    )
    if head.returncode or head.stdout.decode("ascii", errors="replace").strip() != source_revision:
        raise PublicationError("checkout is not the declared exact source revision")

    def read_blob(name: str) -> bytes:
        result = subprocess.run(
            ["git", "show", f"{source_revision}:{name}"],
            cwd=root,
            capture_output=True,
            check=False,
        )
        if result.returncode:
            raise PublicationError(f"source blob unavailable: {name}")
        return result.stdout

    return build_contract(read_blob, source_revision)


def historic_blob_reader(root: Path) -> Callable[[str, str], bytes]:
    """Read declared Forge commits; shallow CI fetches only the two named SHAs."""
    remote = subprocess.run(
        ["git", "remote", "get-url", "origin"], cwd=root, capture_output=True, check=False
    )
    if remote.returncode or remote.stdout.decode("utf-8", errors="replace").strip() not in {
        "https://github.com/szl-holdings/szl-forge",
        "https://github.com/szl-holdings/szl-forge.git",
    }:
        raise PublicationError("historic source lookup requires the canonical Forge origin")
    inspected: set[str] = set()

    def read_blob(revision: str, name: str) -> bytes:
        if not _is_sha(revision, FULL_SHA) or name not in {SOURCE_CARD, SOURCE_BANNER}:
            raise PublicationError("historic card source lookup is outside the closed contract")
        if revision not in inspected:
            if len(inspected) >= 2:
                raise PublicationError("more than two historic card source revisions were requested")
            present = subprocess.run(
                ["git", "cat-file", "-e", f"{revision}^{{commit}}"],
                cwd=root,
                capture_output=True,
                check=False,
            )
            if present.returncode:
                fetched = subprocess.run(
                    ["git", "fetch", "--no-tags", "--depth=1", "origin", revision],
                    cwd=root,
                    capture_output=True,
                    check=False,
                )
                if fetched.returncode:
                    raise PublicationError("declared historic Forge commit could not be fetched")
                present = subprocess.run(
                    ["git", "cat-file", "-e", f"{revision}^{{commit}}"],
                    cwd=root,
                    capture_output=True,
                    check=False,
                )
                if present.returncode:
                    raise PublicationError("declared historic Forge commit is unavailable")
            inspected.add(revision)
        result = subprocess.run(
            ["git", "show", f"{revision}:{name}"],
            cwd=root,
            capture_output=True,
            check=False,
        )
        if result.returncode:
            raise PublicationError("declared historic Forge card blob is unavailable")
        return result.stdout

    return read_blob


def _file_set(info: Any, revision: str) -> None:
    if getattr(info, "sha", None) != revision:
        raise PublicationError("Hub revision readback did not match the requested immutable revision")
    files = {item.rfilename for item in info.siblings}
    if files != HUB_FILES:
        raise PublicationError("Hub model file inventory drifted from the closed Chaski-R4 contract")


def _binding(
    payload: bytes,
    hashes: dict[str, str],
    sizes: dict[str, int],
    read_source_blob: Callable[[str, str], bytes],
) -> None:
    try:
        binding = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise PublicationError("Hub card source binding is invalid JSON") from exc
    if not isinstance(binding, dict):
        raise PublicationError("Hub card source binding must be an object")
    source = binding.get("source")
    if not isinstance(source, dict) or not _is_sha(source.get("revision"), FULL_SHA):
        raise PublicationError("Hub card source binding has no immutable source revision")
    source_revision = source["revision"]
    for name in ("README.md", "holo-banner.svg"):
        declared_blob = read_source_blob(
            source_revision, SOURCE_CARD if name == "README.md" else SOURCE_BANNER
        )
        if sha256(declared_blob) != hashes[name] or len(declared_blob) != sizes[name]:
            raise PublicationError("Hub card asset bytes drifted from the declared immutable Forge source")
    expected = build_source_binding(
        profile="chaski-r4",
        source_revision=source_revision,
        source_assets={
            name: {"sha256": hashes[name], "bytes": sizes[name]}
            for name in ("README.md", "holo-banner.svg")
        },
    )
    if payload != expected:
        raise PublicationError("Hub card source binding differs from the complete Forge publisher contract")


def _snapshot(
    api: Any,
    read_hub: Callable[[str, str], bytes],
    read_source_blob: Callable[[str, str], bytes],
    revision: str,
) -> dict[str, Any]:
    _file_set(api.repo_info(REPO, repo_type="model", revision=revision), revision)
    hashes: dict[str, str] = {}
    sizes: dict[str, int] = {}
    binding_bytes = b""
    for name in sorted(HUB_FILES):
        payload = read_hub(revision, name)
        hashes[name] = sha256(payload)
        sizes[name] = len(payload)
        if name == "szl-source-binding.json":
            binding_bytes = payload
    _binding(binding_bytes, hashes, sizes, read_source_blob)
    return {"hashes": hashes, "sizes": sizes}


def _verify_static(contract: SourceContract, baseline: dict[str, Any], current: dict[str, Any]) -> None:
    files = contract.receipt["files"]
    old = baseline["hashes"]
    live = current["hashes"]
    for name in RECEIPT_FILES:
        if old[name] != files[name]["sha256"]:
            raise PublicationError(f"historic bytes-publication receipt drift: {name}")
        if name != SIDECAR and live[name] != files[name]["sha256"]:
            raise PublicationError(f"frozen Hub artifact drift: {name}")
    if old[".gitattributes"] != live[".gitattributes"]:
        raise PublicationError("Hub .gitattributes drifted from the bytes-publication revision")
    if live["README.md"] != sha256(contract.card) or live["holo-banner.svg"] != sha256(contract.banner):
        raise PublicationError("current Hub card assets drifted from exact Forge source")


def reconcile(
    contract: SourceContract,
    api: Any,
    read_hub: Callable[[str, str], bytes],
    read_source_blob: Callable[[str, str], bytes],
    make_operation: Callable[[str, bytes], Any],
    *,
    on_commit: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Compare every known file before and after a one-path conditional commit."""
    who = api.whoami()
    orgs = who.get("orgs", []) if isinstance(who, dict) else []
    if "SZLHOLDINGS" not in {
        item.get("name") if isinstance(item, dict) else item for item in orgs
    }:
        raise PublicationError("publisher identity is not authorized for SZLHOLDINGS")
    baseline_revision = contract.receipt["bytes_revision"]
    baseline = _snapshot(api, read_hub, read_source_blob, baseline_revision)
    live_info = api.repo_info(REPO, repo_type="model")
    parent = getattr(live_info, "sha", None)
    if not _is_sha(parent, FULL_SHA):
        raise PublicationError("current Hub parent revision is unavailable")
    _file_set(live_info, parent)
    current = _snapshot(api, read_hub, read_source_blob, parent)
    _verify_static(contract, baseline, current)
    sidecar_hash = current["hashes"][SIDECAR]
    desired_hash = sha256(contract.sidecar)
    old_hash = contract.receipt["files"][SIDECAR]["sha256"]
    if sidecar_hash not in {old_hash, desired_hash}:
        raise PublicationError("Hub qualification sidecar has an unrecognized change")
    if api.repo_info(REPO, repo_type="model").sha != parent:
        raise PublicationError("Hub parent changed during preflight")
    if sidecar_hash == desired_hash:
        status = "ALREADY_CURRENT"
        published_revision = parent
    else:
        operation = make_operation(SIDECAR, contract.sidecar)
        # The reviewed 1.23.0 client may return the current tip without a
        # server CAS when an addition becomes a no-op. Readback alone would
        # misattribute another writer's identical sidecar to this publisher.
        if getattr(operation, "_is_committed", None) is not False:
            raise PublicationError("reviewed Hub commit-origin marker is unavailable before write")
        commit = api.create_commit(
            repo_id=REPO,
            repo_type="model",
            revision="main",
            create_pr=False,
            operations=[operation],
            commit_message="chaski-r4: reconcile source-bound qualification sidecar (NOT_PROMOTABLE)",
            commit_description=f"szl-forge@{contract.source_revision}; sidecar only; no weights, card, or promotion",
            parent_commit=parent,
        )
        if getattr(operation, "_is_committed", None) is not True:
            raise PublicationError("Hub client returned without an expected-parent sidecar server commit; publication status UNKNOWN")
        published_revision = getattr(commit, "oid", None)
        if not _is_sha(published_revision, FULL_SHA):
            raise PublicationError("Hub returned no immutable commit revision; publication status UNKNOWN")
        if on_commit:
            on_commit(
                {
                    "status": "COMMITTED_UNVERIFIED",
                    "repo_id": REPO,
                    "source_revision": contract.source_revision,
                    "previous_hub_revision": parent,
                    "hub_revision": published_revision,
                    "changed_paths": [SIDECAR],
                }
            )
        status = "PUBLISHED_AND_READ_BACK"
    after = _snapshot(api, read_hub, read_source_blob, published_revision)
    if after["hashes"][SIDECAR] != desired_hash:
        raise PublicationError("immutable Hub sidecar readback differs from Forge source")
    for name in HUB_FILES - {SIDECAR}:
        if after["hashes"][name] != current["hashes"][name]:
            raise PublicationError(f"unrelated Hub file changed during sidecar publication: {name}")
    if api.repo_info(REPO, repo_type="model").sha != published_revision:
        raise PublicationError("Hub tip changed after immutable readback")
    return {
        "schema": "szl.hf.chaski-r4-qualification-reconciliation/v1",
        "status": status,
        "repo_id": REPO,
        "source_revision": contract.source_revision,
        "source_sidecar_sha256": desired_hash,
        "historic_bytes_revision": baseline_revision,
        "previous_hub_revision": parent,
        "hub_revision": published_revision,
        "changed_paths": [] if status == "ALREADY_CURRENT" else [SIDECAR],
        "file_sha256_at_revision": after["hashes"],
        "promotion": "NOT_PROMOTABLE",
        "publication_eligible": False,
        "autonomy_eligible": False,
        "claim_boundary": "Sidecar byte reconciliation only; no weight, card, promotion, autonomy, deployment, or independent quality claim. Receipts A/B provenance unresolved.",
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=path.parent, delete=False) as temporary:
        temporary.write(payload)
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--publish", action="store_true", help="allow the single guarded Hub sidecar commit")
    args = parser.parse_args()
    pending: dict[str, Any] = {}
    try:
        contract = load_source_contract(ROOT, args.source_revision)
        if not args.publish:
            report = {
                "schema": "szl.hf.chaski-r4-qualification-reconciliation/v1",
                "status": "SOURCE_CHECK_ONLY",
                "source_revision": contract.source_revision,
                "source_sidecar_sha256": sha256(contract.sidecar),
                "historic_bytes_revision": contract.receipt["bytes_revision"],
                "promotion": "NOT_PROMOTABLE",
                "publication_eligible": False,
                "claim_boundary": "Source contract only; Hub state not inspected and no publication attempted.",
            }
        else:
            from huggingface_hub import CommitOperationAdd, HfApi, __version__ as hub_version, hf_hub_download

            if hub_version != "1.23.0":
                raise PublicationError("publication requires the reviewed huggingface_hub 1.23.0 client")

            with tempfile.TemporaryDirectory() as temp_dir:
                def read_hub(revision: str, name: str) -> bytes:
                    result = hf_hub_download(
                        REPO,
                        name,
                        repo_type="model",
                        revision=revision,
                        local_dir=Path(temp_dir) / revision,
                        force_download=True,
                    )
                    return Path(result).read_bytes()

                def on_commit(value: dict[str, Any]) -> None:
                    pending.update(value)
                    _write_report(args.report, value)

                report = reconcile(
                    contract,
                    HfApi(),
                    read_hub,
                    historic_blob_reader(ROOT),
                    lambda name, content: CommitOperationAdd(path_in_repo=name, path_or_fileobj=content),
                    on_commit=on_commit,
                )
        _write_report(args.report, report)
        print(json.dumps({"status": report["status"], "repo_id": REPO, "report": str(args.report)}))
        return 0
    except Exception as exc:
        reason = str(exc) if isinstance(exc, PublicationError) else "provider or runtime error; inspect the named client error in a controlled runner"
        report = {
            "schema": "szl.hf.chaski-r4-qualification-reconciliation/v1",
            "status": "HOLD",
            "source_revision": args.source_revision,
            "repo_id": REPO,
            "error_type": type(exc).__name__,
            "reason": reason,
            "promotion": "NOT_PROMOTABLE",
            "publication_eligible": False,
            "prior_commit_state": pending or None,
        }
        _write_report(args.report, report)
        print(json.dumps({"status": "HOLD", "error_type": type(exc).__name__, "report": str(args.report)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
