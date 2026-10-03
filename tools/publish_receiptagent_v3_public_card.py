#!/usr/bin/env python3
"""One-shot, exact-parent publisher for the public ReceiptAgent v3 README only.

The default is review-only. Publishing is an explicit action from protected
Forge main, with one conditional Hub commit and an immutable byte readback.
No model weights, evaluations, visibility or release status are changed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import prepare_receiptagent_v3_card_sync as card  # noqa: E402


SOURCE_REPOSITORY = "szl-holdings/szl-forge"
SOURCE_MAIN = "https://github.com/szl-holdings/szl-forge.git"
TARGET_REPOSITORY = card.HUB_REPO
RECEIPT_SCHEMA = "szl.receiptagent-v3-public-card-publication/v1"
FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
SOURCE_BOUND_WRITER_FILES = (
    "tools/prepare_receiptagent_v3_card_sync.py",
    "tools/publish_receiptagent_v3_public_card.py",
    "tools/check_receiptagent_v3_card_dispatch.py",
    "tools/acquire_hf_publisher_token.py",
    ".github/workflows/publish-receiptagent-v3-public-card.yml",
)
REQUIRED_HOLD_MARKERS = (
    b"publication_eligible: false",
    b"autonomy_eligible: false",
    b"promotion_effect=NONE",
    b"base model `UNRECORDED`",
    b"No autonomy, deployment, promotion, or flagship selection",
    card.NEW_EVALS.encode("utf-8"),
    card.NEW_SOURCE_NOTE.encode("utf-8"),
)


class PublicationError(RuntimeError):
    """A control failed without publication authority."""


class ReceiptFile:
    """Create an exclusive local receipt and atomically update its state."""

    def __init__(self, path: Path, source_revision: str):
        self.path = path
        self.data: dict[str, Any] = {
            "schema": RECEIPT_SCHEMA,
            "source_repository": SOURCE_REPOSITORY,
            "source_revision": source_revision,
            "target_repository": TARGET_REPOSITORY,
            "expected_hub_parent": card.HUB_PARENT,
            "expected_old_readme_sha256": card.HUB_README_SHA256,
            "candidate_readme_sha256": card.CANDIDATE_README_SHA256,
            "changed_paths": ["README.md"],
            "release_status": "UNQUALIFIED",
            "commit_attempted": False,
            "state": "NOT_ATTEMPTED",
        }
        body = self._encode()
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(body)
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            path.unlink(missing_ok=True)
            raise

    def _encode(self) -> bytes:
        return (json.dumps(self.data, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")

    def transition(self, state: str, **facts: Any) -> None:
        self.data.update(facts)
        self.data["state"] = state
        descriptor, temp_name = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)
        temporary = Path(temp_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(self._encode())
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)


def _git(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command, cwd=ROOT, capture_output=True, text=True,
            encoding="utf-8", timeout=30, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise PublicationError("canonical source lookup failed") from error
    if result.returncode:
        raise PublicationError("canonical source lookup failed")
    return result.stdout.strip()


def assert_current_main(source_revision: str) -> None:
    """Bind publication to local HEAD and freshly observed protected main."""
    if FULL_SHA.fullmatch(source_revision or "") is None:
        raise PublicationError("source revision must be exact 40-hex")
    if _git(["git", "rev-parse", "--verify", "HEAD"]) != source_revision:
        raise PublicationError("checkout does not match source revision")
    lines = _git(["git", "ls-remote", "--exit-code", SOURCE_MAIN, "refs/heads/main"]).splitlines()
    if len(lines) != 1:
        raise PublicationError("canonical main did not expose one exact revision")
    fields = lines[0].split()
    if len(fields) != 2 or fields[1] != "refs/heads/main" or FULL_SHA.fullmatch(fields[0]) is None:
        raise PublicationError("canonical main did not expose one exact revision")
    if fields[0] != source_revision:
        raise PublicationError("publication source no longer owns current main")


def assert_writer_files_committed(source_revision: str) -> None:
    """Reject a modified local publisher or preparer despite a matching HEAD."""
    for name in SOURCE_BOUND_WRITER_FILES:
        try:
            result = subprocess.run(
                ["git", "cat-file", "blob", f"{source_revision}:{name}"],
                cwd=ROOT, capture_output=True, timeout=30, check=False,
            )
            local = (ROOT / name).read_bytes()
        except (OSError, subprocess.TimeoutExpired) as error:
            raise PublicationError("writer source binding unavailable") from error
        if result.returncode or local.replace(b"\r\n", b"\n") != result.stdout:
            raise PublicationError(f"writer differs from exact source: {name}")


def assert_candidate_hold(candidate: bytes, review: dict[str, Any]) -> None:
    if (
        card.digest(candidate) != card.CANDIDATE_README_SHA256
        or review.get("target_repository") != TARGET_REPOSITORY
        or review.get("observed_hub_parent") != card.HUB_PARENT
        or review.get("changed_paths") != ["README.md"]
        or review.get("release_status") != "UNQUALIFIED"
        or review.get("disposition") != "REVIEW_ONLY_NO_HUB_WRITE"
    ):
        raise PublicationError("reviewed README-only candidate identity changed")
    if any(marker not in candidate for marker in REQUIRED_HOLD_MARKERS):
        raise PublicationError("non-promotion boundary is absent from candidate")
    if card.OLD_SOURCE_NOTE.encode("utf-8") in candidate or card.OLD_EVALS.encode("utf-8") in candidate:
        raise PublicationError("stale card summary remains in candidate")


def _provider_module() -> Any:
    import huggingface_hub  # Import only after canonical-source qualification.

    return huggingface_hub


def _repo_head(api: Any) -> str:
    info = api.repo_info(repo_id=TARGET_REPOSITORY, repo_type="model", revision="main")
    if getattr(info, "id", None) != TARGET_REPOSITORY:
        raise PublicationError("Hub target identity changed")
    revision = str(getattr(info, "sha", "") or "").strip().lower()
    if FULL_SHA.fullmatch(revision) is None:
        raise PublicationError("Hub repository did not expose an exact head")
    if getattr(info, "private", None) is not False or getattr(info, "gated", None) is not False:
        raise PublicationError("public Hub visibility changed")
    return revision


def publish(*, source_revision: str, token: str, receipt: ReceiptFile) -> dict[str, Any]:
    """Attempt at most one README-only CAS; never retry ambiguous acceptance."""
    receipt.transition("PRECHECKING")
    try:
        assert_current_main(source_revision)
        assert_writer_files_committed(source_revision)
        candidate, _diff, review = card.prepare(ROOT, source_revision)
        assert_candidate_hold(candidate, review)
        provider = _provider_module()
        if getattr(provider, "__version__", None) != "1.23.0":
            raise PublicationError("publisher requires reviewed huggingface_hub 1.23.0")
        api = provider.HfApi(token=token)
        api.auth_check(repo_id=TARGET_REPOSITORY, repo_type="model", write=True)
        identity = str((api.whoami() or {}).get("name") or "").strip()
        if not identity:
            raise PublicationError("publisher identity unavailable")
        if _repo_head(api) != card.HUB_PARENT:
            raise PublicationError("Hub parent drifted after candidate preparation")
        operation = provider.CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=candidate)
        if getattr(operation, "_is_committed", None) is not False:
            raise PublicationError("reviewed client commit-origin control unavailable")
        assert_current_main(source_revision)
        assert_writer_files_committed(source_revision)
    except Exception as error:
        receipt.transition("BLOCKED_NO_WRITE", failure_type=type(error).__name__)
        raise

    receipt.transition("COMMIT_OUTCOME_UNKNOWN_DO_NOT_RETRY", commit_attempted=True)
    try:
        commit = api.create_commit(
            repo_id=TARGET_REPOSITORY,
            repo_type="model",
            revision="main",
            create_pr=False,
            parent_commit=card.HUB_PARENT,
            operations=[operation],
            commit_message=f"docs: reconcile ReceiptAgent v3 public card from szl-forge@{source_revision}",
            commit_description="README-only historical and additive DEV evidence correction; no model promotion",
        )
        if getattr(operation, "_is_committed", None) is not True:
            raise PublicationError("expected-parent server commit was not confirmed")
        revision = str(getattr(commit, "oid", "") or "").strip().lower()
        if FULL_SHA.fullmatch(revision) is None or revision == card.HUB_PARENT:
            raise PublicationError("Hub commit did not return a new exact revision")
        receipt.transition("COMMITTED_READBACK_PENDING", hub_revision=revision, publisher=identity)
        observed = card._get(f"https://huggingface.co/{TARGET_REPOSITORY}/raw/{revision}/README.md")
        if observed != candidate:
            raise PublicationError("immutable Hub README byte readback mismatch")
        if _repo_head(api) != revision:
            raise PublicationError("Hub head changed before publication confirmation")
        assert_current_main(source_revision)
        receipt.transition("PUBLISHED_AND_READ_BACK", readback_sha256=card.digest(observed))
        return dict(receipt.data)
    except Exception as error:
        receipt.transition("POST_COMMIT_UNVERIFIED_DO_NOT_RETRY", failure_type=type(error).__name__)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--receipt-file", type=Path)
    args = parser.parse_args(argv)
    if not args.publish:
        _candidate, _diff, review = card.prepare(ROOT, args.source_revision)
        print(json.dumps(review, sort_keys=True))
        return 0
    if args.receipt_file is None:
        parser.error("--receipt-file is required for --publish")
    receipt = ReceiptFile(args.receipt_file, args.source_revision)
    try:
        # No credential is read before the fresh protected-main guard.
        assert_current_main(args.source_revision)
        token = os.environ.get("HF_TOKEN", "")
        if not token:
            raise PublicationError("HF_TOKEN is unavailable")
        result = publish(source_revision=args.source_revision, token=token, receipt=receipt)
    except Exception as error:
        if receipt.data["state"] == "NOT_ATTEMPTED":
            receipt.transition("BLOCKED_NO_WRITE", failure_type=type(error).__name__)
        print(json.dumps({"state": receipt.data["state"], "receipt_file": str(receipt.path)}, sort_keys=True))
        return 1
    print(json.dumps({"state": result["state"], "receipt_file": str(receipt.path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
