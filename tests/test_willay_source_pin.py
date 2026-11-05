"""WILLAY's card pin is a main commit, and null base revisions stay null."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL = "ad20ac2c24859156891c391108a86d4f05163f83"
PRIOR_CARD = "4c61d30cf6dfc7a0b4c9673192bf8eb4db0344aa"
PRIOR_CARD_SHA256 = "29b4590af502221d5b0aa9959b10ed2bef03fc982110da2483289c6fca0f2e22"
UNCHANGED = (
    "willay/card/holo-banner.svg",
    "frontier/evaluation/gate_runs/WILLAY_375a007ec961_20260915T040944Z_receipt.json",
    "frontier/evaluation/gate_runs/WILLAY_375a007ec961_20260915T040944Z_heldout.json",
)


def _blob(revision: str, path: str) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", f"{revision}:{path}"], cwd=ROOT, text=True
    ).strip()


def _show(revision: str, path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=ROOT)


def test_historical_banner_and_receipt_blobs_are_unchanged():
    for path in UNCHANGED:
        assert _blob(HISTORICAL, path) == _blob(PRIOR_CARD, path) == _blob("HEAD", path)


def test_authoring_card_no_longer_cites_the_pre_polish_readme_commit():
    card = (ROOT / "willay/card/README.md").read_text(encoding="utf-8")
    assert f"blob/{HISTORICAL}/willay/card/README.md" not in card
    assert f"{HISTORICAL}/willay/card/holo-banner.svg" in card
    assert "leave their base revision null" in card
    assert "main@merge-time" in card
    prior = _show(PRIOR_CARD, "willay/card/README.md")
    assert hashlib.sha256(prior).hexdigest() == PRIOR_CARD_SHA256
    assert prior != (ROOT / "willay/card/README.md").read_bytes()


def test_reviewed_source_revision_resolves_on_main_to_the_pinned_digest():
    profile = json.loads(
        (ROOT / "publishing/curated-model-card-profiles.json").read_text(encoding="utf-8")
    )["profiles"]["willay"]
    revision = profile["reviewed_source_revision"]
    assert revision == "647594764509cb200f4599710e1e2529b00dd2c5"
    subprocess.check_call(
        ["git", "merge-base", "--is-ancestor", revision, "HEAD"], cwd=ROOT
    )
    pinned = _show(revision, "willay/card/README.md")
    assert hashlib.sha256(pinned).hexdigest() == profile["source_sha256"]
    assert hashlib.sha256((ROOT / "willay/card/README.md").read_bytes()).hexdigest() == profile["source_sha256"]


def test_reviewed_adapter_base_revisions_stay_null():
    observation = json.loads(
        (ROOT / "publishing/curated-card-observations/willay-20260930.json").read_text(encoding="utf-8")
    )
    groups = observation["classification"]["adapter_groups"]
    assert [group["base_revision"] for group in groups] == [None, None]
    assert [group["base_revision_status"] for group in groups] == ["UNKNOWN", "UNKNOWN"]
