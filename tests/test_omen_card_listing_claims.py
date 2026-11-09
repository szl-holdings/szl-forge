"""Repository filenames cannot establish the semantics of model weights."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "tools" / "szl_omen_pipeline.py"
SPEC = importlib.util.spec_from_file_location("omen_card_listing", SOURCE)
assert SPEC is not None and SPEC.loader is not None
pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline)

CARD = "---\nlicense: apache-2.0\nlibrary_name: peft\nbase_model_relation: adapter\n---\n\n# Research model\n\nExisting evaluation and release holds.\n"
ROOT_FILES = {"config.json", "model.safetensors"}
LEGACY_NOTE = (
    "<!-- szl:artifact-identity-reconciled -->\n"
    "> **Artifact identity (noted 2026-09-29).** Besides the LoRA adapter, this repository's root carries a "
    "merged full-precision checkpoint (`model.safetensors` + `config.json`, loadable with `transformers`) produced by "
    "the receipted CPU merge of this adapter into its declared base{receipt}. The adapter remains the artifact of "
    "record for every figure on this card; the merged bytes carry no separate held-out receipt and add no claim. "
    "Metadata-only note.\n\n"
)


def card_with_note(note):
    frontmatter, body = CARD.rsplit("---\n", 1)
    return frontmatter + "---\n" + note + body.lstrip("\n")


@pytest.mark.parametrize("files", [ROOT_FILES, ROOT_FILES | {"merge_receipt.json"}])
def test_filename_listing_does_not_assert_unmeasured_model_properties(files):
    after, changes = pipeline.reconcile_card(CARD, files, "2026-11-09")
    assert changes == ["note: artifact-identity blockquote inserted after frontmatter"]
    assert "`model.safetensors`" in after and "`config.json`" in after
    assert "merged full-precision checkpoint" not in after
    assert "loadable with `transformers`" not in after
    assert "receipted CPU merge" not in after
    assert "this adapter into its declared base" not in after
    assert "have not been inspected" in after
    assert "does not establish" in after


def test_receipt_filename_is_reported_without_claiming_its_contents_were_verified():
    after, _ = pipeline.reconcile_card(CARD, ROOT_FILES | {"merge_receipt.json"}, "2026-11-09")
    assert "`merge_receipt.json` is also listed" in after
    assert "contents and bindings have not been verified" in after
    without_receipt, _ = pipeline.reconcile_card(CARD, ROOT_FILES, "2026-11-09")
    assert "`merge_receipt.json`" not in without_receipt


def test_default_note_preserves_frontmatter_holds_and_is_idempotent():
    after, _ = pipeline.reconcile_card(CARD, ROOT_FILES, "2026-11-09")
    assert after.split("---\n", 2)[:2] == CARD.split("---\n", 2)[:2]
    assert after.endswith("# Research model\n\nExisting evaluation and release holds.\n")
    again, changes = pipeline.reconcile_card(after, ROOT_FILES, "2026-11-10")
    assert again == after
    assert changes == ["NOOP: card already reconciled"]


@pytest.mark.parametrize("files", [set(), {"config.json"}, {"model.safetensors"}])
def test_partial_root_listing_does_not_add_a_note(files):
    after, changes = pipeline.reconcile_card(CARD, files, "2026-11-09")
    assert after == CARD
    assert changes[0].startswith("SKIP:")


def test_existing_reviewed_identity_section_is_preserved():
    reviewed = CARD + "\n" + pipeline.NOTE_MARKER + "\nReviewed artifact identity and release holds.\n"
    after, changes = pipeline.reconcile_card(reviewed, ROOT_FILES, "2026-11-09")
    assert after == reviewed
    assert changes == ["NOOP: card already reconciled"]


@pytest.mark.parametrize("old_receipt", ["", " (`merge_receipt.json`)"])
@pytest.mark.parametrize("files", [ROOT_FILES, ROOT_FILES | {"merge_receipt.json"}])
def test_exact_legacy_note_is_replaced_using_current_listing_and_remains_idempotent(old_receipt, files):
    before = card_with_note(LEGACY_NOTE.format(receipt=old_receipt))
    after, changes = pipeline.reconcile_card(before, files, "2026-11-09")
    assert changes == ["note: legacy generated artifact-identity blockquote replaced with listing note"]
    expected, _ = pipeline.reconcile_card(CARD, files, "2026-11-09")
    assert after == expected
    again, changes = pipeline.reconcile_card(after, files, "2026-11-10")
    assert again == after
    assert changes == ["NOOP: card already reconciled"]


@pytest.mark.parametrize("note", [
    LEGACY_NOTE.format(receipt="").replace("Metadata-only note.", "Reviewed note."),
    LEGACY_NOTE.format(receipt="").replace("2026-09-29", "September 29"),
    "Reviewed preface.\n\n" + LEGACY_NOTE.format(receipt=""),
    LEGACY_NOTE.format(receipt="") + "<!-- szl:artifact-identity-reconciled -->\nReviewed second section.\n",
])
def test_custom_moved_or_ambiguous_legacy_sections_are_preserved(note):
    before = card_with_note(note)
    after, changes = pipeline.reconcile_card(before, ROOT_FILES, "2026-11-09")
    assert after == before
    assert changes == ["NOOP: card already reconciled"]
