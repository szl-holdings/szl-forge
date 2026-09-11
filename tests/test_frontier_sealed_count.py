from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from frontier.evaluation.core import canonical_sha256
from frontier.evaluation.sealed_count import (
    COUNTER_PATH,
    KNOWN_BOUNDS,
    SealedCountError,
    apply_seal,
    encode,
    genesis,
    increment,
    load_counter,
    public_view,
    publish_sealed_count,
    validate,
)

GENESIS_PATH = (
    Path(__file__).resolve().parents[1]
    / "frontier/evidence/sealed-run-counter-genesis-2026-09-11.json"
)
FIXTURES = (
    Path(__file__).resolve().parents[1] / "frontier/evaluation/fixtures.v1.json"
)
FIXTURES_DOC = (
    Path(__file__).resolve().parents[1] / "frontier/evaluation/FIXTURES.md"
)
PINNED_FIXTURES_SHA256 = (
    "588299a6d926bb979a7a42316d781f3469220062399aacd0c22d859922ef2ee6"
)
T0 = datetime(2026, 9, 11, 15, 41, 38, tzinfo=timezone.utc)


def test_genesis_is_zero_not_a_filled_silence() -> None:
    doc = genesis()
    validate(doc)
    assert doc["total_sealed"] == 0
    assert doc["days"] == []
    assert doc["updated_at"] is None
    assert doc["parent_sha256"] is None
    assert "pre_counter_silence_is_unavailable_not_zero" in doc["known_bounds"]
    assert doc["known_bounds"] == list(KNOWN_BOUNDS)


def test_genesis_file_matches_in_code_genesis_bytes() -> None:
    on_disk = json.loads(GENESIS_PATH.read_text(encoding="utf-8"))
    validate(on_disk)
    assert encode(on_disk) == encode(genesis())
    assert on_disk["total_sealed"] == 0


def test_load_none_is_genesis_empty_and_garbage_fail_closed() -> None:
    assert load_counter(None)["total_sealed"] == 0
    with pytest.raises(SealedCountError, match="unreadable"):
        load_counter(b"")
    with pytest.raises(SealedCountError, match="unreadable"):
        load_counter(b"{")
    with pytest.raises(SealedCountError, match="not an object"):
        load_counter(b"[1]")


def test_increment_is_strictly_monotonic_and_date_only() -> None:
    first = increment(None, now=T0)
    assert first["total_sealed"] == 1
    assert first["days"] == [{"date": "2026-09-11", "sealed": 1}]
    assert first["parent_sha256"] == genesis()["counter_sha256"]
    second = increment(first, now=T0.replace(hour=18))
    assert second["total_sealed"] == 2
    assert second["days"] == [{"date": "2026-09-11", "sealed": 2}]
    third = increment(second, now=datetime(2026, 9, 12, 1, tzinfo=timezone.utc))
    assert third["total_sealed"] == 3
    assert third["days"] == [
        {"date": "2026-09-11", "sealed": 2},
        {"date": "2026-09-12", "sealed": 1},
    ]
    assert third["parent_sha256"] == second["counter_sha256"]


def test_counter_bytes_never_leak_provider_or_receipt_payload() -> None:
    doc = increment(None, now=T0)
    blob = encode(doc).decode("utf-8")
    for leak in (
        "baseten",
        "glm-5",
        "github-34222679359",
        "score_rate",
        "error_summary",
        "hf_",
    ):
        assert leak not in blob
    with pytest.raises(SealedCountError, match="must not carry key"):
        dirty = dict(doc)
        dirty["provider"] = "baseten"
        dirty["counter_sha256"] = canonical_sha256(
            {k: v for k, v in dirty.items() if k != "counter_sha256"}
        )
        validate(dirty)
    view = public_view(doc)
    assert set(view) <= {
        "schema",
        "epoch",
        "updated_at",
        "total_sealed",
        "days",
        "parent_sha256",
        "counter_sha256",
        "known_bounds",
        "path_in_repo",
    }
    assert view["path_in_repo"] == COUNTER_PATH


def test_naive_datetime_and_backdated_seal_fail_closed() -> None:
    with pytest.raises(SealedCountError, match="timezone-aware"):
        increment(None, now=datetime(2026, 9, 11, 12, 0, 0))
    first = increment(None, now=T0)
    with pytest.raises(SealedCountError, match="before the head"):
        increment(first, now=datetime(2026, 9, 10, tzinfo=timezone.utc))


def test_hash_mismatch_and_unknown_schema_fail_closed() -> None:
    doc = genesis()
    doc["counter_sha256"] = "0" * 64
    with pytest.raises(SealedCountError, match="does not match"):
        validate(doc)
    doc = genesis()
    doc["schema"] = "szl.frontier.sealed-run-counter.v0"
    doc["counter_sha256"] = canonical_sha256(
        {k: v for k, v in doc.items() if k != "counter_sha256"}
    )
    with pytest.raises(SealedCountError, match="unknown sealed-count schema"):
        validate(doc)


def test_publish_uses_injected_io_and_never_sees_a_receipt_dir(
    tmp_path: Path,
) -> None:
    token = "hf_private_value_must_not_appear"
    stored: dict[str, bytes] = {}

    def fetch(_dataset: str) -> bytes | None:
        return stored.get(COUNTER_PATH)

    def commit(_dataset: str, payload: bytes) -> dict[str, str]:
        assert token.encode() not in payload
        stored[COUNTER_PATH] = payload
        return {"commit_oid": "b" * 40, "commit_url": "https://example.test/commit"}

    first = publish_sealed_count(
        token=token,
        dataset_id="SZLHOLDINGS/szl-frontier-evaluation-receipts",
        now=T0,
        fetch_bytes=fetch,
        commit_bytes=commit,
    )
    assert first["status"] == "SEALED_COUNT_ONLY"
    assert first["production_disposition"] == "HOLD"
    assert first["promotion_effect"] == "NONE"
    assert first["total_sealed"] == 1
    assert first["days"] == [{"date": "2026-09-11", "sealed": 1}]
    assert COUNTER_PATH in stored
    payload = json.loads(stored[COUNTER_PATH])
    assert "provider" not in payload
    assert "receipt_sha256" not in payload
    assert tmp_path.exists()
    assert not any(tmp_path.iterdir())

    second = publish_sealed_count(
        token=token,
        dataset_id="SZLHOLDINGS/szl-frontier-evaluation-receipts",
        now=T0,
        fetch_bytes=fetch,
        commit_bytes=commit,
    )
    assert second["total_sealed"] == 2
    assert second["parent_sha256"] == first["counter_sha256"]
    assert token not in json.dumps(second)


def test_publish_without_token_fails_closed() -> None:
    with pytest.raises(SealedCountError, match="HF_TOKEN"):
        publish_sealed_count(token="", dataset_id="SZLHOLDINGS/x")


def test_fixtures_doc_covers_every_answer_equals_without_changing_bytes() -> None:
    import hashlib

    raw = FIXTURES.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == PINNED_FIXTURES_SHA256
    fixtures = json.loads(raw)
    doc = FIXTURES_DOC.read_text(encoding="utf-8")
    assert "paraphrase" in doc.casefold()
    assert "casefold" in doc.casefold()
    assert PINNED_FIXTURES_SHA256 in doc
    for case in fixtures["cases"]:
        assert case["id"] in doc
        assert case["expected"]["answer_equals"] in doc
        assert case["expected"]["decision"] in doc
    assert "does not modify fixtures" in doc.casefold() or "does **not** modify" in doc.casefold()


def test_apply_seal_from_genesis_bytes_roundtrip() -> None:
    first = apply_seal(None, now=T0)
    again = apply_seal(encode(first), now=T0)
    assert again["total_sealed"] == 2
    validate(again)
