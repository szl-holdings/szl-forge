"""Read-only projection of the existing frontier reviewer; never parse trajectories.

The owner configures a local report and its external file digest. A matching hash
is integrity only, not authenticated source provenance or admission to training.
Only fixed, typed aggregate fields reach the workbench. No upstream text, row
hashes, file paths, tool arguments or arbitrary explanatory strings are exposed.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from .safeio import read_regular, strict_json

MAX_REPORT_BYTES = 256 * 1024
SCHEMA = "szl.frontier.ultradata-pilot-review/v1"
_DATASETS = {"openbmb/UltraData-SFT-Agent-2609", "openbmb/UltraData-RL-2609"}
_AUTHORITY = {"productionPromotion", "providerWrites", "trainingLaunch",
              "billableJobCreation", "weightPublication", "toolExecution"}
_REVIEWS = {"rights", "privacy", "decontamination", "semanticDedup",
            "toolArgumentSchema", "outcomeQuality"}
_FIELDS = {"schema", "generatedAt", "sourceRevisionDeclared", "datasetRepoId",
           "datasetRevisionDeclared", "inputSha256", "inputHashMatched",
           "upstreamBytesVerified", "sourceExecutionVerified", "rows", "counts",
           "state", "authorityOrder", "authority", "productionDisposition",
           "signatureState", "reviews", "execution", "rewardVerified",
           "successLabelsInferred", "bounds", "receiptSha256"}


def _need(value: bool) -> None:
    if not value:
        raise ValueError("invalid_corpus_report")


def _digest(value: object, size: int = 64) -> str:
    _need(type(value) is str and re.fullmatch(r"[0-9a-f]{%d}" % size, value) is not None)
    return value


def _integer(value: object, maximum: int) -> int:
    _need(type(value) is int and 0 <= value <= maximum)
    return value


def unavailable_report(state: str = "UNCONFIGURED") -> dict:
    """Unknown inventory is null, never a synthetic zero or readiness signal."""
    _need(state in {"UNCONFIGURED", "INVALID_OR_UNAVAILABLE_REPORT"})
    return {"schema": "szl.model-lab.corpus-view/v1", "state": state,
            "counts": None, "report_generated_at": None, "freshness": "UNKNOWN",
            "training_allowed": False, "execution_authority": False,
            "publication_authority": False, "source_verified": False,
            "signature_verified": False, "reward_verified": False}


def project_report(raw: bytes, expected_sha256: str, *, now: datetime | None = None) -> dict:
    """Validate the producer's report contract and return an allowlisted summary.

    This is not a second dataset reviewer. The producer owns SFT/RL semantics;
    this reader checks the report's internal consistency without executing them.
    Report age is not the age of model, dataset or service observations.
    """
    expected = _digest(expected_sha256)
    _need(type(raw) is bytes and 0 < len(raw) <= MAX_REPORT_BYTES)
    _need(hashlib.sha256(raw).hexdigest() == expected)
    document = strict_json(raw)
    _need(type(document) is dict and set(document) == _FIELDS)
    _need(document["schema"] == SCHEMA)
    claimed = _digest(document["receiptSha256"])
    # Match frontier_completion.core canonicalization, NOT safeio's newline form.
    try:
        body = json.dumps({k: v for k, v in document.items() if k != "receiptSha256"},
                          ensure_ascii=False, allow_nan=False, sort_keys=True,
                          separators=(",", ":")).encode("utf-8")
    except (ValueError, TypeError, UnicodeError, RecursionError, OverflowError) as exc:
        raise ValueError("invalid_corpus_report") from exc
    _need(hashlib.sha256(body).hexdigest() == claimed)
    _need(type(document["datasetRepoId"]) is str and document["datasetRepoId"] in _DATASETS)
    _digest(document["sourceRevisionDeclared"], 40)
    _digest(document["datasetRevisionDeclared"], 40)
    _digest(document["inputSha256"])
    _need(document["inputHashMatched"] is True)
    for field in ("upstreamBytesVerified", "sourceExecutionVerified", "rewardVerified",
                  "successLabelsInferred"):
        _need(document[field] is False)
    _need(document["execution"] == "NONE" and document["signatureState"] == "UNSIGNED"
          and document["productionDisposition"] == "HOLD")
    _need(document["authorityOrder"] == ["GitHub", "Hugging Face", "a-11-oy.com", "a11oy.net"])
    authority, reviews = document["authority"], document["reviews"]
    _need(type(authority) is dict and set(authority) == _AUTHORITY)
    _need(all(value is False for value in authority.values()))
    _need(type(reviews) is dict and set(reviews) == _REVIEWS)
    _need(all(value == "NOT_PERFORMED" for value in reviews.values()))
    bounds = document["bounds"]
    _need(type(bounds) is list and len(bounds) <= 16)
    _need(all(type(value) is str and len(value) <= 4096 for value in bounds))
    counts, rows = document["counts"], document["rows"]
    _need(type(counts) is dict and set(counts) == {"total", "formatChecked", "reviewRequired"})
    _need(type(rows) is list and 1 <= len(rows) <= 256)
    for value in counts.values():
        _integer(value, 256)
    _need(counts["total"] == len(rows))
    checked = 0
    for index, row in enumerate(rows, 1):
        _need(type(row) is dict and set(row) == {"line", "rowSha256", "state", "reasonCode",
                                               "toolCalls", "extraFieldsPresent"})
        _need(_integer(row["line"], 256) == index)
        _digest(row["rowSha256"])
        if row["state"] == "FORMAT_CHECKED":
            _need(row["reasonCode"] is None and type(row["extraFieldsPresent"]) is bool)
            _integer(row["toolCalls"], 1024 * 128)
            checked += 1
        else:
            _need(row["state"] == "REVIEW_REQUIRED"
                  and row["reasonCode"] == "unsupported_or_invalid_record"
                  and row["toolCalls"] is None and row["extraFieldsPresent"] is None)
    _need(counts["formatChecked"] == checked and counts["reviewRequired"] == len(rows) - checked)
    state = "FORMAT_CHECKED_NOT_ADMITTED" if checked == len(rows) else "REVIEW_REQUIRED"
    _need(document["state"] == state)
    generated = document["generatedAt"]
    _need(type(generated) is str and 0 < len(generated) <= 64)
    try:
        reported_at = datetime.fromisoformat(generated.replace("Z", "+00:00"))
        current = now or datetime.now(timezone.utc)
        _need(reported_at.utcoffset() is not None and current.utcoffset() is not None)
        age = (current - reported_at).total_seconds()
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("invalid_corpus_report_time") from exc
    _need(age >= -30)
    return {**unavailable_report(), "state": "REPORT_HASH_CHECKED_NOT_ADMITTED",
            "producer_state": state, "dataset_repo_id": document["datasetRepoId"],
            "source_revision_declared": document["sourceRevisionDeclared"],
            "dataset_revision_declared": document["datasetRevisionDeclared"],
            "report_sha256": expected, "report_generated_at": generated,
            "freshness": "STALE_REPORT" if age > 3600 else "RECENT_REPORT_NOT_RUNTIME_EVIDENCE",
            "counts": {"total": len(rows), "format_checked": checked,
                       "review_required": len(rows) - checked},
            "admission_reviews": "NOT_PERFORMED"}


def load_report(path: Path, expected_sha256: str, *, now: datetime | None = None) -> dict:
    """Read one explicitly configured owner-controlled report, bounded before JSON."""
    _digest(expected_sha256)
    return project_report(read_regular(path, MAX_REPORT_BYTES), expected_sha256, now=now)
