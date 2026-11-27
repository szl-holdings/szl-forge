# SPDX-License-Identifier: Apache-2.0
"""Offline readiness gate for the 2026-11-27 Q5–Q8 training lanes.

Reading a record is the check. This module does not start a job, call a
provider, import a model stack, or write to the Hub. Missing evidence stays
UNKNOWN. Q4 (Chaski 4B) is not an approved lane.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from training.transformers5_image import (
    NEMOTRON_LANE,
    PEFT,
    REJECTED_TRAINER,
    TRAIN_IMAGE,
    TRANSFORMERS,
    TRL,
)

SCHEMA = "szl.forge.q5-q8-lane/v1"
EVIDENCE_SCHEMA = "szl.forge.hf-api-observation/v1"
APPROVAL_DATE = "2026-11-27"
SHA40 = re.compile(r"[0-9a-f]{40}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
REPO_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*\Z")
CAPS = {"per_run_usd": "100", "per_day_usd": "500", "per_month_usd": "5000"}
PROVIDERS = ("salad", "massed")
APPROVED_GPU_HOURS = {"Q5": 16, "Q6": 36}
COMBINED_GPU_HOURS = 12
COMBINED_LANES = ("Q7", "Q8")
PINNED_DATA_LANES = ("Q6", "Q7", "Q8")
Q8_BASE_REPO = "Qwen/Qwen3-1.7B"
CONTAMINATION_NOT_RUN = (
    "GPQA",
    "szl-chaski/rotation heldouts",
    "AbstentionBench",
    "SimpleQA",
    "LiveCodeBench",
    "MTEB",
)
NOT_CHECKED_AS_PUBLISHED = (
    "gpqa_diamond (gated, 403)",
    "SZLHOLDINGS/szl-chaski-heldout-private (private; not on box; no named yes to fetch)",
    "SZLHOLDINGS/szl-heldout-rotation-private (not on box)",
    "AbstentionBench, SimpleQA (not loaded)",
    "LiveCodeBench (Q6 eval suite; not loaded)",
    "MTEB tasks (Q7 eval suite; not loaded)",
)
Q7_SOURCE_IDS = (
    "szl-estate-graph",
    "rag-corpus-v2",
    "doctrine-corpus-v2",
)
_PIN_KEYS = {
    "role",
    "source",
    "path",
    "rows",
    "sha256",
    "sha256_source",
    "lfs_oid",
    "training_eligible",
    "eval_only",
    "hub_repo",
    "hub_revision",
    "api_sha",
    "license",
    "evidence_id",
    "rights_admission",
}
_DATA_CHECK_KEYS = {"status", "embedding_similarity", "contamination_not_run"}

# Names that already exist or are explicitly off limits. A new lane must not
# reuse them. Comparison is case-insensitive because Hub ids are.
DENIED_TARGETS = frozenset(
    item.casefold()
    for item in (
        "SZLHOLDINGS/chaski",
        "SZLHOLDINGS/chaski-r2",
        "SZLHOLDINGS/chaski-r4",
        "SZLHOLDINGS/chaski-5050",
        "SZLHOLDINGS/a11oy-code-r1-candidate",
        "SZLHOLDINGS/a11oy-r1",
        "SZLHOLDINGS/a11oy-r2",
        "SZLHOLDINGS/A11OY-MINI",
        "SZLHOLDINGS/SZL-MiniEmbed-0.8B",
        "SZLHOLDINGS/MiniEmbed-Nano",
        "SZLHOLDINGS/brain-navigator-r2",
        "SZLHOLDINGS/brain-navigator-r2-laptop-lora",
        "SZLHOLDINGS/brain-navigator-r2-estate-sft-v1",
        "SZLHOLDINGS/brain-navigator-v2-trained",
        "SZLHOLDINGS/ReceiptAgent-Nano",
        "SZLHOLDINGS/SZL-Forge-1.5B-ReceiptAgent",
        "SZLHOLDINGS/szl-receiptagent-qwen35-0.8b-v2",
        "SZLHOLDINGS/szl-receiptagent-qwen35-0.8b-sft-v3",
        "SZLHOLDINGS/szl-receiptagent-qwen35-0.8b-v3-authenticated",
    )
)

LANE_FILES = {
    "Q4": "q4-chaski-4b.json",
    "Q5": "q5-receiptagent.manifest.json",
    "Q6": "q6-a11oy-code.manifest.json",
    "Q7": "q7-embedder.manifest.json",
    "Q8": "q8-brain-navigator.manifest.json",
}

_Q4_KEYS = {
    "schema",
    "id",
    "name",
    "status",
    "approval_date",
    "launch_authorized",
    "hub_write",
    "image_push",
    "provider_call",
    "gpu_hours",
    "target_repo",
    "base",
    "datasets",
    "blocker",
}
_LANE_KEYS = {
    "schema",
    "id",
    "name",
    "status",
    "approval_date",
    "launch_authorized",
    "hub_write",
    "image_push",
    "provider_call",
    "train_image",
    "runtime",
    "base",
    "replaced_base",
    "datasets",
    "heldout",
    "refused_corpora",
    "target_repo",
    "target_existence",
    "gpu_hours",
    "spend_caps",
    "dry_run",
    "data_check",
    "blocker",
    "launch_note",
}


class LaneError(ValueError):
    """A readiness record failed closed. The argument is a stable code."""


def root() -> Path:
    return Path(__file__).resolve().parents[1]


def lane_dir() -> Path:
    return root() / "frontier" / "q5-q8-readiness"


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise LaneError("DUPLICATE_JSON_MEMBER")
        result[key] = value
    return result


def _refuse_float(_value: str) -> None:
    raise LaneError("JSON_FLOAT_FORBIDDEN")


def load_json(path: Path) -> dict[str, Any]:
    try:
        parsed = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_pairs,
            parse_float=_refuse_float,
            parse_constant=_refuse_float,
        )
    except LaneError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LaneError("LANE_JSON_UNREADABLE") from exc
    if not isinstance(parsed, dict):
        raise LaneError("LANE_OBJECT_REQUIRED")
    return parsed


def load_evidence(directory: Path | None = None) -> dict[str, Any]:
    path = (directory or lane_dir()) / "hf-api-evidence.json"
    document = load_json(path)
    if document.get("schema") != EVIDENCE_SCHEMA:
        raise LaneError("EVIDENCE_SCHEMA")
    if document.get("observed_at") != APPROVAL_DATE:
        raise LaneError("EVIDENCE_DATE")
    rows = document.get("observations")
    if not isinstance(rows, list) or not rows:
        raise LaneError("EVIDENCE_EMPTY")
    by_id: dict[str, Any] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise LaneError("EVIDENCE_ROW")
        ident = row.get("id")
        if not isinstance(ident, str) or ident in by_id:
            raise LaneError("EVIDENCE_ID")
        by_id[ident] = row
    document["by_id"] = by_id
    return document


def _keys(document: dict[str, Any], expected: set[str]) -> None:
    if set(document) != expected:
        raise LaneError("LANE_KEYS")


def _closed(document: dict[str, Any]) -> None:
    if document.get("launch_authorized") is not False:
        raise LaneError("LAUNCH_AUTHORIZED")
    if document.get("hub_write") is not False:
        raise LaneError("HUB_WRITE")
    if document.get("image_push") is not False:
        raise LaneError("IMAGE_PUSH")
    if document.get("provider_call") is not False:
        raise LaneError("PROVIDER_CALL")


def _sha(value: object, code: str) -> str:
    if not isinstance(value, str) or SHA40.fullmatch(value) is None:
        raise LaneError(code)
    return value


def _observation(evidence: dict[str, Any], ident: object) -> dict[str, Any]:
    if not isinstance(ident, str):
        raise LaneError("EVIDENCE_ID")
    row = evidence["by_id"].get(ident)
    if not isinstance(row, dict):
        raise LaneError("EVIDENCE_ID")
    return row


def _match_base(base: object, evidence: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(base, dict):
        raise LaneError("BASE")
    row = _observation(evidence, base.get("evidence_id"))
    if row.get("kind") != "model":
        raise LaneError("BASE_KIND")
    if base.get("repo") != row.get("repo"):
        raise LaneError("BASE_REPO")
    if base.get("revision") != row.get("sha"):
        raise LaneError("BASE_SHA")
    _sha(base.get("revision"), "BASE_SHA")
    if base.get("license") != row.get("license"):
        raise LaneError("BASE_LICENSE")
    if base.get("license") != "apache-2.0":
        raise LaneError("BASE_LICENSE")
    if base.get("license_name") != row.get("license_name"):
        raise LaneError("BASE_LICENSE_NAME")
    if base.get("license_verification") != "HF_API":
        raise LaneError("BASE_LICENSE_VERIFICATION")
    if base.get("pipeline_tag") != row.get("pipeline_tag"):
        raise LaneError("BASE_PIPELINE")
    return base


def _replaced(value: object, evidence: dict[str, Any], training_repo: str) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        raise LaneError("REPLACED_BASE")
    row = _observation(evidence, value.get("evidence_id"))
    if value.get("repo") != row.get("repo") or value.get("revision") != row.get("sha"):
        raise LaneError("REPLACED_BASE")
    if value.get("license") != row.get("license"):
        raise LaneError("REPLACED_BASE")
    if value.get("license_name") != "qwen-research":
        raise LaneError("REPLACED_BASE")
    if value.get("role") != "INCUMBENT_NOT_THE_TRAINING_BASE":
        raise LaneError("REPLACED_BASE")
    if value.get("repo") == training_repo:
        raise LaneError("REPLACED_BASE_IS_TRAINING_BASE")


def _runtime(document: dict[str, Any]) -> None:
    runtime = document.get("runtime")
    if not isinstance(runtime, dict):
        raise LaneError("RUNTIME")
    if document.get("train_image") != TRAIN_IMAGE or runtime.get("image") != TRAIN_IMAGE:
        raise LaneError("TRAIN_IMAGE")
    if runtime.get("transformers") != TRANSFORMERS:
        raise LaneError("RUNTIME_PIN")
    if runtime.get("peft") != PEFT or runtime.get("trl") != TRL:
        raise LaneError("RUNTIME_PIN")
    if runtime.get("rejected_trainer") != REJECTED_TRAINER:
        raise LaneError("REJECTED_TRAINER")
    if runtime.get("nemotron_lane") != NEMOTRON_LANE:
        raise LaneError("NEMOTRON_LANE")
    if runtime.get("duplicates_nemotron_trainer") is not False:
        raise LaneError("NEMOTRON_DUPLICATE")


def _target(value: object) -> str:
    if not isinstance(value, str) or REPO_ID.fullmatch(value) is None:
        raise LaneError("TARGET_REPO")
    if value.casefold() in DENIED_TARGETS:
        raise LaneError("TARGET_OVERWRITE")
    lowered = value.casefold()
    if lowered.endswith("/chaski") or lowered.endswith("/chaski-r4"):
        raise LaneError("TARGET_OVERWRITE")
    return value


def _spend(value: object) -> None:
    if not isinstance(value, dict):
        raise LaneError("SPEND_CAPS")
    if value.get("currency") != "USD":
        raise LaneError("SPEND_CAPS")
    for key, amount in CAPS.items():
        if value.get(key) != amount:
            raise LaneError("SPEND_CAPS")
    allowed = value.get("allowed_providers")
    if allowed != list(PROVIDERS):
        raise LaneError("PROVIDER")
    if value.get("selected_provider") != "UNKNOWN":
        raise LaneError("PROVIDER")


def _gpu(lane_id: str, value: object) -> int:
    if not isinstance(value, dict):
        raise LaneError("GPU_HOURS")
    estimate = value.get("estimate")
    if not isinstance(estimate, int) or isinstance(estimate, bool) or estimate < 1:
        raise LaneError("GPU_HOURS")
    if value.get("measured") != "UNKNOWN":
        raise LaneError("GPU_HOURS_MEASURED")
    if not isinstance(value.get("basis"), str) or not value["basis"].strip():
        raise LaneError("GPU_HOURS_BASIS")
    if lane_id in APPROVED_GPU_HOURS:
        approved = APPROVED_GPU_HOURS[lane_id]
        if value.get("approved") != approved or value.get("approved_combined") is not None:
            raise LaneError("GPU_HOURS_APPROVED")
        if value.get("approved_scope") != lane_id:
            raise LaneError("GPU_HOURS_APPROVED")
        if estimate > approved:
            raise LaneError("GPU_HOURS_OVER_APPROVAL")
        return estimate
    if lane_id in COMBINED_LANES:
        if value.get("approved") is not None:
            raise LaneError("GPU_HOURS_APPROVED")
        if value.get("approved_combined") != COMBINED_GPU_HOURS:
            raise LaneError("GPU_HOURS_APPROVED")
        if value.get("approved_scope") != "Q7+Q8":
            raise LaneError("GPU_HOURS_APPROVED")
        if estimate > COMBINED_GPU_HOURS:
            raise LaneError("GPU_HOURS_OVER_APPROVAL")
        return estimate
    raise LaneError("GPU_HOURS")


def _heldout_present(heldout: dict[str, Any], datasets: list[Any]) -> bool:
    if heldout.get("status") != "PRESENT" or heldout.get("eval_only") is not True:
        return False
    source = heldout.get("source")
    matched = [
        row
        for row in datasets
        if isinstance(row, dict)
        and row.get("source") == source
        and row.get("eval_only") is True
        and row.get("training_eligible") is False
    ]
    return len(matched) == 1


def _datasets(value: object) -> list[Any]:
    if not isinstance(value, list):
        raise LaneError("DATASETS")
    for row in value:
        if not isinstance(row, dict):
            raise LaneError("DATASETS")
        eligible = row.get("training_eligible")
        eval_only = row.get("eval_only")
        if not isinstance(eligible, bool) or not isinstance(eval_only, bool):
            raise LaneError("DATASETS")
        if eligible and eval_only:
            raise LaneError("HELDOUT_TRAINABLE")
        if eval_only and eligible:
            raise LaneError("HELDOUT_TRAINABLE")
        digest = row.get("sha256")
        if digest is not None and (
            not isinstance(digest, str) or SHA256.fullmatch(digest) is None
        ):
            raise LaneError("DATASET_SHA")
    return value


def _dry_run(value: object) -> None:
    if not isinstance(value, dict):
        raise LaneError("DRY_RUN")
    if value.get("device") != "cpu" or value.get("steps") != 2:
        raise LaneError("DRY_RUN")
    kind = value.get("kind")
    seed = value.get("seed")
    width = value.get("width")
    lr = value.get("lr_milli")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 1:
        raise LaneError("DRY_RUN")
    if not isinstance(width, int) or width < 2 or width > 32:
        raise LaneError("DRY_RUN")
    if not isinstance(lr, int) or lr < 1 or lr > 1000:
        raise LaneError("DRY_RUN")
    if kind == "causal":
        vocab = value.get("vocab")
        seq = value.get("seq")
        if not isinstance(vocab, int) or vocab < 4:
            raise LaneError("DRY_RUN")
        if not isinstance(seq, int) or not 2 <= seq <= 8:
            raise LaneError("DRY_RUN")
        if "batch" in value:
            raise LaneError("DRY_RUN")
        return
    if kind == "contrastive":
        batch = value.get("batch")
        if not isinstance(batch, int) or batch < 2 or batch > 8:
            raise LaneError("DRY_RUN")
        if "vocab" in value or "seq" in value:
            raise LaneError("DRY_RUN")
        return
    raise LaneError("DRY_RUN")


def _validate_q4(document: dict[str, Any]) -> dict[str, Any]:
    _keys(document, _Q4_KEYS)
    if document.get("schema") != SCHEMA or document.get("id") != "Q4":
        raise LaneError("Q4")
    if document.get("name") != "chaski-4b":
        raise LaneError("Q4")
    if document.get("status") != "NOT_APPROVED":
        raise LaneError("Q4_APPROVED")
    if document.get("approval_date") != APPROVAL_DATE:
        raise LaneError("Q4")
    _closed(document)
    if any(document.get(key) is not None for key in ("gpu_hours", "target_repo", "base", "datasets")):
        raise LaneError("Q4_HAS_LANE")
    blocker = document.get("blocker")
    if not isinstance(blocker, str) or "NOT_APPROVED" not in blocker:
        raise LaneError("Q4")
    return document


def _unknown(value: object, code: str) -> None:
    if value != "UNKNOWN":
        raise LaneError(code)


def _data_check(lane_id: str, value: object) -> dict[str, Any] | None:
    if lane_id == "Q5":
        if value is not None:
            raise LaneError("DATA_CHECK")
        return None
    if lane_id not in PINNED_DATA_LANES or not isinstance(value, dict):
        raise LaneError("DATA_CHECK")
    if set(value) != _DATA_CHECK_KEYS:
        raise LaneError("DATA_CHECK")
    if value.get("status") != "PASS_WITH_CONDITIONS":
        raise LaneError("DATA_CHECK")
    if value.get("embedding_similarity") != "UNAVAILABLE":
        raise LaneError("DATA_CHECK")
    if value.get("contamination_not_run") != list(CONTAMINATION_NOT_RUN):
        raise LaneError("DATA_CHECK")
    return value


def _pin_keys(lane_id: str, row: dict[str, Any]) -> None:
    expected = set(_PIN_KEYS)
    if lane_id == "Q6" and row.get("role") == "train":
        expected.add("built_from")
    if lane_id == "Q7" and row.get("role") == "train":
        expected.update({"sources", "attribution"})
    if set(row) != expected:
        raise LaneError("DATASET_KEYS")


def _role_flags(row: dict[str, Any]) -> None:
    role = row.get("role")
    eligible = row.get("training_eligible")
    eval_only = row.get("eval_only")
    if role == "train":
        if eligible is not True or eval_only is not False:
            raise LaneError("DATASET_ROLE")
        return
    if role in {"heldout", "heldout_abstain"}:
        if eligible is not False or eval_only is not True:
            raise LaneError("DATASET_ROLE")
        return
    raise LaneError("DATASET_ROLE")


def _match_file(row: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    observed = _observation(evidence, row.get("evidence_id"))
    if observed.get("kind") != "dataset" or observed.get("private") is not True:
        raise LaneError("DATASET_EVIDENCE")
    if row.get("hub_repo") != observed.get("repo"):
        raise LaneError("DATASET_REPO")
    if row.get("hub_revision") != observed.get("named_revision"):
        raise LaneError("DATASET_REVISION")
    _sha(row.get("hub_revision"), "DATASET_REVISION")
    _unknown(observed.get("api_sha"), "DATASET_API_SHA")
    _unknown(row.get("api_sha"), "DATASET_API_SHA")
    if observed.get("api_http_status") != 401:
        raise LaneError("DATASET_API_SHA")
    if row.get("sha256_source") != "SHA256SUMS" or observed.get("sha256_source") != "SHA256SUMS":
        raise LaneError("DATASET_SHA_SOURCE")
    _unknown(row.get("lfs_oid"), "DATASET_LFS_OID")
    _unknown(observed.get("lfs_oid"), "DATASET_LFS_OID")
    if row.get("license") != observed.get("license"):
        raise LaneError("DATASET_LICENSE")
    if observed.get("license_verification") != "HUB_CARD":
        raise LaneError("DATASET_LICENSE")
    if observed.get("not_checked_as_published") != list(NOT_CHECKED_AS_PUBLISHED):
        raise LaneError("CONTAMINATION_GAPS")
    files = observed.get("files")
    if not isinstance(files, list):
        raise LaneError("DATASET_FILE")
    matched = [
        item
        for item in files
        if isinstance(item, dict) and item.get("path") == row.get("path")
    ]
    if len(matched) != 1:
        raise LaneError("DATASET_FILE")
    item = matched[0]
    digest = row.get("sha256")
    if (
        not isinstance(digest, str)
        or SHA256.fullmatch(digest) is None
        or digest != item.get("sha256")
        or item.get("role") != row.get("role")
    ):
        raise LaneError("DATASET_SHA")
    rows = row.get("rows")
    if isinstance(rows, bool) or not isinstance(rows, int) or rows < 1 or rows != item.get("rows"):
        raise LaneError("DATASET_ROWS")
    if row.get("source") != f"{row.get('hub_repo')}/{row.get('path')}":
        raise LaneError("DATASET_SOURCE")
    if row.get("rights_admission") != "UNKNOWN":
        raise LaneError("RIGHTS_ADMISSION")
    return observed


def _built_from(value: object, evidence: dict[str, Any]) -> None:
    if not isinstance(value, dict):
        raise LaneError("BUILT_FROM")
    if set(value) != {"repo", "revision", "evidence_id"}:
        raise LaneError("BUILT_FROM")
    observed = _observation(evidence, value.get("evidence_id"))
    if value.get("repo") != observed.get("repo") or value.get("revision") != observed.get("sha"):
        raise LaneError("BUILT_FROM")
    _sha(value.get("revision"), "BUILT_FROM")


def _q7_sources(value: object, evidence: dict[str, Any]) -> None:
    if not isinstance(value, list) or len(value) != len(Q7_SOURCE_IDS):
        raise LaneError("DATASET_SOURCES")
    for evidence_id, source in zip(Q7_SOURCE_IDS, value, strict=True):
        if not isinstance(source, dict):
            raise LaneError("DATASET_SOURCES")
        if set(source) != {"repo", "revision", "license", "evidence_id"}:
            raise LaneError("DATASET_SOURCES")
        if source.get("evidence_id") != evidence_id:
            raise LaneError("DATASET_SOURCES")
        observed = _observation(evidence, evidence_id)
        if observed.get("kind") != "dataset" or observed.get("private") is not False:
            raise LaneError("DATASET_SOURCES")
        if source.get("repo") != observed.get("repo"):
            raise LaneError("DATASET_SOURCES")
        if source.get("revision") != observed.get("sha"):
            raise LaneError("DATASET_SOURCES")
        _sha(source.get("revision"), "DATASET_SOURCES")
        if source.get("license") != observed.get("license"):
            raise LaneError("DATASET_SOURCES")
        if observed.get("license_verification") != "HF_API":
            raise LaneError("DATASET_SOURCES")
        if evidence_id == "doctrine-corpus-v2" and observed.get("license") != "cc-by-4.0":
            raise LaneError("DATASET_SOURCES")
        if evidence_id != "doctrine-corpus-v2" and observed.get("license") != "apache-2.0":
            raise LaneError("DATASET_SOURCES")


def _attribution(value: object, evidence: dict[str, Any]) -> None:
    if not isinstance(value, dict):
        raise LaneError("ATTRIBUTION")
    if set(value) != {
        "license",
        "source_repo",
        "source_revision",
        "card_declares_per_row",
        "rows_verified",
    }:
        raise LaneError("ATTRIBUTION")
    doctrine = _observation(evidence, "doctrine-corpus-v2")
    if value.get("license") != "cc-by-4.0" or doctrine.get("license") != "cc-by-4.0":
        raise LaneError("ATTRIBUTION")
    if value.get("source_repo") != doctrine.get("repo"):
        raise LaneError("ATTRIBUTION")
    if value.get("source_revision") != doctrine.get("sha"):
        raise LaneError("ATTRIBUTION")
    _sha(value.get("source_revision"), "ATTRIBUTION")
    if value.get("card_declares_per_row") is not True:
        raise LaneError("ATTRIBUTION")
    _unknown(value.get("rows_verified"), "ATTRIBUTION")


def _bind_pins(
    lane_id: str,
    document: dict[str, Any],
    datasets: list[Any],
    evidence: dict[str, Any],
) -> None:
    """Match Q6–Q8 rows to Hub observations. Unread SHAs stay UNKNOWN."""

    if not datasets:
        raise LaneError("DATASET_EVIDENCE")
    seen: set[str] = set()
    observed: dict[str, Any] | None = None
    for row in datasets:
        if not isinstance(row, dict):
            raise LaneError("DATASETS")
        _pin_keys(lane_id, row)
        _role_flags(row)
        matched = _match_file(row, evidence)
        if observed is None:
            observed = matched
        elif matched is not observed:
            raise LaneError("DATASET_EVIDENCE")
        role = str(row.get("role"))
        if role in seen:
            raise LaneError("DATASET_ROLE")
        seen.add(role)
        if lane_id == "Q6" and role == "train":
            _built_from(row.get("built_from"), evidence)
        if lane_id == "Q7" and role == "train":
            _q7_sources(row.get("sources"), evidence)
            _attribution(row.get("attribution"), evidence)
    if observed is None or document.get("data_check") != observed.get("data_check"):
        raise LaneError("DATA_CHECK")
    expected = {"train", "heldout", "heldout_abstain"} if lane_id == "Q8" else {"train", "heldout"}
    if seen != expected:
        raise LaneError("DATASET_ROLE")


def validate_lane(document: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    """Return the record when it obeys the approval, else raise ``LaneError``."""

    if document.get("id") == "Q4":
        return _validate_q4(document)
    _keys(document, _LANE_KEYS)
    if document.get("schema") != SCHEMA:
        raise LaneError("LANE_SCHEMA")
    lane_id = document.get("id")
    if lane_id not in {"Q5", "Q6", "Q7", "Q8"}:
        raise LaneError("LANE_ID")
    if document.get("approval_date") != APPROVAL_DATE:
        raise LaneError("APPROVAL_DATE")
    _closed(document)
    _runtime(document)
    base = _match_base(document.get("base"), evidence)
    if lane_id == "Q8" and base.get("repo") != Q8_BASE_REPO:
        raise LaneError("Q8_BASE")
    _replaced(document.get("replaced_base"), evidence, str(base.get("repo")))
    if lane_id == "Q6" and document.get("replaced_base") is None:
        raise LaneError("REPLACED_BASE")
    if lane_id != "Q6" and document.get("replaced_base") is not None:
        raise LaneError("REPLACED_BASE")
    datasets = _datasets(document.get("datasets"))
    checked = _data_check(str(lane_id), document.get("data_check"))
    heldout = document.get("heldout")
    if not isinstance(heldout, dict):
        raise LaneError("HELDOUT")
    if heldout.get("independence") != "UNKNOWN":
        raise LaneError("HELDOUT_INDEPENDENCE")
    present = _heldout_present(heldout, datasets)
    status = document.get("status")
    blocker = document.get("blocker")
    if status == "READY_NOT_LAUNCHED":
        if not present or blocker is not None:
            raise LaneError("READY_WITHOUT_HELDOUT")
    elif status == "BLOCKED":
        if present or not isinstance(blocker, str) or not blocker.strip():
            raise LaneError("BLOCKED_WITHOUT_GAP")
        if heldout.get("status") not in {"ABSENT", "UNKNOWN"}:
            raise LaneError("HELDOUT")
    else:
        raise LaneError("LANE_STATUS")
    if lane_id in PINNED_DATA_LANES:
        if status != "READY_NOT_LAUNCHED" or checked is None:
            raise LaneError("LANE_STATUS")
        _bind_pins(str(lane_id), document, datasets, evidence)
    if not isinstance(document.get("refused_corpora"), list):
        raise LaneError("REFUSED_CORPORA")
    _target(document.get("target_repo"))
    if document.get("target_existence") != "UNKNOWN":
        raise LaneError("TARGET_EXISTENCE")
    _gpu(str(lane_id), document.get("gpu_hours"))
    _spend(document.get("spend_caps"))
    _dry_run(document.get("dry_run"))
    note = document.get("launch_note")
    if not isinstance(note, str) or "UNKNOWN" not in note:
        raise LaneError("LAUNCH_NOTE")
    if document.get("launch_authorized") is not False:
        raise LaneError("LAUNCH_AUTHORIZED")
    return document


def combined_gpu_hours(directory: Path, evidence: dict[str, Any]) -> int:
    """Sum Q7 and Q8. Each lane is validated, including the 12-hour ceiling."""

    total = 0
    for lane_id in COMBINED_LANES:
        document = load_json(directory / LANE_FILES[lane_id])
        if document.get("id") != lane_id:
            raise LaneError("LANE_ID")
        validated = validate_lane(document, evidence)
        total += int(validated["gpu_hours"]["estimate"])
    if total > COMBINED_GPU_HOURS:
        raise LaneError("GPU_HOURS_COMBINED")
    return total


def load_lane(lane_id: str, directory: Path | None = None) -> dict[str, Any]:
    if lane_id not in LANE_FILES:
        raise LaneError("LANE_ID")
    directory = directory or lane_dir()
    evidence = load_evidence(directory)
    document = load_json(directory / LANE_FILES[lane_id])
    if document.get("id") != lane_id:
        raise LaneError("LANE_ID")
    validated = validate_lane(document, evidence)
    if lane_id in COMBINED_LANES:
        combined_gpu_hours(directory, evidence)
    return validated


def load_lanes(directory: Path | None = None) -> dict[str, dict[str, Any]]:
    directory = directory or lane_dir()
    evidence = load_evidence(directory)
    loaded: dict[str, dict[str, Any]] = {}
    for lane_id, name in LANE_FILES.items():
        document = load_json(directory / name)
        if document.get("id") != lane_id:
            raise LaneError("LANE_ID")
        loaded[lane_id] = validate_lane(document, evidence)
    combined_gpu_hours(directory, evidence)
    targets = [loaded[lane_id]["target_repo"] for lane_id in ("Q5", "Q6", "Q7", "Q8")]
    if len({item.casefold() for item in targets}) != len(targets):
        raise LaneError("TARGET_REPO")
    return loaded


def readiness_rows(directory: Path | None = None) -> list[dict[str, Any]]:
    """One row per approved lane. Q4 is reported separately by ``load_lane``."""

    lanes = load_lanes(directory)
    rows = []
    for lane_id in ("Q5", "Q6", "Q7", "Q8"):
        lane = lanes[lane_id]
        base = lane["base"]
        rows.append(
            {
                "id": lane_id,
                "base": f"{base['repo']}@{base['revision']}",
                "license": base["license"],
                "heldout": lane["heldout"]["status"],
                "target_repo": lane["target_repo"],
                "gpu_hours": lane["gpu_hours"]["estimate"],
                "status": lane["status"],
                "blocker": lane["blocker"] if lane["blocker"] is not None else lane["launch_note"],
                "launch_authorized": False,
            }
        )
    return rows
