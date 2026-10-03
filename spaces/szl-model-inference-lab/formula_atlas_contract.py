#!/usr/bin/env python3
"""Validate and expose the immutable public Formula Atlas projection."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


SOURCE_ROOT = Path(__file__).resolve().parent
ATLAS_PATH = SOURCE_ROOT / "formula_atlas" / "formula-atlas.v1.json"
ATTRIBUTED_SOURCE_PATH = (
    SOURCE_ROOT / "formula_atlas" / "source-formula-ledger-corpus.json"
)

ATLAS_REPOSITORY = "szl-holdings/szl-formulas"
ATLAS_REVISION = "d03536144d2f23ee2da35c356e2558abe5183779"
ATLAS_SOURCE_PATH = "atlas/formula-atlas.v1.json"
ATLAS_GIT_BLOB_SHA = "1d3737cb3fc32ee17916f9dfe139a286a930a29f"
ATLAS_SHA256 = "765b9e5c9dd8d4fc5c7518a5b2f70cd6008918591cfa5edfc9868453cb30b7a2"
ATLAS_PAYLOAD_SHA256 = (
    "d7b08cde24f57a40de5707c4336ae41eae48aff9e91cb210ec3152ae8d3a8024"
)

ATTRIBUTED_SOURCE = {
    "repository": "szl-holdings/szl-formula-ledger",
    "revision": "ceaef540eba6c5acf85091faf4a20cd9aef480f9",
    "path": "formulas/corpus.json",
    "git_blob_sha": "600654f89cb062320acae4284cb5220ff881eb3c",
    "sha256": "c0b6dfee3233097307c518a57c076e5ba3ea9013f34ab454bef92aeddf0f3634",
    "state": "ARCHIVED_ATTRIBUTED_SOURCE",
}
LOCKED_PROVEN_IDS = {"F1", "F4", "F7", "F11", "F12", "F18", "F19", "F22"}
EXPECTED_CLASSES = {
    "CONJECTURE": 4,
    "DEFINITIONAL": 4,
    "DIMENSIONAL": 3,
    "EMPIRICAL": 4,
    "SYMBOLIC": 15,
}
EXPECTED_DOMAIN_COUNTS = {
    "algebra_number_theory": 4,
    "coding_error_control": 3,
    "dynamics_consensus": 5,
    "energy_entropy_physics": 4,
    "governance_receipts": 5,
    "information_geometry": 2,
    "narrative_lineage": 1,
    "topology_geometry": 1,
    "trust_aggregation": 5,
}
SOURCE_RECORD_KEYS = {"id", "source", "statement", "class", "reported_status"}
ATLAS_RECORD_KEYS = SOURCE_RECORD_KEYS | {
    "admission",
    "locked_proven_membership",
    "quant_domain",
}
SENSITIVE_VALUE = re.compile(
    r"(?:hf_[A-Za-z0-9._-]{16,}|gh[pousr]_[A-Za-z0-9]{16,}|"
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----)"
)


class FormulaAtlasIntegrityError(RuntimeError):
    """The materialized Formula Atlas no longer matches its immutable source."""


def _canonical_text_bytes(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise FormulaAtlasIntegrityError(
            f"formula asset is not a regular file: {path.name}"
        )
    raw = path.read_bytes()
    if b"\0" in raw:
        raise FormulaAtlasIntegrityError(
            f"formula asset contains NUL bytes: {path.name}"
        )
    canonical = raw.replace(b"\r\n", b"\n")
    if b"\r" in canonical:
        raise FormulaAtlasIntegrityError(
            f"formula asset contains a bare CR: {path.name}"
        )
    return canonical


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _git_blob_sha(raw: bytes) -> str:
    header = f"blob {len(raw)}\0".encode("ascii")
    return hashlib.sha1(header + raw, usedforsecurity=False).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _json_object(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FormulaAtlasIntegrityError(f"invalid {label} JSON") from exc
    if not isinstance(value, dict):
        raise FormulaAtlasIntegrityError(f"{label} must be a JSON object")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FormulaAtlasIntegrityError(message)


def _contains_sensitive_value(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            _contains_sensitive_value(key) or _contains_sensitive_value(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_sensitive_value(item) for item in value)
    return isinstance(value, str) and SENSITIVE_VALUE.search(value) is not None


def validate_formula_assets(
    atlas_path: Path = ATLAS_PATH,
    attributed_source_path: Path = ATTRIBUTED_SOURCE_PATH,
) -> dict[str, Any]:
    """Return the atlas only after byte, source, count, and authority validation."""

    atlas_raw = _canonical_text_bytes(atlas_path)
    source_raw = _canonical_text_bytes(attributed_source_path)
    _require(_sha256(atlas_raw) == ATLAS_SHA256, "formula atlas SHA-256 drift")
    _require(
        _git_blob_sha(atlas_raw) == ATLAS_GIT_BLOB_SHA,
        "formula atlas Git blob drift",
    )
    _require(
        _sha256(source_raw) == ATTRIBUTED_SOURCE["sha256"],
        "attributed formula source SHA-256 drift",
    )
    _require(
        _git_blob_sha(source_raw) == ATTRIBUTED_SOURCE["git_blob_sha"],
        "attributed formula source Git blob drift",
    )

    atlas = _json_object(atlas_raw, "formula atlas")
    source = _json_object(source_raw, "attributed formula source")
    _require(
        set(atlas)
        == {
            "attributed_formulas",
            "authority",
            "executable_formulas",
            "payload_sha256",
            "quant_domains",
            "schema",
            "source",
            "state",
            "summary",
        },
        "formula atlas top-level contract drift",
    )
    _require(atlas["schema"] == "szl.formula-quant-atlas/v1", "atlas schema drift")
    _require(
        atlas["state"] == "ATTRIBUTED_REFERENCE_PLUS_EXECUTABLE_KERNEL",
        "atlas state drift",
    )
    _require(atlas["source"] == ATTRIBUTED_SOURCE, "attributed source binding drift")

    expected_payload = atlas.pop("payload_sha256")
    _require(expected_payload == ATLAS_PAYLOAD_SHA256, "atlas payload pin drift")
    _require(
        _sha256(_canonical_json_bytes(atlas)) == expected_payload,
        "atlas canonical payload digest mismatch",
    )
    atlas["payload_sha256"] = expected_payload

    _require(set(source) == {"meta", "formulas"}, "attributed source shape drift")
    source_rows = source["formulas"]
    atlas_rows = atlas["attributed_formulas"]
    _require(
        isinstance(source_rows, list)
        and isinstance(atlas_rows, list)
        and len(source_rows) == len(atlas_rows) == 30,
        "attributed formula count drift",
    )
    _require(
        all(
            isinstance(row, dict) and set(row) == SOURCE_RECORD_KEYS
            for row in source_rows
        ),
        "attributed source record shape drift",
    )
    _require(
        all(
            isinstance(row, dict) and set(row) == ATLAS_RECORD_KEYS
            for row in atlas_rows
        ),
        "atlas record shape drift",
    )
    source_by_id = {row["id"]: row for row in source_rows}
    atlas_by_id = {row["id"]: row for row in atlas_rows}
    _require(
        len(source_by_id) == len(atlas_by_id) == 30
        and set(source_by_id) == set(atlas_by_id),
        "formula identifiers are duplicated or inconsistent",
    )
    for formula_id, row in atlas_by_id.items():
        source_row = source_by_id[formula_id]
        _require(
            all(row[key] == source_row[key] for key in SOURCE_RECORD_KEYS),
            f"formula attribution drift: {formula_id}",
        )
        _require(
            row["locked_proven_membership"]
            == "UNKNOWN_NOT_INFERRED_FROM_REPORTED_STATUS",
            f"formula status promoted locked membership: {formula_id}",
        )
        if row["class"] == "CONJECTURE":
            _require(
                row["admission"] == "OPEN_NOT_EXECUTION_AUTHORITY",
                f"conjecture gained authority: {formula_id}",
            )

    summary = atlas["summary"]
    _require(summary["attributed_formula_count"] == 30, "summary source count drift")
    _require(
        summary["executable_formula_count"] == 21, "summary executable count drift"
    )
    _require(summary["class_counts"] == EXPECTED_CLASSES, "formula class count drift")
    _require(
        summary["quant_domain_counts"] == EXPECTED_DOMAIN_COUNTS,
        "quant domain count drift",
    )
    _require(
        Counter(row["class"] for row in atlas_rows) == EXPECTED_CLASSES,
        "formula classes do not match summary",
    )
    _require(
        Counter(row["quant_domain"] for row in atlas_rows) == EXPECTED_DOMAIN_COUNTS,
        "formula domains do not match summary",
    )
    domains = atlas["quant_domains"]
    _require(isinstance(domains, list) and len(domains) == 9, "quant domain set drift")
    _require(
        {
            row["id"]: row["formula_count"]
            for row in domains
            if isinstance(row, dict)
            and set(row) == {"authority", "formula_count", "id"}
            and row["authority"] == "REFERENCE_AND_CONSTRAINT_INPUT_ONLY"
        }
        == EXPECTED_DOMAIN_COUNTS,
        "quant domain authority or counts drifted",
    )

    executable_rows = atlas["executable_formulas"]
    _require(
        isinstance(executable_rows, list)
        and len(executable_rows) == 21
        and all(
            isinstance(row, dict) and set(row) == {"name", "proof_status"}
            for row in executable_rows
        )
        and len({row["name"] for row in executable_rows}) == 21,
        "executable formula registry drift",
    )
    authority = atlas["authority"]
    _require(
        authority["executable_registry_repository"] == ATLAS_REPOSITORY
        and authority["executable_registry_count"] == 21,
        "executable registry authority drift",
    )
    _require(
        authority["locked_proven_count"] == 8
        and set(authority["locked_proven_ids"]) == LOCKED_PROVEN_IDS,
        "locked-proven authority drift",
    )
    _require(
        authority["lambda_status"] == "CONJECTURE_1_OPEN_ADVISORY_ONLY",
        "Lambda status drift",
    )
    _require(
        authority["f_number_to_executable_registry_mapping"] == "UNKNOWN_NOT_INFERRED",
        "unknown F-number mapping was promoted",
    )
    _require(
        not _contains_sensitive_value(source),
        "formula source contains secret-like data",
    )
    return atlas


def public_formula_atlas() -> dict[str, Any]:
    """Project immutable public metadata without formula statements or graph content."""

    atlas = validate_formula_assets()
    authority = atlas["authority"]
    payload = {
        "schema": "szl.formula-atlas.public/v2",
        "state": "VERIFIED_IMMUTABLE_PUBLIC_PROJECTION",
        "source": {
            "atlas": {
                "repository": ATLAS_REPOSITORY,
                "revision": ATLAS_REVISION,
                "path": ATLAS_SOURCE_PATH,
                "git_blob_sha": ATLAS_GIT_BLOB_SHA,
                "sha256": ATLAS_SHA256,
                "payload_sha256": ATLAS_PAYLOAD_SHA256,
            },
            "attributed_corpus": dict(ATTRIBUTED_SOURCE),
        },
        "counts": {
            **atlas["summary"],
            "quant_domain_count": len(atlas["quant_domains"]),
            "locked_proven_count": authority["locked_proven_count"],
        },
        "truth_and_proof_tiers": {
            "formula_class_counts": dict(atlas["summary"]["class_counts"]),
            "reported_status_authority": "ATTRIBUTED_SOURCE_ONLY",
            "locked_proven_membership": "SEPARATE_EXPLICIT_AUTHORITY_SET",
            "f_number_to_executable_registry_mapping": authority[
                "f_number_to_executable_registry_mapping"
            ],
            "rule": authority["rule"],
        },
        "quant_domains": [dict(row) for row in atlas["quant_domains"]],
        "formula_handles": [
            {
                "id": row["id"],
                "class": row["class"],
                "reported_status": row["reported_status"],
                "quant_domain": row["quant_domain"],
                "admission": row["admission"],
                "locked_proven_membership": row["locked_proven_membership"],
                "source_attribution": row["source"],
            }
            for row in atlas["attributed_formulas"]
        ],
        "executable_formula_handles": [
            dict(row) for row in atlas["executable_formulas"]
        ],
        "proof_authority": {
            "locked_proven_ids": list(authority["locked_proven_ids"]),
            "locked_proven_count": authority["locked_proven_count"],
            "lambda_status": authority["lambda_status"],
        },
        "authority": {
            "formula_may_authorize": False,
            "model_may_authorize": False,
            "public_effectors_enabled": False,
            "human_binding_required": True,
            "public_tools": [],
        },
        "privacy": {
            "content_access": "PUBLIC_FORMULA_METADATA_ONLY",
            "formula_statements_included": False,
            "second_brain_handles_included": False,
            "private_graph_content_included": False,
            "private_retrieval_performed": False,
        },
    }
    _require(
        not _contains_sensitive_value(payload), "public atlas contains secret-like data"
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    if not args.verify:
        parser.error("--verify is required")
    payload = public_formula_atlas()
    print(
        json.dumps(
            {
                "state": payload["state"],
                "counts": payload["counts"],
                "payload_sha256": payload["source"]["atlas"]["payload_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
