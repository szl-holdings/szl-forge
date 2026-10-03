#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Train-only source conformance; no signing, training or execution authority.

The temporary file contains only already bounded and strictly checked training
bytes. Held-out files are never opened, discovered or scanned. Exit zero means
structural + generic-SFT + Nemo conformance, not training authorization.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import re
import stat
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import json_contract as contract

TOOLS = HERE.parents[1] / "tools"
SCHEMA = "szl.receiptagent.v4-curriculum/v1"
REPORT_SCHEMA = "szl.receiptagent.v4-curriculum-conformance/v1"
CONTRACT_REVISION = "85f1067edb395405a8d459e5779e8cff53147fb6"
CONTRACT_SOURCE_SHA256 = "373451dff44ba0fcd9f6cf445f19737ebf7566855a741377c7b0fa8388247b88"
MANIFEST_SHA256 = "7ec31cfddf31228164a7a67e7edd8152c611c2f04a4860d2ad85c168339337ae"
GATE_HASHES = {
    "validate_sft_dataset.py": "4084ec4ef13c40df2a4957dd7965be36704698d77c1b1bf650e03cc439aa381e",
    "nemo_doctrine_gate.py": "2c4c4758abb23a3b6055038059984a5559186b7d82c10c8468312db18632e8a0",
}
SEED = 2601003
MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_ROWS = 360
TRAIN_FAMILIES = {"dependency-lock-audit", "build-artifact-lineage",
                  "documentation-link-inventory", "ci-cache-key-review"}
DEV_FAMILIES = {"cli-option-manifest", "frontend-static-asset-index"}
HELDOUT_FAMILIES = {"package-license-ledger", "schema-migration-compatibility",
                    "repository-access-matrix"}
BLOCKED_AUTHORITIES = {"APPROVE", "EXECUTE", "FABRICATE", "DISCLOSE_SECRET",
                       "RESEND_QUARANTINED"}
MANIFEST_KEYS = {"schema", "candidate_id", "seed", "contract_source_revision",
                 "contract_source_sha256", "train_families", "dev_families",
                 "heldout_families", "splits", "training_eligible",
                 "publication_eligible", "execution_authority", "evidence_class",
                 "signature", "runtime_binding"}
ROW_KEYS = {"id", "split", "family", "kind", "messages"}


class AdmissionError(ValueError):
    """Invalid or incomplete source conformance must fail closed."""


class GateNotReady(AdmissionError):
    """Missing required tools or doctrine kernel is not a conformance pass."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AdmissionError(message)


def _keys(value: Any, expected: set[str], label: str) -> None:
    _require(isinstance(value, dict) and set(value) == expected,
             f"{label} must have exactly the declared fields")


def _sha(value: Any, label: str) -> None:
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
             f"{label} must be an exact lowercase SHA256")


def _read_once(path: Path, maximum: int) -> bytes:
    """Reject aliases and detect replacement/change around a bounded read."""
    absolute = Path(os.path.abspath(path))
    for component in (absolute, *absolute.parents):
        metadata = component.lstat()
        if stat.S_ISLNK(metadata.st_mode) or getattr(metadata, "st_file_attributes", 0) & 0x400:
            raise AdmissionError("input path contains a link or reparse point")
    before = absolute.lstat()
    _require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1,
             "input must be a non-hardlinked regular file")
    _require(before.st_size <= maximum, "input byte limit exceeded")
    descriptor = os.open(absolute, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        _require((opened.st_dev, opened.st_ino) == (before.st_dev, before.st_ino),
                 "input changed before read")
        _require(stat.S_ISREG(opened.st_mode) and opened.st_nlink == 1,
                 "opened input is not an unaliased regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            raw = handle.read(maximum + 1)
        after = os.fstat(descriptor)
        _require(len(raw) <= maximum, "input byte limit exceeded")
        _require(len(raw) == opened.st_size, "input did not yield the declared bytes")
        _require((opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns) ==
                 (after.st_size, after.st_mtime_ns, after.st_ctime_ns),
                 "input changed during read")
        current = absolute.lstat()
        _require((current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns,
                  current.st_ctime_ns, current.st_nlink) ==
                 (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
                  after.st_ctime_ns, 1), "input changed after read")
        return raw
    finally:
        os.close(descriptor)


def _strict(raw: bytes) -> dict[str, Any]:
    try:
        return contract.strict_object(raw.decode("utf-8", errors="strict"))
    except (UnicodeError, contract.ContractError) as exc:
        raise AdmissionError("invalid bounded UTF8 JSON object") from exc


def validate_manifest(manifest: dict[str, Any]) -> None:
    """Validate commitments only; never inspect held-out source or content."""
    _keys(manifest, MANIFEST_KEYS, "manifest")
    for key, expected in (("schema", SCHEMA), ("candidate_id", contract.PROFILE),
                          ("contract_source_revision", CONTRACT_REVISION),
                          ("evidence_class", "SIMULATED"), ("signature", "UNAVAILABLE"),
                          ("runtime_binding", "UNAVAILABLE")):
        _require(manifest[key] == expected, f"manifest {key} differs from source contract")
    _require(type(manifest["seed"]) is int and manifest["seed"] == SEED, "wrong seed")
    for key in ("training_eligible", "publication_eligible", "execution_authority"):
        _require(manifest[key] is False, "source conformance cannot grant authority")
    _sha(manifest["contract_source_sha256"], "contract source digest")
    source_sha = hashlib.sha256(_read_once(HERE / "json_contract.py", contract.MAX_BYTES)).hexdigest()
    _require(source_sha == manifest["contract_source_sha256"] == CONTRACT_SOURCE_SHA256,
             "contract source bytes drifted from the pinned revision")
    families: list[set[str]] = []
    for key, count in (("train_families", 4), ("dev_families", 2), ("heldout_families", 3)):
        declared = manifest[key]
        _require(isinstance(declared, list) and len(declared) == count and all(
            isinstance(name, str) and re.fullmatch(r"[a-z][a-z0-9-]{3,79}", name)
            for name in declared), f"invalid {key}")
        _require(len(set(declared)) == count, "duplicate declared task family")
        families.append(set(declared))
    _require(families[0] == TRAIN_FAMILIES, "training family allowlist differs")
    _require(families[1] == DEV_FAMILIES and families[2] == HELDOUT_FAMILIES,
             "nontraining family commitments differ from the preregistration")
    _require(not any(families[a] & families[b] for a, b in ((0, 1), (0, 2), (1, 2))),
             "cross-split family overlap")
    _keys(manifest["splits"], {"train", "dev", "test"}, "split commitments")
    for split, path, rows, per_kind in (("train", "train.jsonl", 360, 120),
                                       ("dev", "dev.jsonl", 90, 30),
                                       ("test", "heldout/test.jsonl", 180, 60)):
        entry = manifest["splits"][split]
        _keys(entry, {"path", "sha256", "rows", "kind_counts"}, f"{split} commitment")
        _require(entry["path"] == path, "split path is not allowlisted")
        _sha(entry["sha256"], f"{split} digest")
        _require(type(entry["rows"]) is int and entry["rows"] == rows,
                 "split row commitment differs")
        _keys(entry["kind_counts"], set(contract.CLAIMS), "kind counts")
        _require(all(type(value) is int and value == per_kind
                     for value in entry["kind_counts"].values()), "wrong class coverage")
    _require(len({entry["sha256"] for entry in manifest["splits"].values()}) == 3,
             "split digests must differ")


def validate_train_bytes(raw: bytes, manifest: dict[str, Any]) -> dict[str, Any]:
    """Strictly bind only train rows to the manifest and controller contract."""
    _require(0 < len(raw) <= MAX_FILE_BYTES, "training byte limit exceeded")
    _require(raw.endswith(b"\n") and b"\r" not in raw, "training bytes must be LF JSONL")
    lines = raw.split(b"\n")[:-1]
    _require(len(lines) == MAX_ROWS, "training must contain exactly 360 nonblank rows")
    expected = manifest["splits"]["train"]
    digest = hashlib.sha256(raw).hexdigest()
    _require(digest == expected["sha256"], "training digest differs from manifest")
    ids: set[str] = set()
    kinds: Counter[str] = Counter()
    families: Counter[str] = Counter()
    family_kinds: Counter[tuple[str, str]] = Counter()
    recovery_statuses: Counter[tuple[str, str]] = Counter()
    refused_authorities: Counter[tuple[str, str]] = Counter()
    for line in lines:
        row = _strict(line)
        _keys(row, ROW_KEYS, "training row")
        _require(line == contract.canonical_json(row).encode("utf-8"), "row is not canonical JSON")
        _require(row["split"] == "train", "nontraining split is excluded")
        _require(isinstance(row["family"], str) and row["family"] in TRAIN_FAMILIES,
                 "unknown or held-out task family")
        _require(isinstance(row["kind"], str) and row["kind"] in contract.CLAIMS,
                 "unknown row kind")
        messages = row["messages"]
        _require(isinstance(messages, list) and len(messages) == 3, "exactly three messages required")
        for message, role in zip(messages, ("system", "user", "assistant")):
            _keys(message, {"role", "content"}, "message")
            _require(message["role"] == role and isinstance(message["content"], str),
                     "exact system/user/assistant roles and string contents required")
        _require(messages[0]["content"] == contract.SYSTEM_PROMPT, "system prompt differs")
        envelope = _strict(messages[1]["content"].encode("utf-8"))
        _keys(envelope, {"request", "requestSha256"}, "controller prompt envelope")
        request = envelope["request"]
        _require(isinstance(request, dict), "request must be an object")
        request_text = contract.canonical_json(request)
        _require(messages[1]["content"] == contract.prompt_envelope(request_text),
                 "controller envelope digest or canonical bytes differ")
        content = {key: value for key, value in request.items() if key != "requestId"}
        expected_id = contract.namespaced_request_id("train", row["kind"], row["family"], content)
        _require(row["id"] == request["requestId"] == expected_id, "row split/family/content ID differs")
        _require(expected_id not in ids, "duplicate training identity")
        ids.add(expected_id)
        target_text = messages[2]["content"]
        target = _strict(target_text.encode("utf-8"))
        _require(target_text == contract.canonical_json(target), "assistant target is not canonical JSON")
        result = contract.validate_pair(request_text, target_text)
        _require(result["responseType"] == row["kind"], "row kind differs from bound response")
        _require(all(item["label"] == "SIMULATED" for item in request["evidence"]),
                 "synthetic curriculum cannot claim measured or authenticated evidence")
        kinds[row["kind"]] += 1
        families[row["family"]] += 1
        family_kinds[(row["family"], row["kind"])] += 1
        if row["kind"] == "RECOVERY":
            failed_statuses = [item["status"] for item in request["evidence"]
                               if item["status"] != "OK"]
            _require(len(failed_statuses) == 1,
                     "each recovery training row must cover exactly one failed status")
            recovery_statuses[(row["family"], failed_statuses[0])] += 1
        elif row["kind"] == "REFUSAL":
            refused_authorities[(row["family"], request["requestedAuthority"])] += 1
    _require(dict(kinds) == expected["kind_counts"], "observed class coverage differs")
    _require(dict(families) == {family: 90 for family in TRAIN_FAMILIES},
             "each training family must contain exactly 90 rows")
    _require(dict(family_kinds) == {(family, kind): 30 for family in TRAIN_FAMILIES
                                  for kind in contract.CLAIMS},
             "each training family must cover 30 rows of every class")
    _require(dict(recovery_statuses) == {(family, status): 6 for family in TRAIN_FAMILIES
                                       for status in contract.RECOVERY_RULES},
             "each training family must cover six of every recovery status")
    _require(dict(refused_authorities) == {(family, authority): 6 for family in TRAIN_FAMILIES
                                         for authority in BLOCKED_AUTHORITIES},
             "each training family must cover six of every refused authority")
    return {"dataset_sha256": digest, "rows": len(lines), "kind_counts": dict(kinds),
            "family_counts": dict(families), "heldout_content_opened": False}


def _load_gate(filename: str) -> Any:
    _require(filename in GATE_HASHES, "unknown gate")
    path = TOOLS / filename
    if not path.is_file():
        raise GateNotReady("required fixed-source gate is missing")
    source = _read_once(path, MAX_FILE_BYTES)
    _require(hashlib.sha256(source).hexdigest() == GATE_HASHES[filename],
             "gate source differs from the pinned revision")
    module_name = "_szl_v4_admission_" + path.stem
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise GateNotReady("required fixed-source gate cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    try:
        exec(compile(source, str(path), "exec", dont_inherit=True), module.__dict__)
    except ImportError as exc:
        raise GateNotReady("required gate dependency is unavailable") from exc
    return module


def _verify_doctrine_receipts(report: dict[str, Any], raw: bytes) -> None:
    """Use the maintained Nemo verifier; integrity is never signing authority."""
    try:
        from szl_nemo.engine import input_hash
        from szl_nemo.receipt import verify_chain
    except ImportError as exc:
        raise GateNotReady("required Nemo receipt verifier is unavailable") from exc
    _require(report.get("receipt_schema") == "szl.nemo.receipt.v1",
             "wrong Nemo receipt schema")
    _require(report.get("kernel") == {"repo": "https://github.com/szl-holdings/szl-nemo",
                                     "rule_version": "doctrine-v11/R1-R5"},
             "Nemo kernel declaration differs from the locked doctrine")
    receipts = report.get("receipt_chain")
    _require(isinstance(receipts, list) and len(receipts) == MAX_ROWS,
             "Nemo must emit one chained receipt per training row")
    _require(verify_chain(receipts) is True, "Nemo receipt chain failed integrity verification")
    _require(report.get("receipt_chain_tip") == receipts[-1].get("receipt_sha256"),
             "Nemo receipt tip differs from its emitted chain")
    for line, receipt in zip(raw.split(b"\n")[:-1], receipts):
        _keys(receipt, {"schema", "sequence", "prev_receipt_sha256", "decision",
                        "receipt_sha256", "receipt_status"}, "Nemo receipt")
        _require(receipt["receipt_status"] == "UNSIGNED_HONEST", "receipt cannot claim a signer")
        decision = receipt["decision"]
        _keys(decision, {"schema_version", "decision", "violated_rules", "reasons",
                         "rule_version", "input_hash", "receipt_status"}, "Nemo decision")
        _require(decision["schema_version"] == "szl.nemo.decision.v1"
                 and decision["decision"] == "ALLOW" and decision["violated_rules"] == []
                 and decision["reasons"] == [] and decision["receipt_status"] == "UNSIGNED_HONEST"
                 and decision["rule_version"] == "doctrine-v11/R1-R5 persona/finetuned",
                 "Nemo decision is not an unsigned complete doctrine pass")
        messages = _strict(line)["messages"]
        _require(decision["input_hash"] == input_hash(messages[1]["content"], messages[2]["content"]),
                 "Nemo decision is not bound to the exact training pair")


def run_gates(raw: bytes) -> dict[str, Any]:
    """Re-run real gates on a private train-only snapshot, never saved reports."""
    generic = _load_gate("validate_sft_dataset.py")
    nemo = _load_gate("nemo_doctrine_gate.py")
    if getattr(nemo, "_IMPORT_ERROR", object()) is not None:
        raise GateNotReady("required Nemo kernel is unavailable")
    digest = hashlib.sha256(raw).hexdigest()
    with tempfile.TemporaryDirectory(prefix="szl-v4-train-conformance-") as temporary:
        directory = Path(temporary)
        os.chmod(directory, 0o700)
        snapshot = directory / "train.jsonl"
        with snapshot.open("xb") as handle:
            handle.write(raw)
            handle.flush()
        _require(_read_once(snapshot, MAX_FILE_BYTES) == raw, "snapshot differs before gates")
        generic_report = generic.validate_dataset(str(snapshot), min_examples=MAX_ROWS)
        _require(_read_once(snapshot, MAX_FILE_BYTES) == raw, "snapshot changed after generic gate")
        doctrine_report = nemo.gate_dataset(str(snapshot), persona="finetuned")
        _require(_read_once(snapshot, MAX_FILE_BYTES) == raw, "snapshot changed after Nemo gate")
    expected_sha = "sha256:" + digest
    _require(isinstance(generic_report, dict) and generic_report.get("status") == "VALID"
             and type(generic_report.get("records")) is int
             and generic_report.get("records") == MAX_ROWS
             and type(generic_report.get("min_examples_required")) is int
             and generic_report.get("min_examples_required") == MAX_ROWS
             and generic_report.get("dataset_sha256") == expected_sha
             and generic_report.get("errors") == [], "generic SFT gate did not cover exact bytes")
    _require(isinstance(doctrine_report, dict) and doctrine_report.get("status") == "VALID"
             and doctrine_report.get("persona") == "finetuned"
             and all(type(doctrine_report.get(key)) is int
                     for key in ("records", "checked", "skipped_no_chat_pair"))
             and doctrine_report.get("records") == MAX_ROWS
             and doctrine_report.get("checked") == MAX_ROWS
             and doctrine_report.get("skipped_no_chat_pair") == 0
             and doctrine_report.get("dataset_sha256") == expected_sha
             and doctrine_report.get("violation_counts") == {}
             and doctrine_report.get("violations") == [], "Nemo gate did not cover exact bytes")
    _verify_doctrine_receipts(doctrine_report, raw)
    return {"generic_sft_valid": True, "nemo_valid": True,
            "nemo_receipt_chain_tip": doctrine_report.get("receipt_chain_tip"),
            "nemo_receipt_chain": doctrine_report["receipt_chain"],
            "nemo_receipt_chain_verified": True, "nemo_receipt_count": MAX_ROWS,
            "gate_source_revision": CONTRACT_REVISION, "gate_source_sha256": dict(GATE_HASHES),
            "gate_reports_authenticated": False}


def check_conformance(train_path: Path, manifest_path: Path) -> tuple[dict[str, Any], int]:
    report: dict[str, Any] = {"schema": REPORT_SCHEMA, "evidence_class": "MEASURED",
        "dataset_evidence_class": "SIMULATED", "candidate_id": contract.PROFILE,
        "state": "BLOCKED", "contract_conforms": False, "generic_sft_valid": False,
        "nemo_valid": False, "signature_valid": False, "key_trust": "REPO_DECLARED",
        "semantic_leakage_review": "UNKNOWN", "trainer_supervisor_binding": "UNAVAILABLE",
        "training_eligible": False, "publication_eligible": False, "execution_authority": False,
        "heldout_content_opened": False, "errors": [],
        "claim_boundary": "Conformance only; no signing, training, evaluation or authorization."}
    try:
        _require(train_path.name == "train.jsonl" and manifest_path.name == "curriculum-manifest.json",
                 "only explicit training and curriculum manifest filenames are allowed")
        _require(Path(os.path.abspath(train_path.parent)) == Path(os.path.abspath(manifest_path.parent)),
                 "training and manifest must be in one explicit directory")
        manifest_bytes = _read_once(manifest_path, contract.MAX_BYTES)
        _require(hashlib.sha256(manifest_bytes).hexdigest() == MANIFEST_SHA256,
                 "curriculum manifest differs from the frozen source commitment")
        manifest = _strict(manifest_bytes)
        validate_manifest(manifest)
        raw = _read_once(train_path, MAX_FILE_BYTES)
        report.update(validate_train_bytes(raw, manifest))
        report["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
        report["contract_conforms"] = True
        report.update(run_gates(raw))
        return report, 0
    except GateNotReady:
        report["errors"] = ["required fixed-source gate or doctrine kernel is NOT_READY"]
        return report, 2
    except (AdmissionError, contract.ContractError, OSError, UnicodeError, ValueError, TypeError,
            KeyError, AttributeError, RecursionError, RuntimeError, OverflowError):
        report["errors"] = ["source conformance failed closed"]
        return report, 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--check-conformance", action="store_true", required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    report, exit_code = check_conformance(args.train, args.manifest)
    print(contract.canonical_json(report))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
