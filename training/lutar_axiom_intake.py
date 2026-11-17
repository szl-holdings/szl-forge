#!/usr/bin/env python3
"""Offline exact-source Lutar A1-A5 SFT candidate import; NEVER a trainer.

Reads immutable Git blobs from an operator-supplied, already available checkout.
It does not fetch, trust a moving ref, issue rights admission, allocate compute,
load model weights, consume secrets, or publish to Hugging Face.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent
INDEX = ROOT / "lutar_axioms.source.v1.json"
SOURCE_SCHEMA = "szl.forge.lutar-source-intake/v1"
ARTIFACTS = ("curriculum/train.jsonl", "curriculum/holdout.jsonl",
             "curriculum/manifest.json")
HEX40 = re.compile(r"[a-f0-9]{40}\Z")
HEX64 = re.compile(r"[a-f0-9]{64}\Z")
MAX_FILE_BYTES = 400_000
MAX_CASES = 500


class IntakeHold(ValueError):
    """A validation refusal without partial outputs."""


def _unique_pairs(pairs):
    seen = {}
    for key, value in pairs:
        if key in seen:
            raise IntakeHold("duplicate JSON member")
        seen[key] = value
    return seen


def _reject_nonfinite(_):
    raise IntakeHold("non-finite JSON value")


def _parse(raw: bytes):
    try:
        return json.loads(raw.decode("utf-8"),
                          object_pairs_hook=_unique_pairs,
                          parse_constant=_reject_nonfinite)
    except (UnicodeError, json.JSONDecodeError, TypeError) as exc:
        raise IntakeHold("unreadable or malformed JSON") from exc


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def load_index() -> dict:
    index = _parse(INDEX.read_bytes())
    if type(index) is not dict or index.get("schema") != SOURCE_SCHEMA:
        raise IntakeHold("unsupported source index")
    producer, math = index.get("producer"), index.get("mathematics")
    if type(producer) is not dict or type(math) is not dict:
        raise IntakeHold("source provenance incomplete")
    if producer.get("repository") != "szl-holdings/lean-rlvr-proving-ground":
        raise IntakeHold("unexpected corpus authority")
    if math.get("repository") != "szl-holdings/lutar-lean":
        raise IntakeHold("unexpected mathematics authority")
    for value in (producer.get("commit"), math.get("commit"),
                  math.get("axioms_blob"), math.get("invariant_blob")):
        if type(value) is not str or HEX40.fullmatch(value) is None:
            raise IntakeHold("non-immutable source identity")
    if (index.get("train_state") != "CANDIDATE_ONLY"
            or index.get("training_authority") != "NONE"
            or index.get("training_launch") is not False
            or index.get("model_publish") is not False
            or index.get("independent_holdout") is not False
            or index.get("required_minimum") != 32):
        raise IntakeHold("training boundary mismatch")
    files = index.get("files")
    if (type(files) is not list or len(files) != len(ARTIFACTS)
            or [entry.get("path") if type(entry) is dict else None
                for entry in files] != list(ARTIFACTS)):
        raise IntakeHold("source file population mismatch")
    for entry in files:
        if (type(entry.get("sha256")) is not str
                or HEX64.fullmatch(entry["sha256"]) is None
                or type(entry.get("bytes")) is not int
                or not 1 <= entry["bytes"] <= MAX_FILE_BYTES):
            raise IntakeHold("invalid artifact digest or size")
    return index


def read_immutable_git(commit: str, repo: Path, path: str) -> bytes:
    """No network or worktree fallback; immutable Git object bytes only."""
    if not HEX40.fullmatch(commit) or path not in ARTIFACTS:
        raise IntakeHold("invalid immutable Git source")
    if not repo.is_dir():
        raise IntakeHold("local producer Git object store unavailable")
    env = dict(os.environ, GIT_NO_LAZY_FETCH="1", GIT_OPTIONAL_LOCKS="0")
    try:
        checked = subprocess.run(
            ["git", "--no-replace-objects", "-C", str(repo), "rev-parse",
             "--verify", commit + "^{commit}"],
            capture_output=True, timeout=20, env=env, check=True,
        )
        if checked.stdout.decode("ascii").strip() != commit:
            raise IntakeHold("commit identity mismatch")
        found = subprocess.run(
            ["git", "--no-replace-objects", "-C", str(repo), "show",
             commit + ":" + path],
            capture_output=True, timeout=20, env=env, check=True,
        )
    except (OSError, UnicodeError, subprocess.SubprocessError) as exc:
        raise IntakeHold("immutable source object unavailable") from exc
    if not found.stdout or len(found.stdout) > MAX_FILE_BYTES:
        raise IntakeHold("source object outside size limit")
    return found.stdout


def _verify_records(raw: bytes, *, split: str, index: dict, count: int) -> None:
    lines = raw.splitlines()
    if not 32 <= len(lines) <= MAX_CASES or len(lines) != count:
        raise IntakeHold("corpus population or minimum floor mismatch")
    ids = set()
    for line in lines:
        if not line.strip():
            raise IntakeHold("blank corpus record")
        row = _parse(line)
        if type(row) is not dict:
            raise IntakeHold("corpus record is not an object")
        meta = row.get("metadata")
        if (type(meta) is not dict or meta.get("split") != split
                or meta.get("schema") != "szl.lutar.numeric-curriculum/v1"
                or meta.get("source") != {
                    "repository": index["mathematics"]["repository"],
                    "commit": index["mathematics"]["commit"],
                    "axioms_path": "Lutar/Axioms.lean",
                    "axioms_blob": index["mathematics"]["axioms_blob"],
                    "invariant_path": "Lutar/Invariant.lean",
                    "invariant_blob": index["mathematics"]["invariant_blob"],
                    "toolchain": "leanprover/lean4:v4.35.0",
                }
                or meta.get("training_authority") != "NONE"
                or meta.get("review_state") != "CANDIDATE_ONLY"
                or meta.get("proof_status") != "NOT_LEAN_VERIFIED_AS_DATASET"):
            raise IntakeHold("source-bound corpus metadata mismatch")
        identity = meta.get("case_id")
        if type(identity) is not str or HEX64.fullmatch(identity) is None or identity in ids:
            raise IntakeHold("corpus case identity repeated or invalid")
        ids.add(identity)


def verify_payloads(raw: dict, index: dict) -> None:
    """Verify all source bytes and coverage before materializing output."""
    if type(raw) is not dict or set(raw) != set(ARTIFACTS):
        raise IntakeHold("exact source artifact set is required")
    definitions = {entry["path"]: entry for entry in index["files"]}
    for key in ARTIFACTS:
        content = raw[key]
        meta = definitions[key]
        if (type(content) is not bytes or len(content) != meta["bytes"]
                or _sha256(content) != meta["sha256"]):
            raise IntakeHold("source artifact size/digest does not match immutable pin")
    manifest = _parse(raw[ARTIFACTS[-1]])
    if (type(manifest) is not dict
            or manifest.get("schema") != "szl.lutar.numeric-curriculum-manifest/v1"
            or manifest.get("source") != {
                "repository": index["mathematics"]["repository"],
                "commit": index["mathematics"]["commit"],
                "axioms_path": "Lutar/Axioms.lean",
                "axioms_blob": index["mathematics"]["axioms_blob"],
                "invariant_path": "Lutar/Invariant.lean",
                "invariant_blob": index["mathematics"]["invariant_blob"],
                "toolchain": "leanprover/lean4:v4.35.0",
            }
            or manifest.get("training_authority") != "NONE"
            or manifest.get("review_state") != "REVIEW_REQUIRED"
            or manifest.get("lean_proof_of_dataset") is not False
            or manifest.get("independent_holdout") is not False):
        raise IntakeHold("producer manifest maturity/source mismatch")
    for split in ("train", "holdout"):
        meta = manifest.get(split)
        path = "curriculum/" + split + ".jsonl"
        if (type(meta) is not dict or meta.get("sha256") != definitions[path]["sha256"]
                or meta.get("rows") != definitions[path].get("records")):
            raise IntakeHold("producer manifest corpus binding mismatch")
        _verify_records(raw[path], split=split, index=index, count=meta["rows"])


def _forge_validator():
    path = ROOT.parent / "tools" / "validate_sft_dataset.py"
    spec = importlib.util.spec_from_file_location("szl_lutar_sft_validator", path)
    if spec is None or spec.loader is None:
        raise IntakeHold("Forge dataset validator not available")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.validate_dataset


def stage(raw: dict, *, index: dict, output: Path) -> dict:
    """Create one new review-only generation. Failure never promotes a model."""
    verify_payloads(raw, index)
    output = Path(output)
    parent = output.absolute().parent
    if not parent.is_dir() or parent.is_symlink() or output.exists() or output.is_symlink():
        raise IntakeHold("output must be a new directory below an existing local parent")
    staging = Path(tempfile.mkdtemp(prefix=".szl-lutar-candidate-", dir=parent))
    try:
        for name, dest in (("train.jsonl", ARTIFACTS[0]),
                           ("holdout.jsonl", ARTIFACTS[1]),
                           ("manifest.json", ARTIFACTS[2])):
            (staging / name).write_bytes(raw[dest])
        report = _forge_validator()(str(staging / "train.jsonl"),
                                    min_examples=index["required_minimum"],
                                    disjoint_from=[str(staging / "holdout.jsonl")])
        if (report["status"] != "VALID"
                or report["records"] != index["files"][0]["records"]
                or report["cross_dataset_overlaps"] != 0
                or report["duplicate_inputs"] != 0
                or report["duplicate_conversations"] != 0
                or not report["reference_datasets"]
                or report["reference_datasets"][0]["records"] != index["files"][1]["records"]):
            raise IntakeHold("Forge SFT structural/holdout admission failed")
        receipt = {
            "schema": "szl.forge.lutar-candidate-receipt/v1",
            "producer": index["producer"],
            "mathematics": index["mathematics"],
            "train_sha256": index["files"][0]["sha256"],
            "holdout_sha256": index["files"][1]["sha256"],
            "train_rows": report["records"],
            "holdout_rows": report["reference_datasets"][0]["records"],
            "overlaps": report["cross_dataset_overlaps"],
            "state": "SOURCE_BOUND_REVIEW_REQUIRED",
            "rights_admission": "NOT_ISSUED",
            "training_authority": "NONE",
            "training_started": False,
            "model_loaded": False,
            "hub_written": False,
            "release_authorized": False,
        }
        (staging / "source-receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        if output.exists():
            raise IntakeHold("output raced with another writer")
        staging.rename(output)
        return receipt
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer-checkout", type=Path, required=True,
                        help="Local Git checkout containing pinned producer commit object")
    parser.add_argument("--out", type=Path, required=True,
                        help="New local directory for reviewed SFT candidates; never overwrite")
    args = parser.parse_args()
    try:
        index = load_index()
        raw = {path: read_immutable_git(index["producer"]["commit"],
                                        args.producer_checkout, path)
               for path in ARTIFACTS}
        receipt = stage(raw, index=index, output=args.out)
    except IntakeHold as exc:
        print("HOLD: " + str(exc))
        return 2
    print(json.dumps({k: receipt[k] for k in ("schema", "state",
                                             "train_rows", "holdout_rows",
                                             "training_authority")},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
