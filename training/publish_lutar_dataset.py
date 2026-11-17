#!/usr/bin/env python3
"""Publish source-verified first-party Lutar SFT fixtures to PRIVATE HF dataset.

Manual, one-shot operation only. This publisher does not train, issue training
rights, approve models, publish weights, or modify a live inference endpoint.
For any existing dataset (even a differently owned one), fail closed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

from training import lutar_axiom_intake as intake

REPO_ID = "SZLHOLDINGS/lutar-axioms-sft-v1"
REPO_TYPE = "dataset"
SCHEMA = "szl.hub.source-bound-lutar-curriculum/v1"
BOUND_FILES = ("train.jsonl", "holdout.jsonl", "manifest.json", "README.md", "SOURCE_BINDING.json")


def _card(index: dict) -> bytes:
    repo = index["producer"]["repository"]
    commit = index["producer"]["commit"]
    math = index["mathematics"]
    return f"""---
license: apache-2.0
language:
- en
task_categories:
- text-generation
tags:
- lutar
- lean
- mathematical-reasoning
- synthetic
- khipu
pretty_name: SZL Lutar A1–A5 Synthetic Reasoning v1
---

# Lutar A1–A5 Synthetic Reasoning SFT v1

**REVIEW REQUIRED. NOT AN INDEPENDENT EVALUATION, PROOF, OR TRAINING APPROVAL.**

First-party synthetic examples generated from exact original source:
`https://github.com/{repo}/tree/{commit}/curriculum`.

Immutable mathematics authority:
`https://github.com/{math['repository']}/tree/{math['commit']}`.
The canonical axiom and invariant blob identities are stored in
`SOURCE_BINDING.json`. This dataset is a faithful *numeric exercise set*,
not a Lean proof-checking corpus or trained model checkpoint.

There are exactly **160 training** and **80 developer-held diagnostic**
chat JSONL records, with exact hashes in `manifest.json` and
`SOURCE_BINDING.json`. The holdout uses distinct numerical instances from
the same generator and is **not independent**; do not claim generalization
from its score alone.

Ten themes cover A1 monotonicity, A2 homogeneity, A3 diagonal
normalization, A4 upper boundedness, A5 equal-weight permutation
invariance, the max-aggregator **counterexample to unconditional Λ
uniqueness**, nonuniform-weight permutation counterexamples, conditional
evidence admission, mandatory refusal, and zero-axis cases.

**No automatic training rights admission.** Organizational ownership and
an Apache-2.0 license do not alone provide the operator-controlled
`szl.forge.training-admission/v1` receipt that Forge requires. Model
training, HF model publication, and production readiness are **not
established** by this dataset repository, dataset card, or verified bytes.
""" .encode("utf-8")


def _source_binding(index: dict) -> bytes:
    binding = {
        "schema": SCHEMA,
        "repo_id": REPO_ID,
        "repo_type": REPO_TYPE,
        "private_at_creation": True,
        "producer": index["producer"],
        "mathematics": index["mathematics"],
        "source_files": {e["path"].split("/")[-1]: {
            "sha256": e["sha256"], "bytes": e["bytes"],
            "rows": e.get("records"),
        } for e in index["files"]},
        "review_state": "REVIEW_REQUIRED",
        "training_authority": "NONE",
        "independent_holdout": False,
        "weights_published": False,
        "source_license": "Apache-2.0",
    }
    return (json.dumps(binding, sort_keys=True, indent=2) + "\n").encode("utf-8")


def build_bundle(*, producer_checkout: Path) -> tuple[dict, dict[str, bytes]]:
    """Exact immutable source, no network fallback, no generated replacement."""
    index = intake.load_index()
    raw = {
        path: intake.read_immutable_git(index["producer"]["commit"], producer_checkout, path)
        for path in intake.ARTIFACTS
    }
    intake.verify_payloads(raw, index)
    bundle = {
        "train.jsonl": raw["curriculum/train.jsonl"],
        "holdout.jsonl": raw["curriculum/holdout.jsonl"],
        "manifest.json": raw["curriculum/manifest.json"],
        "README.md": _card(index),
        "SOURCE_BINDING.json": _source_binding(index),
    }
    if set(bundle) != set(BOUND_FILES):
        raise intake.IntakeHold("publication file population changed")
    return index, bundle


def validate_bundle(bundle: dict[str, bytes], *, workdir: Path, index: dict) -> None:
    for name, value in bundle.items():
        if name not in BOUND_FILES or type(value) is not bytes:
            raise intake.IntakeHold("invalid publication input")
        (workdir / name).write_bytes(value)
    report = intake._forge_validator()(
        str(workdir / "train.jsonl"), min_examples=32,
        disjoint_from=[str(workdir / "holdout.jsonl")],
    )
    if (report.get("status") != "VALID" or report.get("records") != 160
            or report.get("cross_dataset_overlaps") != 0
            or report.get("duplicate_inputs") != 0
            or not report.get("reference_datasets")
            or report["reference_datasets"][0].get("records") != 80):
        raise intake.IntakeHold("source publisher refused SFT contract")
    bound = intake._parse(bundle["SOURCE_BINDING.json"])
    if (bound.get("schema") != SCHEMA
            or bound.get("repo_id") != REPO_ID
            or bound.get("producer") != index["producer"]
            or bound.get("mathematics") != index["mathematics"]
            or bound.get("training_authority") != "NONE"):
        raise intake.IntakeHold("published provenance or authority mismatch")


def publish_private(*, producer_checkout: Path, receipt_file: Path,
                    api=None, downloader=None) -> dict:
    """No existing repo overwrite, no output receipt until verified readback."""
    receipt_file = Path(receipt_file)
    if receipt_file.exists() or receipt_file.is_symlink() or not receipt_file.parent.is_dir():
        raise intake.IntakeHold("receipt path must be new in an existing local directory")
    index, files = build_bundle(producer_checkout=producer_checkout)
    with tempfile.TemporaryDirectory(prefix="szl-lutar-hf-private-") as folder:
        stage = Path(folder)
        validate_bundle(files, workdir=stage, index=index)
        if api is None:
            from huggingface_hub import HfApi
            api = HfApi()
        if downloader is None:
            from huggingface_hub import hf_hub_download
            downloader = hf_hub_download
        if api.repo_exists(REPO_ID, repo_type=REPO_TYPE):
            raise intake.IntakeHold("dataset already exists; never overwrite another revision")
        api.create_repo(repo_id=REPO_ID, repo_type=REPO_TYPE,
                        private=True, exist_ok=False)
        api.upload_folder(
            folder_path=str(stage), repo_id=REPO_ID, repo_type=REPO_TYPE,
            commit_message="dataset: exact first-party Lutar A1-A5 review corpus",
        )
        revision = api.repo_info(repo_id=REPO_ID, repo_type=REPO_TYPE).sha
        if (not isinstance(revision, str)
                or intake.HEX40.fullmatch(revision) is None):
            raise intake.IntakeHold("Hub immutable dataset revision unavailable")
        remote = api.repo_info(repo_id=REPO_ID, repo_type=REPO_TYPE)
        if remote.private is not True:
            raise intake.IntakeHold("dataset privacy was not preserved")
        # Independent authenticated provider bytes at the immutable Hub revision.
        for name, body in files.items():
            content = downloader(repo_id=REPO_ID, repo_type=REPO_TYPE,
                                 filename=name, revision=revision)
            if hashlib.sha256(Path(content).read_bytes()).digest() != hashlib.sha256(body).digest():
                raise intake.IntakeHold("Hub readback disagrees with source bytes")
        receipt = {
            "schema": "szl.hub.lutar-sft-publish-receipt/v1",
            "repo_id": REPO_ID, "hub_revision": revision,
            "private": True,
            "files": {name: hashlib.sha256(body).hexdigest()
                      for name, body in files.items()},
            "source_commit": index["producer"]["commit"],
            "status": "PRIVATE_BYTES_VERIFIED_REVIEW_REQUIRED",
            "training_authority": "NONE",
            "trained": False,
            "model_published": False,
        }
    with receipt_file.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer-checkout", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, default=None)
    parser.add_argument("--publish-private", action="store_true")
    args = parser.parse_args()
    try:
        if not args.publish_private:
            index, files = build_bundle(producer_checkout=args.producer_checkout)
            with tempfile.TemporaryDirectory(prefix="szl-lutar-dryrun-") as folder:
                validate_bundle(files, workdir=Path(folder), index=index)
            print(json.dumps({
                "status": "LOCAL_VALIDATION_ONLY",
                "repo_id": REPO_ID, "private": True,
                "files": sorted(files), "training_authority": "NONE",
                "source_commit": index["producer"]["commit"],
            }, sort_keys=True))
        else:
            if args.receipt is None:
                raise intake.IntakeHold("--receipt is required for publication")
            result = publish_private(producer_checkout=args.producer_checkout,
                                     receipt_file=args.receipt)
            print(json.dumps({k: result[k] for k in
                              ("repo_id", "hub_revision", "status", "training_authority")},
                             sort_keys=True))
    except (intake.IntakeHold, OSError) as exc:
        print("HOLD: " + str(exc))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
