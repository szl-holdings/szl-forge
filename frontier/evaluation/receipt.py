# SPDX-License-Identifier: Apache-2.0
"""Receipt construction and optional Hugging Face dataset publication."""
from __future__ import annotations

import argparse
import json
import platform
import re
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence
from urllib.parse import quote

from .core import SYSTEM_PROMPT, aggregate, canonical_sha256, compare, file_sha256
from .provider import BASELINE_URL, ROUTER_URL


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_receipt(
    *,
    args: argparse.Namespace,
    candidate: Mapping[str, Any],
    fixtures_sha256: str,
    source_checks: Mapping[str, Any],
    baseline_identity: Mapping[str, Any],
    baseline_records: Sequence[Mapping[str, Any]],
    candidate_records: Sequence[Mapping[str, Any]],
    fallback: Mapping[str, Any],
    provider: str,
    provider_attempts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    baseline_metrics = aggregate(baseline_records)
    candidate_metrics = aggregate(candidate_records)
    fallback_transport_pass = bool(fallback.get("transport_pass"))
    fallback_semantic_pass = bool(fallback.get("semantic_safety_pass"))
    completed = (
        bool(source_checks.get("pass"))
        and baseline_metrics["successful_call_rate"] == 1.0
        and candidate_metrics["successful_call_rate"] == 1.0
        and fallback_transport_pass
        and fallback_semantic_pass
        and bool(fallback.get("pass"))
    )
    here = Path(__file__).resolve().parent
    receipt: dict[str, Any] = {
        "schema": "szl.nemo.frontier-qualification-receipt.v1",
        "generated_at": utc_now(),
        "run_identity": {
            "run_id": args.run_id,
            "source_repository": args.source_repository,
            "source_revision": args.source_revision,
            "suite": args.suite,
            "runner_sha256": file_sha256(here / "runner.py"),
            "core_sha256": file_sha256(here / "core.py"),
            "source_module_sha256": file_sha256(here / "source.py"),
            "provider_module_sha256": file_sha256(here / "provider.py"),
            "receipt_module_sha256": file_sha256(here / "receipt.py"),
        },
        "candidate_id": candidate["candidate_id"],
        "model_id": candidate["upstream_model_id"],
        "model_revision": candidate["upstream_revision"],
        "provider": provider,
        "provider_route": f"{candidate['upstream_model_id']}:{provider}",
        "provider_attempts": list(provider_attempts),
        "provider_execution_revision": "UNAVAILABLE",
        "runtime_engine": (
            "Hugging Face Inference Providers OpenAI-compatible router"
        ),
        "runtime_version": "UNAVAILABLE",
        "client": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "provider_hardware_fingerprint": "UNAVAILABLE",
        "baseline_identity": baseline_identity,
        "fixture_set_sha256": fixtures_sha256,
        "configuration_sha256": canonical_sha256(
            {
                "system_prompt": SYSTEM_PROMPT,
                "providers": list(args.providers),
                "suite": args.suite,
                "router_url": ROUTER_URL,
                "baseline_url": BASELINE_URL,
            }
        ),
        "seed": "UNAVAILABLE_PROVIDER_DEPENDENT",
        "baseline_results_sha256": canonical_sha256(list(baseline_records)),
        "candidate_results_sha256": canonical_sha256(list(candidate_records)),
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "comparison": compare(candidate_metrics, baseline_metrics),
        "metric_truth_labels": {
            "request_latency_and_response_scoring": "MEASURED",
            "source_files": "MEASURED",
            "provider_failure_transport": "MEASURED",
            "baseline_fallback_transport": "MEASURED",
            "fallback_semantic_guard": "DETERMINISTIC_VERIFIED",
            "upstream_benchmark_claims": "NOT_USED",
            "provider_execution_revision": "UNAVAILABLE",
            "provider_hardware": "UNAVAILABLE",
        },
        "fallback_evidence_sha256": canonical_sha256(dict(fallback)),
        "fallback_evidence": dict(fallback),
        "fallback_qualification": {
            "transport_pass": fallback_transport_pass,
            "semantic_safety_pass": fallback_semantic_pass,
            "production_fallback_qualified": (
                fallback_transport_pass and fallback_semantic_pass
            ),
        },
        "violated_invariants": (
            [] if completed else ["evaluation_evidence_incomplete"]
        ),
        "known_bounds": [
            "provider_execution_weight_revision_not_attested_by_chat_response",
            "provider_server_hardware_not_disclosed",
            "provider_runtime_build_not_disclosed",
            "integrity_receipt_is_unsigned",
        ],
        "decision": (
            "EVIDENCE_COMPLETE_REVIEW_REQUIRED" if completed else "HOLD"
        ),
        "production_disposition": "HOLD",
        "promotion_effect": "NONE",
        "signature_status": "UNSIGNED_HONEST",
        "authenticity_not_established": True,
        "receipt_sha256": None,
    }
    receipt["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    return receipt


PUBLIC_EXAMPLE = (
    "https://github.com/szl-holdings/governed-receipt-spec/tree/"
    "320983d22e76fc9b26af0b2cd20799c5000543fc/examples/public-single-cell"
)
EVIDENCE_FILES = ("bundle.json", "receipt.json", "summary.json")


def dataset_card(
    dataset_id: str, path_in_repo: str, receipt: Mapping[str, Any]
) -> str:
    """Describe this publication using its actual receipt and source revision."""
    identity = receipt["run_identity"]
    revision = str(identity["source_revision"])
    repository = str(identity["source_repository"])
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("publication requires an immutable source revision")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("invalid source repository")
    source = f"https://github.com/{repository}/blob/{revision}"
    run = (
        f"https://huggingface.co/datasets/{quote(dataset_id, safe='/')}/"
        f"blob/main/{quote(path_in_repo, safe='/')}"
    )
    count = receipt["candidate_metrics"]["case_count"]
    return (
        "---\nlicense: apache-2.0\n"
        "pretty_name: SZL Frontier Evaluation Receipts\n"
        "tags:\n- evaluation\n- reproducibility\n- provenance\n---\n\n"
        "# Checkable language-model evaluation records\n\n"
        "We publish the inputs, measured responses, scoring results and file "
        "hashes from small language-model evaluations. These records help "
        "readers inspect what was tested and check whether files changed.\n\n"
        f"**Start with a real-data example:** [public single-cell calculation]"
        f"({PUBLIC_EXAMPLE}). It summarizes all 3,072 submitted cell columns "
        "from GEO GSE85241 with four source donor labels retained. Two "
        "calculations produced identical summary bytes; changed output and "
        "receipt-payload controls were rejected against the retained reference. "
        "This separate example performs no gene signature scoring or "
        "biological inference.\n\n"
        "## Inspect the run published with this card\n\n"
        f"- [Results summary]({run}/summary.json)\n"
        f"- [Complete response and evidence bundle]({run}/bundle.json)\n"
        f"- [Unsigned computation record]({run}/receipt.json)\n"
        f"- [Exact evaluation source]({source}/frontier/evaluation/runner.py) "
        f"and [test inputs]({source}/frontier/evaluation/fixtures.v1.json)\n\n"
        f"This run includes {count} synthetic public test cases per model. "
        "The full suite has four cases covering a supported answer, an "
        "unsupported question, an approval boundary and bounded arithmetic; "
        "the shorter smoke suite uses two. Calls go to a candidate model and "
        "a live baseline. These are language-model checks, not biological data "
        "or a broad benchmark. Scores describe only the recorded cases and "
        "do not establish method superiority.\n\n"
        "## What the records establish\n\n"
        "The records identify source files and bind measured responses and "
        "results with SHA-256 hashes. They are **unsigned**: hashes detect "
        "changes against a retained reference, but do not authenticate an "
        "author or prevent replacement of every file and hash together. "
        "The provider's executed weight revision, server hardware and runtime "
        "build are not attested. Language-model responses are not promised "
        "to repeat byte for byte.\n\n"
        "If a required check cannot run or fails, the evaluation is reported "
        "as incomplete. A completed evaluation still requires review and "
        "does not authorize production use. No biological findings, clinical "
        "validity, safety certification or superiority over AUCell, UCell or "
        "Seurat is claimed.\n\n"
        "## Publication schedule\n\n"
        "The workflow publishes after evaluation-related changes reach main, "
        "on Sundays at 06:17 UTC, and on manual runs with publication enabled. "
        "Pull-request checks do not publish. The card and finalized evidence "
        "files are committed together; publication errors are reported as "
        "failures. Each run has a unique directory, and previous run files "
        "are not overwritten by this publisher.\n"
    )


def publish(
    output: Path,
    token: str,
    dataset_id: str,
    path_in_repo: str,
    receipt: Mapping[str, Any],
) -> dict[str, str]:
    from huggingface_hub import CommitOperationAdd, HfApi

    # Snapshot only finalized evidence, never arbitrary files from a work folder.
    # This also excludes source caches, credentials, bytecode, stale publication
    # receipts, DATASET_CARD.md and the still-open runner-output.jsonl tee stream.
    parts = PurePosixPath(path_in_repo).parts
    if (
        not path_in_repo.startswith("runs/")
        or "/".join(parts) != path_in_repo
        or any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", p) for p in parts)
    ):
        raise ValueError("publication requires a safe run directory")
    files: dict[str, bytes] = {}
    for name in EVIDENCE_FILES:
        path = output / name
        if output.is_symlink() or path.is_symlink() or not path.is_file():
            raise ValueError(f"missing or linked finalized evidence: {name}")
        files[name] = path.read_bytes()
        if token and token.encode("utf-8") in files[name]:
            raise ValueError("publication credential found in finalized evidence")
    recorded = json.loads(files["receipt.json"])
    bundle = json.loads(files["bundle.json"])
    summary = json.loads(files["summary.json"])
    digest = canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    if (
        recorded != receipt
        or bundle.get("receipt") != receipt
        or summary.get("receipt_sha256") != digest
        or receipt.get("receipt_sha256") != digest
    ):
        raise ValueError("finalized evidence does not match the publication receipt")
    for records in ("baseline_results", "candidate_results"):
        if canonical_sha256(bundle.get(records)) != receipt.get(f"{records}_sha256"):
            raise ValueError(f"finalized {records} do not match the receipt")
    expected_summary = {
        key: receipt[key]
        for key in (
            "candidate_id", "model_id", "provider", "baseline_metrics",
            "candidate_metrics", "comparison", "decision",
            "production_disposition", "receipt_sha256",
        )
    }
    expected_summary["suite"] = receipt["run_identity"]["suite"]
    if summary != expected_summary:
        raise ValueError("finalized summary does not match the publication receipt")
    card = dataset_card(dataset_id, path_in_repo, receipt).encode("utf-8")

    api = HfApi(token=token)
    api.create_repo(
        repo_id=dataset_id,
        repo_type="dataset",
        private=False,
        exist_ok=True,
    )
    info = api.repo_info(repo_id=dataset_id, repo_type="dataset", revision="main")
    parent = str(info.sha or "")
    if not re.fullmatch(r"[0-9a-f]{40}", parent):
        raise ValueError("dataset head could not be identified before publication")
    if any(
        entry.rfilename == path_in_repo
        or entry.rfilename.startswith(path_in_repo + "/")
        for entry in info.siblings
    ):
        raise ValueError("run directory already exists; choose a new run identity")
    operations = [
        CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card)
    ] + [
        CommitOperationAdd(
            path_in_repo=f"{path_in_repo}/{name}", path_or_fileobj=data
        )
        for name, data in files.items()
    ]
    # One commit binds the public entry point to the exact bundle. The expected
    # parent rejects concurrent drift; failures propagate without a success record.
    commit = api.create_commit(
        repo_id=dataset_id,
        repo_type="dataset",
        revision="main",
        parent_commit=parent,
        operations=operations,
        commit_message=(
            f"eval: publish {receipt['candidate_id']} receipt "
            f"{str(receipt['receipt_sha256'])[:12]}"
        ),
    )
    oid = str(getattr(commit, "oid", "") or "")
    if not re.fullmatch(r"[0-9a-f]{40}", oid):
        raise ValueError("publication returned no immutable commit identity")
    published = f"https://huggingface.co/datasets/{dataset_id}/blob/{oid}"
    return {
        "dataset_id": dataset_id,
        "path_in_repo": path_in_repo,
        "commit_url": str(getattr(commit, "commit_url", "") or ""),
        "commit_oid": oid,
        "card_url": f"{published}/README.md",
        "summary_url": f"{published}/{path_in_repo}/summary.json",
        "bundle_url": f"{published}/{path_in_repo}/bundle.json",
        "receipt_url": f"{published}/{path_in_repo}/receipt.json",
    }
