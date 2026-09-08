# SPDX-License-Identifier: Apache-2.0
"""Receipt construction and optional Hugging Face dataset publication."""
from __future__ import annotations

import argparse
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

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
    completed = (
        bool(source_checks.get("pass"))
        and baseline_metrics["successful_call_rate"] == 1.0
        and candidate_metrics["successful_call_rate"] == 1.0
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
            "upstream_benchmark_claims": "NOT_USED",
            "provider_execution_revision": "UNAVAILABLE",
            "provider_hardware": "UNAVAILABLE",
        },
        "fallback_evidence_sha256": canonical_sha256(dict(fallback)),
        "fallback_evidence": dict(fallback),
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


def publish(
    output: Path,
    token: str,
    dataset_id: str,
    path_in_repo: str,
    receipt: Mapping[str, Any],
) -> dict[str, str]:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(
        repo_id=dataset_id,
        repo_type="dataset",
        private=False,
        exist_ok=True,
    )
    card = output / "DATASET_CARD.md"
    card.write_text(
        "---\nlicense: apache-2.0\n"
        "pretty_name: SZL Frontier Evaluation Receipts\n"
        "tags:\n- evaluation\n- model-governance\n- reproducibility\n---\n\n"
        "# SZL Frontier Evaluation Receipts\n\n"
        "Content-addressed model evaluation evidence. Receipts do not grant "
        "production authority; A11oy remains the consequential-action "
        "admission layer.\n",
        encoding="utf-8",
    )
    try:
        api.upload_file(
            path_or_fileobj=str(card),
            path_in_repo="README.md",
            repo_id=dataset_id,
            repo_type="dataset",
            commit_message=(
                "docs: initialize frontier evaluation receipt dataset"
            ),
        )
    except Exception:
        pass
    commit = api.upload_folder(
        folder_path=str(output),
        path_in_repo=path_in_repo,
        repo_id=dataset_id,
        repo_type="dataset",
        commit_message=(
            f"eval: publish {receipt['candidate_id']} receipt "
            f"{str(receipt['receipt_sha256'])[:12]}"
        ),
        ignore_patterns=[
            "DATASET_CARD.md",
            ".source-cache/**",
            "**/__pycache__/**",
            "**/*.pyc",
        ],
    )
    return {
        "dataset_id": dataset_id,
        "path_in_repo": path_in_repo,
        "commit_url": str(getattr(commit, "commit_url", "") or ""),
        "commit_oid": str(getattr(commit, "oid", "") or ""),
    }
