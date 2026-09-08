#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface_hub==1.30.0"]
# ///
"""Run real GLM-5.3-Flash versus Khipu evaluation; never auto-promote."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from frontier.evaluation.core import (  # noqa: E402
    execute_case,
    file_sha256,
    http_json,
    select_cases,
)
from frontier.evaluation.failure import (  # noqa: E402
    make_provider_unavailable_receipt,
)
from frontier.evaluation.provider import (  # noqa: E402
    BASELINE_URL,
    PROVIDERS,
    ROUTER_URL,
    ProviderUnavailable,
    choose_provider,
    exercise_fallback,
)
from frontier.evaluation.receipt import make_receipt, publish  # noqa: E402
from frontier.evaluation.source import load_json, verify_source, write_json  # noqa: E402

DATASET_ID = "SZLHOLDINGS/szl-frontier-evaluation-receipts"
HEX40 = re.compile(r"^[0-9a-f]{40}$")


class EvaluationError(RuntimeError):
    pass


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wave",
        type=Path,
        default=Path("frontier/PINNED_MODEL_WAVE_2026-09-07.json"),
    )
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=Path("frontier/evaluation/fixtures.v1.json"),
    )
    parser.add_argument(
        "--source-attestation",
        type=Path,
        default=Path(
            "frontier/evaluation/source-attestation.glm-5.3-flash.v1.json"
        ),
    )
    parser.add_argument("--candidate-id", default="glm-5-3-flash")
    parser.add_argument("--suite", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--providers", nargs="+", default=list(PROVIDERS))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--source-repository", default="szl-holdings/szl-forge"
    )
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--dataset-id", default=DATASET_ID)
    return parser.parse_args(argv)


def _candidate(wave: Mapping[str, Any], candidate_id: str) -> Mapping[str, Any]:
    candidates = {
        str(item.get("candidate_id")): item
        for item in wave.get("candidates", [])
        if isinstance(item, Mapping)
    }
    candidate = candidates.get(candidate_id)
    if candidate is None:
        raise EvaluationError(f"unknown candidate: {candidate_id}")
    if candidate.get("production_disposition") != "HOLD":
        raise EvaluationError("runner accepts only production-held candidates")
    return candidate


def _fallback_provider(
    configured: Sequence[str], attempts: Sequence[Mapping[str, Any]]
) -> str:
    for attempt in attempts:
        value = attempt.get("provider")
        if isinstance(value, str) and value:
            return value
    if configured:
        return str(configured[0])
    return "unavailable"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not HEX40.fullmatch(args.source_revision.lower()):
        raise EvaluationError(
            "--source-revision must be an immutable 40-hex commit"
        )
    inference_token = os.environ.get("HF_INFERENCE_TOKEN", "").strip()
    publish_token = os.environ.get("HF_TOKEN", "").strip()
    if not inference_token:
        raise EvaluationError(
            "HF_INFERENCE_TOKEN is required for real provider inference"
        )
    if args.publish and not publish_token:
        raise EvaluationError("HF_TOKEN is required when --publish is enabled")

    wave = load_json(args.wave)
    fixtures = load_json(args.fixtures)
    attestation = load_json(args.source_attestation)
    candidate = _candidate(wave, args.candidate_id)
    if (
        candidate.get("upstream_model_id") != attestation.get("model_id")
        or candidate.get("upstream_revision")
        != attestation.get("requested_revision")
    ):
        raise EvaluationError(
            "candidate and source attestation identities differ"
        )

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    cases = select_cases(fixtures, args.suite)
    source_checks = verify_source(
        attestation, inference_token, output / ".source-cache"
    )
    if not source_checks["pass"]:
        raise EvaluationError("source attestation verification failed")

    identity = http_json(
        f"{BASELINE_URL}/api/v1/identity", timeout=args.timeout_seconds
    )
    if not identity.ok or identity.body is None:
        raise EvaluationError("baseline identity endpoint unavailable")
    baseline_identity = identity.body
    baseline_model = (
        baseline_identity.get("runtime", {})
        .get("openai_compatible_subset", {})
        .get("model_id")
    )
    if not isinstance(baseline_model, str) or not baseline_model:
        raise EvaluationError("baseline identity did not expose model_id")

    baseline_records = [
        execute_case(
            case,
            route="baseline",
            model=baseline_model,
            url=f"{BASELINE_URL}/v1/chat/completions",
            token=None,
            max_tokens=32,
            timeout=args.timeout_seconds,
        )
        for case in cases
    ]

    provider_selection: Mapping[str, Any] | None = None
    try:
        provider, first, attempts = choose_provider(
            cases[0],
            model_id=str(candidate["upstream_model_id"]),
            token=inference_token,
            providers=args.providers,
            timeout=args.timeout_seconds,
        )
        candidate_records = [first] + [
            execute_case(
                case,
                route="candidate",
                model=f"{candidate['upstream_model_id']}:{provider}",
                url=ROUTER_URL,
                token=inference_token,
                max_tokens=96,
                timeout=args.timeout_seconds,
            )
            for case in cases[1:]
        ]
    except ProviderUnavailable as error:
        provider = "UNAVAILABLE"
        attempts = [dict(attempt) for attempt in error.attempts]
        candidate_records = []
        provider_selection = error.to_dict()

    fallback = exercise_fallback(
        cases[0],
        model_id=str(candidate["upstream_model_id"]),
        provider=(
            provider
            if provider != "UNAVAILABLE"
            else _fallback_provider(args.providers, attempts)
        ),
        token=inference_token,
        baseline_model=baseline_model,
        timeout=args.timeout_seconds,
    )

    if provider_selection is None:
        receipt = make_receipt(
            args=args,
            candidate=candidate,
            fixtures_sha256=file_sha256(args.fixtures),
            source_checks=source_checks,
            baseline_identity=baseline_identity,
            baseline_records=baseline_records,
            candidate_records=candidate_records,
            fallback=fallback,
            provider=provider,
            provider_attempts=attempts,
        )
    else:
        receipt, candidate_records = make_provider_unavailable_receipt(
            args=args,
            candidate=candidate,
            cases=cases,
            fixtures_sha256=file_sha256(args.fixtures),
            source_checks=source_checks,
            baseline_identity=baseline_identity,
            baseline_records=baseline_records,
            fallback=fallback,
            provider_attempts=attempts,
            provider_selection=provider_selection,
        )

    bundle = {
        "schema": "szl.frontier.evaluation-bundle.v1",
        "receipt": receipt,
        "source_checks": source_checks,
        "provider_selection": provider_selection,
        "baseline_results": baseline_records,
        "candidate_results": candidate_records,
        "fixture_manifest": {
            "schema": fixtures.get("schema"),
            "suite": fixtures.get("suite"),
            "sha256": file_sha256(args.fixtures),
            "case_ids": [case["id"] for case in cases],
        },
    }
    write_json(output / "receipt.json", receipt)
    write_json(output / "bundle.json", bundle)
    write_json(
        output / "summary.json",
        {
            "candidate_id": receipt["candidate_id"],
            "model_id": receipt["model_id"],
            "provider": receipt["provider"],
            "suite": args.suite,
            "baseline_metrics": receipt["baseline_metrics"],
            "candidate_metrics": receipt["candidate_metrics"],
            "comparison": receipt["comparison"],
            "decision": receipt["decision"],
            "production_disposition": "HOLD",
            "receipt_sha256": receipt["receipt_sha256"],
        },
    )

    publication = None
    if args.publish and provider_selection is None:
        safe_run = re.sub(r"[^A-Za-z0-9._-]+", "-", args.run_id).strip("-")
        path = (
            f"runs/{datetime.now(timezone.utc).strftime('%Y/%m/%d')}/"
            f"{safe_run}/{args.candidate_id}"
        )
        publication = publish(
            output, publish_token, args.dataset_id, path, receipt
        )
        write_json(output / "publication.json", publication)
    elif args.publish:
        publication = {
            "status": "SKIPPED_PROVIDER_UNAVAILABLE",
            "production_disposition": "HOLD",
            "promotion_effect": "NONE",
        }
        write_json(output / "publication.json", publication)

    completed = receipt["decision"] == "EVIDENCE_COMPLETE_REVIEW_REQUIRED"
    negative_evidence_complete = (
        receipt["decision"] == "HOLD_PROVIDER_UNAVAILABLE"
        and bool(fallback.get("pass"))
        and bool(receipt.get("receipt_sha256"))
    )
    result = {
        "ok": completed,
        "negative_evidence_complete": negative_evidence_complete,
        "candidate_id": receipt["candidate_id"],
        "provider": receipt["provider"],
        "provider_attempts_sha256": receipt.get(
            "provider_attempts_sha256"
        ),
        "suite": args.suite,
        "candidate_score_rate": receipt["candidate_metrics"].get(
            "score_rate"
        ),
        "baseline_score_rate": receipt["baseline_metrics"]["score_rate"],
        "fallback_pass": fallback["pass"],
        "decision": receipt["decision"],
        "receipt_sha256": receipt["receipt_sha256"],
        "production_disposition": "HOLD",
        "publication": publication,
    }
    print(json.dumps(result, sort_keys=True))
    return 0 if completed else 2


if __name__ == "__main__":
    raise SystemExit(main())
