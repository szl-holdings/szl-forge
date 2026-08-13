#!/usr/bin/env python3
"""Revalidate and compare exact base, v2, and v3 reports without promoting them."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluate_candidate import (
    absolute_gate,
    evaluation_split,
    recompute_counts,
    validate_refusal,
    validate_structured,
    verify_report_digest,
)
from train_candidate import (
    QualificationError,
    fresh_exact_source,
    load_committed_json,
    sanitized_error,
    sha256_bytes,
    sha256_json,
)


CASE_RESULT_KEYS = (
    "parsed",
    "schemaValid",
    "requestBound",
    "dispositionCorrect",
    "authoritySafe",
    "evidenceExact",
    "effortContractExact",
    "recoveryExact",
    "claimExact",
    "refusalContractExact",
    "reasoningTagsAbsent",
    "unsupportedEvidenceCount",
    "casePass",
)


def load_report(path: Path, expected_kind: str) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    verify_report_digest(report, f"{expected_kind} evaluation report")
    if report.get("schema") != "szl.frontier-eval-run/v3":
        raise QualificationError(f"{expected_kind} report schema is unsupported")
    if report.get("modelKind") != expected_kind:
        raise QualificationError(
            f"expected {expected_kind}, got {report.get('modelKind')}"
        )
    if report.get("state") != "MEASURED_EVALUATION_COMPLETED_UNATTESTED":
        raise QualificationError(f"{expected_kind} evaluation is not completed")
    if report.get("split") != "TEST":
        raise QualificationError(f"{expected_kind} report is not a test result")
    if report.get("comparisonEligible") is not False:
        raise QualificationError(f"{expected_kind} raw report crossed comparison boundary")
    if report.get("authenticatedEvaluationEnvelopePresent") is not False:
        raise QualificationError(f"{expected_kind} report authentication state drifted")
    if report.get("receiptEligible") is not False:
        raise QualificationError(f"{expected_kind} unsigned report crossed receipt boundary")
    if report.get("publicationEligible") is not False:
        raise QualificationError(f"{expected_kind} unsigned report crossed publication boundary")
    return report


def verify_model_identity(
    report: dict[str, Any],
    expected_kind: str,
    candidate: dict[str, Any],
) -> None:
    identity = report.get("model") or {}
    implementation = candidate["actual_training_base"]
    required_base = {
        "kind": expected_kind,
        "baseRole": "PINNED_UNSLOTH_IMPLEMENTATION_BASE",
        "baseRepoId": implementation["repo_id"],
        "baseRevision": implementation["revision"],
        "loadIn4Bit": implementation["load_in_4bit"],
        "upstreamByteEquivalenceVerified": False,
    }
    for key, expected in required_base.items():
        if identity.get(key) != expected:
            raise QualificationError(f"{expected_kind} model identity {key} differs")
    if expected_kind == "base":
        if any(key.startswith("adapter") for key in identity):
            raise QualificationError("base report unexpectedly contains an adapter identity")
    elif expected_kind == "v2":
        predecessor = candidate["predecessor"]
        expected = {
            "adapterRepoId": predecessor["repo_id"],
            "adapterRevision": predecessor["release_revision"],
            "adapterModelSha256": predecessor["adapter_model_sha256"],
        }
        for key, value in expected.items():
            if identity.get(key) != value:
                raise QualificationError(f"v2 model identity {key} differs")
        if int(identity.get("adapterTensorCount", 0)) < 1:
            raise QualificationError("v2 adapter tensor count is absent")
    else:
        adapter_sha = identity.get("adapterAggregateSha256")
        if not isinstance(adapter_sha, str) or len(adapter_sha) != 64:
            raise QualificationError("v3 adapter aggregate digest is absent")
        if identity.get("adapterSource") != "LOCAL_ATTESTATION_PENDING":
            raise QualificationError("v3 adapter source state differs")
        training_sha = report.get("trainingReportSha256")
        if not isinstance(training_sha, str) or len(training_sha) != 64:
            raise QualificationError("v3 training report digest is absent")


def revalidate_report(
    report: dict[str, Any],
    *,
    expected_kind: str,
    rows: list[dict[str, Any]],
    response_validator: Any,
    protocol_sha: str,
    source_commit: str,
    candidate: dict[str, Any],
) -> tuple[dict[str, int], dict[str, float]]:
    if report.get("candidateId") != candidate["candidate_id"]:
        raise QualificationError(f"{expected_kind} candidate identity differs")
    if report.get("runtimePackages") != candidate["runtime_lock"]:
        raise QualificationError(f"{expected_kind} runtime package lock differs")
    if (report.get("source") or {}).get("revision") != source_commit:
        raise QualificationError(f"{expected_kind} source revision differs")
    if (report.get("protocol") or {}).get("protocolSha256") != protocol_sha:
        raise QualificationError(f"{expected_kind} evaluation protocol differs")
    verify_model_identity(report, expected_kind, candidate)
    cases = report.get("cases")
    if not isinstance(cases, list) or len(cases) != len(rows):
        raise QualificationError(
            f"{expected_kind} report does not contain the full {len(rows)}-case roster"
        )
    case_ids = [case.get("caseId") for case in cases]
    expected_ids = [row["caseId"] for row in rows]
    if case_ids != expected_ids or len(case_ids) != len(set(case_ids)):
        raise QualificationError(f"{expected_kind} ordered case roster differs")

    recomputed_cases: list[dict[str, Any]] = []
    for row, stored in zip(rows, cases, strict=True):
        if stored.get("kind") != row["kind"]:
            raise QualificationError(f"{expected_kind} case kind differs at {row['caseId']}")
        if stored.get("promptSha256") != sha256_json(row["messages"]):
            raise QualificationError(f"{expected_kind} prompt digest differs at {row['caseId']}")
        output = stored.get("output")
        if not isinstance(output, str):
            raise QualificationError(f"{expected_kind} output is absent at {row['caseId']}")
        if stored.get("outputSha256") != sha256_bytes(output.encode("utf-8")):
            raise QualificationError(f"{expected_kind} output digest differs at {row['caseId']}")
        if row["kind"] == "REFUSAL":
            computed = validate_refusal(output, row)
        else:
            computed = validate_structured(output, row, response_validator)
        for key in CASE_RESULT_KEYS:
            if key in computed and stored.get(key) != computed[key]:
                raise QualificationError(
                    f"{expected_kind} stored result {key} differs at {row['caseId']}"
                )
        recomputed_cases.append({"kind": row["kind"], **computed})
    counts, rates = recompute_counts(recomputed_cases)
    if report.get("counts") != counts:
        raise QualificationError(f"{expected_kind} reported counts were not recomputed")
    if report.get("rates") != rates:
        raise QualificationError(f"{expected_kind} reported rates were not recomputed")
    if report.get("absoluteGatePassed") is not absolute_gate(counts):
        raise QualificationError(f"{expected_kind} absolute gate flag differs")
    return counts, rates


def compare(
    base: dict[str, Any],
    v2: dict[str, Any],
    v3: dict[str, Any],
    *,
    rows: list[dict[str, Any]],
    response_validator: Any,
    protocol_sha: str,
    source_commit: str,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    reports = {"base": base, "v2": v2, "v3": v3}
    recomputed: dict[str, dict[str, Any]] = {}
    for kind, report in reports.items():
        counts, rates = revalidate_report(
            report,
            expected_kind=kind,
            rows=rows,
            response_validator=response_validator,
            protocol_sha=protocol_sha,
            source_commit=source_commit,
            candidate=candidate,
        )
        recomputed[kind] = {"counts": counts, "rates": rates}
    v3_counts = recomputed["v3"]["counts"]
    v2_counts = recomputed["v2"]["counts"]
    required_improvement = candidate["evaluation_protocol"][
        "required_strict_case_improvement_over_v2"
    ]
    strict_delta = v3_counts["strictCasePass"] - v2_counts["strictCasePass"]
    if not absolute_gate(v3_counts):
        raise QualificationError("v3 did not pass every preregistered absolute gate")
    if strict_delta < required_improvement:
        raise QualificationError(
            f"v3 strict-case improvement {strict_delta} is below {required_improvement}"
        )
    if v3_counts["authoritySafe"] < v2_counts["authoritySafe"]:
        raise QualificationError("v3 trails v2 on authority safety")
    report = {
        "schema": "szl.frontier-comparison/v2",
        "candidateId": candidate["candidate_id"],
        "state": "UNAUTHENTICATED_COMPARISON_CRITERIA_SATISFIED",
        "measuredAt": datetime.now(timezone.utc).isoformat(),
        "sourceRevision": source_commit,
        "protocolSha256": protocol_sha,
        "caseCount": len(rows),
        "inputReports": {
            kind: reports[kind]["reportSha256"] for kind in ("base", "v2", "v3")
        },
        "recomputedResults": recomputed,
        "strictCaseImprovementOverV2": strict_delta,
        "requiredStrictCaseImprovementOverV2": required_improvement,
        "absoluteGatePassed": True,
        "authoritySafetyNoRegression": True,
        "comparisonCriteriaSatisfied": True,
        "integrityDigestIsAuthentication": False,
        "authenticatedComparisonEnvelopePresent": False,
        "requiresAuthenticatedSignerRevalidation": True,
        "receiptEligible": False,
        "publicationEligible": False,
        "autonomyEligible": False,
        "claimBoundary": (
            "This unauthenticated recomputation shows local criteria satisfaction on one "
            "committed public, project-authored suite. It cannot establish candidate "
            "provenance or authorize a receipt, publication, runtime, or autonomy claim."
        ),
    }
    report["reportSha256"] = sha256_json(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--v2", type=Path, required=True)
    parser.add_argument("--v3", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        fresh_exact_source(args.source_commit)
        candidate = load_committed_json(args.source_commit, "candidate.json")
        rows, split_evidence = evaluation_split(
            args.source_commit, "test", candidate
        )
        report = compare(
            load_report(args.base, "base"),
            load_report(args.v2, "v2"),
            load_report(args.v3, "v3"),
            rows=rows,
            response_validator=split_evidence["responseValidator"],
            protocol_sha=split_evidence["protocol"]["protocolSha256"],
            source_commit=args.source_commit,
            candidate=candidate,
        )
        code = 0
    except Exception as exc:  # noqa: BLE001 - fail closed with bounded evidence
        report = {
            "schema": "szl.frontier-comparison/v2",
            "state": "BLOCKED",
            "measuredAt": datetime.now(timezone.utc).isoformat(),
            "fatal": sanitized_error(exc),
            "comparisonCriteriaSatisfied": False,
            "receiptEligible": False,
            "publicationEligible": False,
            "autonomyEligible": False,
        }
        report["reportSha256"] = sha256_json(report)
        code = 1
    args.report.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return code


if __name__ == "__main__":
    sys.exit(main())
