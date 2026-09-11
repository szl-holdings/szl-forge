"""Plan preflight and full-chain evidence closure. Never an action authorizer."""
from __future__ import annotations

from typing import Any
from .core import AUTHORITY_ORDER, DENIED, age_seconds, digest, finite, integer, receipt, require, sha256, text

STAGES = ("source", "artifact", "runtime", "browser", "proof")


def plan_preflight(plan: dict[str, Any]) -> dict[str, Any]:
    """Check structure and resource ceilings, not the authenticity of approvals.

    The actual scheduler must verify principal identity, budget authorization,
    storage rights, tenant scope and provider policy through the existing control
    plane. A local JSON object cannot grant them. This helper launches nothing.
    """
    require(type(plan) is dict, "plan_object_required")
    required = {"schema", "runId", "candidateId", "sourceRevision", "modelRevision",
                "datasetSha256", "runtimeImageDigest", "maxWallSeconds", "maxCostUSD",
                "maxDownloadBytes", "maxInputTokens", "maxOutputTokens", "requestedGPUCount",
                "approvedBudgetUSD", "approvalReceiptId", "privateData", "rightsReviewed"}
    require(set(plan) == required and plan["schema"] == "szl.evaluation-plan.v1", "plan_schema")
    text(plan["runId"], 80); text(plan["candidateId"], 200)
    digest(plan["sourceRevision"], 40); digest(plan["modelRevision"], 40)
    digest(plan["datasetSha256"])
    image_digest = plan["runtimeImageDigest"]
    require(type(image_digest) is str and image_digest.startswith("sha256:"), "image_digest_required")
    digest(image_digest[7:])
    integer(plan["maxWallSeconds"], 1, 86400, "wall_seconds")
    integer(plan["maxDownloadBytes"], 0, 10**15, "download_bytes")
    integer(plan["maxInputTokens"], 1, 1048576, "input_tokens")
    integer(plan["maxOutputTokens"], 1, 1048576, "output_tokens")
    integer(plan["requestedGPUCount"], 0, 64, "gpu_count")
    for flag in ("privateData", "rightsReviewed"):
        require(type(plan[flag]) is bool, "plan_boolean_required")
    requested, approved = finite(plan["maxCostUSD"]), finite(plan["approvedBudgetUSD"])
    require(0 <= requested <= approved, "budget_exceeded")
    require(plan["privateData"] is False, "private_data_requires_separate_existing_admission")
    require(plan["rightsReviewed"] is True, "rights_not_reviewed")
    text(plan["approvalReceiptId"], 200)
    return receipt({"schema": "szl.plan-preflight.v1", "planSha256": sha256(plan),
                    "structureValid": True, "approvalAuthenticityVerified": False,
                    "schedulerMayLaunch": False, "productionPromotion": False,
                    "nextGate": "EXISTING_AUTHENTICATED_SCHEDULER_MUST_VALIDATE_APPROVAL"})


def summarize_closure(records: list[dict[str, Any]], *, expected_product_source: str,
                      expected_recipe_sha256: str, now: str, max_age_seconds: int = 3600) -> dict[str, Any]:
    """Assess supplied source-bound stage observations without promoting models.

    Input is not trusted solely because its hashes agree. No stage represents a
    universal health claim. The proof repository's own SHA is a different identity
    from the product SHA it records; both must be explicit in its evidence payload.
    """
    digest(expected_product_source, 40); digest(expected_recipe_sha256)
    integer(max_age_seconds, 1, 86400, "freshness_budget")
    require(type(records) is list and len(records) <= len(STAGES), "closure_record_bound")
    indexed = {}
    for record in records:
        require(type(record) is dict and set(record) == {"stage", "state", "observedAt", "productSourceRevision", "recipeSha256", "evidenceSha256"}, "closure_stage_schema")
        stage = record["stage"]
        require(stage in STAGES and stage not in indexed, "duplicate_or_unknown_stage")
        require(record["state"] in {"PASS", "FAIL", "UNAVAILABLE"}, "closure_stage_state")
        digest(record["evidenceSha256"]); digest(record["productSourceRevision"], 40)
        digest(record["recipeSha256"])
        age = age_seconds(record["observedAt"], now=now)
        state = record["state"]
        reason = "observation_reported"
        if age < -60 or age > max_age_seconds:
            state, reason = "UNAVAILABLE", "stale_or_future_evidence"
        elif record["productSourceRevision"] != expected_product_source or record["recipeSha256"] != expected_recipe_sha256:
            state, reason = "FAIL", "source_or_recipe_mismatch"
        indexed[stage] = {"state": state, "reason": reason, "evidenceSha256": record["evidenceSha256"]}
    statuses = {stage: indexed.get(stage, {"state": "UNAVAILABLE", "reason": "missing_stage"}) for stage in STAGES}
    complete = all(row["state"] == "PASS" for row in statuses.values())
    return receipt({"schema": "szl.chain-closure-review.v1", "authorityOrder": list(AUTHORITY_ORDER),
                    "expectedProductSource": expected_product_source, "expectedRecipeSha256": expected_recipe_sha256,
                    "stages": statuses, "reportedStageClosure": "COMPLETE" if complete else "PARTIAL",
                    "outerEvidenceAuthenticityVerified": False,
                    "productionDisposition": "HOLD", "authority": dict(DENIED)})


def quantization_canary(baseline: dict[str, Any], candidate: dict[str, Any], *,
                        max_absolute_kl_increase: float, max_case_regressions: int = 0) -> dict[str, Any]:
    """Screen comparable same-budget observations; retain an incumbent on failure.

    This conservative rule compares at exactly equal declared bytes. A production
    multi-size Pareto-curve evaluator must be built separately as described in the
    payload; interpolating mismatched file sizes here would fabricate comparability.
    """
    fields = {"modelRevision", "tokenizerSha256", "templateSha256", "calibrationSha256", "holdoutSha256",
              "runnerSha256", "hardwareProfileSha256", "tensorSchemaSha256", "generatorSha256",
              "weightSha256", "bytes", "kl", "caseRegressions"}
    for row in (baseline, candidate):
        require(type(row) is dict and set(row) == fields, "quant_canary_schema")
        digest(row["modelRevision"], 40)
        for key in fields - {"modelRevision", "bytes", "kl", "caseRegressions"}:
            digest(row[key])
        integer(row["bytes"], 1, 10**15, "quant_bytes")
        require(finite(row["kl"]) >= 0, "negative_kl")
        integer(row["caseRegressions"], 0, 10**7, "case_regressions")
    invariant = fields - {"weightSha256", "kl", "caseRegressions", "generatorSha256"}
    require(all(baseline[key] == candidate[key] for key in invariant), "quant_comparison_not_controlled")
    tolerance = finite(max_absolute_kl_increase)
    require(tolerance >= 0, "negative_tolerance")
    integer(max_case_regressions, 0, 10**7, "regression_budget")
    require(baseline["calibrationSha256"] != baseline["holdoutSha256"], "calibration_holdout_overlap_identity")
    passed = candidate["kl"] <= baseline["kl"] + tolerance and candidate["caseRegressions"] <= max_case_regressions
    return receipt({"schema": "szl.quant-canary-review.v1", "localRulePassed": passed,
                    "selection": "CANDIDATE_FOR_FURTHER_EVALUATION" if passed else "KEEP_INCUMBENT",
                    "baselineSha256": sha256(baseline), "candidateSha256": sha256(candidate),
                    "statisticalSignificanceEstablished": False, "productionPromotion": False})
