"""Bind a reviewed proposal to one typed, public synthetic experiment.

This is local research tooling. The caller supplies the recipe and asserts that
an operator reviewed its relationship to the report. Neither assertion is
authenticated here. No report text is interpreted as code or as a recipe.
"""
from __future__ import annotations

import hashlib

from inference import research_cycle as cycle
from inference import research_investigator as investigator


MANIFEST_SCHEMA = "szl.research-admission-manifest/v1"
RESULT_SCHEMA = "szl.research-admission-result/v1"
KIND = "lexical_retrieval_v1"
REPORT_KEYS = {
    "schema", "state", "question", "max_turns", "system_prompt_sha256",
    "final_prompt_sha256", "corpus_sha256", "execution_place", "proposal",
    "trace", "evidence", "experiment_executed", "novelty",
    "semantic_quality", "source_authenticity", "training_eligible",
    "publication_eligible", "autonomy_eligible",
}


class AdmissionError(ValueError):
    """An input does not satisfy the local research admission contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AdmissionError(message)


def _json_bytes(raw: bytes, limit: int) -> object:
    _require(type(raw) is bytes, "exact file bytes required")
    return cycle.strict_json(raw, limit)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def admit(manifest_bytes: bytes, report_bytes: bytes, suite_bytes: bytes) -> dict:
    """Evaluate exactly one caller-supplied recipe against a pinned suite.

    All inputs are exact file bytes. The returned HOLD record is unsigned and
    grants no production, training, publication, or autonomous authority.
    """
    manifest = _json_bytes(manifest_bytes, 8192)
    _require(type(manifest) is dict and set(manifest) == {
        "schema", "kind", "review_assertion", "investigator_report_sha256",
        "suite_sha256", "recipe",
    }, "manifest fields invalid")
    _require(manifest["schema"] == MANIFEST_SCHEMA, "manifest schema invalid")
    _require(manifest["kind"] == KIND, "experiment kind invalid")
    _require(manifest["review_assertion"] == "OPERATOR_REVIEWED", "operator review assertion required")
    for field, raw in (("investigator_report_sha256", report_bytes), ("suite_sha256", suite_bytes)):
        _require(type(raw) is bytes, "exact file bytes required")
        expected = manifest[field]
        _require(type(expected) is str and cycle.HEX64.fullmatch(expected), "exact SHA256 required")
        _require(_sha256(raw) == expected, f"{field} mismatch")

    report = _json_bytes(report_bytes, cycle.MAX_INPUT)
    _require(type(report) is dict and set(report) == REPORT_KEYS, "investigator report fields invalid")
    _require(report["schema"] == investigator.SCHEMA, "investigator report schema invalid")
    _require(report["state"] == "PROPOSAL_REQUIRES_REVIEW", "report is not a proposal requiring review")
    _require(type(report["proposal"]) is dict and bool(report["proposal"]), "proposal missing")
    for field in ("experiment_executed", "training_eligible", "publication_eligible", "autonomy_eligible"):
        _require(report[field] is False, f"report {field} must be false")
    _require(report["novelty"] == "NOT_ESTABLISHED", "report novelty status invalid")
    _require(report["semantic_quality"] == "NOT_EVALUATED", "report quality status invalid")
    _require(report["source_authenticity"] == "CALLER_SUPPLIED_NOT_INDEPENDENTLY_VERIFIED",
             "report source boundary invalid")

    suite = cycle.validate_suite(_json_bytes(suite_bytes, cycle.MAX_INPUT))
    selected = cycle.recipe(manifest["recipe"])

    # The existing cycle is the fixed evaluator. SYNTHETIC_REPLAY describes its
    # mechanism, not the origin of this human-supplied recipe or its review.
    evaluator = cycle.run_cycle(suite, lambda _context: cycle.canonical(selected).decode("utf-8"),
                                mode="SYNTHETIC_REPLAY", attempts=1)
    _require(evaluator["attempts"][0]["status"] in {"EVALUATED", "DUPLICATE"},
             "typed recipe was not evaluated")
    result = {
        "schema": RESULT_SCHEMA,
        "kind": KIND,
        "manifest_sha256": _sha256(manifest_bytes),
        "investigator_report_sha256": _sha256(report_bytes),
        "suite_file_sha256": _sha256(suite_bytes),
        "recipe_sha256": cycle.digest(selected),
        "recipe_origin": "CALLER_SUPPLIED_AFTER_REVIEW",
        "review_provenance": "CALLER_ASSERTED_NOT_AUTHENTICATED",
        "report_check": "PINNED_BYTES_STATE_AND_NONEXECUTION_FLAGS_ONLY",
        "evaluator_mechanism": "SYNTHETIC_REPLAY_OF_CALLER_SUPPLIED_RECIPE",
        "evaluator_receipt_sha256": evaluator["receipt_sha256"],
        "evaluator_result": evaluator,
        "validation_outcome": evaluator["decision"],
        "evaluation_scope": cycle.SCOPE,
        "production_disposition": "HOLD",
        "training_admission": False,
        "publication_eligible": False,
        "autonomy_eligible": False,
        "investigator_proposal_executed": False,
        "synthetic_evaluation_executed": True,
        "signature_status": "UNSIGNED_LOCAL",
        "limits": [
            "operator review, report authorship, and source truth are not authenticated",
            "the relationship between report prose and the typed recipe is caller asserted",
            "public validation is reusable; repeated trials can overfit it",
            "synthetic retrieval results establish no production or model quality claim",
        ],
    }
    result["receipt_sha256"] = cycle.digest(result)
    return result
