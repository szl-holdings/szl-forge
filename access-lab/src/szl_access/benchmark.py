"""Owned synthetic regression cases; not a held-out model qualification benchmark."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any, Callable

from .engine import apply_proposal, deterministic_proposal


RECIPE_VERSION = "szl-access-synthetic-regression-v1"
PROVENANCE = "SZL-authored synthetic HTML; no scraped source or personal data"
REPAIRED = "VERIFIED_SCOPED_REPAIR"
REVIEW = "REVIEW_REQUIRED"
UNCHANGED = "NO_CHANGE"

# Explicit authored fixture expectations, not identifiers inferred by the engine.
EXPECTED_REPAIR_IDS = {
    "repair-text": ("contact-name",),
    "repair-email": ("contact-email",),
    "repair-form": ("applicant",),
    "repair-div": ("query",),
    "repair-fieldset": ("location",),
    "repair-required": ("contact",),
    "repair-escaped": ("interest",),
    "repair-single-quotes": ("town",),
    "repair-whitespace": ("organization",),
    "repair-two-fields": ("first", "last"),
    "repair-two-forms": ("member", "volunteer"),
    "repair-mixed": ("region",),
}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def benchmark_cases() -> list[dict[str, Any]]:
    """Return fresh public fixtures and explicit labels, never a training split.

    Family labels are split-group candidates for a future separately collected
    benchmark. This small public suite was used during development and is NOT
    unseen evaluation data. None of its examples establish WCAG conformance.
    """
    definitions = [
        ("repair-text", "text-adjacency", "Contact name", '<label>Contact name</label><input id="contact-name" type="text">', REPAIRED, 1),
        ("repair-email", "email-adjacency", "Email field", '<label>Email</label>\n<input id="contact-email" type="email">', REPAIRED, 1),
        ("repair-form", "form-wrapper", "Application form", '<form><label>Applicant</label><input id="applicant" type="text"></form>', REPAIRED, 1),
        ("repair-div", "div-wrapper", "Search field", '<div><label>Search</label> <input id="query" type="search"></div>', REPAIRED, 1),
        ("repair-fieldset", "fieldset-wrapper", "Fieldset", '<fieldset><label>Location</label><input id="location" type="text"></fieldset>', REPAIRED, 1),
        ("repair-required", "boolean-attribute", "Required contact", '<label>Contact</label><input id="contact" type="text" required>', REPAIRED, 1),
        ("repair-escaped", "escaped-label", "Escaped text", '<label>Arts &amp; crafts</label><input id="interest" type="text">', REPAIRED, 1),
        ("repair-single-quotes", "single-quoted-attribute", "Single-quoted HTML", "<label>Town</label><input id='town' type='text'>", REPAIRED, 1),
        ("repair-whitespace", "whitespace-adjacency", "Whitespace-separated field", '<label>Organization</label>\n\t  <input id="organization" type="text">', REPAIRED, 1),
        ("repair-two-fields", "multiple-fields", "Two fields", '<form><label>First name</label><input id="first" type="text"><label>Last name</label><input id="last" type="text"></form>', REPAIRED, 2),
        ("repair-two-forms", "distinct-forms", "Independent forms", '<form><label>Member</label><input id="member" type="text"></form><form><label>Volunteer</label><input id="volunteer" type="text"></form>', REPAIRED, 2),
        ("repair-mixed", "mixed-existing-and-missing", "Preserve existing binding", '<form><label for="city">City</label><input id="city" type="text"><label>Region</label><input id="region" type="text"></form>', REPAIRED, 1),
        ("review-script", "active-script", "Script requires review", '<label>Name</label><input id="name"><script>alert(1)</script>', REVIEW, 0),
        ("review-style", "style-element", "Style requires review", '<style>input{display:none}</style><label>Name</label><input id="name">', REVIEW, 0),
        ("review-event", "event-attribute", "Event handler requires review", '<label>Name</label><input id="name" onclick="submit()">', REVIEW, 0),
        ("review-external", "external-action", "External action requires review", '<form action="https://example.invalid/submit"><label>Name</label><input id="name"></form>', REVIEW, 0),
        ("review-duplicate-id", "duplicate-identifier", "Duplicate IDs", '<label>First</label><input id="duplicate"><label>Second</label><input id="duplicate">', REVIEW, 0),
        ("review-unclosed", "malformed-markup", "Unclosed label", '<label>Name<input id="name">', REVIEW, 0),
        ("review-nested-forms", "nested-forms", "Nested forms", '<form><form><label>Name</label><input id="name"></form></form>', REVIEW, 0),
        ("review-hidden", "hidden-control", "Hidden control", '<label>Name</label><input id="name" type="hidden">', REVIEW, 0),
        ("review-aria", "aria-semantics", "Existing ARIA semantics", '<label>Name</label><input id="name" aria-label="Legal name">', REVIEW, 0),
        ("review-two-labels", "ambiguous-labels", "Two candidate labels", '<label>First</label><label>Second</label><input id="name">', REVIEW, 0),
        ("review-nonadjacent", "nonadjacent-label", "Intervening markup", '<label>Name</label><div>Help</div><input id="name">', REVIEW, 0),
        ("review-svg", "foreign-markup", "SVG requires review", '<svg></svg><label>Name</label><input id="name">', REVIEW, 0),
        ("review-unquoted", "unquoted-attribute", "Unquoted attribute", '<label>Name</label><input id=name>', REVIEW, 0),
        ("review-missing-id", "missing-identifier", "Control without ID", '<label>Name</label><input type="text">', REVIEW, 0),
        ("review-nested-label", "complex-label", "Nested label content", '<label><strong>Name</strong></label><input id="name">', REVIEW, 0),
        ("review-inline-style", "inline-style", "Inline style requires review", '<label>Name</label><input id="name" style="display:none">', REVIEW, 0),
        ("review-cross-parent", "cross-parent-association", "Different parents", '<div><label>Name</label></div><div><input id="name"></div>', REVIEW, 0),
        ("review-bad-reference", "broken-existing-binding", "Label references absent control", '<label for="absent">Name</label><input id="name">', REVIEW, 0),
        ("unchanged-bound", "existing-text-binding", "Already labeled", '<label for="name">Name</label><input id="name" type="text">', UNCHANGED, 0),
        ("unchanged-email", "existing-email-binding", "Already labeled email", '<label for="email">Email</label><input id="email" type="email">', UNCHANGED, 0),
        ("unchanged-form", "existing-form-binding", "Bound form", '<form><label for="member">Member</label><input id="member" type="text"></form>', UNCHANGED, 0),
        ("unchanged-div", "existing-div-binding", "Bound div", '<div><label for="town">Town</label><input id="town" type="text"></div>', UNCHANGED, 0),
        ("unchanged-fieldset", "existing-fieldset-binding", "Bound fieldset", '<fieldset><label for="region">Region</label><input id="region" type="text"></fieldset>', UNCHANGED, 0),
        ("unchanged-empty-form", "no-controls", "No fields to repair", '<form></form>', UNCHANGED, 0),
    ]
    cases = []
    for case_id, family, title, source, state, bindings in definitions:
        expected_output = source
        for control_id in EXPECTED_REPAIR_IDS.get(case_id, ()):
            expected_output = expected_output.replace("<label>", '<label for="' + control_id + '">', 1)
        cases.append({
            "id": case_id,
            "family": family,
            "split_group": family,
            "title": title,
            "source": source,
            "source_sha256": _sha(source),
            "expected_state": state,
            "expected_bindings": bindings,
            "expected_output": expected_output,
            "expected_output_sha256": _sha(expected_output),
            "provenance": PROVENANCE,
            "split": "public_development_regression_not_held_out",
        })
    return cases


def examples() -> list[dict[str, str]]:
    """Small built-in UI examples, including one explicit refusal boundary."""
    selected = {"repair-text", "repair-two-fields", "repair-two-forms", "unchanged-bound", "review-aria"}
    return [
        {"id": case["id"], "title": case["title"], "source": case["source"]}
        for case in benchmark_cases()
        if case["id"] in selected
    ]


def run_benchmark(
    proposer: Callable[[str], dict[str, Any]] | None = None,
    *,
    methodology: str = "deterministic",
) -> dict[str, Any]:
    """Evaluate proposals through the same deterministic gate as the application.

    A caller may supply another proposer for a paired comparison. Its display
    name is untrusted attribution, not proof of which model/weights executed.
    The builtin path makes no downloads, training calls, network calls, or HTML
    execution. A caller-supplied proposer owns its separate execution boundary.
    """
    proposal_function = deterministic_proposal if proposer is None else proposer
    if proposer is None and methodology != "deterministic":
        raise ValueError("A non-deterministic methodology requires an explicit proposer")
    cases = benchmark_cases()
    recipe_sha256 = _sha(json.dumps(cases, sort_keys=True, separators=(",", ":")))
    rows: list[dict[str, Any]] = []
    for case in cases:
        source = case["source"]
        problems: list[str] = []
        state = "ERROR"
        observed_output_sha256 = None
        try:
            result = apply_proposal(source, proposal_function(source))
            state = result.get("state", "MISSING_STATE")
            output = result.get("output_html")
            observed_output_sha256 = result.get("output_sha256")
            if state != case["expected_state"]:
                problems.append("state_mismatch")
            if result.get("source_sha256") != case["source_sha256"]:
                problems.append("source_hash_mismatch")
            if not isinstance(output, str) or observed_output_sha256 != _sha(output):
                problems.append("output_hash_mismatch")
            if output != case["expected_output"]:
                problems.append("exact_output_mismatch")
            if len(result.get("applied_bindings", [])) != case["expected_bindings"]:
                problems.append("binding_count_mismatch")
            if case["expected_state"] != REPAIRED and output != source:
                problems.append("non_repair_mutated_source")
            if case["expected_state"] == REPAIRED and isinstance(output, str):
                if output == source:
                    problems.append("repair_did_not_change_source")
                second = apply_proposal(output, deterministic_proposal(output))
                if second.get("state") != UNCHANGED or second.get("output_html") != output:
                    problems.append("repair_not_idempotent")
        except Exception as exc:  # A broken proposer is a failed case, never a pass.
            problems.append("execution_error:" + type(exc).__name__)
        rows.append(
            {
                "id": case["id"],
                "family": case["family"],
                "split_group": case["split_group"],
                "source_sha256": case["source_sha256"],
                "output_sha256": observed_output_sha256,
                "expected_state": case["expected_state"],
                "observed_state": state,
                "state_correct": state == case["expected_state"],
                "passed": not problems,
                "problems": problems,
            }
        )
    passed = sum(row["passed"] for row in rows)
    return {
        "schema": RECIPE_VERSION,
        "recipe_sha256": recipe_sha256,
        "methodology": methodology,
        "proposer_identity": "builtin_deterministic" if proposer is None else "caller_supplied_not_independently_verified",
        "status": "PASS" if passed == len(rows) else "FAIL",
        "count": len(rows),
        "passed": passed,
        "state_correct": sum(row["state_correct"] for row in rows),
        "failed_case_ids": [row["id"] for row in rows if not row["passed"]],
        "expected_state_counts": dict(sorted(Counter(case["expected_state"] for case in cases).items())),
        "provenance": PROVENANCE,
        "rows": rows,
        "limitations": [
            "Public synthetic development regression suite; not a held-out generalization benchmark.",
            "No trained weights are created or qualified by this result.",
            "Static restricted HTML label-binding scope, not WCAG certification or assistive-technology testing.",
            "A future model comparison needs pinned artifact identities and independently collected, family-separated held-out cases.",
        ],
    }


def main() -> int:
    result = run_benchmark()
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
