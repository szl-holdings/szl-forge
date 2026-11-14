# SPDX-License-Identifier: Apache-2.0
"""Negative-path readiness tests: no offline projection may grant admission."""
from __future__ import annotations

import copy
import datetime as dt

import pytest

from tools.evidence_readiness_gate import project

NOW = dt.datetime(2026, 11, 14, 1, tzinfo=dt.timezone.utc)


def valid():
    return {
        "schema": "szl.evidence-readiness/v1",
        "source_sha": "a" * 40,
        "artifact_sha256": "b" * 64,
        "dataset_sha256": "c" * 64,
        "grader_sha256": "d" * 64,
        "observed_at": "2026-11-14T00:00:00Z",
        "execution_admitted": False,
        "checks": [{"name": "heldout_eval", "passed": 5, "total": 5, "state": "PASS"}],
    }


def test_pass_never_grants_runtime_authority():
    result = project(valid(), now=NOW)
    assert result["state"] == "PASS"
    assert result["execution_authorized"] is False
    assert result["production_ready"] is False


@pytest.mark.parametrize("field,value", [
    ("source_sha", "main"), ("artifact_sha256", "f" * 40),
    ("dataset_sha256", "x" * 64), ("grader_sha256", "f" * 63),
    ("execution_admitted", True), ("execution_admitted", None),
    ("observed_at", "2026-11-14T00:00:00"),
    ("observed_at", "2026-11-15T00:00:00Z"),
    ("schema", "szl.evidence-readiness/v2"),
])
def test_bad_identity_and_authority_fail_closed(field, value):
    record = valid()
    record[field] = value
    with pytest.raises(ValueError):
        project(record, now=NOW)


@pytest.mark.parametrize("check", [
    {"name": "heldout_eval", "passed": 0, "total": 0, "state": "PASS"},
    {"name": "heldout_eval", "passed": 1, "total": 2, "state": "PASS"},
    {"name": "heldout_eval", "passed": 1, "total": 1, "state": "UNAVAILABLE"},
    {"name": "heldout_eval", "passed": True, "total": 1, "state": "PASS"},
    {"name": "heldout_eval", "passed": -1, "total": 1, "state": "FAIL"},
    {"name": "heldout_eval", "passed": 1, "total": 0, "state": "FAIL"},
    {"name": "heldout_eval", "passed": 1, "total": 1, "state": "PROMOTED"},
    {"name": "heldout_eval", "passed": 1, "total": 1, "state": "PASS", "approval": True},
])
def test_invalid_checks_fail_closed(check):
    record = valid()
    record["checks"] = [check]
    with pytest.raises(ValueError):
        project(record, now=NOW)


def test_missing_denominator_unavailable_not_zero_score():
    record = valid()
    record["checks"][0] = {"name": "heldout_eval", "passed": 0, "total": 0, "state": "UNAVAILABLE"}
    result = project(record, now=NOW)
    assert result["state"] == "UNAVAILABLE"
    assert result["checks"][0]["total"] == 0


def test_stale_pass_downgraded():
    record = valid()
    record["observed_at"] = "2026-11-12T00:00:00Z"
    assert project(record, now=NOW)["state"] == "STALE"


def test_fail_dominates_staleness_and_unavailable():
    record = valid()
    record["observed_at"] = "2026-11-12T00:00:00Z"
    record["checks"] = [
        {"name": "heldout_eval", "passed": 0, "total": 2, "state": "FAIL"},
        {"name": "runtime", "passed": 0, "total": 0, "state": "UNAVAILABLE"},
    ]
    assert project(record, now=NOW)["state"] == "FAIL"


def test_duplicate_check_rejected():
    record = valid()
    record["checks"].append(copy.deepcopy(record["checks"][0]))
    with pytest.raises(ValueError):
        project(record, now=NOW)


def test_unexpected_fields_rejected():
    record = valid()
    record["production_ready"] = True
    with pytest.raises(ValueError):
        project(record, now=NOW)
