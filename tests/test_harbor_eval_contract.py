from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "harbor_eval_contract", ROOT / "frontier" / "harbor_eval_contract.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

D = "a" * 64
E = "b" * 64
P = "c" * 64
I = "d" * 64
O = "e" * 64


def run(status: str = "PASS", *, output: str = O) -> object:
    return MODULE.RunEvidence(
        input_sha256=D,
        output_sha256=output,
        preset_sha256=P,
        image_sha256=I,
        environment_sha256=E,
        status=status,
    )


def evidence(*runs: object, public: bool = True) -> object:
    return MODULE.EvaluationEvidence(
        upstream_repository=MODULE.UPSTREAM_REPOSITORY,
        upstream_revision=MODULE.UPSTREAM_REVISION,
        public_non_secret_fixture=public,
        runs=tuple(runs),
    )


def test_exact_upstream_source_is_fixed() -> None:
    assert MODULE.UPSTREAM_REPOSITORY == "huggingface/harbor-hf"
    assert MODULE.UPSTREAM_REVISION == "ba67b21d625227abf084eba7b605737e4a057575"
    assert MODULE.FRONTIER_MERGE == "8675f27c4af4fb57a988cf49e57a3204699794f0"


def test_repeatable_pass_remains_evaluation_only() -> None:
    result = MODULE.evaluate(evidence(run(), run()))
    assert result["reproducible"] is True
    assert result["disposition"] == "EVALUATION"
    assert result["productionAuthorized"] is False
    assert result["automaticPromotionAuthorized"] is False


def test_one_run_cannot_establish_repeatability() -> None:
    try:
        evidence(run())
    except MODULE.HarborContractError:
        pass
    else:
        raise AssertionError("single-run evidence was accepted")


def test_unavailable_is_honest_hold_not_fake_pass() -> None:
    result = MODULE.evaluate(evidence(run("UNAVAILABLE"), run("UNAVAILABLE")))
    assert result["available"] is False
    assert result["reproducible"] is False
    assert result["disposition"] == "HOLD"


def test_output_drift_is_not_reproducible() -> None:
    result = MODULE.evaluate(evidence(run(), run(output="f" * 64)))
    assert result["allRunsPassed"] is True
    assert result["sameOutputs"] is False
    assert result["reproducible"] is False
    assert result["disposition"] == "HOLD"


def test_private_or_secret_fixture_is_rejected() -> None:
    try:
        evidence(run(), run(), public=False)
    except MODULE.HarborContractError:
        pass
    else:
        raise AssertionError("private/non-public fixture was accepted")


def test_wrong_upstream_revision_is_rejected() -> None:
    try:
        MODULE.EvaluationEvidence(
            upstream_repository=MODULE.UPSTREAM_REPOSITORY,
            upstream_revision="0" * 40,
            public_non_secret_fixture=True,
            runs=(run(), run()),
        )
    except MODULE.HarborContractError:
        pass
    else:
        raise AssertionError("wrong upstream revision was accepted")


def test_operational_authority_is_rejected() -> None:
    try:
        MODULE.EvaluationEvidence(
            upstream_repository=MODULE.UPSTREAM_REPOSITORY,
            upstream_revision=MODULE.UPSTREAM_REVISION,
            public_non_secret_fixture=True,
            runs=(run(), run()),
            deployment_authorized=True,
        )
    except MODULE.HarborContractError:
        pass
    else:
        raise AssertionError("deployment authority leaked into evaluation evidence")
