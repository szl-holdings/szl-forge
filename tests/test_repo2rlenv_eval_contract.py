from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "repo2rlenv_eval_contract", ROOT / "frontier" / "repo2rlenv_eval_contract.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

A = "a" * 64
B = "b" * 64
C = "c" * 64
D = "d" * 64
SOURCE_SHA = "e" * 40


def receipt(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "upstream_repository": MODULE.UPSTREAM_REPOSITORY,
        "upstream_revision": MODULE.UPSTREAM_REVISION,
        "source_repository": "pallets/click",
        "source_revision": SOURCE_SHA,
        "source_event_id": "pr:123",
        "source_visibility": "PUBLIC_NON_SECRET",
        "pipeline": "pr_runtime",
        "bootstrap_sha256": A,
        "task_sha256": B,
        "verifier_sha256": C,
        "reward_sha256": D,
        "provider_called": False,
        "hub_write_performed": False,
        "training_admitted": False,
        "oracle_leak_detected": False,
        "status": "PASS",
    }
    row.update(overrides)
    return row


def rejected(**overrides: object) -> None:
    try:
        MODULE.validate_task_receipt(receipt(**overrides))
    except MODULE.SynthesisContractError:
        return
    raise AssertionError("unsafe synthesis receipt was accepted")


def test_exact_functional_release_is_fixed() -> None:
    assert MODULE.UPSTREAM_REPOSITORY == "huggingface/Repo2RLEnv"
    assert MODULE.UPSTREAM_RELEASE == "v0.8.7"
    assert MODULE.UPSTREAM_REVISION == "d1f7677265b10ae5deec1b9cc43425d715564d47"
    assert MODULE.FRONTIER_MERGE == "2094061aa783c8c7aa89adc7b530a41de3469208"


def test_only_stable_runtime_pipelines_are_accepted() -> None:
    assert MODULE.validate_task_receipt(receipt(pipeline="pr_runtime"))["pipeline"] == "pr_runtime"
    assert MODULE.validate_task_receipt(receipt(pipeline="commit_runtime"))["pipeline"] == "commit_runtime"
    rejected(pipeline="equivalence_tests")
    rejected(pipeline="cve_patches")


def test_provider_hub_training_and_private_source_are_denied() -> None:
    rejected(provider_called=True)
    rejected(hub_write_performed=True)
    rejected(training_admitted=True)
    rejected(source_visibility="PRIVATE")


def test_oracle_leakage_fails_closed() -> None:
    rejected(oracle_leak_detected=True)


def test_wrong_upstream_or_moving_source_is_rejected() -> None:
    rejected(upstream_revision="0" * 40)
    rejected(source_revision="main")


def test_unavailable_remains_hold() -> None:
    result = MODULE.evaluate_repeatability([
        receipt(status="UNAVAILABLE"),
        receipt(status="UNAVAILABLE"),
    ])
    assert result["available"] is False
    assert result["reproducible"] is False
    assert result["disposition"] == "HOLD"


def test_stable_repeatability_remains_evaluation_only() -> None:
    result = MODULE.evaluate_repeatability([receipt(), receipt()])
    assert result["reproducible"] is True
    assert result["disposition"] == "EVALUATION"
    assert result["productionAuthorized"] is False
    assert result["publicationAuthorized"] is False
    assert result["trainingAuthorized"] is False
    assert result["automaticPromotionAuthorized"] is False


def test_task_or_reward_drift_fails_repeatability() -> None:
    task_drift = MODULE.evaluate_repeatability([receipt(), receipt(task_sha256="f" * 64)])
    assert task_drift["stableTask"] is False
    assert task_drift["disposition"] == "HOLD"
    reward_drift = MODULE.evaluate_repeatability([receipt(), receipt(reward_sha256="f" * 64)])
    assert reward_drift["stableReward"] is False
    assert reward_drift["disposition"] == "HOLD"


def test_single_receipt_cannot_establish_repeatability() -> None:
    try:
        MODULE.evaluate_repeatability([receipt()])
    except MODULE.SynthesisContractError:
        pass
    else:
        raise AssertionError("single receipt established repeatability")
