"""Fail-closed contract for the exact-source Hugging Face Harbor evaluation lane.

This module does not invoke Harbor, publish artifacts, access credentials, or authorize
production. It validates evidence produced by a separately sandboxed evaluator.
"""
from __future__ import annotations

import dataclasses
import re
from typing import Iterable

UPSTREAM_REPOSITORY = "huggingface/harbor-hf"
UPSTREAM_REVISION = "ba67b21d625227abf084eba7b605737e4a057575"
FRONTIER_MERGE = "8675f27c4af4fb57a988cf49e57a3204699794f0"
SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class HarborContractError(ValueError):
    """Malformed, unsafe, or incomplete evidence cannot become permission."""


def _sha256(value: str, field: str) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise HarborContractError(f"{field} must be an exact lowercase sha256 digest")
    return value


@dataclasses.dataclass(frozen=True, slots=True)
class RunEvidence:
    input_sha256: str
    output_sha256: str
    preset_sha256: str
    image_sha256: str
    environment_sha256: str
    status: str

    def __post_init__(self) -> None:
        for field in (
            "input_sha256",
            "output_sha256",
            "preset_sha256",
            "image_sha256",
            "environment_sha256",
        ):
            _sha256(getattr(self, field), field)
        if self.status not in {"PASS", "FAIL", "UNAVAILABLE"}:
            raise HarborContractError("status must be PASS, FAIL, or UNAVAILABLE")


@dataclasses.dataclass(frozen=True, slots=True)
class EvaluationEvidence:
    upstream_repository: str
    upstream_revision: str
    public_non_secret_fixture: bool
    runs: tuple[RunEvidence, ...]
    production_route_authorized: bool = False
    provider_write_authorized: bool = False
    secret_access_authorized: bool = False
    deployment_authorized: bool = False
    autonomous_merge_authorized: bool = False
    model_qualification_authorized: bool = False

    def __post_init__(self) -> None:
        if self.upstream_repository != UPSTREAM_REPOSITORY:
            raise HarborContractError("unexpected upstream repository")
        if self.upstream_revision != UPSTREAM_REVISION or not SHA40.fullmatch(self.upstream_revision):
            raise HarborContractError("evaluation must bind the admitted exact upstream revision")
        if self.public_non_secret_fixture is not True:
            raise HarborContractError("first Harbor lane accepts only public non-secret fixtures")
        if len(self.runs) < 2:
            raise HarborContractError("repeatability requires at least two runs")
        denied = (
            self.production_route_authorized,
            self.provider_write_authorized,
            self.secret_access_authorized,
            self.deployment_authorized,
            self.autonomous_merge_authorized,
            self.model_qualification_authorized,
        )
        if any(value is not False for value in denied):
            raise HarborContractError("evaluation evidence cannot grant operational authority")


def evaluate(evidence: EvaluationEvidence) -> dict[str, object]:
    """Summarize bounded evidence without converting benchmark success to production."""
    statuses = [run.status for run in evidence.runs]
    available = all(status != "UNAVAILABLE" for status in statuses)
    passed = available and all(status == "PASS" for status in statuses)
    same_inputs = len({run.input_sha256 for run in evidence.runs}) == 1
    same_outputs = len({run.output_sha256 for run in evidence.runs}) == 1
    same_environment = len({
        (run.preset_sha256, run.image_sha256, run.environment_sha256)
        for run in evidence.runs
    }) == 1
    reproducible = passed and same_inputs and same_outputs and same_environment
    return {
        "upstreamRepository": evidence.upstream_repository,
        "upstreamRevision": evidence.upstream_revision,
        "frontierMerge": FRONTIER_MERGE,
        "runCount": len(evidence.runs),
        "available": available,
        "allRunsPassed": passed,
        "sameInputs": same_inputs,
        "sameOutputs": same_outputs,
        "sameEnvironment": same_environment,
        "reproducible": reproducible,
        "disposition": "EVALUATION" if reproducible else "HOLD",
        "productionAuthorized": False,
        "automaticPromotionAuthorized": False,
    }


def main(argv: Iterable[str] | None = None) -> int:
    del argv
    print(f"Harbor contract pinned to {UPSTREAM_REPOSITORY}@{UPSTREAM_REVISION}")
    print("No benchmark execution or production authority is performed by this module.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
