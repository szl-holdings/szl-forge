"""Fail-closed contract for Harbor private Hugging Face Dataset evaluation.

The contract models evidence booleans and immutable identities only. It does not
accept credentials, clone a repository, call Hugging Face, run inference, train,
publish, deploy, or authorize production.
"""
from __future__ import annotations

import dataclasses
import re
from typing import Iterable

UPSTREAM_REPOSITORY = "huggingface/harbor-hf"
UPSTREAM_REVISION = "8685727908e8d9ee5ccde48586864a851e24ff79"
PRIOR_UPSTREAM_REVISION = "271303d7a21fa595a5a71e46ce5524b0aff6bb9f"
FRONTIER_ISSUE = 83
FORGE_ISSUE = 227
SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class HarborPrivateDatasetContractError(ValueError):
    """Malformed or authority-bearing evidence cannot become permission."""


@dataclasses.dataclass(frozen=True, slots=True)
class PrivateDatasetEvidence:
    upstream_repository: str
    upstream_revision: str
    dataset_repo_id: str
    dataset_revision: str
    dataset_content_sha256: str
    synthetic_or_non_sensitive_fixture: bool
    credential_host_restricted_to_huggingface: bool
    credential_in_clone_url: bool
    credential_in_argv: bool
    credential_in_persisted_git_config: bool
    credential_in_output_or_logs: bool
    control_token_visible_to_catalog_discovery: bool
    control_token_visible_to_trial_environment: bool
    inference_or_github_credential_reused: bool
    missing_token_fails_before_inference: bool
    denied_access_fails_before_inference: bool
    missing_git_lfs_fails_before_source_resolution: bool
    lfs_materialization_failure_fails_before_inference: bool
    revision_drift_fails_closed: bool
    read_only_access_proved: bool
    mutation_attempt_denied: bool
    revocation_denies_new_retrieval: bool
    cached_checkout_cleanup_proved: bool
    jobs_scan_bounded: bool
    partial_history_explicit: bool
    inference_performed: bool = False
    training_authorized: bool = False
    publication_authorized: bool = False
    deployment_authorized: bool = False
    production_corpus_authorized: bool = False
    automatic_promotion_authorized: bool = False

    def __post_init__(self) -> None:
        if self.upstream_repository != UPSTREAM_REPOSITORY:
            raise HarborPrivateDatasetContractError("unexpected upstream repository")
        if self.upstream_revision != UPSTREAM_REVISION or not SHA40.fullmatch(self.upstream_revision):
            raise HarborPrivateDatasetContractError("evaluation must bind exact admitted Harbor revision")
        if "/" not in self.dataset_repo_id or not self.dataset_repo_id.strip():
            raise HarborPrivateDatasetContractError("dataset_repo_id must be explicit")
        if not SHA40.fullmatch(self.dataset_revision):
            raise HarborPrivateDatasetContractError("dataset_revision must be exact SHA-40")
        if not SHA256.fullmatch(self.dataset_content_sha256):
            raise HarborPrivateDatasetContractError("dataset_content_sha256 must be exact SHA-256")
        denied = (
            self.inference_performed,
            self.training_authorized,
            self.publication_authorized,
            self.deployment_authorized,
            self.production_corpus_authorized,
            self.automatic_promotion_authorized,
        )
        if any(value is not False for value in denied):
            raise HarborPrivateDatasetContractError("bounded evidence cannot contain operational authority")


def evaluate(evidence: PrivateDatasetEvidence) -> dict[str, object]:
    secret_safe = all(
        (
            evidence.credential_host_restricted_to_huggingface,
            not evidence.credential_in_clone_url,
            not evidence.credential_in_argv,
            not evidence.credential_in_persisted_git_config,
            not evidence.credential_in_output_or_logs,
            not evidence.control_token_visible_to_catalog_discovery,
            not evidence.control_token_visible_to_trial_environment,
            not evidence.inference_or_github_credential_reused,
        )
    )
    fail_closed = all(
        (
            evidence.missing_token_fails_before_inference,
            evidence.denied_access_fails_before_inference,
            evidence.missing_git_lfs_fails_before_source_resolution,
            evidence.lfs_materialization_failure_fails_before_inference,
            evidence.revision_drift_fails_closed,
        )
    )
    read_only = evidence.read_only_access_proved and evidence.mutation_attempt_denied
    revocation_safe = evidence.revocation_denies_new_retrieval and evidence.cached_checkout_cleanup_proved
    observation_safe = evidence.jobs_scan_bounded and evidence.partial_history_explicit
    fixture_safe = evidence.synthetic_or_non_sensitive_fixture

    bounded_pass = all((secret_safe, fail_closed, read_only, revocation_safe, observation_safe, fixture_safe))
    blockers: list[str] = []
    if not fixture_safe:
        blockers.append("fixture_not_bounded_non_sensitive")
    if not secret_safe:
        blockers.append("credential_isolation_or_non_disclosure_failed")
    if not fail_closed:
        blockers.append("source_prerequisite_failure_not_fail_closed")
    if not read_only:
        blockers.append("read_only_boundary_not_proved")
    if not revocation_safe:
        blockers.append("revocation_or_cleanup_not_proved")
    if not observation_safe:
        blockers.append("moving_jobs_observation_not_bounded")

    return {
        "upstreamRepository": evidence.upstream_repository,
        "upstreamRevision": evidence.upstream_revision,
        "priorUpstreamRevision": PRIOR_UPSTREAM_REVISION,
        "frontierIssue": FRONTIER_ISSUE,
        "forgeIssue": FORGE_ISSUE,
        "datasetRepoId": evidence.dataset_repo_id,
        "datasetRevision": evidence.dataset_revision,
        "datasetContentSha256": evidence.dataset_content_sha256,
        "secretIsolationPassed": secret_safe,
        "failClosedPrerequisitesPassed": fail_closed,
        "readOnlyBoundaryPassed": read_only,
        "revocationAndCleanupPassed": revocation_safe,
        "movingJobsObservationBounded": observation_safe,
        "boundedContractPassed": bounded_pass,
        "blockers": blockers,
        "disposition": "EVALUATION" if bounded_pass else "HOLD",
        "productionAuthorized": False,
        "automaticPromotionAuthorized": False,
    }


def main(argv: Iterable[str] | None = None) -> int:
    del argv
    print(f"Harbor private-Dataset contract pinned to {UPSTREAM_REPOSITORY}@{UPSTREAM_REVISION}")
    print("No credential, repository access, inference, training, publication or deployment is performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
