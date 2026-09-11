"""Fail-closed contract for Harbor provider credential-reference evaluation.

This module models reviewed metadata only. It does not accept provider credentials,
call a provider, launch a worker, publish artifacts, or authorize production.
"""
from __future__ import annotations

import dataclasses
import re
from typing import Iterable

UPSTREAM_REPOSITORY = "huggingface/harbor-hf"
UPSTREAM_REVISION = "271303d7a21fa595a5a71e46ce5524b0aff6bb9f"
PRIOR_UPSTREAM_REVISION = "ba67b21d625227abf084eba7b605737e4a057575"
FRONTIER_ISSUE = 72
FORGE_ISSUE = 222
SHA40 = re.compile(r"^[0-9a-f]{40}$")
REFERENCE = re.compile(r"^[A-Z][A-Z0-9_]{2,127}$")
PROTOCOLS = frozenset({"native", "chat-completions", "responses"})


class HarborProviderContractError(ValueError):
    """Malformed or authority-bearing evidence cannot become permission."""


@dataclasses.dataclass(frozen=True, slots=True)
class ProviderReferenceEvidence:
    upstream_repository: str
    upstream_revision: str
    reference_name: str
    contains_secret_value: bool
    attempted_value_persistence_rejected: bool
    review_invalidated_recipe_change: bool
    review_invalidated_model_change: bool
    review_invalidated_connection_change: bool
    review_invalidated_registry_change: bool
    review_invalidated_presence_change: bool
    expected_revision_conflict_denied: bool
    ambiguous_save_followed_by_readback: bool
    blind_retry_used: bool
    legacy_fallback_used: bool
    dispatch_grant_rechecked: bool
    restart_grant_rechecked: bool
    independent_egress_policy_enforced: bool
    controller_count: int
    protocol: str
    reviewed_base_url: bool
    model_base_url_bound: bool
    running_worker_remote_revocation_supported: bool
    provider_call_performed: bool = False
    deployment_authorized: bool = False
    worker_launch_authorized: bool = False
    production_route_authorized: bool = False
    automatic_promotion_authorized: bool = False

    def __post_init__(self) -> None:
        if self.upstream_repository != UPSTREAM_REPOSITORY:
            raise HarborProviderContractError("unexpected upstream repository")
        if self.upstream_revision != UPSTREAM_REVISION or not SHA40.fullmatch(self.upstream_revision):
            raise HarborProviderContractError("evaluation must bind the exact admitted upstream revision")
        if not REFERENCE.fullmatch(self.reference_name):
            raise HarborProviderContractError("reference_name must be a synthetic environment-style name")
        if self.protocol not in PROTOCOLS:
            raise HarborProviderContractError("unsupported protocol")
        if not isinstance(self.controller_count, int) or isinstance(self.controller_count, bool):
            raise HarborProviderContractError("controller_count must be an integer")
        denied = (
            self.provider_call_performed,
            self.deployment_authorized,
            self.worker_launch_authorized,
            self.production_route_authorized,
            self.automatic_promotion_authorized,
        )
        if any(value is not False for value in denied):
            raise HarborProviderContractError("bounded evidence cannot contain operational authority")


def evaluate(evidence: ProviderReferenceEvidence) -> dict[str, object]:
    """Evaluate the bounded safety contract without promoting it to production."""
    invalidation_complete = all(
        (
            evidence.review_invalidated_recipe_change,
            evidence.review_invalidated_model_change,
            evidence.review_invalidated_connection_change,
            evidence.review_invalidated_registry_change,
            evidence.review_invalidated_presence_change,
        )
    )
    protocol_binding_valid = (
        evidence.protocol == "native"
        or (evidence.reviewed_base_url and evidence.model_base_url_bound)
    )
    reference_only = not evidence.contains_secret_value and evidence.attempted_value_persistence_rejected
    conflict_safe = (
        evidence.expected_revision_conflict_denied
        and evidence.ambiguous_save_followed_by_readback
        and not evidence.blind_retry_used
    )
    admission_safe = (
        not evidence.legacy_fallback_used
        and evidence.dispatch_grant_rechecked
        and evidence.restart_grant_rechecked
    )
    single_writer = evidence.controller_count == 1
    egress_safe = evidence.independent_egress_policy_enforced

    bounded_pass = all(
        (
            reference_only,
            invalidation_complete,
            conflict_safe,
            admission_safe,
            single_writer,
            egress_safe,
            protocol_binding_valid,
        )
    )
    blockers: list[str] = []
    if not reference_only:
        blockers.append("reference_only_contract_failed")
    if not invalidation_complete:
        blockers.append("review_invalidation_incomplete")
    if not conflict_safe:
        blockers.append("revision_or_ambiguous_save_control_failed")
    if not admission_safe:
        blockers.append("fallback_or_grant_recheck_failed")
    if not single_writer:
        blockers.append("single_writer_invariant_failed")
    if not egress_safe:
        blockers.append("independent_egress_policy_missing")
    if not protocol_binding_valid:
        blockers.append("protocol_binding_invalid")
    if evidence.running_worker_remote_revocation_supported is not True:
        blockers.append("already_running_worker_remote_revocation_unavailable")

    return {
        "upstreamRepository": evidence.upstream_repository,
        "upstreamRevision": evidence.upstream_revision,
        "priorUpstreamRevision": PRIOR_UPSTREAM_REVISION,
        "frontierIssue": FRONTIER_ISSUE,
        "forgeIssue": FORGE_ISSUE,
        "referenceOnly": reference_only,
        "reviewInvalidationComplete": invalidation_complete,
        "revisionConflictSafe": conflict_safe,
        "noLegacyFallbackAndGrantRecheck": admission_safe,
        "singleWriter": single_writer,
        "independentEgressPolicy": egress_safe,
        "protocolBindingValid": protocol_binding_valid,
        "boundedContractPassed": bounded_pass,
        "blockers": blockers,
        "disposition": "EVALUATION" if bounded_pass else "HOLD",
        "productionAuthorized": False,
        "automaticPromotionAuthorized": False,
    }


def main(argv: Iterable[str] | None = None) -> int:
    del argv
    print(f"Harbor provider-reference contract pinned to {UPSTREAM_REPOSITORY}@{UPSTREAM_REVISION}")
    print("No provider credential, network call, worker launch or deployment is performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
