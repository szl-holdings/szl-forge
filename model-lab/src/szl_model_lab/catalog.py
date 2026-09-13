"""Proposed model tracks, not another authoritative estate/model registry.

The existing Forge publishing/model-source-bindings.json remains authoritative.
No catalog entry here asserts that a Hugging Face repository or weights exist.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass

@dataclass(frozen=True)
class Track:
    slug: str
    display_name: str
    target: str
    features: tuple[str, ...]
    implementation: str
    proposed_hf_id: str | None
    boundary: str

TRACKS = {
    "router": Track(
        "router", "A11OY Router · outcome scorer", "task_succeeded",
        ("input_load", "output_budget", "queue_utilization", "memory_utilization",
         "prior_quality_rate", "latency_pressure", "unit_cost", "domain_match"),
        "CPU tabular MLP research baseline", "SZLHOLDINGS/A11OY-Router",
        "Scores only. szl-router and its deterministic admission checks retain authority.",
    ),
    "invariant": Track(
        "invariant", "A11OY Invariant · violation predictor", "violation_observed",
        ("action_scope", "evidence_gap", "policy_distance", "privilege_level",
         "data_sensitivity", "dependency_churn", "verification_age", "rollback_difficulty"),
        "CPU tabular MLP research baseline", "SZLHOLDINGS/A11OY-Invariant",
        "Predicts a labeled violation, not a proof. Never overrides szl-invariants.",
    ),
}

DEFERRED_TRACKS = (
    {"slug": "yarqa", "display_name": "YARQA encoder research", "state": "DESIGN_ONLY",
     "reason": "Canonical v0 attention is CPU-only and non-causal. Encoder integration, gradients, padding and partition tests are not implemented in this addition.",
     "proposed_hf_id": "SZLHOLDINGS/YARQA-1"},
    {"slug": "kernel-family", "display_name": "SZL kernel-derived model family", "state": "CATALOG_ONLY",
     "reason": "szl-kernels is an executable suite, not one trainable architecture. Preserve existing MiniEmbed and A11OY-MINI identities.",
     "proposed_hf_id": None},
)

def catalog() -> list[dict]:
    return [dict(asdict(t), state="BLUEPRINT_NOT_TRAINED", publication_eligible=False)
            for t in TRACKS.values()] + [dict(t, publication_eligible=False) for t in DEFERRED_TRACKS]

def track_for(slug: str) -> Track:
    try:
        return TRACKS[slug]
    except KeyError as exc:
        raise ValueError("unsupported_model_track") from exc
