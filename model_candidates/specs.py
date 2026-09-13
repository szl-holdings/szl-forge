"""Source-only candidate specifications, not a second estate artifact registry.

The published-asset authority remains portfolio/model_portfolio.json and the
existing Forge release gates. Proposed Hub names below are not reserved repos.
Importing this module requires neither Torch nor a connection to the mesh.
"""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any

# Ordered normalized inputs. Missing values must not silently become zero.
FEATURES = (
    "task_complexity", "context_load", "tool_requirement", "quality_prior",
    "latency_pressure", "cost_pressure", "evidence_freshness", "capacity_headroom",
)
RISK_FEATURES = (
    "authorization_coverage", "policy_scope_coverage", "evidence_freshness",
    "provenance_coverage", "sensitive_data_fraction", "external_egress_fraction",
    "tool_privilege_fraction", "contract_schema_coverage",
)
RISK_LABELS = (
    "contract_violation", "unsupported_claim", "privacy_violation", "tool_misuse",
)
SPECS: dict[str, dict[str, Any]] = {
    "router": {
        "proposed_hf_id": "SZLHOLDINGS/A11OY-Router",
        "architecture": "candidate-utility-mlp/v1",
        "purpose": "Rank already-eligible routes; never authorize or dispatch.",
        "config": {"input_dim": 8, "hidden_dim": 32},
        "feature_order": list(FEATURES),
        "targets": ["eligible_winner_index"],
    },
    "invariant-risk": {
        "proposed_hf_id": "SZLHOLDINGS/A11OY-Invariant",
        "architecture": "advisory-risk-mlp/v1",
        "purpose": "Predict outcome risks; deterministic invariants remain outside weights.",
        "config": {"input_dim": 8, "hidden_dim": 32, "label_count": 4},
        "feature_order": list(RISK_FEATURES),
        "targets": list(RISK_LABELS),
    },
    "yarqa-causal": {
        "proposed_hf_id": "SZLHOLDINGS/YARQA-1",
        "architecture": "fixed-canal-causal-reference/v1",
        "purpose": "Byte-level causal research model, not the original CPU YARQA kernel.",
        "config": {
            "vocab_size": 257, "width": 64, "heads": 4, "layers": 2,
            "canal_width": 32, "max_length": 128,
        },
        "feature_order": ["byte_ids_0_to_255", "bos_id_256"],
        "targets": ["next_token_id"],
    },
}


def candidate_spec(key: str) -> dict[str, Any]:
    """Return an isolated recipe; callers cannot mutate the shared definition."""
    if key not in SPECS:
        raise ValueError("unknown candidate")
    return {
        "key": key,
        **copy.deepcopy(SPECS[key]),
        "state": "SOURCE_ONLY_NOT_TRAINED",
        "training_run": None,
        "weights_revision": None,
        "benchmark": None,
        "runtime_eligible": False,
        "publication_eligible": False,
        "authority": "existing Forge portfolio and release gates",
    }


def candidate_specs() -> list[dict[str, Any]]:
    return [candidate_spec(key) for key in SPECS]


def mesh_declaration() -> dict[str, Any]:
    """Public source coordinates only, NEVER an observation of running devices."""
    return {
        "state": "DECLARED_NOT_OBSERVED",
        "observed_at": None,
        "source_revision": "4845411b81f6aa32c9802dae0b851928fc4f36c6",
        "pool": "sovereign-llm",
        "nodes": ["omen", "betterwithage"],
        "connections": {
            "pool": "szl-holdings/a11oy:box-scripts/litellm_config.yaml",
            "containers": "szl-holdings/a11oy:box-scripts/docker-compose.yml",
            "private_transport": "szl-holdings/a11oy:box-scripts/tailscale_acl.json",
            "remote_fallback": "szl-holdings/a11oy:docs/LAPTOP_ONLY_WIRING.md",
            "policy_gateway": "szl-holdings/szl-router:router_control.app",
            "training": "szl-holdings/szl-forge:local-compute/",
            "coordination_not_gpu_pool": "szl-holdings/szl-mesh",
        },
        "constraints": [
            "No network probe or runtime mutation is performed by this workbench.",
            "Inference load balancing does not pool VRAM for distributed training.",
            "OpenRouter is remote inference, not owned training hardware.",
        ],
    }


def encode_features(key: str, rows: list[Mapping[str, float]]) -> list[list[float]]:
    """Encode named features in the admitted order; reject omissions and extras.

    This is a numeric/schema check, NOT provenance or dataset admission. Training
    still needs a frozen normalization recipe, split and observed outcome labels.
    """
    if key not in ("router", "invariant-risk"):
        raise ValueError("candidate does not use named vector features")
    order = candidate_spec(key)["feature_order"]
    limit = 64 if key == "router" else 32
    if not isinstance(rows, list) or not 1 <= len(rows) <= limit:
        raise ValueError("invalid feature row count")
    encoded = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != set(order):
            raise ValueError("missing, extra or incorrect feature names")
        values = []
        for name in order:
            value = row[name]
            if type(value) not in (float, int) or not 0 <= value <= 1:
                raise ValueError("normalized real numbers required; booleans are not scores")
            if not math.isfinite(value):
                raise ValueError("nonfinite feature")
            values.append(float(value))
        encoded.append(values)
    return encoded
