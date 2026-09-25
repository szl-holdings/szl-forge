from __future__ import annotations

import json
from pathlib import Path


PLAN = Path("frontier/transformers-pp-tp-5474a55e/evaluation-plan.json")
EXPECTED_REVISION = "5474a55e920f358d8382f3ecd3377edca979baa1"


def load_plan() -> dict:
    return json.loads(PLAN.read_text(encoding="utf-8"))


def test_exact_upstream_source_and_hold_state() -> None:
    plan = load_plan()
    assert plan["schema"] == "szl.transformers-pp-tp-evaluation/v1"
    assert plan["state"] == "EVALUATION_HOLD"
    assert plan["upstream"] == {
        "repository": "huggingface/transformers",
        "revision": EXPECTED_REVISION,
        "pull_request": 48155,
        "multimodal_pp_tp_claimed": False,
    }


def test_no_production_authority_is_granted() -> None:
    authority = load_plan()["authority"]
    assert authority == {
        "production_authorization": False,
        "route_default_change": False,
        "automatic_promotion": False,
        "weight_rehosting": False,
    }


def test_topology_matrix_is_explicit_and_fsdp_isolated() -> None:
    topologies = {item["name"]: item for item in load_plan()["topologies"]}
    assert topologies["single"] == {"name": "single", "pp": 1, "tp": 1, "fsdp": 1}
    assert topologies["pp_only"] == {"name": "pp_only", "pp": 2, "tp": 1, "fsdp": 1}
    assert topologies["tp_only"] == {"name": "tp_only", "pp": 1, "tp": 2, "fsdp": 1}
    assert topologies["pp_tp"] == {"name": "pp_tp", "pp": 2, "tp": 2, "fsdp": 1}
    assert all(item["fsdp"] == 1 for item in topologies.values())


def test_required_evidence_precedes_performance_claims() -> None:
    plan = load_plan()
    correctness = set(plan["correctness_gates"])
    negative = set(plan["negative_gates"])
    performance = set(plan["performance_after_correctness_only"])
    assert {
        "world_size_equals_pp_times_tp",
        "named_mesh_dimensions_pp_tp",
        "deterministic_partition_ownership",
        "process_group_global_rank_translation",
        "stage_boundary_shape_dtype",
        "no_layer_omission_or_duplication",
        "generation_termination",
        "cross_stage_error_propagation",
        "task_specific_output_parity",
    } <= correctness
    assert {
        "reject_incompatible_device_map",
        "reject_world_size_mismatch",
        "reject_mps_tensor_parallel",
        "reject_fsdp_combined_with_pp_or_tp",
        "process_loss_fails_closed",
        "timeout_fails_closed",
        "cancellation_fails_closed",
        "malformed_mesh_fails_closed",
    } <= negative
    assert performance == {
        "latency_same_hardware",
        "throughput_same_hardware",
        "peak_memory_same_hardware",
    }


def test_exact_runtime_pins_are_mandatory() -> None:
    assert set(load_plan()["required_pins"]) == {
        "transformers_revision",
        "torch_version",
        "model_repo",
        "model_revision",
        "tokenizer_revision",
        "hardware_topology",
        "launch_command",
    }
