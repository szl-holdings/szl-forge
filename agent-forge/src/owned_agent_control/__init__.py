"""Owned Agent Control public package metadata."""

from __future__ import annotations

__version__ = "2.0.0"

from .controller import (
    ContextGenerationError,
    ControlError,
    CrossStepConsistency,
    EnrichedContextGenerator,
    EntropyDepthAllocator,
    apply_isolation,
    build_request,
    create_state_database,
    doctor,
    export_a11oy_context_evidence,
    generate_context,
    initialize_state,
    read_context_trace,
    record_context_trace,
    register_agent,
    register_demo,
    require_stabilized_context,
    self_test,
    sign_request_with_key,
    start_agent,
    state_paths,
    target_status,
    verify_request,
)

__all__ = [
    "__version__",
    "ContextGenerationError",
    "ControlError",
    "CrossStepConsistency",
    "EnrichedContextGenerator",
    "EntropyDepthAllocator",
    "apply_isolation",
    "build_request",
    "create_state_database",
    "doctor",
    "export_a11oy_context_evidence",
    "generate_context",
    "initialize_state",
    "read_context_trace",
    "record_context_trace",
    "register_agent",
    "register_demo",
    "require_stabilized_context",
    "self_test",
    "sign_request_with_key",
    "start_agent",
    "state_paths",
    "target_status",
    "verify_request",
]
