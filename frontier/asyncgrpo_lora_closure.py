"""Recorded PEFT / vLLM closure taken from the exact TRL pin.

These are the extras declared by huggingface/trl at
f540773f5250c816e992ae3d35a41142ef3625c0 (pyproject.toml
[project.optional-dependencies] peft / vllm). They are constraints, not
invented exact versions. A job-local install must use these specs and must
not pick a newer-than-allowed vLLM or an unrecorded pin.
"""
from __future__ import annotations

import re

from frontier.asyncgrpo_lora_contract import (
    EvaluationContractError,
    TRL_BASELINE_VERSION,
    TRL_REPOSITORY,
    TRL_REVISION,
)

PEFT_SPEC = "peft>=0.13.0"
VLLM_SPEC = "vllm>=0.19.1,<=0.28.0"
VLLM_EXTRA_SPECS = ("aiohttp>=3.13.3", "requests")
TRL_GIT_REQUIREMENT = (
    f"trl @ git+https://github.com/{TRL_REPOSITORY}.git@{TRL_REVISION}"
)
CLOSURE_SOURCE_PATH = "pyproject.toml"
CLOSURE_SOURCE_SECTION = "project.optional-dependencies"
STABLE_BASELINE_REVISION = "3d9261f1fec9f9a8140099c78a65c7da73dce79c"
ADAPTER_ONLY_UPSTREAM_COMMIT = (
    f"https://github.com/{TRL_REPOSITORY}/commit/{TRL_REVISION}"
)
ADAPTER_ONLY_UPSTREAM_ISSUE = "https://github.com/huggingface/trl/issues/5975"
ADAPTER_ONLY_UPSTREAM_PR = "https://github.com/huggingface/trl/pull/7017"

# Source-inspected at the pinned revision: AsyncGRPOTrainer._sync_weight_lora,
# save_lora_adapter, load-before-evict, max_staleness+2. This is not a GPU run.
ADAPTER_ONLY_API_VERIFIED_AT_PIN = True


def recorded_requirements_text() -> str:
    extras = "\n".join(VLLM_EXTRA_SPECS)
    return (
        f"# Job-local only. Never install into the production interpreter.\n"
        f"# Recorded from {TRL_REPOSITORY}@{TRL_REVISION} {CLOSURE_SOURCE_PATH}\n"
        f"# [{CLOSURE_SOURCE_SECTION}] peft / vllm. Constraints, not invented pins.\n"
        f"{TRL_GIT_REQUIREMENT}\n"
        f"{PEFT_SPEC}\n"
        f"{VLLM_SPEC}\n"
        f"{extras}\n"
    )


def recorded_closure_identity() -> dict[str, object]:
    return {
        "repository": TRL_REPOSITORY,
        "revision": TRL_REVISION,
        "stableBaselineVersion": TRL_BASELINE_VERSION,
        "stableBaselineRevision": STABLE_BASELINE_REVISION,
        "sourcePath": CLOSURE_SOURCE_PATH,
        "sourceSection": CLOSURE_SOURCE_SECTION,
        "peftSpec": PEFT_SPEC,
        "vllmSpec": VLLM_SPEC,
        "vllmExtraSpecs": list(VLLM_EXTRA_SPECS),
        "trlRequirement": TRL_GIT_REQUIREMENT,
        "inventedExactVersions": False,
        "adapterOnlyApiVerifiedAtPin": ADAPTER_ONLY_API_VERIFIED_AT_PIN,
        "adapterOnlyUpstreamCommit": ADAPTER_ONLY_UPSTREAM_COMMIT,
        "adapterOnlyUpstreamIssue": ADAPTER_ONLY_UPSTREAM_ISSUE,
        "adapterOnlyUpstreamPr": ADAPTER_ONLY_UPSTREAM_PR,
    }


def _numeric_version(value: object) -> tuple[int, int, int]:
    if type(value) is not str or not value:
        raise EvaluationContractError("version must be a nonempty string")
    match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", value)
    if match is None:
        raise EvaluationContractError("version is not a dotted numeric triple")
    major = int(match.group(1))
    minor = int(match.group(2))
    patch = int(match.group(3) or "0")
    return (major, minor, patch)


def spec_satisfied(version: object, spec: str) -> bool:
    """Check a recorded PEP 508 numeric floor/cap. Rejects bools and floats."""
    if type(spec) is not str or not spec:
        raise EvaluationContractError("spec must be a nonempty string")
    try:
        observed = _numeric_version(version)
    except EvaluationContractError:
        return False
    remainder = spec.strip()
    named = re.match(r"^[A-Za-z0-9_.-]+(?=[<>=])", remainder)
    if named is not None:
        remainder = remainder[named.end() :]
    if not remainder:
        raise EvaluationContractError("unsupported version specifier")
    for clause in remainder.split(","):
        token = clause.strip()
        if token.startswith(">="):
            if observed < _numeric_version(token[2:]):
                return False
        elif token.startswith("<="):
            if observed > _numeric_version(token[2:]):
                return False
        elif token.startswith("=="):
            if observed != _numeric_version(token[2:]):
                return False
        else:
            raise EvaluationContractError("unsupported version specifier")
    return True
