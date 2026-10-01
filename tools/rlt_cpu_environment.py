#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Bind the RLT CPU lane to its project declaration, not a second version pin.

This checks local package identity only. It is not wheel-byte provenance, model
quality, hardware qualification, or production/publication authorization.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import re
import sys
import tomllib
from typing import Any, Sequence

PROJECT = Path(__file__).resolve().parents[1] / "frontier/rlt/pyproject.toml"
MAX_PROJECT_BYTES = 65536
EXACT_TORCH = re.compile(r"torch==([0-9]+\.[0-9]+\.[0-9]+)(?:\+cpu)?", re.ASCII)


class ContractError(ValueError):
    """A fixed diagnostic code; never reflect arbitrary source or provider text."""


def declared_cpu_requirement(raw: bytes) -> str:
    if type(raw) is not bytes or not 0 < len(raw) <= MAX_PROJECT_BYTES:
        raise ContractError("PROJECT_BYTES_INVALID")
    try:
        dependencies = tomllib.loads(raw.decode("utf-8"))["project"]["dependencies"]
    except (UnicodeError, tomllib.TOMLDecodeError, KeyError, TypeError) as exc:
        raise ContractError("PROJECT_SCHEMA_INVALID") from exc
    if type(dependencies) is not list or any(type(item) is not str for item in dependencies):
        raise ContractError("DEPENDENCIES_INVALID")
    candidates = [item for item in dependencies if re.match(r"(?i)^torch(?:\b|\[)", item.strip())]
    if len(candidates) != 1:
        raise ContractError("ONE_TORCH_DECLARATION_REQUIRED")
    match = EXACT_TORCH.fullmatch(candidates[0])
    if match is None:
        raise ContractError("EXACT_STABLE_TORCH_PIN_REQUIRED")
    return f"torch=={match.group(1)}+cpu"


def validate_runtime(requirement: str, distribution: Any, module_version: Any,
                     cuda: Any, hip: Any) -> None:
    if type(requirement) is not str or EXACT_TORCH.fullmatch(requirement) is None or not requirement.endswith("+cpu"):
        raise ContractError("CPU_REQUIREMENT_INVALID")
    expected = requirement.removeprefix("torch==")
    if type(distribution) is not str or distribution != expected:
        raise ContractError("INSTALLED_DISTRIBUTION_MISMATCH")
    if not isinstance(module_version, str) or module_version != expected:
        raise ContractError("IMPORTED_MODULE_MISMATCH")
    # cuda.is_available() is insufficient: a CUDA wheel on a CPU runner returns
    # false too. Check build metadata for both supported GPU backend families.
    if cuda is not None or hip is not None:
        raise ContractError("NON_CPU_BUILD")


def read_project() -> bytes:
    if PROJECT.is_symlink() or not PROJECT.is_file():
        raise ContractError("PROJECT_NOT_REGULAR")
    with PROJECT.open("rb") as handle:
        raw = handle.read(MAX_PROJECT_BYTES + 1)
    declared_cpu_requirement(raw)
    return raw


def run(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("requirement", "verify"))
    args = parser.parse_args(argv)
    try:
        raw = read_project()
        requirement = declared_cpu_requirement(raw)
        if args.action == "requirement":
            print(requirement)
            return 0
        distribution = importlib.metadata.version("torch")
        if distribution != requirement.removeprefix("torch=="):
            raise ContractError("INSTALLED_DISTRIBUTION_MISMATCH")
        torch = importlib.import_module("torch")
        validate_runtime(requirement, distribution, torch.__version__,
                         torch.version.cuda, torch.version.hip)
        if read_project() != raw:
            raise ContractError("PROJECT_CHANGED_DURING_VERIFICATION")
    except Exception as exc:  # Fail closed, without reflecting import-error text.
        code = str(exc) if isinstance(exc, ContractError) else "ENVIRONMENT_UNAVAILABLE"
        print(json.dumps({"state": "FAIL_CPU_ENVIRONMENT", "reason": code,
                          "production_authorized": False}), file=sys.stderr)
        return 2
    print(json.dumps({"schema": "szl.rlt.cpu-environment/v1",
                      "state": "PASS_CPU_PACKAGE_IDENTITY_ONLY",
                      "requirement": requirement,
                      "distribution_version": distribution,
                      "module_version": str(torch.__version__),
                      "cuda_build": None, "hip_build": None,
                      "project_sha256": hashlib.sha256(raw).hexdigest(),
                      "wheel_bytes_verified": False,
                      "production_authorized": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
