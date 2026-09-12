"""Run exact-source PEFT on deterministic CPU fixtures, never on user models.

Copyright 2026 SZL Holdings. SPDX-License-Identifier: Apache-2.0

The command has no path/URL/model arguments and no publisher. Temporary exports
are deleted on exit. CPU malformed-shape fixtures are NOT distributed gathering.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import copy
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import re
import socket
import subprocess
import tempfile
from typing import Any
from unittest.mock import patch
import warnings

PEFT_REVISION = "78bce7cb48f800a7ad0d352b68a46302e13e1687"
SOURCES = {
    "peft": ("0.20.1.dev0", "huggingface/peft", PEFT_REVISION),
    "transformers": ("5.17.0", "huggingface/transformers", "856157a2f3e9594954310df18fdccc31ffddebe9"),
    "accelerate": ("1.15.0", "huggingface/accelerate", "6afc1e5ee217051fde702b23de2813344dc0fd33"),
    "huggingface-hub": ("1.31.0", "huggingface/huggingface_hub", "495b17c8529614759ae0f1ccf1ebe9a61c148b7c"),
}
VERSIONS = {"torch": "2.10.0+cpu", "safetensors": "0.7.0", "numpy": "2.3.5"}
SAVE_AND_LOAD_BLOB = "f576ffdfd33dd8a81b9be50c97f8a7b399477d69"
BASE_ID = "szl-fixture/linear-lora"
SHAPES = {"base_model.model.linear.lora_A.weight": [2, 3],
          "base_model.model.linear.lora_B.weight": [4, 2]}
ROOT = Path(__file__).resolve().parents[1]


class EvaluationError(ValueError):
    """Stable reason code; do not expose provider or environment details."""


def require(value: bool, code: str) -> None:
    if not value:
        raise EvaluationError(code)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def requirements() -> str:
    """Installation preparation only. No package is installed by this module."""
    return "\n".join([
        "numpy==2.3.5", "safetensors==0.7.0",
        *(f"{name} @ git+https://github.com/{repo}.git@{revision}"
          for name, (_, repo, revision) in SOURCES.items()),
    ]) + "\n"


def validate_install(name: str, version: str, document: Any) -> dict[str, str]:
    require(name in SOURCES, "UNEXPECTED_PACKAGE")
    expected_version, repository, revision = SOURCES[name]
    require(version == expected_version, "PACKAGE_VERSION_MISMATCH")
    require(type(document) is dict, "INSTALL_PROVENANCE_MISSING")
    vcs = document.get("vcs_info")
    require(type(vcs) is dict and vcs.get("vcs") == "git", "INSTALL_NOT_GIT_BOUND")
    require(document.get("url") == f"https://github.com/{repository}.git" and
            vcs.get("commit_id") == revision, "INSTALL_SOURCE_MISMATCH")
    return {"version": version, "repository": repository, "revision": revision}


def check_installations() -> dict[str, Any]:
    result: dict[str, Any] = {}
    try:
        for name in SOURCES:
            dist = metadata.distribution(name)
            text = dist.read_text("direct_url.json")
            require(text is not None and len(text) <= 16384, "INSTALL_PROVENANCE_MISSING")
            result[name] = validate_install(name, dist.version, json.loads(text))
        for name, version in VERSIONS.items():
            require(metadata.version(name) == version, "DEPENDENCY_VERSION_MISMATCH")
            result[name] = {"version": version}
    except metadata.PackageNotFoundError as exc:
        raise EvaluationError("DEPENDENCY_UNAVAILABLE") from exc
    except json.JSONDecodeError as exc:
        raise EvaluationError("INSTALL_PROVENANCE_INVALID_JSON") from exc
    return result


def source_revision() -> str:
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True, timeout=10).strip()
    require(re.fullmatch(r"[0-9a-f]{40}", revision) is not None, "INVALID_FORGE_SOURCE")
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                   cwd=ROOT, text=True, timeout=10)
    require(not dirty.strip(), "DIRTY_FORGE_SOURCE")
    expected = os.environ.get("SZL_EVAL_SOURCE_SHA")
    require(expected is None or expected == revision, "FORGE_SOURCE_MOVED")
    return revision


@contextmanager
def deny_network():
    """Catch Python socket egress. This is NOT an operating-system sandbox."""
    def denied(*_args: Any, **_kwargs: Any) -> None:
        raise EvaluationError("UNEXPECTED_NETWORK_ATTEMPT")
    with ExitStack() as stack:
        for owner, key in ((socket.socket, "connect"), (socket.socket, "connect_ex"),
                           (socket, "create_connection"), (socket, "getaddrinfo")):
            stack.enter_context(patch.object(owner, key, denied))
        yield


def bounded_read(path: Path, maximum: int) -> bytes:
    """Only called for fixed names inside this command's private temporary root."""
    require(not path.is_symlink() and path.is_file(), "EXPORT_FILE_MISSING_OR_SYMLINK")
    with path.open("rb") as stream:
        data = stream.read(maximum + 1)
    require(0 < len(data) <= maximum, "EXPORT_SIZE_LIMIT")
    return data


def run_cpu(revision: str) -> dict[str, Any]:
    require(re.fullmatch(r"[0-9a-f]{40}", revision) is not None, "INVALID_FORGE_SOURCE")
    dependencies = check_installations()
    # Imports follow provenance validation; never download a pretrained model.
    import torch
    import peft
    from peft import LoraConfig, PeftModel, PeftWarning, get_peft_model
    from peft.utils.save_and_load import _validate_lora_adapter_state_dict
    from safetensors.torch import load, save
    from tools.evaluate_peft_export import inspect_adapter, MAX_BYTES, MAX_HEADER

    upstream_bytes = Path(peft.utils.save_and_load.__file__).read_bytes()
    blob = hashlib.sha1(b"blob " + str(len(upstream_bytes)).encode() + b"\0" + upstream_bytes).hexdigest()
    require(blob == SAVE_AND_LOAD_BLOB, "UPSTREAM_CRITICAL_FILE_MISMATCH")
    torch.set_num_threads(1)
    torch.manual_seed(1742)
    torch.use_deterministic_algorithms(True)

    class TinyLinear(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(3, 4, bias=False)
            with torch.no_grad():
                self.linear.weight.copy_(torch.arange(12, dtype=torch.float32).reshape(4, 3) / 16)
            self.name_or_path = BASE_ID

        def forward(self, x):
            return self.linear(x)

    base_bytes = save(TinyLinear().state_dict())
    sample = torch.tensor([[0.25, -0.5, 1.0], [1.0, 0.0, -0.25]])

    def make_model(*, dora: bool = False):
        config = LoraConfig(target_modules=["linear"], r=2, lora_alpha=4,
                            lora_dropout=0.0, bias="none", use_dora=dora, revision=revision)
        model = get_peft_model(TinyLinear(), config)
        model.peft_config["default"].base_model_name_or_path = BASE_ID
        with torch.no_grad():
            model.base_model.model.linear.lora_A["default"].weight.fill_(0.25)
            model.base_model.model.linear.lora_B["default"].weight.fill_(0.5)
        return model.eval()

    def shape_warnings(caught) -> list[Any]:
        return [w for w in caught if issubclass(w.category, PeftWarning) and
                "DeepSpeed ZeRO-3 / FSDP shards" in str(w.message)]

    def export(model, directory: Path, *, explicit: bool = False):
        kwargs = {"state_dict": model.state_dict()} if explicit else {}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.save_pretrained(str(directory), safe_serialization=True,
                                  save_embedding_layers=False, **kwargs)
        data = bounded_read(directory / "adapter_model.safetensors", MAX_BYTES)
        config = bounded_read(directory / "adapter_config.json", MAX_HEADER)
        return data, config, shape_warnings(caught)

    def inspect(data: bytes, config: bytes, provenance: str = "INFERRED"):
        return inspect_adapter(data, config, {
            "artifactSha256": sha256(data), "configSha256": sha256(config),
            "baseModelId": BASE_ID, "baseModelRevision": revision,
            "peftRevision": PEFT_REVISION, "expectedTensorShapes": copy.deepcopy(SHAPES),
            "stateDictProvenance": provenance,
        })

    with deny_network(), tempfile.TemporaryDirectory(prefix="szl-peft-cpu-") as temporary:
        root = Path(temporary)
        model = make_model()
        with torch.no_grad():
            before = model(sample).clone()
        good, config, notices = export(model, root / "valid")
        require(not notices, "VALID_EXPORT_SHAPE_WARNING")
        structure = inspect(good, config)
        require(structure["state"] == "PASS", "VALID_EXPORT_STRUCTURE_REJECTED")
        base = TinyLinear()
        base.load_state_dict(load(base_bytes), strict=True)
        require(save(base.state_dict()) == base_bytes, "BASE_BYTES_CHANGED")
        reloaded = PeftModel.from_pretrained(base, str(root / "valid"),
                                             local_files_only=True, is_trainable=False).eval()
        with torch.no_grad():
            after = reloaded(sample)
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        require(torch.isfinite(after).all().item(), "RELOAD_NONFINITE")
        require(bounded_read(root / "valid" / "adapter_model.safetensors", MAX_BYTES) == good,
                "RELOAD_MUTATED_ADAPTER_BYTES")
        # A deliberately different base must not be mistaken for the same model.
        wrong_base = TinyLinear()
        with torch.no_grad():
            wrong_base.linear.weight.add_(1)
        require(sha256(save(wrong_base.state_dict())) != sha256(base_bytes), "WRONG_BASE_NOT_DISTINCT")
        wrong = PeftModel.from_pretrained(wrong_base, str(root / "valid"),
                                         local_files_only=True, is_trainable=False).eval()
        with torch.no_grad():
            require(not torch.equal(wrong(sample), before), "WRONG_BASE_NEGATIVE_INEFFECTIVE")

        cases = []
        for side in ("lora_A", "lora_B"):
            for case, shape in (("flat", (8,)), ("empty", (0, 2)), ("scalar", ())):
                for explicit in (False, True):
                    broken = make_model()
                    layer = getattr(broken.base_model.model.linear, side)["default"]
                    layer.weight.data = torch.zeros(shape, dtype=torch.float32)
                    data, cfg, notices = export(broken, root / f"{side}-{case}-{explicit}", explicit=explicit)
                    require(bool(notices) is (not explicit), "UPSTREAM_WARNING_BOUNDARY_CHANGED")
                    observed = load(data)[f"base_model.model.linear.{side}.weight"]
                    require(tuple(observed.shape) == shape, "MALFORMED_EXPORT_NOT_PRESERVED")
                    verdict = inspect(data, cfg, "EXPLICIT" if explicit else "INFERRED")
                    require(verdict["state"] == "REJECTED" and verdict["reason"] == "UNGATHERED_AB_SHAPE",
                            "MALFORMED_EXPORT_ACCEPTED")
                    require(verdict["publicationAuthorized"] is False, "PUBLICATION_AUTHORITY_ESCALATED")
                    cases.append({"side": side, "shapeCase": case, "explicitStateDict": explicit,
                                  "warningObserved": bool(notices), "writeCompleted": True,
                                  "inspectorReason": verdict["reason"], "sha256": sha256(data)})

        # Method-specific vectors must not trigger the upstream A/B shape heuristic.
        for key in ("model.linear.lora_magnitude_vector", "model.linear.lora_E"):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                _validate_lora_adapter_state_dict({key: torch.ones(4)})
            require(not shape_warnings(caught), "AUXILIARY_VECTOR_FALSE_POSITIVE")
        dora_data, dora_config, dora_notices = export(make_model(dora=True), root / "dora")
        require(not dora_notices, "DORA_FALSE_POSITIVE")
        dora_shapes = {**copy.deepcopy(SHAPES), "base_model.model.linear.lora_magnitude_vector": [4]}
        dora_binding = {"artifactSha256": sha256(dora_data), "configSha256": sha256(dora_config),
                        "baseModelId": BASE_ID, "baseModelRevision": revision,
                        "peftRevision": PEFT_REVISION, "expectedTensorShapes": dora_shapes,
                        "stateDictProvenance": "INFERRED"}
        dora_verdict = inspect_adapter(dora_data, dora_config, dora_binding)
        require(dora_verdict["state"] == "UNSUPPORTED", "DORA_SCOPE_NOT_HELD")

    return {
        "schema": "szl.forge.peft-cpu-execution.v1", "state": "CPU_EVALUATION_PASS",
        "sourceRevision": revision, "upstreamRevision": PEFT_REVISION,
        "evidenceClass": "UNSIGNED_EXECUTION_OBSERVATION_REQUIRES_OUTER_CI_BINDING",
        "installedSourceAssurance": "PEP610_LOCAL_METADATA_NOT_INDEPENDENT_ATTESTATION",
        "dependencies": dependencies,
        "resolvedEnvironment": {d.metadata["Name"]: d.version for d in metadata.distributions()
                                if d.metadata.get("Name")},
        "python": platform.python_version(), "platform": platform.system(), "device": "cpu",
        "baseIdentity": {"kind": "SYNTHETIC_RECIPE_NOT_HUB_MODEL", "recipeGitRevision": revision,
                         "safetensorsSha256": sha256(base_bytes), "parameterCount": 12},
        "validExport": {"artifactSha256": sha256(good), "configSha256": sha256(config),
                        "structure": structure["state"], "peftReload": "PASS",
                        "maxAbsoluteError": float((before - after).abs().max()),
                        "wrongBaseNegative": "PASS", "adapterBytesUnchanged": True},
        "malformedExports": cases, "auxiliaryVectorHeuristic": "PASS", "doraScope": dora_verdict["state"],
        "loadedPeftSaveAndLoadSha256": sha256(upstream_bytes),
        "loadedPeftSaveAndLoadGitBlob": blob,
        "remaining": ["REAL_ZERO3_FSDP_GATHERING", "AUTHENTICATED_EXTERNAL_MODEL_LINEAGE",
                      "IMMUTABLE_PROVIDER_READBACK", "DISTRIBUTED_CANCELLATION_FAILURE_ROLLBACK",
                      "MODEL_QUALITY_QUALIFICATION", "INDEPENDENT_PUBLISHER_ADMISSION"],
        "productionDisposition": "HOLD", "productionAuthorized": False,
        "publicationAuthorized": False, "automaticPromotion": False, "providerWrites": 0,
        "trainingExecuted": False, "pretrainedWeightsDownloaded": False,
    }


def validate_observation(report: dict[str, Any], revision: str) -> None:
    """Independently reject partial executions and authority-bearing reports."""
    require(report.get("state") == "CPU_EVALUATION_PASS", "INCOMPLETE_CPU_EVALUATION")
    require(report.get("sourceRevision") == revision and report.get("upstreamRevision") == PEFT_REVISION,
            "OBSERVATION_SOURCE_MISMATCH")
    require(report.get("loadedPeftSaveAndLoadGitBlob") == SAVE_AND_LOAD_BLOB,
            "UPSTREAM_CRITICAL_FILE_MISMATCH")
    for flag in ("productionAuthorized", "publicationAuthorized", "automaticPromotion",
                 "trainingExecuted", "pretrainedWeightsDownloaded"):
        require(report.get(flag) is False, "OBSERVATION_AUTHORITY_ESCALATION")
    require(type(report.get("providerWrites")) is int and report["providerWrites"] == 0 and
            report.get("productionDisposition") == "HOLD", "OBSERVATION_AUTHORITY_ESCALATION")
    good = report.get("validExport", {})
    require(good.get("structure") == good.get("peftReload") == good.get("wrongBaseNegative") == "PASS" and
            type(good.get("maxAbsoluteError")) is float and good["maxAbsoluteError"] == 0.0 and
            good.get("adapterBytesUnchanged") is True, "RELOAD_EVIDENCE_INCOMPLETE")
    cases = report.get("malformedExports")
    require(type(cases) is list and len(cases) == 12, "NEGATIVE_MATRIX_INCOMPLETE")
    expected = {(side, case, explicit) for side in ("lora_A", "lora_B")
                for case in ("flat", "empty", "scalar") for explicit in (False, True)}
    observed = set()
    for case in cases:
        require(type(case) is dict and type(case.get("explicitStateDict")) is bool,
                "NEGATIVE_CASE_INVALID")
        identity = (case.get("side"), case.get("shapeCase"), case["explicitStateDict"])
        require(identity in expected and identity not in observed, "NEGATIVE_CASE_DUPLICATE_OR_UNKNOWN")
        observed.add(identity)
        require(case.get("writeCompleted") is True and
                case.get("warningObserved") is (not case["explicitStateDict"]) and
                case.get("inspectorReason") == "UNGATHERED_AB_SHAPE", "NEGATIVE_CASE_FAILED")
    require(observed == expected and report.get("auxiliaryVectorHeuristic") == "PASS" and
            report.get("doraScope") == "UNSUPPORTED", "AUXILIARY_SCOPE_INCOMPLETE")
    require(report.get("baseIdentity", {}).get("kind") == "SYNTHETIC_RECIPE_NOT_HUB_MODEL",
            "SYNTHETIC_BASE_MISLABELED")
    require(report.get("remaining") == ["REAL_ZERO3_FSDP_GATHERING", "AUTHENTICATED_EXTERNAL_MODEL_LINEAGE",
                                       "IMMUTABLE_PROVIDER_READBACK", "DISTRIBUTED_CANCELLATION_FAILURE_ROLLBACK",
                                       "MODEL_QUALITY_QUALIFICATION", "INDEPENDENT_PUBLISHER_ADMISSION"],
            "QUALIFICATION_BOUNDS_CHANGED")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("requirements", "run"))
    args = parser.parse_args(argv)
    if args.command == "requirements":
        print(requirements(), end="")
        return 0
    try:
        revision = source_revision()
        report = run_cpu(revision)
        validate_observation(report, revision)
    except Exception as exc:
        # Unexpected failures are not converted into PASS; no raw environment,
        # provider messages, filenames or credential-bearing exception text.
        reason = str(exc) if isinstance(exc, EvaluationError) else type(exc).__name__
        print(json.dumps({"schema": "szl.forge.peft-cpu-execution.v1", "state": "BLOCKED",
                          "reason": reason, "productionDisposition": "HOLD",
                          "productionAuthorized": False, "publicationAuthorized": False,
                          "automaticPromotion": False}, sort_keys=True))
        return 2
    print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
