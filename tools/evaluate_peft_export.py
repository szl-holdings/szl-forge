"""Bounded, offline serialized linear-LoRA inspection; never a publisher.

Copyright 2026 SZL Holdings. SPDX-License-Identifier: Apache-2.0

This is an SZL evaluation of artifact bytes, not execution of PEFT's warning.
The byte API accepts already-acquired immutable inputs. It neither opens paths
nor loads pickle, imports model code, calls providers, or authenticates a base.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from typing import Any

UPSTREAM_REVISION = "78bce7cb48f800a7ad0d352b68a46302e13e1687"
MAX_BYTES = 4 * 1024 * 1024
MAX_HEADER = 64 * 1024
MAX_TENSORS = 256
WIDTHS = {"F16": 2, "BF16": 2, "F32": 4, "F64": 8}
LORA = re.compile(r"(?P<module>[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*)\.(?P<side>lora_A|lora_B)\.weight")


class InspectionError(ValueError):
    """Stable, secret-free reason code for rejected evaluation input."""


def need(condition: bool, code: str) -> None:
    if not condition:
        raise InspectionError(code)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def strict_object(raw: bytes) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            need(key not in result, "DUPLICATE_JSON_KEY")
            result[key] = value
        return result

    def invalid_constant(_: str) -> None:
        raise InspectionError("NONFINITE_JSON_NUMBER")

    def finite_float(text: str) -> float:
        value = float(text)
        need(math.isfinite(value), "NONFINITE_JSON_NUMBER")
        return value

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                           parse_constant=invalid_constant, parse_float=finite_float)
    except InspectionError:
        raise
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise InspectionError("INVALID_JSON") from exc
    need(type(value) is dict, "JSON_OBJECT_REQUIRED")
    return value


def finite_values(dtype: str, payload: memoryview) -> bool:
    if dtype == "BF16":
        return all((word & 0x7F80) != 0x7F80 for (word,) in struct.iter_unpack("<H", payload))
    return all(math.isfinite(value) for (value,) in
               struct.iter_unpack({"F16": "<e", "F32": "<f", "F64": "<d"}[dtype], payload))


def parse_tensors(raw: bytes) -> dict[str, dict[str, Any]]:
    """Inspect a bounded floating-point safetensors subset, without allocation.

    Exact contiguous payload accounting rejects holes, overlap, trailing bytes,
    truncated tensors and forged lengths. Unsupported dtypes fail closed. This
    is intentionally not a replacement for the upstream safetensors loader.
    """
    need(type(raw) is bytes and 8 < len(raw) <= MAX_BYTES, "ARTIFACT_SIZE_OR_TYPE")
    header_size = struct.unpack("<Q", raw[:8])[0]
    need(1 < header_size <= MAX_HEADER and 8 + header_size <= len(raw), "HEADER_SIZE")
    header_bytes = raw[8:8 + header_size]
    need(header_bytes.startswith(b"{"), "HEADER_NOT_OBJECT")
    header = strict_object(header_bytes)
    metadata = header.pop("__metadata__", {})
    need(type(metadata) is dict and all(type(k) is str and type(v) is str
                                       for k, v in metadata.items()), "INVALID_METADATA")
    need(0 < len(header) <= MAX_TENSORS, "TENSOR_COUNT")
    payload = memoryview(raw)[8 + header_size:]
    intervals = []
    result = {}
    for name, info in header.items():
        need(type(name) is str and 0 < len(name) <= 256, "TENSOR_NAME")
        need(type(info) is dict and set(info) == {"dtype", "shape", "data_offsets"}, "TENSOR_DESCRIPTOR")
        dtype, shape, offsets = info["dtype"], info["shape"], info["data_offsets"]
        need(type(dtype) is str and dtype in WIDTHS, "UNSUPPORTED_DTYPE")
        need(type(shape) is list and len(shape) <= 4 and
             all(type(d) is int and 0 <= d <= MAX_BYTES for d in shape), "INVALID_SHAPE")
        need(type(offsets) is list and len(offsets) == 2 and
             all(type(x) is int for x in offsets), "INVALID_OFFSETS")
        start, end = offsets
        need(0 <= start <= end <= len(payload), "OFFSET_BOUNDS")
        need(math.prod(shape) * WIDTHS[dtype] == end - start, "TENSOR_BYTE_LENGTH")
        intervals.append((start, end, name))
        result[name] = {"dtype": dtype, "shape": shape,
                        "finite": finite_values(dtype, payload[start:end])}
    cursor = 0
    for start, end, _ in sorted(intervals):
        need(start == cursor, "PAYLOAD_GAP_OR_OVERLAP")
        cursor = end
    need(cursor == len(payload), "UNACCOUNTED_PAYLOAD")
    return result


def inspect_adapter(artifact: bytes, config_bytes: bytes, binding: dict[str, Any]) -> dict[str, Any]:
    """Return scoped evidence. Even a structural PASS leaves production HOLD.

    Binding is an independent caller declaration, not a signature or proof of
    model lineage. A model reload, distributed gather and PEFT execution remain
    NOT_RUN; callers may not promote this report into those missing predicates.
    """
    report: dict[str, Any] = {
        "schema": "szl.forge.peft-export-structure.v1",
        "candidate": "peft-distributed-lora-export-78bce7c",
        "upstreamRevision": UPSTREAM_REVISION,
        "scope": "BOUNDED_SERIALIZED_LINEAR_LORA_STRUCTURE_ONLY",
        "state": "REJECTED", "reason": None,
        "disposition": "EVALUATION_HOLD",
        "bindingAssurance": "CALLER_DECLARED_NOT_AUTHENTICATED",
        "upstreamWarningExecution": "NOT_RUN",
        "distributedGather": "NOT_RUN", "exactBaseModelReload": "NOT_RUN",
        "providerReadback": "NOT_RUN", "modelQuality": "NOT_MEASURED",
        "publicationAuthorized": False, "productionAuthorized": False,
        "automaticPromotion": False, "providerWrites": 0,
        "auxiliaryKeysOutsideABShapeHeuristic": [],
    }
    try:
        need(type(config_bytes) is bytes and 0 < len(config_bytes) <= MAX_HEADER, "CONFIG_SIZE_OR_TYPE")
        need(type(binding) is dict and set(binding) == {
            "artifactSha256", "configSha256", "baseModelId", "baseModelRevision",
            "peftRevision", "expectedTensorShapes", "stateDictProvenance",
        }, "BINDING_SCHEMA")
        need(type(artifact) is bytes and 8 < len(artifact) <= MAX_BYTES, "ARTIFACT_SIZE_OR_TYPE")
        for key in ("artifactSha256", "configSha256"):
            need(type(binding[key]) is str and re.fullmatch(r"[0-9a-f]{64}", binding[key]) is not None,
                 "BINDING_DIGEST")
        need(binding["peftRevision"] == UPSTREAM_REVISION, "PEFT_REVISION_MISMATCH")
        need(type(binding["baseModelRevision"]) is str and
             re.fullmatch(r"[0-9a-f]{40}", binding["baseModelRevision"]) is not None, "BASE_REVISION_NOT_EXACT")
        need(type(binding["baseModelId"]) is str and
             re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", binding["baseModelId"]) is not None,
             "BASE_MODEL_ID")
        need(type(binding["stateDictProvenance"]) is str and binding["stateDictProvenance"] in
             {"INFERRED", "EXPLICIT", "UNKNOWN", "SYNTHETIC"}, "PROVENANCE_CLASS")
        report["stateDictProvenance"] = binding["stateDictProvenance"]
        report["declaredBase"] = {"repository": binding["baseModelId"], "revision": binding["baseModelRevision"]}
        report["artifactSha256"] = digest(artifact)
        report["configSha256"] = digest(config_bytes)
        need(report["artifactSha256"] == binding["artifactSha256"], "ARTIFACT_DIGEST_MISMATCH")
        need(report["configSha256"] == binding["configSha256"], "CONFIG_DIGEST_MISMATCH")
        config = strict_object(config_bytes)
        need(config.get("base_model_name_or_path") == binding["baseModelId"] and
             config.get("revision") == binding["baseModelRevision"], "CONFIG_BASE_BINDING_MISMATCH")
        tensors = parse_tensors(artifact)
        report["tensorCount"] = len(tensors)
        expected = binding["expectedTensorShapes"]
        need(type(expected) is dict and 0 < len(expected) <= MAX_TENSORS, "EXPECTED_INVENTORY_REQUIRED")
        need(set(expected) == set(tensors), "TENSOR_INVENTORY_MISMATCH")
        pairs: dict[str, dict[str, dict[str, Any]]] = {}
        auxiliary = []
        unsupported = []
        for name, tensor in tensors.items():
            dims = expected[name]
            need(type(dims) is list and len(dims) <= 4 and
                 all(type(d) is int and 0 <= d <= MAX_BYTES for d in dims), "EXPECTED_SHAPE_INVALID")
            match = LORA.fullmatch(name)
            if match:
                need(len(tensor["shape"]) >= 2 and all(d > 0 for d in tensor["shape"]), "UNGATHERED_AB_SHAPE")
                if len(tensor["shape"]) != 2:
                    unsupported.append(name)
                pairs.setdefault(match["module"], {})[match["side"]] = tensor
            elif {"lora_magnitude_vector", "lora_E"}.intersection(name.split(".")):
                auxiliary.append(name)
            else:
                unsupported.append(name)
            need(tensor["shape"] == dims, "EXPECTED_SHAPE_MISMATCH")
            need(tensor["finite"], "NONFINITE_TENSOR_PAYLOAD")
        report["auxiliaryKeysOutsideABShapeHeuristic"] = sorted(auxiliary)
        need(pairs != {}, "NO_LORA_AB_TENSORS")
        for pair in pairs.values():
            need(set(pair) == {"lora_A", "lora_B"}, "INCOMPLETE_AB_PAIR")
        # Valid DoRA/AdaLoRA auxiliary vectors are not malformed A/B tensors.
        # They require a different evaluator and cannot receive linear-LoRA PASS.
        if (auxiliary or unsupported or config.get("peft_type") != "LORA" or
                config.get("use_dora", False) is not False or
                config.get("bias", "none") != "none" or
                config.get("rank_pattern") not in (None, {}) or
                config.get("modules_to_save") not in (None, [])):
            report.update(state="UNSUPPORTED", reason="OUTSIDE_LINEAR_LORA_SCOPE")
            return report
        rank = config.get("r")
        need(type(rank) is int and 0 < rank <= MAX_BYTES, "INVALID_CONFIG_RANK")
        for pair in pairs.values():
            a, b = pair["lora_A"], pair["lora_B"]
            need(a["shape"][0] == b["shape"][1] == rank, "AB_RANK_MISMATCH")
            need(a["dtype"] == b["dtype"], "AB_DTYPE_MISMATCH")
        report.update(state="PASS", reason="STRUCTURE_ONLY", moduleCount=len(pairs))
    except InspectionError as exc:
        report["reason"] = str(exc)
    return report


def fixture(tensors: dict[str, tuple[list[int], list[float]]], *, dtype: str = "F32",
            config_updates: dict[str, Any] | None = None,
            provenance: str = "SYNTHETIC") -> tuple[bytes, bytes, dict[str, Any]]:
    """Make small independent synthetic bytes; never an export from a real model."""
    header, payload, shapes = {}, bytearray(), {}
    for name, (shape, values) in sorted(tensors.items()):
        start = len(payload)
        if dtype == "BF16":
            payload.extend(b"".join(struct.pack("<I", struct.unpack("<I", struct.pack("<f", value))[0])[2:]
                                    for value in values))
        else:
            fmt = {"F16": "e", "F32": "f", "F64": "d"}[dtype]
            payload.extend(struct.pack("<" + fmt * len(values), *values))
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [start, len(payload)]}
        shapes[name] = shape
    encoded = json.dumps(header, separators=(",", ":"), sort_keys=True).encode()
    encoded += b" " * (-len(encoded) % 8)
    artifact = struct.pack("<Q", len(encoded)) + encoded + payload
    config = {"base_model_name_or_path": "szl-fixture/linear-lora", "revision": "a" * 40,
              "peft_type": "LORA", "r": 2, "bias": "none", "use_dora": False}
    config.update(config_updates or {})
    config_bytes = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    binding = {"artifactSha256": digest(artifact), "configSha256": digest(config_bytes),
               "baseModelId": "szl-fixture/linear-lora", "baseModelRevision": "a" * 40,
               "peftRevision": UPSTREAM_REVISION, "expectedTensorShapes": shapes,
               "stateDictProvenance": provenance}
    return artifact, config_bytes, binding


def main() -> int:
    """Run only the fixed offline fixture; write evidence to stdout, not a path."""
    case = fixture({"base.model.linear.lora_A.weight": ([2, 3], [0.25] * 6),
                    "base.model.linear.lora_B.weight": ([4, 2], [0.5] * 8)})
    report = inspect_adapter(*case)
    print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))
    return 0 if report["state"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
