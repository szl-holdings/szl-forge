"""Optional official-loader cross-check; CPU fixtures, never a PEFT model load.

Copyright 2026 SZL Holdings. SPDX-License-Identifier: Apache-2.0
Run with the exact optional versions recorded in the evidence. The default
repository suite stays dependency-free; this lane fails if dependencies differ.
"""
from __future__ import annotations

import importlib.metadata as metadata
import json

from tools.evaluate_peft_export import digest, fixture, inspect_adapter


def main() -> int:
    versions = {name: metadata.version(name) for name in ("torch", "safetensors", "numpy")}
    expected = {"torch": "2.10.0+cpu", "safetensors": "0.7.0", "numpy": "2.3.5"}
    if versions != expected:
        raise RuntimeError("optional_interop_dependency_pins_differ")
    import torch
    from safetensors.torch import load, save

    a = "base.model.layers.0.q_proj.lora_A.weight"
    b = "base.model.layers.0.q_proj.lora_B.weight"
    checks = []
    for label, dtype in (("F16", torch.float16), ("BF16", torch.bfloat16),
                         ("F32", torch.float32), ("F64", torch.float64)):
        tensors = {a: torch.full((2, 3), 0.25, dtype=dtype), b: torch.full((4, 2), 0.5, dtype=dtype)}
        raw, config, binding = fixture({a: ([2, 3], [0.25] * 6), b: ([4, 2], [0.5] * 8)}, dtype=label)
        decoded = load(raw)
        if any(not torch.equal(decoded[key], tensors[key]) for key in tensors):
            raise RuntimeError("official_loader_fixture_mismatch")
        official = save(tensors)
        result = inspect_adapter(official, config, {**binding, "artifactSha256": digest(official)})
        if result["state"] != "PASS":
            raise RuntimeError("official_serialization_rejected")
        x = torch.arange(6, dtype=torch.float64).reshape(2, 3) / 8
        direct = (x @ tensors[a].double().T) @ tensors[b].double().T
        readback = (x @ decoded[a].double().T) @ decoded[b].double().T
        torch.testing.assert_close(direct, readback, rtol=0, atol=0)
        checks.append({"dtype": label, "fixtureDecodedByOfficialLoader": True,
                       "officialSerializationInspected": True,
                       "linearFixtureMaxAbsoluteError": float((direct - readback).abs().max()),
                       "officialArtifactSha256": digest(official), "fixtureArtifactSha256": digest(raw)})
    print(json.dumps({"evidenceClass": "LOCAL_OFFLINE_FIXTURE_INTEROPERABILITY_NOT_MODEL_QUALIFICATION",
                      "dependencies": versions, "checks": checks, "peftExecution": "NOT_RUN",
                      "distributedGather": "NOT_RUN", "baseModelReload": "NOT_RUN",
                      "productionDisposition": "HOLD", "productionAuthorized": False},
                     indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
