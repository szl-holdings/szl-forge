---
library_name: kernels
license: apache-2.0
tags:
- kernel
- governance
- ouroboros
- loop-tax
- provenance
- doi:10.5281/zenodo.19944926
szl-review:
  artifact_class: SOFTWARE_KERNEL
  maturity: SOFTWARE_LIMITED
  trained_weights_present: false
  evaluation: HISTORICAL_RECORDS_NOT_RERUN
  source_mirror_parity: NOT_VERIFIED_THIS_REVIEW
  reviewed_at: "2026-09-30"
---

<p align="center"><img src="https://raw.githubusercontent.com/szl-holdings/szl-forge/main/szl-ouroboros/card/holo-banner.svg" alt="szl-ouroboros software provenance illustration" width="100%"/></p>

# szl-ouroboros

**Account for bounded-loop timing and preserve measurement gaps**

**Software kernel · limited maturity · Apache-2.0 declared · no trained weights in the reviewed tree**

A Python package for reconstructing bounded-loop traces and calculating timing fields from supplied attempt windows and an optional measured wall time.

## Artifact and source identity

- Model-type card mirror: [reviewed tree](https://huggingface.co/SZLHOLDINGS/szl-ouroboros/tree/91e68d606a6273dd46db600f83938e61322b2b37), revision `91e68d606a6273dd46db600f83938e61322b2b37`
- First-class kernel package: [reviewed tree](https://huggingface.co/kernels/SZLHOLDINGS/szl-ouroboros/tree/194970667faa1292ce09c7ae5880a1918f5cddeb), revision `194970667faa1292ce09c7ae5880a1918f5cddeb`
- Declared canonical GitHub source: [observed source tree](https://github.com/szl-holdings/szl-ouroboros/tree/f4c9df3840a84c767b7e5fa1c29aa25f00cc0457), revision `f4c9df3840a84c767b7e5fa1c29aa25f00cc0457`
- Source paths: `torch-ext/szl_ouroboros/` and `build/torch-universal/szl_ouroboros/` in the canonical repository
- Inspection date: 2026-09-30 UTC

These are distinct repository revisions. The source repository is reachable; this review does not establish byte-for-byte parity, a reproducible build, approved publication, or the identity of a deployed process. Use the release's source binding and publication receipt to establish that chain.

The reviewed mirror contains Python source, build metadata and historical evidence records. Its listed tree contains no `model.joblib`, `.safetensors`, `.gguf`, `.onnx`, `.pt`, `.pth` or `.bin` model artifact. Historical surrogate descriptions in [MODEL_PROVENANCE.json](https://huggingface.co/SZLHOLDINGS/szl-ouroboros/blob/91e68d606a6273dd46db600f83938e61322b2b37/MODEL_PROVENANCE.json) and [TRAINING_RECEIPT.json](https://huggingface.co/SZLHOLDINGS/szl-ouroboros/blob/91e68d606a6273dd46db600f83938e61322b2b37/TRAINING_RECEIPT.json) do not establish present downloadable weights. Surrogate replay is blocked by the missing artifact. Do not load executable pickle/joblib artifacts from this card mirror.

## Intended use

Offline performance accounting for benign software retry workflows. Keep measured inputs, derived arithmetic, and counterfactual quantities visibly separate.

## Public interface

build_loop_trace; loop_tax; classify_exit; explain; selfcheck

## Usage after publication verification

The example below is source-informed and was **not executed in this card review**. Install the client and dependencies qualified by the relevant release. Loading remote kernel Python is code execution. Review the exact package and use the immutable **kernel-package** revision from verified publication evidence. Do not substitute the model-mirror SHA or GitHub source SHA.

```python
import os
import re
from kernels import get_kernel

revision = os.environ.get("SZL_VERIFIED_KERNEL_REVISION", "")
if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
    raise ValueError("Set the kernel revision from verified publication evidence")

kernel = get_kernel(
    "SZLHOLDINGS/szl-ouroboros", revision=revision, trust_remote_code=True
)
attempts = [
    {"ok": False, "latency_ms": 220},
    {"ok": True, "latency_ms": 900},
]
trace = kernel.build_loop_trace(attempts, wall_ms=1300, max_budget=4)
print(trace["modelMs"], trace["overheadMs"])
```

The revision-format check is input validation only. It does not verify signatures, provenance, dependencies or runtime compatibility. A self-check exercises its programmed assertions; it is not product qualification.

## Evidence and evaluation limits

[BENCH.laptop-blackwell.json](https://huggingface.co/SZLHOLDINGS/szl-ouroboros/blob/91e68d606a6273dd46db600f83938e61322b2b37/BENCH.laptop-blackwell.json), [TRAINING_RECEIPT.json](https://huggingface.co/SZLHOLDINGS/szl-ouroboros/blob/91e68d606a6273dd46db600f83938e61322b2b37/TRAINING_RECEIPT.json), and [PROMOTION_READINESS_AUDIT.json](https://huggingface.co/SZLHOLDINGS/szl-ouroboros/blob/91e68d606a6273dd46db600f83938e61322b2b37/PROMOTION_READINESS_AUDIT.json) retain owner-produced historical records. Each result applies only to its stated artifact, inputs, environment and date. This review did not rerun those tests, import the kernel, verify signatures, download weights, or benchmark throughput, energy or end-to-end behavior. Historical surrogate fidelity must not be presented as current kernel quality or current model availability.

## Limitations

This card covers the kernel artifact, not the separate TypeScript ouroboros product or every GitHub automation in the source repository. Accounting does not itself execute or enforce an application retry policy. When wall time is absent, overheadMs remains unavailable. serializationTaxMs is counterfactual, not a realized saving; deadHopMs is an upper-bound interpretation, not measured prefetch savings. The historical surrogate does not replace exact arithmetic.

Lambda uniqueness remains Conjecture 1, open and advisory. Receipt consistency, signatures and proof labels do not establish model accuracy, system safety, or certification. Preserve missing, failed, historical, and unverified states in downstream displays.

## License

The reviewed [LICENSE](https://huggingface.co/SZLHOLDINGS/szl-ouroboros/blob/91e68d606a6273dd46db600f83938e61322b2b37/LICENSE) contains the Apache License 2.0 text, matching the card declaration. This observation does not independently establish rights to every artifact, training-data permission, or downstream-use clearance. Preserve upstream notices and check dependency licenses separately.

## Citation and verification

Cite the exact artifact URL, revision, evidence file, date, and test scope relevant to the claim. Prefer immutable links above over branch links or a live status badge. Report an evidence gap as an evidence gap.
