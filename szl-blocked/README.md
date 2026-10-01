---
library_name: kernels
license: apache-2.0
tags:
- kernel
- eu-ai-act
- compliance
- annex-iv
- governance
- doi:10.5281/zenodo.19944926
szl-review:
  artifact_class: SOFTWARE_KERNEL
  maturity: SOFTWARE_LIMITED
  trained_weights_present: false
  evaluation: HISTORICAL_RECORDS_NOT_RERUN
  source_mirror_parity: NOT_VERIFIED_THIS_REVIEW
  reviewed_at: "2026-09-30"
---

<p align="center"><img src="https://raw.githubusercontent.com/szl-holdings/szl-forge/main/szl-blocked/card/holo-banner.svg" alt="szl-blocked software provenance illustration" width="100%"/></p>

# szl-blocked

**Explicit policy outcomes, with blocked calls recorded**

**Software kernel · limited maturity · Apache-2.0 declared · no trained weights in the reviewed tree**

A Python governance kernel that applies a supplied policy before calling a function and records the outcome in a receipt chain. The companion szl_euaiact package generates an Annex IV-style documentation draft.

## Artifact and source identity

- Model-type card mirror: [reviewed tree](https://huggingface.co/SZLHOLDINGS/szl-blocked/tree/730a238648ac69aa2a69c11428e06354c48d09e2), revision `730a238648ac69aa2a69c11428e06354c48d09e2`
- First-class kernel package: [reviewed tree](https://huggingface.co/kernels/SZLHOLDINGS/szl-blocked/tree/b8b15cde2d3d3d7c6ef30467d6cacc6d219a1349), revision `b8b15cde2d3d3d7c6ef30467d6cacc6d219a1349`
- Declared canonical GitHub source: [observed source tree](https://github.com/szl-holdings/szl-blocked/tree/96d520d51ad867ec5e0f62b5063fbe85c17cef7e), revision `96d520d51ad867ec5e0f62b5063fbe85c17cef7e`
- Source paths: `torch-ext/szl_blocked/` and `build/torch-universal/szl_blocked/` in the canonical repository
- Inspection date: 2026-09-30 UTC

These are distinct repository revisions. The source repository is reachable; this review does not establish byte-for-byte parity, a reproducible build, approved publication, or the identity of a deployed process. Use the release's source binding and publication receipt to establish that chain.

The reviewed mirror contains Python source, build metadata and historical evidence records. Its listed tree contains no `model.joblib`, `.safetensors`, `.gguf`, `.onnx`, `.pt`, `.pth` or `.bin` model artifact. Historical surrogate descriptions in [MODEL_PROVENANCE.json](https://huggingface.co/SZLHOLDINGS/szl-blocked/blob/730a238648ac69aa2a69c11428e06354c48d09e2/MODEL_PROVENANCE.json) and [TRAINING_RECEIPT.json](https://huggingface.co/SZLHOLDINGS/szl-blocked/blob/730a238648ac69aa2a69c11428e06354c48d09e2/TRAINING_RECEIPT.json) do not establish present downloadable weights. Surrogate replay is blocked by the missing artifact. Do not load executable pickle/joblib artifacts from this card mirror.

## Intended use

Research and development of local, explicitly configured software policy gates and documentation workflows. Inspect the policy and test the denial path before integration.

## Public interface

governed_call; GovernedGate; deny_by_default; deny_if_flag; deny_if_action_in; UnifiedReceiptChain

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
    "SZLHOLDINGS/szl-blocked", revision=revision, trust_remote_code=True
)
chain = kernel.UnifiedReceiptChain()
policy = kernel.deny_if_action_in({"write_protected"})
result = kernel.governed_call(lambda: "example", policy, chain,
    request={"action": "write_protected"})
assert result.blocked is True and result.output is None
```

The revision-format check is input validation only. It does not verify signatures, provenance, dependencies or runtime compatibility. A self-check exercises its programmed assertions; it is not product qualification.

## Evidence and evaluation limits

[BENCH.laptop-blackwell.json](https://huggingface.co/SZLHOLDINGS/szl-blocked/blob/730a238648ac69aa2a69c11428e06354c48d09e2/BENCH.laptop-blackwell.json), [TRAINING_RECEIPT.json](https://huggingface.co/SZLHOLDINGS/szl-blocked/blob/730a238648ac69aa2a69c11428e06354c48d09e2/TRAINING_RECEIPT.json), and [PROMOTION_READINESS_AUDIT.json](https://huggingface.co/SZLHOLDINGS/szl-blocked/blob/730a238648ac69aa2a69c11428e06354c48d09e2/PROMOTION_READINESS_AUDIT.json) retain owner-produced historical records. Each result applies only to its stated artifact, inputs, environment and date. This review did not rerun those tests, import the kernel, verify signatures, download weights, or benchmark throughput, energy or end-to-end behavior. Historical surrogate fidelity must not be presented as current kernel quality or current model availability.

## Limitations

A recorded allow decision does not certify the requested action or the surrounding application. The Annex IV output is a draft skeleton requiring human completion and review; it does not establish legal compliance. This package is not identified as the A11oy production pre-action core. No throughput or CUDA acceleration claim is made.

Lambda uniqueness remains Conjecture 1, open and advisory. Receipt consistency, signatures and proof labels do not establish model accuracy, system safety, or certification. Preserve missing, failed, historical, and unverified states in downstream displays.

## License

The reviewed [LICENSE](https://huggingface.co/SZLHOLDINGS/szl-blocked/blob/730a238648ac69aa2a69c11428e06354c48d09e2/LICENSE) contains the Apache License 2.0 text, matching the card declaration. This observation does not independently establish rights to every artifact, training-data permission, or downstream-use clearance. Preserve upstream notices and check dependency licenses separately.

## Citation and verification

Cite the exact artifact URL, revision, evidence file, date, and test scope relevant to the claim. Prefer immutable links above over branch links or a live status badge. Report an evidence gap as an evidence gap.
