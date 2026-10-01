---
library_name: kernels
license: apache-2.0
tags:
- kernel
- governance
- formulas
- proof-status
- lean4
- lambda-aggregate
- doi:10.5281/zenodo.19944926
szl-review:
  artifact_class: SOFTWARE_KERNEL
  maturity: SOFTWARE_LIMITED
  trained_weights_present: false
  evaluation: HISTORICAL_RECORDS_NOT_RERUN
  source_mirror_parity: NOT_VERIFIED_THIS_REVIEW
  reviewed_at: "2026-09-30"
---

<p align="center"><img src="https://raw.githubusercontent.com/szl-holdings/szl-forge/main/szl-formulas/card/holo-banner.svg" alt="szl-formulas software provenance illustration" width="100%"/></p>

# szl-formulas

**Formula implementations with their declared proof scope intact**

**Software kernel · limited maturity · Apache-2.0 declared · no trained weights in the reviewed tree**

A Python formula registry and composition package. The reviewed Hub package declares 21 registry entries and separately preserves eight named canonical proof identifiers.

## Artifact and source identity

- Model-type card mirror: [reviewed tree](https://huggingface.co/SZLHOLDINGS/szl-formulas/tree/5e4519bb648c47bc7eb4aea4ba5fe80d92570883), revision `5e4519bb648c47bc7eb4aea4ba5fe80d92570883`
- First-class kernel package: [reviewed tree](https://huggingface.co/kernels/SZLHOLDINGS/szl-formulas/tree/636339753ea625bd27fd35a992699dba635c1cbb), revision `636339753ea625bd27fd35a992699dba635c1cbb`
- Declared canonical GitHub source: [observed source tree](https://github.com/szl-holdings/szl-formulas/tree/65c800b59249f27559de575bc66f5054fa585fa5), revision `65c800b59249f27559de575bc66f5054fa585fa5`
- Source paths: `torch-ext/szl_formulas/` and `build/torch-universal/szl_formulas/` in the canonical repository
- Inspection date: 2026-09-30 UTC

These are distinct repository revisions. The source repository is reachable; this review does not establish byte-for-byte parity, a reproducible build, approved publication, or the identity of a deployed process. Use the release's source binding and publication receipt to establish that chain.

The reviewed mirror contains Python source, build metadata and historical evidence records. Its listed tree contains no `model.joblib`, `.safetensors`, `.gguf`, `.onnx`, `.pt`, `.pth` or `.bin` model artifact. Historical surrogate descriptions in [MODEL_PROVENANCE.json](https://huggingface.co/SZLHOLDINGS/szl-formulas/blob/5e4519bb648c47bc7eb4aea4ba5fe80d92570883/MODEL_PROVENANCE.json) and [TRAINING_RECEIPT.json](https://huggingface.co/SZLHOLDINGS/szl-formulas/blob/5e4519bb648c47bc7eb4aea4ba5fe80d92570883/TRAINING_RECEIPT.json) do not establish present downloadable weights. Surrogate replay is blocked by the missing artifact. Do not load executable pickle/joblib artifacts from this card mirror.

## Intended use

Research and inspection of numerical formula implementations and their declared proof-status metadata. Follow each claim to the exact formal source and assumptions before citing it.

## Public interface

registry_count; lambda_aggregate; LOCKED_PROVEN_FORMULA_IDS; run_governed_loop; PROOF_STATUS; proof_status; selfcheck

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
    "SZLHOLDINGS/szl-formulas", revision=revision, trust_remote_code=True
)
print(kernel.registry_count())
print(sorted(kernel.LOCKED_PROVEN_FORMULA_IDS))
print(kernel.lambda_status())
# Counts and labels do not prove a registry-to-formal-theorem mapping.
```

The revision-format check is input validation only. It does not verify signatures, provenance, dependencies or runtime compatibility. A self-check exercises its programmed assertions; it is not product qualification.

## Evidence and evaluation limits

[BENCH.laptop-blackwell.json](https://huggingface.co/SZLHOLDINGS/szl-formulas/blob/5e4519bb648c47bc7eb4aea4ba5fe80d92570883/BENCH.laptop-blackwell.json), [TRAINING_RECEIPT.json](https://huggingface.co/SZLHOLDINGS/szl-formulas/blob/5e4519bb648c47bc7eb4aea4ba5fe80d92570883/TRAINING_RECEIPT.json), and [PROMOTION_READINESS_AUDIT.json](https://huggingface.co/SZLHOLDINGS/szl-formulas/blob/5e4519bb648c47bc7eb4aea4ba5fe80d92570883/PROMOTION_READINESS_AUDIT.json) retain owner-produced historical records. Each result applies only to its stated artifact, inputs, environment and date. This review did not rerun those tests, import the kernel, verify signatures, download weights, or benchmark throughput, energy or end-to-end behavior. Historical surrogate fidelity must not be presented as current kernel quality or current model availability.

## Limitations

The eight F-identifiers are a separate canonical set. A mapping from those identifiers to the 21 registry function names is not established, so this is not a claim that eight of these 21 implementations are proved. A PROOF_STATUS label is a declaration at its stated scope, not verification of this Python implementation. The Lambda aggregate remains advisory and uniqueness remains an open conjecture. The package does not establish CUDA acceleration, model accuracy, or safety.

Lambda uniqueness remains Conjecture 1, open and advisory. Receipt consistency, signatures and proof labels do not establish model accuracy, system safety, or certification. Preserve missing, failed, historical, and unverified states in downstream displays.

## License

The reviewed [LICENSE](https://huggingface.co/SZLHOLDINGS/szl-formulas/blob/5e4519bb648c47bc7eb4aea4ba5fe80d92570883/LICENSE) contains the Apache License 2.0 text, matching the card declaration. This observation does not independently establish rights to every artifact, training-data permission, or downstream-use clearance. Preserve upstream notices and check dependency licenses separately.

## Citation and verification

Cite the exact artifact URL, revision, evidence file, date, and test scope relevant to the claim. Prefer immutable links above over branch links or a live status badge. Report an evidence gap as an evidence gap.
