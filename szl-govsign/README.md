---
library_name: kernels
license: apache-2.0
tags:
- kernel
- signing
- attestation
- dsse
- in-toto
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

<p align="center"><img src="https://raw.githubusercontent.com/szl-holdings/szl-forge/main/szl-govsign/card/holo-banner.svg" alt="szl-govsign software provenance illustration" width="100%"/></p>

# szl-govsign

**Signed envelopes with an explicit verification boundary**

**Software kernel · limited maturity · Apache-2.0 declared · no trained weights in the reviewed tree**

A Python package for creating and verifying ECDSA P-256 signatures over DSSE/in-toto-shaped envelopes containing an SZL governance predicate.

## Artifact and source identity

- Model-type card mirror: [reviewed tree](https://huggingface.co/SZLHOLDINGS/szl-govsign/tree/13ce04b3b3e490e9bee6453df7f83fdcf0f3b5c9), revision `13ce04b3b3e490e9bee6453df7f83fdcf0f3b5c9`
- First-class kernel package: [reviewed tree](https://huggingface.co/kernels/SZLHOLDINGS/szl-govsign/tree/e498c5042c1f721a1bbd8a4b9fdaa3756001e78d), revision `e498c5042c1f721a1bbd8a4b9fdaa3756001e78d`
- Declared canonical GitHub source: [observed source tree](https://github.com/szl-holdings/szl-govsign/tree/7bff004d12347ac23ae60ffbfb78ce7f9ef15829), revision `7bff004d12347ac23ae60ffbfb78ce7f9ef15829`
- Source paths: `torch-ext/szl_govsign/` and `build/torch-universal/szl_govsign/` in the canonical repository
- Inspection date: 2026-09-30 UTC

These are distinct repository revisions. The source repository is reachable; this review does not establish byte-for-byte parity, a reproducible build, approved publication, or the identity of a deployed process. Use the release's source binding and publication receipt to establish that chain.

The reviewed mirror contains Python source, build metadata and historical evidence records. Its listed tree contains no `model.joblib`, `.safetensors`, `.gguf`, `.onnx`, `.pt`, `.pth` or `.bin` model artifact. Historical surrogate descriptions in [MODEL_PROVENANCE.json](https://huggingface.co/SZLHOLDINGS/szl-govsign/blob/13ce04b3b3e490e9bee6453df7f83fdcf0f3b5c9/MODEL_PROVENANCE.json) and [TRAINING_RECEIPT.json](https://huggingface.co/SZLHOLDINGS/szl-govsign/blob/13ce04b3b3e490e9bee6453df7f83fdcf0f3b5c9/TRAINING_RECEIPT.json) do not establish present downloadable weights. Surrogate replay is blocked by the missing artifact. Do not load executable pickle/joblib artifacts from this card mirror.

## Intended use

Research and development of signed software provenance receipts. Establish public-key trust independently and verify the exact envelope before relying on a signature.

## Public interface

attest; verify; build_governance_predicate; generate_ephemeral_keypair; selfcheck

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
    "SZLHOLDINGS/szl-govsign", revision=revision, trust_remote_code=True
)
print(kernel.selfcheck())  # diagnostic output; not a release or trust verdict
```

The revision-format check is input validation only. It does not verify signatures, provenance, dependencies or runtime compatibility. A self-check exercises its programmed assertions; it is not product qualification.

## Evidence and evaluation limits

[BENCH.laptop-blackwell.json](https://huggingface.co/SZLHOLDINGS/szl-govsign/blob/13ce04b3b3e490e9bee6453df7f83fdcf0f3b5c9/BENCH.laptop-blackwell.json), [TRAINING_RECEIPT.json](https://huggingface.co/SZLHOLDINGS/szl-govsign/blob/13ce04b3b3e490e9bee6453df7f83fdcf0f3b5c9/TRAINING_RECEIPT.json), and [PROMOTION_READINESS_AUDIT.json](https://huggingface.co/SZLHOLDINGS/szl-govsign/blob/13ce04b3b3e490e9bee6453df7f83fdcf0f3b5c9/PROMOTION_READINESS_AUDIT.json) retain owner-produced historical records. Each result applies only to its stated artifact, inputs, environment and date. This review did not rerun those tests, import the kernel, verify signatures, download weights, or benchmark throughput, energy or end-to-end behavior. Historical surrogate fidelity must not be presented as current kernel quality or current model availability.

## Limitations

Signature verification establishes integrity relative to the selected public key; external identity and key provenance require separate evidence. It does not establish the truth of a predicate, safety, legal compliance, or model quality. No SLSA level or external verifier compatibility is established by a self-check. The historical structural surrogate cannot replace cryptographic verification. Runtime code requires cryptography; the first-class package metadata contains inconsistent dependency declarations that need qualification.

Lambda uniqueness remains Conjecture 1, open and advisory. Receipt consistency, signatures and proof labels do not establish model accuracy, system safety, or certification. Preserve missing, failed, historical, and unverified states in downstream displays.

## License

The reviewed [LICENSE](https://huggingface.co/SZLHOLDINGS/szl-govsign/blob/13ce04b3b3e490e9bee6453df7f83fdcf0f3b5c9/LICENSE) contains the Apache License 2.0 text, matching the card declaration. This observation does not independently establish rights to every artifact, training-data permission, or downstream-use clearance. Preserve upstream notices and check dependency licenses separately.

## Citation and verification

Cite the exact artifact URL, revision, evidence file, date, and test scope relevant to the claim. Prefer immutable links above over branch links or a live status badge. Report an evidence gap as an evidence gap.
