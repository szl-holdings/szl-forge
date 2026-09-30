---
library_name: kernels
license: apache-2.0
tags:
- kernel
- governance
- invariants
- provenance
- receipts
- ed25519
- doi:10.5281/zenodo.19944926
szl-review:
  artifact_class: SOFTWARE_KERNEL
  maturity: SOFTWARE_LIMITED
  trained_weights_present: false
  evaluation: HISTORICAL_RECORDS_NOT_RERUN
  source_mirror_parity: NOT_VERIFIED_THIS_REVIEW
  reviewed_at: "2026-09-30"
---

<p align="center"><img src="https://raw.githubusercontent.com/szl-holdings/szl-forge/main/szl-invariants/card/holo-banner.svg" alt="szl-invariants software provenance illustration" width="100%"/></p>

# szl-invariants

**Replay receipt and ledger checks without hiding missing evidence**

**Software kernel · limited maturity · Apache-2.0 declared · no trained weights in the reviewed tree**

A Python package that checks a supplied receipt/ledger export and optional sample lineage against eight named self-consistency checks. It represents missing evidence and verification failures explicitly.

## Artifact and source identity

- Model-type card mirror: [reviewed tree](https://huggingface.co/SZLHOLDINGS/szl-invariants/tree/4ba2b95956cb8a836e7c9f5886b3da778c667f1a), revision `4ba2b95956cb8a836e7c9f5886b3da778c667f1a`
- First-class kernel package: [reviewed tree](https://huggingface.co/kernels/SZLHOLDINGS/szl-invariants/tree/78d04afef253e526326d6b0f4e944fa3d4b1990c), revision `78d04afef253e526326d6b0f4e944fa3d4b1990c`
- Declared canonical GitHub source: [observed source tree](https://github.com/szl-holdings/szl-invariants/tree/e9621c5d95e1b6a336981346e473980cfe2d037b), revision `e9621c5d95e1b6a336981346e473980cfe2d037b`
- Source paths: `torch-ext/szl_invariants/` and `build/torch-universal/szl_invariants/` in the canonical repository
- Inspection date: 2026-09-30 UTC

These are distinct repository revisions. The source repository is reachable; this review does not establish byte-for-byte parity, a reproducible build, approved publication, or the identity of a deployed process. Use the release's source binding and publication receipt to establish that chain.

The reviewed mirror contains Python source, build metadata and historical evidence records. Its listed tree contains no `model.joblib`, `.safetensors`, `.gguf`, `.onnx`, `.pt`, `.pth` or `.bin` model artifact. Historical surrogate descriptions in [MODEL_PROVENANCE.json](https://huggingface.co/SZLHOLDINGS/szl-invariants/blob/4ba2b95956cb8a836e7c9f5886b3da778c667f1a/MODEL_PROVENANCE.json) and [TRAINING_RECEIPT.json](https://huggingface.co/SZLHOLDINGS/szl-invariants/blob/4ba2b95956cb8a836e7c9f5886b3da778c667f1a/TRAINING_RECEIPT.json) do not establish present downloadable weights. Surrogate replay is blocked by the missing artifact. Do not load executable pickle/joblib artifacts from this card mirror.

## Intended use

Offline inspection of software-run exports you are authorized to inspect. Supply the correct public key and optional lineage samples; retain the original export alongside the report.

## Public interface

run_invariants; load_jsonl; verify_ed25519; list_checks; selfcheck

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
    "SZLHOLDINGS/szl-invariants", revision=revision, trust_remote_code=True
)
rows = kernel.load_jsonl("runs_export.jsonl")
report = kernel.run_invariants(rows, samples=None, pubkey=None)
print(report["summary"])
# Missing key or sample evidence must not be displayed as a pass.
```

The revision-format check is input validation only. It does not verify signatures, provenance, dependencies or runtime compatibility. A self-check exercises its programmed assertions; it is not product qualification.

## Evidence and evaluation limits

[BENCH.laptop-blackwell.json](https://huggingface.co/SZLHOLDINGS/szl-invariants/blob/4ba2b95956cb8a836e7c9f5886b3da778c667f1a/BENCH.laptop-blackwell.json), [TRAINING_RECEIPT.json](https://huggingface.co/SZLHOLDINGS/szl-invariants/blob/4ba2b95956cb8a836e7c9f5886b3da778c667f1a/TRAINING_RECEIPT.json), and [PROMOTION_READINESS_AUDIT.json](https://huggingface.co/SZLHOLDINGS/szl-invariants/blob/4ba2b95956cb8a836e7c9f5886b3da778c667f1a/PROMOTION_READINESS_AUDIT.json) retain owner-produced historical records. Each result applies only to its stated artifact, inputs, environment and date. This review did not rerun those tests, import the kernel, verify signatures, download weights, or benchmark throughput, energy or end-to-end behavior. Historical surrogate fidelity must not be presented as current kernel quality or current model availability.

## Limitations

The checks do not prove export completeness, source-system honesty, or answer accuracy. Missing key or sample evidence must remain indeterminate. Preserve HOLDS, VIOLATED, KEY_ROTATED, NO_DATA and UNAVAILABLE distinctly. It is not a load-time tensor-signature enforcement product. A recent GitHub source commit is not proof that its fixes were published to the Hub; use the authorized publication workflow and immutable provider-byte evidence.

Lambda uniqueness remains Conjecture 1, open and advisory. Receipt consistency, signatures and proof labels do not establish model accuracy, system safety, or certification. Preserve missing, failed, historical, and unverified states in downstream displays.

## License

The reviewed [LICENSE](https://huggingface.co/SZLHOLDINGS/szl-invariants/blob/4ba2b95956cb8a836e7c9f5886b3da778c667f1a/LICENSE) contains the Apache License 2.0 text, matching the card declaration. This observation does not independently establish rights to every artifact, training-data permission, or downstream-use clearance. Preserve upstream notices and check dependency licenses separately.

## Citation and verification

Cite the exact artifact URL, revision, evidence file, date, and test scope relevant to the claim. Prefer immutable links above over branch links or a live status badge. Report an evidence gap as an evidence gap.
