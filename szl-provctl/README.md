---
library_name: kernels
license: apache-2.0
tags:
- kernel
- provenance
- supply-chain
- in-toto
- slsa
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

<p align="center"><img src="https://raw.githubusercontent.com/szl-holdings/szl-forge/main/szl-provctl/card/holo-banner.svg" alt="szl-provctl software provenance illustration" width="100%"/></p>

# szl-provctl

**Inspect the declared provenance graph, edge by edge**

**Software kernel · limited maturity · Apache-2.0 declared · no trained weights in the reviewed tree**

A Python provenance package with receipt-chain graph traversal and helpers for in-toto Statement v1 and SLSA provenance v1-shaped records. It can attach an available NVML energy observation to a kernel invocation.

## Artifact and source identity

- Model-type card mirror: [reviewed tree](https://huggingface.co/SZLHOLDINGS/szl-provctl/tree/8d2082d4b9901b1b92c9efd86113fb760a64bf1e), revision `8d2082d4b9901b1b92c9efd86113fb760a64bf1e`
- First-class kernel package: [reviewed tree](https://huggingface.co/kernels/SZLHOLDINGS/szl-provctl/tree/424418b95e5d3403278c53189403e098662d4a3b), revision `424418b95e5d3403278c53189403e098662d4a3b`
- Declared canonical GitHub source: [observed source tree](https://github.com/szl-holdings/szl-provctl/tree/ce4a4cd36abba999de44b774a559b6fe33cf329e), revision `ce4a4cd36abba999de44b774a559b6fe33cf329e`
- Source paths: `torch-ext/szl_provctl/` and `build/torch-universal/szl_provctl/` in the canonical repository
- Inspection date: 2026-09-30 UTC

These are distinct repository revisions. The source repository is reachable; this review does not establish byte-for-byte parity, a reproducible build, approved publication, or the identity of a deployed process. Use the release's source binding and publication receipt to establish that chain.

The reviewed mirror contains Python source, build metadata and historical evidence records. Its listed tree contains no `model.joblib`, `.safetensors`, `.gguf`, `.onnx`, `.pt`, `.pth` or `.bin` model artifact. Historical surrogate descriptions in [MODEL_PROVENANCE.json](https://huggingface.co/SZLHOLDINGS/szl-provctl/blob/8d2082d4b9901b1b92c9efd86113fb760a64bf1e/MODEL_PROVENANCE.json) and [TRAINING_RECEIPT.json](https://huggingface.co/SZLHOLDINGS/szl-provctl/blob/8d2082d4b9901b1b92c9efd86113fb760a64bf1e/TRAINING_RECEIPT.json) do not establish present downloadable weights. Surrogate replay is blocked by the missing artifact. Do not load executable pickle/joblib artifacts from this card mirror.

## Intended use

Offline inspection of declared software-run relationships and experimental provenance record construction. Independently validate exported statements with the external verifier that will consume them.

## Public interface

UnifiedReceiptChain; ProvenanceDAG; verify_dag; statement_from_chain; slsa_statement; measure_kernel_energy; selfcheck

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
    "SZLHOLDINGS/szl-provctl", revision=revision, trust_remote_code=True
)
chain = kernel.UnifiedReceiptChain()
chain.emit("example", "inspect", {"record": "synthetic"})
dag = kernel.ProvenanceDAG()
dag.add_run("example", chain)
report = kernel.verify_dag(dag)
print(report)
```

The revision-format check is input validation only. It does not verify signatures, provenance, dependencies or runtime compatibility. A self-check exercises its programmed assertions; it is not product qualification.

## Evidence and evaluation limits

[BENCH.laptop-blackwell.json](https://huggingface.co/SZLHOLDINGS/szl-provctl/blob/8d2082d4b9901b1b92c9efd86113fb760a64bf1e/BENCH.laptop-blackwell.json), [TRAINING_RECEIPT.json](https://huggingface.co/SZLHOLDINGS/szl-provctl/blob/8d2082d4b9901b1b92c9efd86113fb760a64bf1e/TRAINING_RECEIPT.json), and [PROMOTION_READINESS_AUDIT.json](https://huggingface.co/SZLHOLDINGS/szl-provctl/blob/8d2082d4b9901b1b92c9efd86113fb760a64bf1e/PROMOTION_READINESS_AUDIT.json) retain owner-produced historical records. Each result applies only to its stated artifact, inputs, environment and date. This review did not rerun those tests, import the kernel, verify signatures, download weights, or benchmark throughput, energy or end-to-end behavior. Historical surrogate fidelity must not be presented as current kernel quality or current model availability.

## Limitations

Hash-chain consistency does not authenticate the producer, prove truthful inputs, or prove that the graph is complete. Producing SLSA-shaped JSON does not establish a SLSA level. External verifier interoperability was not exercised in this review. In the inspected Hub _dag.py, a missing dependency can raise KeyError during topological sorting before the declared missing_dependency result is reached; handle malformed inputs as failure, not success. Energy is unavailable when the measurement backend is unavailable.

Lambda uniqueness remains Conjecture 1, open and advisory. Receipt consistency, signatures and proof labels do not establish model accuracy, system safety, or certification. Preserve missing, failed, historical, and unverified states in downstream displays.

## License

The reviewed [LICENSE](https://huggingface.co/SZLHOLDINGS/szl-provctl/blob/8d2082d4b9901b1b92c9efd86113fb760a64bf1e/LICENSE) contains the Apache License 2.0 text, matching the card declaration. This observation does not independently establish rights to every artifact, training-data permission, or downstream-use clearance. Preserve upstream notices and check dependency licenses separately.

## Citation and verification

Cite the exact artifact URL, revision, evidence file, date, and test scope relevant to the claim. Prefer immutable links above over branch links or a live status badge. Report an evidence gap as an evidence gap.
