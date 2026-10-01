---
license: apache-2.0
language: [en]
base_model: SZLHOLDINGS/SZL-Khipu-1.5B
base_model_relation: quantized
pipeline_tag: text-generation
library_name: llama.cpp
tags: [gguf, llama.cpp, research, proposal-only, retrieval]
---

<p align="center"><img src="https://raw.githubusercontent.com/szl-holdings/szl-forge/main/khipu-gguf/card/holo-banner.svg" alt="Khipu GGUF research artifact illustration" width="100%"/></p>

# Khipu 1.5B · GGUF

**Compact, proposal-only retrieval plans over supplied handles**

**Derived quantizations · limited research/demo scope · production promotion blocked**

GGUF derivatives of `SZLHOLDINGS/SZL-Khipu-1.5B`, itself a fine-tune of `Qwen/Qwen2.5-1.5B-Instruct`. The model proposes a JSON retrieval plan over candidate handles supplied by a controller. It does not contain or authorize access to a private graph, execute the plan, or issue trusted receipts.

## Artifact identity

Reviewed Hub revision: [`7c39154b22ccb5e2151b4dd53e0d36965dfbc920`](https://huggingface.co/SZLHOLDINGS/SZL-Khipu-1.5B-GGUF/tree/7c39154b22ccb5e2151b4dd53e0d36965dfbc920), inspected 2026-09-30 UTC.

The following exact file identities are recorded in [publication.json](https://huggingface.co/SZLHOLDINGS/SZL-Khipu-1.5B-GGUF/blob/7c39154b22ccb5e2151b4dd53e0d36965dfbc920/publication.json). File presence was checked against the reviewed tree; this review did not download or independently hash the GGUF bytes.

| File | Bytes recorded | SHA-256 recorded |
|---|---:|---|
| `SZL-Khipu-1.5B-F16.gguf` | 3093668768 | `2348ee342efe639e100f3fb31a3dc11b8c12d8c43ecfe45e18041b9f94c71a12` |
| `SZL-Khipu-1.5B-Q4_K_M.gguf` | 986047904 | `13c1a1993063e1dff92f7413ccf48eaca6d48efc8801ae9af35961ae3396623a` |
| `SZL-Khipu-1.5B-Q5_K_M.gguf` | 1125049760 | `3bf460ac163c5dc952c273999c38a41349e3e6d666e4b713aed22c996860fd4c` |
| `SZL-Khipu-1.5B-Q8_0.gguf` | 1646572448 | `6aff1087f64631679f4cdf032613aee6911dbde38cd3bac6b81bf63741a56f0d` |

File size does not equal peak runtime memory. This card makes no ranking of quantization quality, speed or energy efficiency because no comparative post-quantization evaluation was established.

## Evidence scope

The repository carries owner-signed training/evaluation receipts from the pre-quantized model and a repository-declared public key. The publication record reports that these signatures validated against that key; this review did not repeat signature verification. These are Ed25519 signatures over canonical JSON, not DSSE envelopes. External signer identity is not independently established.

The recorded pre-quantized held-out evaluation has plan-validity 11/11, grounding 4/5 and abstention **2/6**. Those results belong to the evaluated artifact described in the receipt. They do not establish behavioral performance for any of these GGUF files. Keep the abstention failure visible.

An unsigned publication record also describes a historical best-effort CPU service probe with two runs and a median latency of 1467.5 ms. That tiny service observation is not a current uptime assertion, an SLA, a benchmark, restart reproducibility, or a post-quantization quality test. Instrumented energy and process-peak memory are unavailable in that record. Current service operation was not tested in this card review.

## Source and lineage

The source of this mirror's card and banner is the separate Forge publisher input
`khipu-gguf/card/README.md`. A card-only correction does not attest the GGUF
build, publish weights, or qualify a runtime.


[publication.json](https://huggingface.co/SZLHOLDINGS/SZL-Khipu-1.5B-GGUF/blob/7c39154b22ccb5e2151b4dd53e0d36965dfbc920/publication.json) identifies the source curriculum, schema and receipt tree at [`szl-forge@cddba1ad9887211d6fd87386e98bb6707f16dbf5/khipu`](https://github.com/szl-holdings/szl-forge/tree/cddba1ad9887211d6fd87386e98bb6707f16dbf5/khipu). It records upstream base revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` and fine-tuned parent revision `759c6112b1acae67f30351ef4e652e07671a42fb`.

This identifies a source relationship; it does not establish a reproducible quantization build. No owner-signed release receipt covering these GGUF bytes is reported. A parent-model signature does not cover a derived GGUF unless that exact file digest is included in its signed subject.

## Bounded usage

Use a reviewed, pinned llama.cpp-compatible runtime and the exact selected file. The prompt contract is [khipu.schema.json](https://huggingface.co/SZLHOLDINGS/SZL-Khipu-1.5B-GGUF/blob/7c39154b22ccb5e2151b4dd53e0d36965dfbc920/khipu.schema.json): a query plus candidate handles, followed by a proposed NAVIGATE or ABSTAIN plan. Validate output schema and handle membership independently. The controller retains access, approval and execution authority.

An immutable artifact retrieval example, not an inference or deployment test:

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="SZLHOLDINGS/SZL-Khipu-1.5B-GGUF",
    filename="SZL-Khipu-1.5B-Q4_K_M.gguf",
    revision="7c39154b22ccb5e2151b4dd53e0d36965dfbc920",
)
print(path)
```

Retrieval may consume approximately 986 MB of storage for the selected file plus cache overhead. Hash-check against the reviewed evidence before use. This example was not executed in the review; confirm disk and runtime memory headroom before running it. A downloaded file is not a validated model.

## Limitations and license

- Proposal-only, bounded research/demo use; no autonomous or high-stakes operation
- No post-quantization quality result, independent certification, quantified energy result or reproducible-build claim
- Runtime outputs are unsigned unless separately signed by an authorized external component
- Quantized numerics can differ; do not transfer parent results without evaluation
- Apache-2.0 is declared and [LICENSE](https://huggingface.co/SZLHOLDINGS/SZL-Khipu-1.5B-GGUF/blob/7c39154b22ccb5e2151b4dd53e0d36965dfbc920/LICENSE) is present; preserve base-model and dependency notices
- Lambda remains an open advisory conjecture; this card claims no SLSA level, government approval, ATO, or blanket trust percentage
