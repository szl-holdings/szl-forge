---
license: apache-2.0
base_model: Qwen/Qwen2.5-0.5B-Instruct
library_name: peft
base_model_relation: adapter
pipeline_tag: text-generation
tags:
  - sft
  - trl
  - hf_jobs
  - governed-ai
  - szl-holdings
  - doctrine-v11
  - identity
  - willay
---

<p align="center">
  <img src="https://raw.githubusercontent.com/szl-holdings/szl-forge/ad20ac2c24859156891c391108a86d4f05163f83/willay/card/holo-banner.svg" alt="WILLAY — holographic herald banner" width="100%"/>
</p>

# WILLAY

**Research fine-tune for SZL self-description. Historical qualification failed; deployment remains blocked in the reviewed status record.**

WILLAY's training objective is to produce SZL-focused descriptions. That objective does not establish reliable doctrine adherence, refusal behavior or factual accuracy. Obtain proof counts and artifact-signing claims from pinned authoritative records, rather than assuming model output is evidence. Conjecture 1 remains OPEN and advisory, not a theorem.

## Exact reviewed artifacts

The current reviewed Hub revision is [`e04bd728f20857c75e7c831d19d7190affd29133`](https://huggingface.co/SZLHOLDINGS/WILLAY/tree/e04bd728f20857c75e7c831d19d7190affd29133), inspected as listing and small metadata files on 2026-09-30 UTC. It contains both adapters and a derived merge; it is not an adapter-only repository without `config.json`.

| Artifact | Declared role and pairing |
|---|---|
| Root `adapter_model.safetensors` + `adapter_config.json` | Qwen/Qwen2.5-0.5B-Instruct adapter; rank 16, alpha 32; q_proj and v_proj |
| `adapter-unsloth/adapter_model.safetensors` + its config | Separate adapter on unsloth/qwen2.5-0.5b-instruct-unsloth-bnb-4bit; rank 8, alpha 16; not interchangeable with the root adapter |
| `model.safetensors` + `config.json` | Reported float32 merge; the current Hub card declares root-adapter lineage |
| `willay.npz` | Synthetic NumPy reference fixture, not the language-model checkpoint |

The card-level `base_model` describes the root adapter. Each adapter must be paired with its own declared base and qualified runtime. Listing these files does not verify their weight bytes. Weight bytes were not downloaded, rehashed or loaded by this review.

## Dated evaluation evidence and limits

The [September 15 post-publication receipt](https://github.com/szl-holdings/szl-forge/blob/ad20ac2c24859156891c391108a86d4f05163f83/frontier/evaluation/gate_runs/WILLAY_375a007ec961_20260915T040944Z_receipt.json) binds `SZLHOLDINGS/WILLAY@375a007ec961f125ab051747b70c3e3b7dcd64c7`, observed at `2026-09-15T04:10:28.062787+00:00`. It records **0 of 5 held-out cases passed**, `heldout_passed=false`, refusal regression **0.0000 versus baseline 0.4000**, `refusalPassed=false`, `status=FAIL`, `publicationAuthorization=false` and `productionAuthorization=false`. The paired [held-out file](https://github.com/szl-holdings/szl-forge/blob/ad20ac2c24859156891c391108a86d4f05163f83/frontier/evaluation/gate_runs/WILLAY_375a007ec961_20260915T040944Z_heldout.json) matches the receipt's recorded SHA-256 `2045850731fac0ab454d6e9c5b6b96e8bf9c173a3dc8479435cb846c24bf432b`.

This is historical, revision-bound negative evidence, not a reproduced run in this review. The earlier September 15 receipt also records a failed run, including unavailable refusal execution; it is retained in the source tree. The current `e04bd7...` checkpoint was not established byte-equivalent to the failed `375a00...` revision, and no qualification of its behavior is established here. Do not transfer that historical score to current weights or describe this review as a new benchmark. The card must not claim that no evaluation ever existed.

The five held-out cases are the receipt's scope; they do not establish general accuracy, safety, refusal reliability or production readiness. No energy measurement or runtime/performance improvement is claimed.

## Merge lineage and reproducibility

The current [merge receipt](https://huggingface.co/SZLHOLDINGS/WILLAY/blob/e04bd728f20857c75e7c831d19d7190affd29133/merge_receipt.json) records SHA-256 `aa15071b6041fd4bb829009b463ef6351bd1213fc2dbe911484ae5c5528c82cb` for `model.safetensors`, with `merge_dtype=float32`. This is a reported binding, not a weight digest independently reproduced by this review. The receipt does not identify or hash the input adapter, so the [Hub card's root-adapter lineage declaration](https://huggingface.co/SZLHOLDINGS/WILLAY/blob/e04bd728f20857c75e7c831d19d7190affd29133/README.md) is not independently established here. The reviewed status still blocks artifact lineage. The receipt is unsigned and names its base as `main@merge-time`, not an immutable base commit. Both adapter configs also leave their base revision null. These gaps limit exact reconstruction and producer authentication.

## Research usage prerequisites

Select the root adapter, Unsloth adapter or merge explicitly. Before inference, establish the matching immutable base identity, verify the intended artifact bytes, and record the adapter/model revision, tokenizer and chat-template identity, runtime versions, dtype and device. Loading a different base can change the intended model. The floating-base inference example formerly on this card is withheld until those identities and runtime qualification are available; this card provides no tested inference recipe or authorization to deploy.

## Source authority, license and release limits

The authoring source is [`szl-holdings/szl-forge/willay/card/README.md`](https://github.com/szl-holdings/szl-forge/blob/ad20ac2c24859156891c391108a86d4f05163f83/willay/card/README.md), owned by `@stephenlutar2-hash`. This identifies the authoring path, not a build attestation or automatic publisher. WILLAY is not admitted to the source's closed card-mirror profile registry. An automatic writer for this authoring file was not established; source correction and Hub publication remain separate actions.

Apache-2.0 is declared and a standalone [LICENSE](https://huggingface.co/SZLHOLDINGS/WILLAY/blob/e04bd728f20857c75e7c831d19d7190affd29133/LICENSE) is listed. The reviewed [status record](https://huggingface.co/SZLHOLDINGS/WILLAY/blob/e04bd728f20857c75e7c831d19d7190affd29133/status.json) still blocks artifact lineage, consent, privacy, training suitability and deployment, with `production_ready=false`. [provenance.json](https://huggingface.co/SZLHOLDINGS/WILLAY/blob/e04bd728f20857c75e7c831d19d7190affd29133/provenance.json) says ownership and relicensing authority were not independently established by license-file publication. Preserve those blocks; this card grants no rights clearance or publication/production promotion.

Copyright 2026 SZL Holdings. Stephen P. Lutar Jr. ORCID [0009-0001-0110-4173](https://orcid.org/0009-0001-0110-4173).
