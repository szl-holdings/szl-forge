---
license: apache-2.0
language:
  - en
pipeline_tag: feature-extraction
library_name: sentence-transformers
base_model: Qwen/Qwen3-Embedding-0.6B
tags:
  - szl-holdings
  - series-a
  - doctrine-v11
  - governed-ai
  - feature-extraction
  - sentence-transformers
  - roadmap
szl:
  doctrine: v11-LOCKED
  lean: "749/14/163"
  lambda: "Conjecture 1 — advisory, never a theorem"
  evidence_ceiling: 0.97
  artifact_class: ENCODER
  originality: FINETUNE_DISCLOSED_BASE
  collection: "SZL Fall 2026 — Original Cuts"
  owner: Stephen Lutar
  lane: "NINA (FORGE-class)"
  jobs: UNKNOWN
  weights: UNAVAILABLE
  evals: UNKNOWN
  ndcg10: UNKNOWN
  publication_eligible: false
  not_a11oy_chakana_wiring: true
---

# Chakana

**One line.** Bridge embeddings for szl-lake / doctrine / killinchu text retrieval.

| | |
|---|---|
| **Artifact** | GitHub recipe. Hub [`SZLHOLDINGS/chakana`](https://huggingface.co/SZLHOLDINGS/chakana) is a stub (README + `.gitattributes`). **No weights.** |
| **Originality** | Intended SZL fine-tune of a disclosed Apache embedding base. Cut is original SZL pairs. We do not republish Qwen tensors. |
| **Base** | `Qwen/Qwen3-Embedding-0.6B` (Apache-2.0). Alternate MIT: `BAAI/bge-m3`. Never KaLM. Never EmbeddingGemma. |
| **Library** | `sentence-transformers` · `feature-extraction` |
| **Owner / lane** | Stephen Lutar · NINA (FORGE-class) |
| **HF Jobs** | **UNKNOWN** — no job id in this checkout |
| **Eval** | Frozen in-house **nDCG@10**. Status **UNKNOWN** (held-out not run). |
| **License** | `apache-2.0` |
| **Doctrine** | v11 LOCKED (749 / 14 / 163). Λ = Conjecture 1. Evidence ceiling 0.97. |

> **Fashion rule.** Silhouette from Qwen3-Embedding / MTEB leaders. Cut is original SZL pairs. We do not republish someone else's tensors and we do not paste leaderboard numbers.

Forge trainer: [`chakana/train_chakana.py`](./train_chakana.py). sentence-transformers Matryoshka 256/512/1024 over MultipleNegativesRankingLoss. Seed 11.

**Not** a11oy CHAKANA wiring / tinkuy (Andean-cross topology under a11oy organs). **Not** MiniEmbed in `szl-kernels`.

## Intended use

- **Who:** szl-lake / doctrine / killinchu retrieval behind a controller
- **What:** `query → vector`, cosine retrieve, optional Matryoshka truncate 256/512/1024
- **Where:** after admitted SZL pairs exist at `SZLHOLDINGS/chakana-pairs` (or a local jsonl of those pairs)

## What it is NOT

- Not a loadable fine-tune today. Do not `from_pretrained("SZLHOLDINGS/chakana")` expecting weights.
- Not a Qwen3-Embedding rehost.
- Not a11oy CHAKANA wiring / tinkuy.
- Not MiniEmbed.
- Not an autonomous agent.

## Evaluation

**Status: UNKNOWN.** Protocol when a job and a held-out SZL pair slice exist: frozen in-house nDCG@10. If we have not run that slice, this card stays UNKNOWN. No MTEB paste.

## Training

- Recipe: sentence-transformers. Script: `train_chakana.py`.
- Data: rights-admitted SZL query-positive pairs only. Default dataset id `SZLHOLDINGS/chakana-pairs` (does not exist on Hub as of this checkout). `szl-lake` is receipts; `rag-corpus-v1` is a BGE-indexed corpus, not pairs. Third-party MTEB/BEIR is refused.
- Jobs: UNKNOWN. Launcher prints `hf jobs uv run --secrets HF_TOKEN` and **refuses to fire**.
- Trackio: planned when `HF_TOKEN` is present. No dashboard URL until a job exists.
- GitHub stamp only. Hub README is not recut from this checkout.

## Limitations

- No admitted pair file in this repository. No encoder bytes. nDCG@10 UNKNOWN. `publication_eligible: false`. Λ = Conjecture 1. Trust ceiling 0.97.
