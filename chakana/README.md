# Chakana — SZLHOLDINGS/chakana

**Bridge embeddings. ROADMAP. Jobs UNKNOWN. No weights.**

NINA (FORGE-class) owns this embedding lane. Owner: Stephen Lutar.

This kit is query→vector for szl-lake / doctrine / killinchu text retrieval,
with cosine retrieve and optional Matryoshka 256/512/1024. Hub
[`SZLHOLDINGS/chakana`](https://huggingface.co/SZLHOLDINGS/chakana) is a stub
(README + `.gitattributes` only). This checkout does not PUT Hub and does
not fire a Job.

| | |
|---|---|
| **SKU** | `SZLHOLDINGS/chakana` |
| **Base** | `Qwen/Qwen3-Embedding-0.6B` (Apache-2.0). Alternate MIT: `BAAI/bge-m3` |
| **Library** | sentence-transformers · feature-extraction |
| **Status** | **ROADMAP** |
| **HF Jobs** | **UNKNOWN** — this PR does not fire a job and does not invent an id |
| **Eval** | frozen in-house nDCG@10 · **UNKNOWN** (held-out not run) |
| **publication_eligible** | **false** |
| **Hub PUT** | **false** |
| **Doctrine** | v11 LOCKED (749 / 14 / 163). Λ = Conjecture 1. Evidence ceiling 0.97 |
| **Seed** | 11 |

> **Fashion rule.** Silhouette from Qwen3-Embedding. Cut is original SZL pairs.
> We do not republish Qwen tensors. We do not paste MTEB numbers.

## Disambiguation

**a11oy CHAKANA wiring / tinkuy is not this model.** That path
(`a11oy/organs/amaru/**/chakana_wiring.py`) is Andean-cross topology wiring.
Do not treat it as ALIGNED with this encoder. Do not edit it from this kit.
MiniEmbed in `szl-kernels` is a different, smaller artifact.

## What this is

A first-class sentence-transformers recipe: status by default, `--train` on
owner metal or HF Jobs, honest eval stamp, Python serve, unsigned receipt
stubs, and a Jobs launcher that **refuses to fire**.

Training data is rights-admitted SZL query-positive pairs only. Default
dataset id is `SZLHOLDINGS/chakana-pairs` (absent on Hub as of this checkout).
`szl-lake` is a receipt lake, not pairs. `rag-corpus-v1` is a BGE-indexed
chunk corpus, not query-positive pairs. Third-party MTEB/BEIR is refused as
if it were SZL.

## What this is NOT

- Not a trained encoder and not a Hub weight drop
- Not a Qwen3-Embedding rehost
- Not a11oy CHAKANA wiring / tinkuy
- Not MiniEmbed
- Not a publication of nDCG@10 or any MTEB score
- Not a Hub overwrite or Jobs launch from this checkout

## Commands (local, no Hub write)

```bash
python chakana/train_chakana.py
python chakana/eval_chakana.py
python chakana/serve_chakana.py --check
python chakana/jobs/launch_chakana_job.py
python chakana/sign_receipt.py stub --check
```

`--train` runs sentence-transformers when admitted pairs exist (local jsonl
or `CHAKANA_PAIRS_DATASET`, default `SZLHOLDINGS/chakana-pairs`). Missing
pairs write `SKIP-NO-ADMITTED-PAIRS` and do not invent metrics. It still does
not PUT Hub and still leaves `publication_eligible` false.

`--run-job` on the launcher exits 2 (`refusing`). The printed command is
Jobs-first:

```bash
hf jobs uv run --flavor a10g-large --timeout 2h --secrets HF_TOKEN \
  chakana/train_chakana.py --train
```

Trackio is planned when `HF_TOKEN` is present. No dashboard URL until a real
job exists.

Eval `--run` scores frozen in-house nDCG@10 only when a local encoder and
`held_out.jsonl` both exist. Otherwise status stays **UNKNOWN**.
