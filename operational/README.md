# operational/ — SZL Forge operational training & remediation package

**Control before action. Evidence after.** Doctrine v11 LOCKED. Λ = Conjecture 1.

This package is the operational layer that sits on top of the existing lane
scripts (`chakana/`, `tinku/`, `chaski_r2/`, `khipu_r2/`, `receiptagent/`,
`willay/`). It does not replace them. It:

1. Builds the controlled, source-bound candidate corpus from canonical repo
   documents (`build_curated_datasets.py`).
2. Drives the lane trainers in smoke mode with explicit, local, admitted
   files (`train_chakana.py`, `train_tinku.py`).
3. Computes base-vs-candidate held-out metrics and writes receipts
   (`evaluate_retrieval.py`, `evaluate_reranker.py`).
4. Builds remediation curricula for known model failures
   (`remediate_chaski.py`, `remediate_khipu_r3.py`).
5. Runs sealed / doctrine evaluation harnesses that fail closed to
   `UNKNOWN` when their sealed inputs are absent
   (`evaluate_receiptagent_v3.py`, `evaluate_willay.py`).
6. Orchestrates the whole pass and writes one provenance receipt
   (`run_all.py`).

## Hard rules (non-negotiable)

- **No Hub PUT.** Nothing here uploads weights, datasets, or cards.
- **No third-party benchmark data.** MTEB/BEIR/MS MARCO and similar names are
  refused as training sources. Held-out numbers come only from in-house
  frozen slices; absent slice ⇒ metric is `UNKNOWN`, never pasted.
- **Admitted bases only.** Chakana: `Qwen/Qwen3-Embedding-0.6B` (Apache-2.0)
  or `BAAI/bge-m3` (MIT). Tinku: `Qwen/Qwen3-Reranker-0.6B`. KaLM and
  EmbeddingGemma are refused.
- **`publication_eligible` stays `false`** until a MEASURED in-house held-out
  nDCG@10 (retrieval) or equivalent sealed result exists and is receipted.
- Remediation outputs are `CANDIDATE_REQUIRES_REVIEW`. They are curricula,
  not trained models.
- Default mode is **smoke**: no GPU required, no job fire, tiny limits.
  Pass `--full` / `--train` only on provisioned hardware.

## Layout

```
operational/
  README.md                      this file
  requirements.txt               pinned-enough deps for the operational layer
  build_curated_datasets.py      controlled corpus builder (cross-platform)
  train_chakana.py               drives chakana/train_chakana.py --train
  train_tinku.py                 smoke CrossEncoder trainer (admitted triples)
  evaluate_retrieval.py          base-vs-candidate MRR/Recall/nDCG@10
  evaluate_reranker.py           base-vs-candidate MAP/MRR@10/nDCG@10
  evaluate_receiptagent_v3.py    sealed-test runner (--unseal required)
  evaluate_willay.py             doctrine suite runner (fails closed UNKNOWN)
  remediate_chaski.py            JSON-schema + refusal-contract curriculum
  remediate_khipu_r3.py          abstention + near-miss curriculum
  run_all.py                     orchestrator + run receipt
  out/                           generated artifacts (git-ignored)
```

## Quick start (smoke, no GPU)

```bash
python -m pip install -r operational/requirements.txt
python operational/run_all.py            # build + UNKNOWN-eval receipts
python operational/run_all.py --train    # additionally fire lane trainers
```

Outputs land in `operational/out/` and one `run-receipt.json` summarizes
every step with per-step `status` and `evidence` pointers.
