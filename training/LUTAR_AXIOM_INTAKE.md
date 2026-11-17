# Lutar A1–A5 → SZL Forge SFT: immutable-source intake

**2026-11-17 · Stage: SOURCE_BOUND_REVIEW_REQUIRED · Not a training launch.**

The canonical 160-train / 80-developer-diagnostic-holdout chat corpus lives in
[`szl-holdings/lean-rlvr-proving-ground`](https://github.com/szl-holdings/lean-rlvr-proving-ground/tree/b23ea5fe7e63d6b7402ce3a6f0a6aa5e1fa56b2c/curriculum)
at protected main commit `b23ea5fe7e63d6b7402ce3a6f0a6aa5e1fa56b2c`.
The upstream PR #10 was green on both push and pull_request checks and merged
through normal branch policy. Both datasets are **first-party synthetic arithmetic
fixtures**; they are not Lean-checked proof terms, new discoveries, or an
independently authored generalization holdout.

The math source is `szl-holdings/lutar-lean@c2ca813fca00ecf58bcc5f505f2aa4237fbf2481`
with precisely pinned `Lutar/Axioms.lean` and `Lutar/Invariant.lean` blobs.
The train examples cover A1 monotonicity, A2 positive homogeneity, A3
normalization, A4 maximum bound, A5 equal-weight permutation invariance,
conditional admission, zero-axis behavior, **max-aggregator counterexamples to
unconditional uniqueness**, and unequal-weight counterexamples to naive A5.
The unconditional Lambda uniqueness conjecture remains false under just A1–A5.

## Source-to-training wiring

`training/lutar_axioms.source.v1.json` is the Forge-owned immutable source
index. Its train, holdout, and manifest hashes bind the producer source.
`training/lutar_axiom_intake.py` reads **Git object bytes at that exact
commit only** from an already-available local Git checkout. It never uses a
moving branch, silently downloads a model, consults or prints credentials,
allocates a GPU, or writes to the Hub.

```sh
# Precondition: a previously reviewed local checkout has the pinned commit object.
python training/lutar_axiom_intake.py \
  --producer-checkout /path/to/lean-rlvr-proving-ground \
  --out /path/to/a/new/local/lutar-sft-review
```

The importer verifies every byte against the reviewed index, exact source
claim, mandatory `training_authority: NONE`, row identities, and 32-example
floor. It invokes **the existing** `tools/validate_sft_dataset.py` for
conversation structure, duplicates, and train/holdout prompt separation.
A new directory is created only after all checks pass; existing output is
never overwritten. It writes candidate `train.jsonl`, `holdout.jsonl`,
`manifest.json`, and `source-receipt.json`.

The output `SOURCE_BOUND_REVIEW_REQUIRED` means that the input bytes,
structure, and developer-split distinction were reproduced; it **does not**
mean the training data was independently reviewed, that its source digest
authenticates an author, or that it is safe to promote to model weights.
The internal numerical reward oracle cannot replace `referee.py`'s Lean
kernel lane or an independent proof checker.

## Admission gates before a real GPU fine-tune

1. Independent data/claims review with rights, privacy and attribution
   checked. A signed operator-controlled `szl.forge.training-admission/v1`
   receipt must bind the **actual admitted dataset's** immutable Hub revision
   and exact train digest. This importer's receipt does not issue one.
2. Confirm an exact permitted student model commit, `kind=sft`, local-only
   output, the existing Forge `launch_contract.py`, 32+ admitted examples,
   a separately sealed held-out corpus, an exclusive GPU lease, maximum cost,
   and a model-family-compatible QLoRA/LoRA hyperparameter plan.
3. Preflight the bounded run on the GPU owner, then execute existing
   `training/train_wave.py --config ...`. Preserve seed, corpus digests,
   checkpoints, training receipt, actual losses, and real energy evidence or
   `null` where unavailable. No direct Hugging Face publication.
4. After independent evaluation, use the canonical model release gate,
   signatures, and post-deployment exact-source readback. Neither successful
   local training nor a new HF model card alone proves production readiness.

**No additional trainer, publisher, bypass, new secret, or paid job is
introduced.** The review candidate is now located on Forge's real, already
validated SFT input path, but remains barred from GPU execution until the
remaining owner-controlled admissions are satisfied.

## Optional private Hugging Face dataset publication

The canonical, GitHub-sourced corpus can now be published through the
**manual, one-shot** `training/publish_lutar_dataset.py` path. The publisher
never moves source authority from GitHub and requires the immutable producer
commit `b23ea5fe7e63d6b7402ce3a6f0a6aa5e1fa56b2c` to exist locally.

```sh
# Network-free source/holdout validation; no credential used.
python3 -m training.publish_lutar_dataset \
  --producer-checkout /path/to/lean-rlvr-proving-ground

# Explicitly authorized PRIVATE dataset creation only, never public.
python3 -m training.publish_lutar_dataset \
  --producer-checkout /path/to/lean-rlvr-proving-ground \
  --publish-private --receipt /path/to/a/new/local/publish-receipt.json
```

Target repo is fixed to `SZLHOLDINGS/lutar-axioms-sft-v1` (private).
The publisher refuses to overwrite an existing dataset, validates the existing
Forge SFT holdout contract, uploads all five versioned artifacts and then
downloads every artifact at the exact returned Hub commit before issuing
a receipt. The receipt is a **byte-parity witness, not an admission, a
cryptographic signature, or training permission**. Partial upload/auth errors
require operator investigation; the program never silently retries a new
revision or downgrades privacy. Do not distribute the card or modify
the private repository's visibility without separate authorization.


## Distinct from the existing Flagship55 axiom corpus

The existing [Flagship55 curriculum specification](../docs/frontier-20261115/flagship55-curriculum-spec.md)
already pins `SZLHOLDINGS/axiom-invariant-corpus-v2` (94 declaration
records at Lutar source `c2ca813`), with a separate 93-row SFT mirror.
That tier is **verbatim statement context** and stays in its own admission
scope. This 160-row synthetic numerical A1–A5 exercise set is **additional**,
not a replacement, alias, or approved Flagship55 training mixture. Do not
count the two datasets together, claim one independent of the other, or
change flagship training without its owner-controlled provenance/heldout
and budget review. The private Hub publisher introduced here does not
touch either existing dataset.
