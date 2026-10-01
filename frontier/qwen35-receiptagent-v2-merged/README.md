# qwen35-receiptagent-v2-merged — merged full-model SKU lane

Canonical admission lane for the first merged full-model SKU in the estate:
`SZLHOLDINGS/szl-receiptagent-qwen35-0.8b-v2-merged` (published research artifact).

The [2026-09-24 publication followup](publication-followup.20260924.json)
records an immutable Hub history readback. The first observed commit containing
the six model files is `138a1c78eaa8553f12abf2c86387571be6861882`; the card was
added in `6f7dc8e2215b689aa2b7d2a28d794d29f5ba3079`. Current main at observation
was `b770792751f00069fb042e04ee60f28f9faf5ebf`. All six model file identities
are unchanged across those revisions. The safetensors SHA-256 from Hub LFS
metadata is `3435ba19eafc0bc0781067f31e8192ff84a5b21d307aca89573af880a30feb68`.
Large weights were not downloaded for this readback.

The original [publication receipt](publication.json) remains unchanged. This
followup supplies the missing observed revision identities; it does not
retroactively bind or rerun the historical 5/5 local equivalence evaluation.
Production remains HOLD and promotion remains NONE.

## What this is

A single-repo, no-adapter-required build of the v2 ReceiptAgent: the
frozen v2 LoRA adapter merged into the correct multimodal base
(`unsloth/Qwen3.5-0.8B`, `language_model.linear_attn` architecture) via an
architecture-correct, fail-closed merge.

## Why the merge needed salvaging

The v2 adapter was trained via Unsloth FastVisionModel on the multimodal
wrapper; its 372 LoRA keys target `model.language_model.layers.N.linear_attn.*`
without the `.default` segment. A merge that loads the text-only base silently
drops every key and produces a byte-copy of base. The 2026-09-12 frontier delta
already excluded that empty merge from publication; this lane replaces it with
a verified merge.

## Evidence chain (all receipts committed on main)

1. Key map: 372 checkpoint keys → 372 live modules, 0 unmatched, 0 missing
   (`receiptagent-v2/salvage_fix.20260913.log`)
2. Merge: max weight delta 0.00422266 across 186 tensors, 852,985,920 params,
   load-verified (`receiptagent-v2/salvage_receipt.20260913.json`, PR #306)
3. Bounded local equivalence: 5/5 (100%) exact-match behavioral equivalence vs the
   adapter-on-multimodal-base on real chat-template eval prompts
   (`receiptagent-v2/qual_v2_salvage_receipt.20260913.json`, PR #307)

## What this lane does NOT do

- Does not overwrite `SZLHOLDINGS/szl-receiptagent-qwen35-0.8b-v2` — that repo
  is the FROZEN COMPARATOR pinned by the v3 candidate, model-source-bindings,
  local-compute verification, portfolio, and tournament rosters
- Does not republish or overwrite weights; the existing ~3.4 GB merged
  safetensors publication is observed in the separate immutable followup
- Grants no production route, default, or inference authority
