# qwen35-receiptagent-v2-merged — merged full-model SKU lane

Canonical admission lane for the first merged full-model SKU in the estate:
`SZLHOLDINGS/szl-receiptagent-qwen35-0.8b-v2-merged` (not yet published).

## What this is

A single-repo, no-adapter-required build of the qualified v2 ReceiptAgent: the
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
3. Qualification: 5/5 (100%) exact-match behavioral equivalence vs the
   adapter-on-multimodal-base on real chat-template eval prompts
   (`receiptagent-v2/qual_v2_salvage_receipt.20260913.json`, PR #307)

## What this lane does NOT do

- Does not overwrite `SZLHOLDINGS/szl-receiptagent-qwen35-0.8b-v2` — that repo
  is the FROZEN COMPARATOR pinned by the v3 candidate, model-source-bindings,
  local-compute verification, portfolio, and tournament rosters
- Does not upload weights: the ~3.4 GB merged safetensors remain local until
  the canonical publisher executes publication after this lane is admitted
- Grants no production route, default, or inference authority
