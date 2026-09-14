---
library_name: transformers
license: apache-2.0
base_model: unsloth/Qwen3.5-0.8B
pipeline_tag: text-generation
---

# szl-receiptagent-qwen35-0.8b-v2-merged

**Status: QUALIFIED, NOT YET PUBLISHED** — this card is the admission draft;
the repository goes live only through the canonical publisher.

Merged full-model build of the governed, proposal-only ReceiptAgent v2: the
qualified LoRA adapter merged into the correct multimodal base, so it runs
without PEFT.

## Evidence

- Merge key map: 372/372 keys matched, 0 dropped (the original broken merge
  dropped all 372 and was excluded from publication before ever shipping)
- Max weight delta vs base: 0.00422 across 186 tensors — a real merge, not a
  byte-copy
- Behavioral equivalence: 100% exact-match vs the adapter on real eval prompts
  under greedy decoding

## Boundary

Proposal-only. Every output is an evidence-bound, approval-gated decision
DRAFT (`decision=DRAFT`, `approvalRequired=true`, `executed=false`). The model
never finalizes, never executes, and never fabricates citations or receipts.

## Predecessor

Frozen comparator: `SZLHOLDINGS/szl-receiptagent-qwen35-0.8b-v2`
(adapter SHA-256 `885fc29f...`). This SKU does not replace it; it is derived
from it.
