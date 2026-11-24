# szl-forge domain leader study r1 (propose-only)

Date: 2026-11-24. Method: web-verified leader landscape + governing-seat ratification (Kimi nonce T6-RULING-20261124-K1). Study pinned to szl-forge HEAD 9f5bc9c02a4b55ab706e1ea5f52a7260e3383aa2, fetched 2026-11-24. This card is documentation only; it changes no code, CI, config, or existing PR status. All absorb actions are FUTURE PROPOSALS (non-authorizing); implementation and merges remain separately reviewed and gated. Λ Conjecture 1: OPEN (advisory, never a closed theorem). No speedup, energy, or proven-trust claims; upstream performance claims are labeled as upstream claims only.

## Domain: estate model factory - QLoRA kit that fine-tunes an open base model into SZL-1 on estate-owned hardware

## Leaders studied (web-verified 2026-11-24)
1. QLoRA (https://arxiv.org/abs/2305.14314) - moat: 4-bit NormalFloat (NF4) quantization with double quantization and paged optimizers; the upstream paper claims 65B finetuning on a single 48GB GPU with downstream performance matching 16-bit finetuning (upstream paper claim, not estate-verified).
2. Unsloth (https://github.com/unslothai/unsloth) - moat: open-source finetuning library with optimized QLoRA kernels; upstream claims up to 2x faster finetuning and lower memory usage on supported GPUs (upstream claim, not estate-verified).

## Absorb actions (FUTURE PROPOSALS, non-authorizing)
1. (QLoRA) Propose a documentation-only mapping of szl-forge training waves to the QLoRA NF4/double-quantization math, with each wave's quantization settings recorded as observable-training receipts and proof-banner publication kept mandatory. Cost: one documentation pass plus receipt-template review; limit: no code, config, dataset, or training-run changes, and no claim that upstream 65B/48GB or 16-bit-parity results transfer to SZL-1.
2. (Unsloth) Propose honest-labeling discipline: any future benchmark comparing estate training runs against upstream speed/memory claims must carry measured receipts, hardware and base-model identifiers, and explicit separation of upstream claims from estate observations. Cost: benchmark-receipt schema and reviewer checklist drafting only; limit: no benchmark is run or implied here, no kernel adoption is authorized, and no unverified speedup or memory-saving statement may be published.
3. (Baseline) Propose preserving append-only receipts, head-sha pinning, dedupe against landed cards, and stop-on-duplicate/conflict behavior for every later study wave. Cost: documentation and review time only; limit: this card authorizes no merge, implementation, integration, deployment, signature-wall bypass, or change to excluded lanes.

## Do not do (per governing seat)

Do not merge, bypass the signature wall, or claim implementation beyond this card.
