# chaski/frontier-op1 — frontier op1 lane evidence (mirrored receipts)

Coding-back per the owner directive of 2026-11-24: audit all SZLHOLDINGS models and code them back into the
szl-holdings org (front end cards, back end pure Python). This directory collects the op1 frontier-lane run
evidence mirrored byte-exact from the sibling sovereign/dream training lane that owns these model repos.

## Lane models (SZLHOLDINGS, private; census 2026-11-24)

- salad-chaski-frontier-4b-op1 / salad-chaski-frontier-9b-op1
- salad-chaski-frontier-9b-op1-r2 / -r3 / -r9 / -r11 / -r12
- salad-chaski-frontier-4b-narrow1 / -9b-narrow1 (mid-publish at audit time, cards pending)
- salad-chaski-preflight
- trained on SZLHOLDINGS/szl-frontier-openprob-mix-v1 (pinned in receipts); LoRA q/k/v/o r=32 over Qwen bases per card

## Mirrored here (byte-exact, fetched 2026-11-24)

- receipts/4b-op1.train_receipt.json (3471 B)
- receipts/9b-op1-r11.train_receipt.json (3108 B)
- logs/9b-op1-r11.train_log.txt

## Honest boundaries

- The training driver for this lane is NOT mirrored here: it lives with the owning sovereign/dream lane (device/Salad side).
  The general chaski trainer lineage in this repo (chaski/train_chaski.py and family) may or may not be the exact producer
  of these runs; no identity is claimed without a receipt that names this path. Fail-closed.
- Salad evidence lanes remain parked (replicas 0) per owner order; nothing here unparks, deploys, or authorizes serving.
- Registry note: these models are recorded unmapped in szl-asset-registry data/intake_new_models_20261124.json (PR #34,
  merged 2026-11-24). Canonical-source adjudication stays with the owning tools + owner.
- Sibling repos were not modified; only additive files under chaski/frontier-op1/.

Audit receipt: SZLHOLDINGS/wave-logs mesh/2026-11-24/ouroboros/hf-audit-20261124/ (commit be25681).