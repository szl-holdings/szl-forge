# HF Frontier Delta — 2026-09-14

## Estate publish

Seven standalone merged checkpoints published to huggingface.co/SZLHOLDINGS on 2026-09-12 (~22:40 EDT):
chaski_r2, chaski_5050, brain_navigator_r2, khipu_r3, willay, khipu_r2, szl-receiptagent-qwen35-0.8b-v3.
All training and merging done locally on a single RTX 5050 laptop GPU (8GB VRAM).

## Incident: empty merges (5 of 7 repos)

**Detection (2026-09-13, ~08:04 EDT):** held-out eval on the chaski-r2 flagship returned 0/5 on the trained task and 6/6 on refusals. Refusals are base-prompt behavior; the trained task is not. That gap indicated the training was absent from the published weights.

**Audit:** tensor-level comparison against the base model showed all 320 tensors byte-equal in 5 repos: chaski_r2, chaski_5050, brain_navigator_r2, khipu_r3, willay. The remaining 2 repos (khipu_r2, szl-receiptagent-qwen35-0.8b-v3 — different base, standard key format) were genuine merges and reproduced their published eval numbers exactly.

**Root cause:** adapters trained via Unsloth on the Qwen3.5-0.8B multimodal build save LoRA keys with a different module path and no adapter-name segment. The merge step loaded adapters through a non-strict key-matching path that silently matched zero keys. Output checkpoints were byte-for-byte copies of the base model. No error was raised. (Related upstream: PEFT issue #5220 on non-strict load/copy semantics.)

**Classification:** silent merge no-op — infrastructure failure, not training failure. Training data, adapters, and evals were intact.

**Fix (2026-09-14, 09:08 EDT):** manual LoRA merge, W' = W + (alpha/r) * B @ A, applied directly. 96/96 target modules applied with non-zero weight deltas, verified per-module. All 5 affected repos republished with corrected merge receipts.

**Post-fix verification:** chaski-r2 reran held-out gates: 5/5 trained task, 6/6 refusals — matches the trained baseline exactly.

## Hardening

- Zero delta on any target module now hard-fails the merge (assert, not warn).
- Held-out eval on the merged artifact is a publish gate; adapter-level or training metrics do not satisfy it.
- Merge receipts now record modules applied and weight-delta checksums.

## Security note

Two PATs burned by webhook callers: PAT ending 25cc burned by mcp+lk3kksr7b@huggingface.co at 2026-09-11 18:05:59; PAT ending 3e26 burned by operational@szlholdings.com. Revocation pending at time of this delta.
