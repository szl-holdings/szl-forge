# Payload 002 — TRL AsyncGRPO LoRA / vLLM evaluation

Tracking: `szl-forge#224`, `szl-frontier#76/#77`.

Exact upstream source: `huggingface/trl@f540773f5250c816e992ae3d35a41142ef3625c0`. Do not follow moving main. Stable TRL 1.13 from merged tooling lane #216 remains the comparison baseline.

## Implement

Extend the existing `inference/hf_tooling.py` / installed-runtime workflow with a separately selectable AsyncGRPO-LoRA lane. Install the exact TRL commit and a compatible exact PEFT/vLLM closure into a fresh job-local environment. Capture PEP 610/install provenance and package closure before running probes. Do not add these dependencies to the production runtime.

Use a tiny local model or deterministic synthetic policy where practical; use fixed non-secret rollout fixtures. A hardware-dependent test that cannot run on the available runner must emit `UNAVAILABLE` with the missing prerequisite, never PASS.

Measure both sync paths under equivalent conditions: adapter-only and merged fallback. Record transferred bytes, sync pause, generation throughput, policy version, max staleness, adapter rank/capacity, loaded/evicted adapter identities and checkpoint presence in a machine-readable receipt. Keep upstream performance numbers labeled upstream; only locally measured numbers may become SZL evidence.

## Negative paths

Cover server without LoRA, runtime adapter update disabled, unsupported/insufficient max rank, insufficient max_loras, shared-path unavailable, path escape/symlink, partially written adapter, load failure, load-before-evict ordering, server restart, cache eviction, checkpoint missing, and hybrid recurrent-state/log-prob mismatch. Verify the serving cache is ephemeral and never treated as the durable trained artifact.

## Authority

No provider credentials, external production server, Hub write, production training/serving route, autonomous merge/deploy, branch-protection mutation, or automatic promotion. If all evaluation checks pass, disposition is still EVALUATION; production requires a successor qualification wave with exact model/data/runtime/policy/rollback/proof evidence.
