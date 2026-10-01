# Storage-aware research in the existing Forge Model Lab

This source adds a bounded Python CPU expert-store/cache/forward reference, an offline Spark KV planner, a CLI action, and authenticated Python-rendered views. It does not install Edge0, Spark, SAS, a learned prerouter, INT4 kernels, Recover-LoRA, CUDA or MLX. Synthetic test tensors are not trained SZL weights. No upstream model is downloaded or executed by this integration.

## Study, adapt, measure

Prior art: https://github.com/Edge0-AI/Edge0 and https://huggingface.co/Edge0/Edge0-35B-A3B-preview . Some upstream prerouting modes replace executed expert indices; they are not interchangeable with route-preserving prefetch. This original SZL reference uses hints ONLY to prepare storage. The caller's actual route, weights and content-addressed bundle select the experts. A missed hint loads the required expert or rejects an over-budget route; it never substitutes a convenient expert.

The owner-controlled store contains per-expert FP32 Safetensors gate/up/down matrices. A manifest digest binds every expert ID, byte hash and size. Each read checks the complete payload. This is not Edge0's packed INT4 format. Parent, tokenizer, quantizer, prerouter and adapter lineage require separate immutable-byte and rights review.

The cache is per-instance and lock-serialized. Hints use spare capacity; leases protect in-use payloads. A route exceeding capacity is rejected rather than truncated. Staging is synchronous: asynchronous I/O/compute overlap is NOT implemented or measured. Serialized byte accounting excludes tensor copies, Python/allocator overhead, OS page cache and GPU buffers. Root directories and ancestors must be owner-controlled; this is not hostile-process isolation.

## Dense Spark: independent context planning

Primary sources: https://huggingface.co/XHToken/Spark-X2.5-4B and https://github.com/XHToken/Spark-X2.5 . Dense Spark has no MoE experts to stream. The planner validates a supplied config and SHA-256 without executing custom code. Its model revision is explicitly operator-declared, not provider-attested.

The default reference transcribes public config dimensions: 36 layers, 9 full and 27 sliding, 4 KV heads, head dimension 256, sliding window 512, maximum 1,048,576 positions. This is not immutable provider-byte readback or a silent replacement of the recorded szl-frontier intake at frontier/waves/2026-09-08-kimi-hy4-spark.json, revision 5e10fcc0286756aebf7c41dc52c1e42d95c70281.

KV bytes = 2 * batch * KV_heads * head_dim * element_bytes * cached_slots. Both full allocation and conditional sliding eviction are shown. A sliding attention mask does not prove the engine frees KV slots. Use full allocation until the selected engine is qualified. Weights, workspace, actual free memory, output-token reservation and concurrent-process headroom are separate requirements; absent observations stay unknown, never zero.

## Existing commands and interface

Use the existing package in an isolated environment with its reviewed dependency contract:

```bash
szl-model-lab storage-plan --tokens 32768 --batch 1
szl-model-lab storage-plan --config /owner/config.json --config-sha256 "$CONFIG_SHA256" --model-revision "$MODEL_SHA" --tokens 131072
szl-model-lab serve --port 8765
```

Parent operator Basic authentication and SZL_LAB_ACCESS_TOKEN protect GET /storage and GET /api/storage/plan. Only bounded tokens/batch parameters are accepted. Existing host, no-store and CSP controls remain. There is no upload, arbitrary URL, shell, training, provider-dispatch or publication endpoint. Keep the workbench private/loopback; this is not a public authentication deployment.

## Qualification and existing ownership

Forge owns experiments; szl-frontier owns canonical intake/deduplication; szl-serve owns admitted backend adapters; szl-router owns policy/route eligibility; a11oy owns pool qualification and product delivery; a11oy-net owns proof projection. No second registry, router, workbench, pool or publisher is introduced.

For expert streaming compare resident, on-demand and exact-route hint modes. Separate logical reads from physical SSD I/O; record cold/warm cache, RSS/VRAM scope, prefill/decode, numerical and task results. Route-changing prerouting is a separately named approximation experiment. For Spark measure actual KV allocation, incremental context levels, retrieval/tool quality, reset isolation, cancellation and OOM behavior on the real backend. Recover-LoRA requires separately authorized training, legitimate data, held-out splits and base/quant binding. Owner training remains paused.

## Later frontier proposals are not installed by this change

- Tencent SAS (https://huggingface.co/tencent/Simple-Attention-Sparsification) selects approximate attention blocks using router-only checkpoints and a specialized backend. It is not the exact-route expert cache. Compare matched dense/sparse quality and memory before any adoption.
- NVIDIA Open-SWE-Traces (https://huggingface.co/datasets/nvidia/Open-SWE-Traces) requires task/repository holdout isolation, originating license review, and inert treatment of tools and patches. Its resolved=-1 state is unknown, not true. Reference patches must not leak into evaluation inputs. No corpus is ingested here.
- Nemotron math (https://huggingface.co/nvidia/Nemotron-3-Labs-Ultra-Math-SFT) contributes a candidate/verifier/refinement pattern, not a proof that an SZL verifier or production model exists.
- Reef and ALTK consistency belong in existing experiment/release machinery: separately version candidates, retain missing repeats, distinguish mean success, all-runs success and at-least-one success, and keep release authority outside learner output.
- ShadowPEFT, TRL/vLLM/Omni, GGUF/AutoRound, AuK, Agnes/YARQA/RLT, OUI and Granite remain distinct compatibility, consent, quality and product-binding workstreams. Their discovery or source-contract state is not runtime qualification.

## Publication and current release boundary

Only the existing blueprint SOURCE_FILES closure is extended so projected source imports and templates remain complete. Publisher functions, destinations, credentials, workflows, identities and weights are unchanged. Do not rehost Edge0/Spark weights as SZL-owned models. Reviewed GitHub source precedes immutable HF byte readback, product verification, and proof projection; GitHub and HF commit IDs are linked, not equated.

The old A11oy #2152 HTML defect has successful full-container evidence on its updated feature branch; do not reapply the old middleware ZIP or call that historical defect still unfixed. That PR remains separate from this private-workbench change and still needs its own admission/publication controls. Passing this planner does not close its holds or establish estate-wide operation.
