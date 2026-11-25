# Leader-Study Upgrade Plan - W27 loop-8 - 2026-11-23

Status: PROPOSE-ONLY DRAFT - DO NOT MERGE. Owner-gated: merging requires governing approval and the estate signature-wall owner unlock.

Repo: szl-forge - SZL Forge: QLoRA kit that fine-tunes an open base model into SZL-1 on SZL-owned hardware. Dataset, Unsloth training script, Ollama Modelfile, runbook.

Frame: OUROBOROS loop-8 leader study. Governing seat: Kimi kimi-k3.1-pro (json_ok, think off). Advisory seats: Astra/Sol/Claude community rails (provenance unverified, advisory only). Leader evidence web-verified 2026-11-23.

## Leader moats to absorb
- QLoRA doctrine: 4-bit NormalFloat quantized base with LoRA adapters, enabling single 12GB consumer GPU fine-tune of an 8B instruction model in under an hour. Moat: Memory-efficiency math plus a published, reproducible recipe that made consumer-GPU fine-tuning credible.
- HF PEFT + bitsandbytes: Canonical LoRA/QLoRA primitives: target_modules q/k/v/o/gate/up/down_proj, lora_alpha 32, gradient checkpointing. Moat: Reference implementations and broad ecosystem trust; the de facto correctness baseline every kit is diffed against.
- Unsloth: Custom Triton kernels, 2x-5x training acceleration, fused checkpointing, dropout-0 optimization. Moat: Hand-written kernel-level speed and memory wins that are hard to replicate without deep systems work.
- axolotl: Declarative multi-GPU YAML pipelines for fine-tuning. Moat: Config-as-contract reproducibility and scaled-out orchestration beyond a single script.

## Honest gaps (grounded in repo facts)
- pushed_at is null, so there is no evidence in the facts of any committed training code, run history, or that the py_compile verify command currently passes.
- The ReceiptAgent evidence state is truncated mid-sentence ('owner-signed training/evaluation receip'), so the promised weight hashes and signed receipts cannot be confirmed as present or complete.
- The description promises dataset, Unsloth training script, Modelfile, and runbook, but the facts only name training/train_wave.py, training/dpo_wave.py, and training/eval_wave.py; the dataset, Modelfile, and runbook files are not evidenced.
- No declared multi-GPU or declarative-config surface appears in the facts, so axolotl-style pipeline reproducibility is unproven.
- The readme asserts 'observable training waves, receipt-carrying launches, and proof-banner publications,' but no receipt schema, signing mechanism, or verification command for those receipts is shown in the facts.
- Scope is split across agent-forge and clinical-gateway with strong boundary language, yet no facts show automated enforcement that PHI/clinical content is excluded from training data.

## Dosed upgrade plan
- Dose 1 (propose only): push an initial commit and set a real pushed_at, then make the 60s verify command actually run green in CI so the repo's core claim is evidenced.
- Dose 2: check in a declarative run-config schema (axolotl-style YAML) that pins base model, NF4 quantization, LoRA target_modules, lora_alpha 32, gradient checkpointing, and Unsloth version, and have train_wave.py load it.
- Dose 3: add the missing dataset, Ollama Modelfile, and runbook artifacts the description advertises, each with a stated data-boundary check excluding PHI/clinical content.
- Dose 4: define and complete the truncated receipt contract — a signed training/evaluation receipt schema plus expected public weight hashes for SZL-Forge-1.5B-ReceiptAgent — and a verify command that checks hashes against the HF weights.
- Dose 5: add a qualification gate that fails a launch unless receipts verify and the evidence state moves off MEASURED_LIMITED for the claims being published.
- Risk note: Highest risk is an evidence gap, not a capability gap: the readme makes strong observability and proof claims while pushed_at is null and the receipt evidence is truncated, so external verifiers cannot yet distinguish a runnable, receipt-backed factory from an unexecuted recipe. Declarative-config and signing changes also touch the trust boundary and must be reviewed before any real training or publication.
- Propose-only note: All steps are proposals for human review; no files were changed and no run, hash, or signature was executed or verified in this audit.

## Advisory notes (unverified rails, non-binding)
- astra: szl-router: Verifiable routing provenance via cryptographic receipts. szl-forge: Domain-specific data curation pipelines for high-signal QLoRA. hatun-mcp: Formal semantic mapping between doctrine and tool schemas.
- sol: szl-router: Verifiable routing provenance via cryptographic receipts. szl-forge: Domain-specific QLoRA optimization presets for sovereign hardware. hatun-mcp: Formal doctrine-to-tool mapping for deterministic server behavior.
- claude: szl-router: Cryptographic proof of routing intent to prevent provider-side steering. szl-forge: Automated synthetic data curation loops tied to routing receipt failures. hatun-mcp: Formal verification of doctrine compliance for tool-call execution.

## DO NOT MERGE
This plan is a proposal artifact of the 10-pass OUROBOROS loop (loop-8 of 10). No code, no CI, no behavior changes in this file. A reviewer with authority may convert to a work order later.