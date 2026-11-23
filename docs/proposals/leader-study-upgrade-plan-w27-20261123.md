# Proposal: szl-forge upgrade plan from leader-study loop-1

- Date: 2026-11-23 (UTC)
- Status: PROPOSE-ONLY. This document does not authorize any merge or deploy. DO NOT MERGE without owner signature; the estate merge wall is owner-gated.
- Provenance: typed by the W26 leader-study loop-1 council (Kimi governing; Astra + Sol advisory), grounded in repo facts (GitHub GraphQL) and domain-leader research dated 2026-11-22/23. Published receipts: HF SZLHOLDINGS/wave-logs mesh/2026-11-22/frontier-push/w26-solo-build/ commit 8c01e7ed. Leader research marked DRAFT until independently verified.
- Honest scope: authored from the README head + repo facts only; no code changed; secrets-free; no unbacked claims.

## Measured archetype
A sovereign, evidence-bound model factory that wraps QLoRA/DPO fine-tuning and Ollama publication in a fail-closed governance layer, prioritizing verifiable receipts and bounded execution over raw training speed or model breadth.

## Prior art: normative publications
- LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021): Freezing pre-trained model weights and injecting trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks.
- QLoRA: Efficient Finetuning of Quantized LLMs (Dettmers et al., 2023): Backpropagating gradients through a frozen, 4-bit quantized pretrained language model into Low Rank Adapters, enabling fine-tuning of large models on limited hardware while preserving full 16-bit finetuning task performance.

## Prior art: community/framework leaders
- Unsloth (Unsloth AI) [canonical](https://github.com/unslothai/unsloth) - The core training engine for the estate; provides the hand-written kernels and low-VRAM efficiency required for single-GPU sovereign training and GGUF export.
- Axolotl (Axolotl AI / Open Access AI Collective) [canonical](https://github.com/axolotl-ai-cloud/axolotl) - The standard for config-driven YAML training; szl-forge should adopt its declarative recipe patterns to ensure reproducibility and reduce imperative script drift.
- TorchTune (PyTorch / Meta) [canonical](https://github.com/pytorch/torchtune) - Provides PyTorch-native, composable recipes that align with the estate's need for explicit, source-controlled training logic without heavy framework abstraction.
- LLaMA-Factory (LLaMA-Factory Community) - Sets the benchmark for unified UI and broad model support; highlights a gap in szl-forge's operator experience and multi-architecture flexibility.
- TRL + PEFT (Hugging Face) - The canonical reference for LoRA/QLoRA/DPO implementations; szl-forge must prove its custom training waves match or exceed these reference behaviors.

## Measured gaps vs the leaders
- Imperative script lock-in: Unlike Axolotl or TorchTune, the repo relies on specific Python scripts (train_wave.py) rather than declarative, versioned configuration files, making parameter auditing harder.
- Narrow hardware envelope: The focus on Unsloth/single-GPU creates a dependency on specific kernel optimizations, lacking the hardware-agnostic fallback paths found in TorchTune or raw HF TRL.
- Missing automated eval gates: While receipts are mentioned, there is no evidence of integration with standardized evaluation harnesses (e.g., lm-evaluation-harness) to automatically block publication on regression.
- Limited model architecture support: The kit appears tuned for specific SZL artifacts (e.g., 1.5B), lacking the broad, plug-and-play model coverage of LLaMA-Factory or Axolotl.
- Opaque 'Wave' semantics: The 'training wave' terminology is estate-specific and lacks mapping to standard industry concepts (SFT, DPO, RLHF), increasing the cognitive load for external auditors.
- UI/Observability deficit: Compared to LLaMA-Factory's unified UI, szl-forge appears CLI-centric, potentially limiting real-time operator intervention during long training runs.

## Ranked upgrade plan (dosed, small-PR sized)
1. Extract hyperparameters from training/*.py into signed, schema-validated YAML recipe files (e.g., recipes/szl-1-sft-v1.yaml). | expected effect: Decouples code from configuration, enabling cryptographic signing of exact training parameters before a run begins (readback-or-it-didnt-happen). | effort: Small | priority: P0
2. Implement a 'Reference Parity' test suite that runs a tiny model (e.g., 10M params) through szl-forge and vanilla HF TRL with identical seeds/data. | expected effect: Proves that Unsloth optimizations do not degrade model quality or alter training dynamics compared to the canonical reference. | effort: Medium | priority: P1
3. Add a standardized `eval_harness` adapter to `training/eval_wave.py` that outputs results in the `lm-evaluation-harness` JSON format. | expected effect: Allows SZL receipts to be compared directly against public leaderboards and standard benchmarks, moving from MEASURED_LIMITED to verifiable context. | effort: Small | priority: P1
4. Create a `Modelfile` linter that validates the Ollama configuration against the training receipt (e.g., ensuring temperature/stop tokens match eval settings). | expected effect: Prevents 'configuration drift' between the qualified model artifact and the served inference behavior. | effort: Small | priority: P2
5. Document a 'Kernel Fallback' path using standard PEFT/Transformers for when Unsloth does not support a target architecture. | expected effect: Ensures the forge remains fail-closed but functional on new hardware or model types, reducing vendor lock-in to Unsloth kernels. | effort: Medium | priority: P2

## Advisory seat notes
- ASTRA verdict: AMEND; Map 'Wave' semantics to industry standard terms (SFT, DPO, RLHF) to ensure auditability.; Transition from imperative scripts (train_wave.py) to declarative config patterns (Axolotl-style) to close the parameter auditing gap.
- SOL verdict: AMEND; Map 'Wave' semantics to industry standards (SFT, DPO, RLHF) in the archetype or a dedicated mapping table to satisfy the 'honesty over agreement' doctrine for external auditors.; Formalize the 'automated eval gates' as a P0 requirement, specifically integrating the `szl-eval-harness` blocking ledger as a hard gate before Ollama publication.

## DRAFT warnings
- The 'Domain Leader Research' is dated 2026-11-22 and marked DRAFT; claims regarding Unsloth's current performance or feature set are unverified against live documentation.
- The README head is truncated; the full capabilities of `agent-forge/` and `clinical-gateway/` are inferred from the summary table and not fully audited.
- The specific implementation details of `training/train_wave.py`, `dpo_wave.py`, and `eval_wave.py` are not provided; gaps regarding code structure are inferred from the file names and 'recipe' description.
- The claim 'MEASURED_LIMITED' for the ReceiptAgent weights is accepted as ground truth from the text, but the actual receipts and hashes were not verified.
- The existence and content of the HuggingFace demo at https://huggingface.co/SZLHOLDINGS were not independently verified.
- The arXiv identifiers (2106.09685, 2305.14314) are taken from the provided notes and were not resolved.
