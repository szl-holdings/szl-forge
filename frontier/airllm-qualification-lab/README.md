# Independent AirLLM qualification lab

This Forge research component contains independently written, provider-free
qualification controls for memory residency and LoRA recomputation. It studies
mechanisms described by AirLLM and the cited prior art. It does not load or train
a real model, and it does not change any published model or candidate gate.

## Evidence boundary

- Byte budgets and phase totals are MODELED; artificial dense-layer execution
  and failure fixtures are SIMULATED.
- Actual Python test commands and their outcomes can be MEASURED software
  evidence. They do not qualify an AirLLM package, CUDA backend or real model.
- Every result keeps `training_eligible=false`, `publication_eligible=false`
  and `execution_authority=false`.
- No model/tokenizer files, training curriculum or held-out gate are loaded.
  No remote code, provider API, dependency installation, GPU call, signing key,
  container mutation or cleanup is used by this lab.
- Proposal hashes identify these supplied declarations, not verified weight
  bytes. Manifest completeness and real numerical parity remain UNKNOWN.
- This adds no locked formula or Lean proof; established mathematical and
  systems ideas remain attributed. Model quality and novelty remain UNKNOWN.

## What is implemented

`residency_plan.py` walks declared construction, forward, backward, optimizer
and checkpoint phases. Reservations explicitly identify owners, lifetimes,
device ordinals, byte counts and persistence. It is a serial-phase arithmetic
oracle, not an allocator, scheduler, lease service or admission controller.
Concurrent sessions and cross-phase pipelines need a later runtime model.
Within-phase layer/prefetch overlap must be represented by simultaneous entries.

GPU and host budgets must describe available reservation capacity, not total
installed memory. Disk budget means currently free bytes. Retained originals
already on disk count toward gross footprint but do not consume free bytes a
second time. New originals, derived shards, temporary conversion buffers and
checkpoint outputs do consume free capacity at their simultaneous peaks.
Persistent assets cannot disappear between their allocation and checkpoint;
already-allocated persistent disk assets must appear in every phase.

Unknown bytes or reserves stay UNKNOWN. Known lower bounds exceeding capacity
stay BLOCKED even when another quantity is unknown. Invalid booleans, floats,
negative sizes, duplicate names, device mismatches and declared target-coverage
mismatches are rejected. CUDA UUID binding and real exclusivity are UNKNOWN.
The abort function only returns simulated names; it never frees or deletes data.

`toy_lora_oracle.py` and its tests are an independent, standard-library dense
LoRA algebra fixture. Its resident and recomputing paths, finite-difference
gradient oracle, explicit supervision masks and checkpoint/optimizer checks
are artificial software controls. They are not Qwen, PEFT, Transformers or the
production AirLLM `_StreamedModule` path, and cannot substitute for those tests.

## Reproduce the software checks

From this directory with Python 3.11 or later; no third-party packages are needed:

```shell
python -I -B -m unittest discover -s . -p 'test_*.py' -v
```

The pull request records the source revision and actual command outcome.
`test_residency_plan.py` exercises capacity, persistence and refusal boundaries;
`test_toy_lora_oracle.py` checks analytical gradients against finite differences,
recomputation parity, malformed inputs and checkpoint resume. These software
checks cannot establish real-model evaluation or runtime readiness.

The path-scoped `AirLLM qualification lab` workflow runs these controls on Python
3.11 and 3.12 without installing model libraries. Its results are software
evidence only; it does not train, evaluate or publish model weights.

## Recommended real-model progression

1. Bind the reviewed source to an isolated, pinned runtime and an admitted
   compute profile. This component grants no installation or training authority.
2. Qualify the native resident base first: declared v4 target
   `unsloth/Qwen3.5-0.8B` at `23c69c53358a07516b5827588b3fdb12ae78fd65`.
   Bind actual configuration, conditional-generation loader, tokenizer/template,
   device/dtype and complete tensor bytes. A base-only run is not a v4 release.
3. Compare the pinned AirLLM production path on exactly the same assets and
   precision, with remote acquisition and original deletion disabled. Compare
   logits/tokens/cache state before memory or speed. QLoRA-4bit and streamed BF16
   are separate experiments, not a controlled streaming-only comparison.
4. Compare intended adapter coverage, supervised masks, loss, every gradient,
   optimizer update and resumed checkpoint before real fine-tuning. Nonzero
   dropout requires deterministic recomputation state.
5. Train only the admitted curriculum. Preserve all held-out sets, unchanged
   thresholds, provenance, immutable candidate identity and proposal-only output.
   Run the actual model gates; do not substitute loss or toy tests for them.
6. Only then consider a larger open base on storage/RAM/compute that has actually
   been admitted. Improvement must be demonstrated by held-out outcomes and
   controlled comparisons, not parameter count or low VRAM alone.

## Source lineage and prior art

The current studied AirLLM source is commit
`1c8d2630c9373a6d345febe8ef93e05c58bdfb21`, tree
`20b928cfdfab9943ad3f3fcee3bfc0dcdbb6c054`. The repository is Apache-2.0;
model licenses and checkpoint provenance remain separate. No upstream source
code was copied into this implementation.

- [AirLLM pinned source](https://github.com/lyogavin/airllm/tree/1c8d2630c9373a6d345febe8ef93e05c58bdfb21)
- [ZeRO-Inference](https://www.deepspeed.ai/2022/09/09/zero-inference.html)
- [FlexGen](https://arxiv.org/abs/2303.06865)
- [Accelerate big-model inference](https://huggingface.co/docs/accelerate/concept_guides/big_model_inference)
- [LoRA](https://arxiv.org/abs/2106.09685)
- [Activation recomputation](https://arxiv.org/abs/1604.06174)

Low-VRAM loading is a systems capability, not an independent claim about task
quality, safety, throughput, energy, production readiness or benchmark rank.
