# Independent AirLLM qualification lab

This Forge research component contains independently written qualification
controls for memory residency and LoRA recomputation. It also has an opt-in,
local-only AirLLM inference smoke path. It studies mechanisms described by
AirLLM and the cited prior art. It does not train a real model or change any
published model or candidate gate.

## Evidence boundary

- Byte budgets and phase totals are MODELED; artificial dense-layer execution
  and failure fixtures are SIMULATED.
- Actual Python test commands and their outcomes can be MEASURED software
  evidence. They do not qualify an AirLLM package, CUDA backend or real model.
- Every result keeps `training_eligible=false`, `publication_eligible=false`
  and `execution_authority=false`.
- The modeled controls do not load model/tokenizer files, training curriculum or
  held-out gates. They use no remote code, provider API, dependency installation,
  GPU call, signing key, container mutation or cleanup.
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

## Opt-in local AirLLM smoke

`local_airllm.py` is a separate executable seam. It accepts only the exact
`openbmb/MiniCPM5-2B` revision and eight file hashes in Forge's
`frontier/minicpm5/evidence/2026-09-07-a10g-smoke.json`. That source evidence
reports `SMOKE_FAIL` and `HOLD`; matching bytes do not turn it into a qualified
model. The smoke uses the base checkpoint only. It does not load an SZL adapter,
run a held-out gate, or replace the recommended v4 Qwen3.5 comparison above.

The manifest must enumerate those eight files and give absolute paths for the
existing local snapshot, a fresh split directory, a task-owned scratch root,
the AirLLM wheel and a GPU UUID. The scratch root must already contain
`.szl-airllm-task-owned.json` with schema `szl.airllm.scratch-owner/v1` and the
same model ID and revision; the split directory must be a nonexistent direct
child. It must live under the task work parent as a sibling of the Forge
checkout. The marker is a **DECLARED** ownership assertion, not proof of
exclusive filesystem access. The receipt must be outside the checkpoint and
scratch root. There is no
automatic download, installation, cleanup or deletion. `delete_original` is
always `False`; its appearance as a manifest field is refused.

AirLLM is pinned to [v4.0.0](https://github.com/lyogavin/airllm/tree/ff35db207a0c559af9aa95d686057c3fe84f1d40).
The [PyPI wheel](https://pypi.org/project/airllm/4.0.0/) SHA-256 is
`155e7c286acbfb5f82a358970d980efcefff1e3fcd8a4ebf3c5850137a31d841`.
Preflight checks that wheel and every installed `airllm/` package file against
it. Other dependency versions are observed, not hash-locked. Numerical parity
and package behavior remain unqualified.

Upstream [AutoModel](https://github.com/lyogavin/airllm/blob/ff35db207a0c559af9aa95d686057c3fe84f1d40/air_llm/airllm/auto_model.py)
sets `trust_remote_code=True`; its base loader and tokenizer can also use that
setting. The guard therefore accepts only the byte-pinned, local built-in
`LlamaForCausalLM` configuration, rejects `auto_map` in every JSON file,
rejects `.py` and pickle-format checkpoint files, and sets Hub/Transformers
offline variables before import. The worker rehashes the files before loading.
This restricts the known dynamic-code paths for this exact snapshot; an
unreviewed model ID, tokenizer class, or mutable shared snapshot is refused.

The source's [splitter](https://github.com/lyogavin/airllm/blob/ff35db207a0c559af9aa95d686057c3fe84f1d40/air_llm/airllm/utils.py)
loads an entire safetensors shard into host RAM. This checkpoint has a single
5,033,557,096-byte shard, so low GPU memory does not imply low host RAM or
disk. Preflight measures current free disk, available host RAM and UUID-bound
GPU memory. Conservative MODELED bounds require `2 × weight bytes + 4 GiB`
free disk, at least the larger of `2 × weight bytes + 2 GiB` and `largest shard
+ 4 GiB` available host RAM, and 3 GiB free GPU memory. A passing bound only
permits an isolated experiment; it is not a guarantee that the run will fit.

Run preflight with the local manifest and a fresh receipt path:

```shell
python -I -B local_airllm.py --manifest /absolute/manifest.json --receipt /absolute/PREFLIGHT.json
```

Add `--run` only when preflight is `READY_FOR_LOCAL_EXPERIMENT_ONLY` and the
checkpoint/scratch ownership is still current. The worker runs offline, binds
the GPU UUID, gives AirLLM a task-owned layer path and `delete_original=False`,
then generates **one token** from a fixed synthetic prompt under a five-minute
timeout. Its token hash, elapsed time and CUDA peak are a local software smoke,
not a quality score, energy measurement, model release or execution authority.
Common Hugging Face, Torch, Triton, CUDA and temporary cache paths are directed
under task scratch before imports; third-party shared-cache mutation remains
`UNKNOWN`. A timeout is `UNKNOWN_AFTER_ATTEMPT`, and partial split files remain
for inspection without automatic cleanup.
The attempted run records pre/post free disk and RAM when observable. A future
qualification must compare native-resident and AirLLM logits/tokens on
identical assets; this one-token smoke does not perform that comparison.

## Manual hosted CPU plumbing smoke

`airllm-hosted-cpu-smoke.yml` is a separate, manually dispatched GitHub Actions
experiment on `main`; it never runs for a pull request. It exercises AirLLM
4.0.0 on the public Apache-2.0 `HuggingFaceTB/SmolLM2-135M-Instruct` base at
immutable revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`. This is not
the MiniCPM checkpoint above and does not change its failed/HOLD result.

`hosted_cpu_smoke.py` binds exactly eight top-level files by size and SHA-256,
including the 269,060,552-byte safetensors file. It rejects extra files,
Python/pickle checkpoint files, `auto_map` anywhere in the selected JSON, an
unreviewed architecture or tokenizer, a different installed AirLLM package,
and insufficient fresh disk/RAM. The AirLLM wheel is the same pinned PyPI wheel
and source commit identified above. Dependencies are exact-version observed,
but their full transitive hash closure is **not verified**.

After acquisition and rehashing, the worker sets Hub/Transformers offline mode
before importing AirLLM, uses CPU float32, a fixed synthetic prompt, exactly
one new token, `delete_original=False`, task-specific cache/split directories,
and a 240-second subprocess timeout. The runner retains separate prepare and
attempt receipts as an artifact, including source SHA, selected bytes, package
versions, free disk/RAM, elapsed time, and only a hash of token IDs. The run
receipt also records the SHA-256 of the prepare receipt bytes, the SHA-256 of
the executed harness after comparing it with its checked-out Git blob, the
SHA-256 of the fixed prompt and canonical one-record SAMPLE input, and a
generation record with `do_sample` false and seed `0`; the prepare record keeps
`seed_applied` false, and the run record sets it true only after the worker
applies the Python/Torch seed.
The run records `base_revision`, inference-only loss as `UNAVAILABLE`/`NOT_RUN`,
and metrics as `UNAVAILABLE` until a worker succeeds. A run refuses a prepare
receipt from a different harness, prompt, input, model, seed, or source SHA.
The input is synthetic and is not a training or held-out evaluation dataset.
Failed and timed-out attempts stay `BLOCKED` or `UNKNOWN_AFTER_ATTEMPT`; the
split remains for inspection until the ephemeral runner ends.

Receipt schema v2 is an unsigned, nonauthoritative chain. `prepare_receipt_sha256`
and `previous_receipt_sha256` are raw-byte digests, not signatures, signer trust,
authorization, model qualification, or publication evidence. The earlier v1
hosted run remains historical v1 evidence and is not retroactively upgraded.

A pass establishes only one hosted CPU plumbing smoke. It does not establish
native-versus-AirLLM parity, QLoRA support, model quality, held-out abstention,
GPU fit, energy, training, Hugging Face publication, or production authority.

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
