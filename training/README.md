# Training waves (single-GPU SFT)

`train_wave.py` runs one supervised fine-tuning wave on a single A6000-class
GPU using the Unsloth backend (QLoRA 4-bit or LoRA bf16).

```bash
python training/launch_contract.py --config path/to/wave.json
python training/train_wave.py --config path/to/wave.json
```

`training/w30/distill-r2.blocked.json` is a blocked draft. It does not launch.

Behavior:
- The launch contract runs before a model import, a teacher call, training, or a Hub write.
- It requires an operator admission receipt, immutable student and dataset revisions, a disjoint holdout file, and 32 accepted train examples. Thirty-two examples do not waive another failed gate.
- The existing operator admission must also bind the exact **SHA-256 of the holdout JSONL** (`holdout_sha256`). Missing, malformed, unreadable or byte-mismatched holdout evidence fails closed before any model import. A same-dataset developer holdout remains **not an independent generalization test**, and a checksum is **not an authenticated operator signature**.
- Explicit `--model` / `--datasets` flags are not a launch path.
- Hyperparameter flags are a cross-check. A flag that disagrees with the validated config is rejected before a model import. A flag that repeats the config is accepted. `--max-samples` is rejected.
- Fails closed on missing HF_TOKEN, missing CUDA, or malformed examples. Those checks are reached only after the contract passes.
- Writes `train-receipt.json` (pinned student, dataset, corpus sha256, hyperparameters, final loss)
  next to the checkpoints so runs stay evidence-carrying per doctrine.
- Direct model publication is disabled. Set `"local_only": true` in the SFT config to keep `target_repo` available for the read-only target identity/family preflight while writing training artifacts locally. `local_only` must be a JSON boolean; it defaults to `false`. A config without `target_repo` also trains locally, but cannot pass the target-specific Hub preflight. Omit `--repo-id` for local DPO training. Direct publication requests fail before reading `HF_TOKEN` or importing the model stack, rather than after spending a complete training run.
- Local checkpoints and `train-receipt.json` remain the training outputs. Export the artifacts and use `.github/workflows/model-publish-gate.yml` for a separately approved publication with held-out evaluation, BOM, signing, and conformance checks. A training receipt is not publication approval.
- `--max-samples` is rejected. It cannot shrink the corpus after the floor is counted.

Datasets must provide `messages` or `conversations` fields compatible with the
model's chat template.



## Lutar A1–A5 — source-bound synthetic curriculum now available for review

The immutable 160-train / 80-held-out Lutar arithmetic/chat curriculum from
`lean-rlvr-proving-ground@b23ea5fe7e63d6b7402ce3a6f0a6aa5e1fa56b2c`
can be **materialized into a new local Forge SFT review directory** using
`training/lutar_axiom_intake.py`. See
[`LUTAR_AXIOM_INTAKE.md`](./LUTAR_AXIOM_INTAKE.md) for exact source digests,
nonuniform-weight and uniqueness negative controls, the numerical oracle,
reproduction commands, and authority restrictions. The importer reuses this
repository's actual SFT validator and never starts a training wave.

The candidate manifest and receipt deliberately retain
`training_authority=NONE`: rights admission, immutable Hub dataset/student
pins, an independent evaluation set, GPU authorization and the standard
`launch_contract.py` remain prerequisites for fine-tuning. Do not convert a
valid synthetic fixture into a claim of trained weights or deployed inference.


## Materials Frontier — five challenge-linked synthetic science lanes

The [Materials Frontier SFT candidate](MATERIALS_FRONTIER_SFT.md) adds a
deterministic, first-party procedural curriculum for cuprates, semiconductor
interfaces, quantum chemistry, light-driven materials, and thermal interfaces.
The source-only generator creates **240 candidate train examples, 60 dev
examples, and 60 diagnostic examples** with exact unit-checked arithmetic,
symmetry/conservation tests, and scientific abstention. It runs the existing
Forge SFT validator and writes only to a new private directory.

No external test questions, protected benchmark solutions, scientific
experimental records, confidential source, model weights, paid GPU jobs,
Cloudflare access settings, HF publications, PyPI distributions or contest
submissions are consumed or created. Diagnostic splits share one generator
and are **not independent scientific heldouts**. The existing rights, model,
independent-evaluation, budget and training launch gates remain required.

Offline reproduction:

    python -m pytest -q tests/test_materials_frontier_seed.py
    python training/materials_frontier_seed.py --output-dir /new/private/materials-review

## TRL SFTTrainer API compatibility

The canonical SFT and DPO runners pass the tokenizer through TRL's
`processing_class` parameter, not the removed legacy `tokenizer` keyword.
SFTConfig uses `max_length`. The load dtype matches the trainer precision
(`bfloat16` or `float16` on CUDA, full precision on a CPU dry-run) and 4-bit
loading is refused without a CUDA mixed-precision dtype. A missing chat
template or padding token fails closed before `train()` or generation.
This is checked against TRL `0.23.1` and Transformers `4.57.6` by
`tests/test_sft_trl_compat_contract.py` and, when that stack is installed,
`tests/test_trainer_cpu_dry_run.py`. A green unit test is not a GPU training
result, a rights admission, or publication permission.

## Q5–Q8 readiness (not a launch)

`frontier/q5-q8-readiness/` records the 2026-11-27 lanes. Q4 is not approved.
The records pin the transformers 5.22.2 image shared with
`kallpa/nemotron-h-trainer` and reject the TRL 0.23.1 Unsloth path. They do
not start training. `training/q5_q8_lanes.py` is the offline gate, and
`training/q5_q8_dry_run.py` takes two CPU steps on a tiny random config.
