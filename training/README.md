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

