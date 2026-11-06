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
- Fails closed on missing HF_TOKEN, missing CUDA, or malformed examples. Those checks are reached only after the contract passes.
- Writes `train-receipt.json` (pinned student, dataset, corpus sha256, hyperparameters, final loss)
  next to the checkpoints so runs stay evidence-carrying per doctrine.
- A Hub push happens only after training, and only to `target_repo` in the config.
- `--max-samples` is rejected. It cannot shrink the corpus after the floor is counted.

Datasets must provide `messages` or `conversations` fields compatible with the
model's chat template.
