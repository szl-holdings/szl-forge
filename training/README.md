# Training waves (single-GPU SFT)

`train_wave.py` runs one supervised fine-tuning wave on a single A6000-class
GPU using the Unsloth backend (QLoRA 4-bit or LoRA bf16).

```bash
export HF_TOKEN=...            # write-scoped, SZLHOLDINGS member
python training/train_wave.py \
  --model SZLHOLDINGS/SZL-Khipu-1.5B \
  --datasets SZLHOLDINGS/governed-sft-mix-v2,SZLHOLDINGS/szl-1-doctrine-sft \
  --method qlora --epochs 3 --lr 2e-4 --seqlen 4096 --pack \
  --out out/khipu-sft
```

Behavior:
- Fails closed on missing HF_TOKEN, missing CUDA, or malformed examples.
- Writes `train-receipt.json` (model, datasets, hyperparameters, final loss)
  next to the checkpoints so runs stay evidence-carrying per doctrine.
- Optional `--repo-id` pushes the merged model to the HuggingFace Hub.
- `--max-samples N` caps the corpus for smoke runs.

Datasets must provide `messages` or `conversations` fields compatible with the
model's chat template.
