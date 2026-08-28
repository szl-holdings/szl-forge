# Chaski — SZLHOLDINGS/chaski

Messenger LLM. Proposal-only. GitHub path: `chaski/train_chaski.py`.

| | |
|---|---|
| **base_model** | `Qwen/Qwen3.5-0.8B` (Apache-2.0) |
| **Recipe** | Unsloth QLoRA, seed 11, LoRA r=16, 64 steps, jsonl-only `SZLHOLDINGS/szl-1-doctrine-sft` / `szl_dataset.jsonl` |
| **Status** | CUTTING until an adapter file lands |
| **Attempt 3 receipt** | [training_receipt.json](https://huggingface.co/SZLHOLDINGS/chaski/blob/main/training_receipt.json) — train_loss MEASURED `1.782708187121898`, weights UNAVAILABLE |
| **Live job** | [`6a91bb7c984507d9db4ea0a4`](https://huggingface.co/jobs/SZLHOLDINGS/6a91bb7c984507d9db4ea0a4) upload_folder adapter + merged 16-bit |

Serve studio remains `spaces/szl-model-inference-lab` for **Khipu GGUF only**. This directory does not pin Chaski there.

`A11OY-MINI` is a later GGUF of this 0.8B cut after adapters land. Not Khipu.
