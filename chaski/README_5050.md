# Chaski-5050 — `SZLHOLDINGS/chaski-5050`

Separate SKU. **Not** live [`SZLHOLDINGS/chaski`](https://huggingface.co/SZLHOLDINGS/chaski). Do not overwrite live Chaski. Do not recut the disclosed Qwen3.5-0.8B base onto another instruct family.

| | |
|---|---|
| **Hub id** | `SZLHOLDINGS/chaski-5050` |
| **Hub commit** | [`c907ebe6e1fa900021be7b6fec19b38ec45be574`](https://huggingface.co/SZLHOLDINGS/chaski-5050/tree/c907ebe6e1fa900021be7b6fec19b38ec45be574) |
| **Live Chaski** | `SZLHOLDINGS/chaski` — a different artifact. This kit refuses that id. |
| **Base** | `Qwen/Qwen3.5-0.8B` (`CANONICAL_BASE`, Apache-2.0). Do not recut onto another instruct family. |
| **Hardware** | owner GPU / local RTX 5050. **Not an HF Job.** job `local-5050`. |
| **Recipe** | Unsloth LoRA bf16. QLoRA forbidden. `load_in_4bit=False`, `load_in_16bit=True`, r=16, **alpha=16**, seed=11, `warmup_steps=6`, 3 epochs, batch 1, ga 4, seq 2048, `adamw_8bit`, unsloth gradient checkpointing, `report_to=none` |
| **Data** | jsonl-only `szl_dataset.jsonl`. Refuses `SZL_ESTATE_MANAGED.json`. `dataset_sha256` `ddc5594bfb1c78449ba40a263f5ac41d21c896c3c7ed7346341c7c080611a243`. |
| **Weights** | AVAILABLE. `adapter_model.safetensors` present on Hub. |
| **Train loss** | MEASURED `2.228136855544466` (`train_runtime` 883.2224s, 3 epochs, 41 rows). train metric, not an eval. |
| **Training label** | `REPORTED owner-metal` until a signed receipt exists. |
| **SKU** | **NOT MEASURED.** Do not stamp this model as MEASURED. |
| **Evals** | none-this-run. Not 5/5. `publication_eligible: false`. |
| **Lab** | Do not load into the Khipu lab. No tok/s claims. |
| **A11OY-MINI** | GGUF of **live** Chaski, not this 5050 kit. |

## Scripts

- `chaski/train_chaski_bf16_5050.py` — default `--status`. `--train` is local GPU only. QLoRA forbidden.
- `chaski/eval_chaski_5050.py` — none-this-run without a local `chaski-5050-adapter`.
- `chaski/serve_chaski_5050.py` — proposal-only. Serve pin is `SZLHOLDINGS/chaski-5050`, not live Chaski.
- `chaski/HF_MODEL_CARD_5050.md` — house card for this SKU. Not live Chaski.

GitHub stamp only. No Hub PUT from this checkout. Hub README is not recut from this checkout.
