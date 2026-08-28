# Chaski-5050 — `SZLHOLDINGS/chaski-5050`

Separate SKU. **Not** live [`SZLHOLDINGS/chaski`](https://huggingface.co/SZLHOLDINGS/chaski). Do not overwrite live Chaski. Do not recut the disclosed Qwen3.5-0.8B base onto another instruct family.

| | |
|---|---|
| **Hub id** | `SZLHOLDINGS/chaski-5050` |
| **Live Chaski** | `SZLHOLDINGS/chaski` — a different artifact. This kit refuses that id. |
| **Base** | `Qwen/Qwen3.5-0.8B` (`CANONICAL_BASE`, Apache-2.0). Do not recut onto another instruct family. |
| **Hardware** | owner GPU / local RTX 5050. **Not an HF Job.** |
| **Recipe** | Unsloth LoRA bf16. QLoRA forbidden. `load_in_4bit=False`, `load_in_16bit=True`, r=16, **alpha=16**, seed=11, `warmup_steps=6`, 3 epochs, batch 1, ga 4, seq 2048, `adamw_8bit`, unsloth gradient checkpointing, `report_to=none` |
| **Data** | jsonl-only `szl_dataset.jsonl`. Refuses `SZL_ESTATE_MANAGED.json`. |
| **Training label** | `REPORTED owner-metal` until a signed receipt exists. |
| **Evals** | none-this-run. Not 5/5. `publication_eligible: false`. Train loss is not an eval. |
| **Lab** | Do not load into the Khipu lab. No tok/s claims. |
| **A11OY-MINI** | GGUF of **live** Chaski, not this 5050 kit. |

## Scripts

- `chaski/train_chaski_bf16_5050.py` — default `--status`. `--train` is local GPU only. QLoRA forbidden.
- `chaski/eval_chaski_5050.py` — none-this-run without a local `chaski-5050-adapter`.
- `chaski/serve_chaski_5050.py` — proposal-only. Serve pin is `SZLHOLDINGS/chaski-5050`, not live Chaski.
- `chaski/HF_MODEL_CARD_5050.md` — house card for this SKU. Not live Chaski.

No Hub PUT from this checkout. Receipt only.
