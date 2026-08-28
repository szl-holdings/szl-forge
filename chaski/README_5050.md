# Chaski-5050 — `SZLHOLDINGS/chaski-5050`

Separate SKU. Byte-aligned to the owner-GPU recipe. **Not** live `SZLHOLDINGS/chaski`. **Not** an HF Job. `job=local-5050`.

| | |
|---|---|
| **Hub id** | `SZLHOLDINGS/chaski-5050` |
| **Forbidden** | `SZLHOLDINGS/chaski` — never overwrite |
| **Base** | `Qwen/Qwen3.5-0.8B` (`CANONICAL_BASE`) |
| **LoRA** | r=16, **alpha=16** (not live Chaski 16/32) |
| **Train** | seed=11, 3 epochs, batch=1, ga=4, seq=2048, lr=2e-4, `load_in_4bit=False`, `load_in_16bit=True`, QLoRA forbidden, `report_to=none`, `push_to_hub=False` during train |
| **Data** | jsonl-only `szl_dataset.jsonl` from `SZLHOLDINGS/szl-1-doctrine-sft`. Refuses `SZL_ESTATE_MANAGED.json`. |
| **Card label** | `REPORTED owner-metal` until a signed receipt exists. `train_loss` may be MEASURED as a train metric, not an eval. |
| **Evals** | none-this-run. Not 5/5. `publication_eligible: false`. |
| **Lab** | No Khipu lab pin. No tok/s claims. |
| **A11OY-MINI** | GGUF of **live** Chaski, not this 5050 kit. |

## Scripts

- `chaski/train_chaski_bf16_5050.py` — same recipe as the owner-GPU file. Default status. `--train` is local GPU only. FORGE uploads adapters after.
- `chaski/eval_chaski_5050.py` — none-this-run without a local adapter.
- `chaski/serve_chaski_5050.py` — proposal-only pin `SZLHOLDINGS/chaski-5050`.

No Hub PUT from this checkout.
