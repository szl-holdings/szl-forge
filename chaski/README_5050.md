# Chaski-5050 — `SZLHOLDINGS/chaski-5050`

**NEW Hub id.** Local RTX 5050 Unsloth LoRA. **Not** live [`SZLHOLDINGS/chaski`](https://huggingface.co/SZLHOLDINGS/chaski). **Not** an alias of live Chaski. **Not** `A11OY-MINI`.

| | |
|---|---|
| **Hub id** | `SZLHOLDINGS/chaski-5050` |
| **Live Chaski** | `SZLHOLDINGS/chaski` — a different artifact. Do not push this kit there. |
| **Base** | `Qwen/Qwen3.5-0.8B` (`CANONICAL_BASE`, Apache-2.0) |
| **Hardware** | local RTX 5050. **Not an HF Job.** |
| **Recipe** | Unsloth LoRA bf16. `load_in_4bit=False`, `load_in_16bit=True`, r=16, alpha=32, seed=11, 3 epochs, batch 1, ga 4, seq 2048, `adamw_8bit`, unsloth gradient checkpointing, `report_to=none` |
| **Data** | jsonl-only `szl_dataset.jsonl`. Refuses `SZL_ESTATE_MANAGED.json`. |
| **Evals** | none-this-run without a local adapter. Train loss MEASURED is not an eval. Not 5/5. |
| **A11OY-MINI** | no. That ROADMAP GGUF belongs to live Chaski 0.8B, not this kit. |

## Scripts

- `chaski/train_chaski_bf16_5050.py` — default `--status`. `--train` is local GPU only.
- `chaski/eval_chaski_5050.py` — none-this-run without a local `chaski-5050-adapter`.
- `chaski/serve_chaski_5050.py` — proposal-only. Serve pin is `SZLHOLDINGS/chaski-5050`, not live Chaski.

No Hub PUT from this checkout. Receipt only.
