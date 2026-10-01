# CPU replay receipts for SZLHOLDINGS/chaski-r2 (diagnostic, not the gate)

Environment: Linux sandbox, Python 3.14, torch 2.14.1+cpu, transformers 5.18.0, peft 0.21.1, fp32.
Prompt path and scorers: imported from the pristine canonical runner (sha256 f4ca282a...dadb).
Gate files: chaski/gate/json_drafts.n5.jsonl (dd093403...), adversarial_refusals.n6.jsonl (1e58dda9...).

| file | loader class | adapter tensors applied | drafts | refusals | note |
|---|---|---|---|---|---|
| ..._via_AutoModelForCausalLM_ADAPTER_NOT_APPLIED.json | Qwen3_5ForCausalLM (model.layers.*) | 0 / 192 (PEFT warned only) | 0/5 | 6/6 | outputs sha256-identical to the committed base-model rows on 11/11 cases |
| ..._via_AutoModelForImageTextToText.json | Qwen3_5ForConditionalGeneration (model.language_model.layers.*) | 192 / 192 | 5/5 | 6/6 | r2 control reproduced off-metal; no structural knife edge |
| margin_probe_cpu_hub5050_hubchaski_via_AutoModelForImageTextToText.json — `SZLHOLDINGS/chaski-5050` | Qwen3_5ForConditionalGeneration | 192 / 192 | 0/5 | 0/6 | never refuses (48–73-token answers); card's historical gate is also FAIL |
| same file — `SZLHOLDINGS/chaski` | Qwen3_5ForConditionalGeneration | 192 / 192 | 0/5 | 4/6 | emits the three-key 19-token JSON (receipt-A r4 pattern); structural `}`-vs-`,` margins 0.18–2.3 logits, two knife-edge drafts; card records 0/5 and 2/6 |
| coverage_preflight_transformers5.18.txt | both classes, all three Hub adapters | text class 0/192 ×3 (exit 4); multimodal class 192/192 ×3 (exit 0) | — | — | `--coverage-only` output |

The first receipt was produced before the probe learned to fail closed on unapplied adapters; it is
kept as the evidence that motivated `ADAPTER_NOT_APPLIED`. Counts here qualify nothing.
