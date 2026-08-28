---
license: apache-2.0
language:
  - en
pipeline_tag: text-generation
library_name: llama.cpp
base_model: Qwen/Qwen3.5-0.8B
# No Hub quantized relation. Live Chaski is not publication-eligible.
# This SKU is not a quantized Hub child until a .gguf exists on SZLHOLDINGS/A11OY-MINI.
tags:
  - szl-holdings
  - series-a
  - doctrine-v11
  - governed-ai
  - proposal-only
  - gguf
  - roadmap
szl:
  doctrine: v11-LOCKED
  lean: "749/14/163"
  lambda: "Conjecture 1 — advisory, never a theorem"
  artifact_class: GGUF
  originality: QUANT_OF_SZL_ORIGINAL
  parent: SZLHOLDINGS/chaski
  forbidden_parent: SZLHOLDINGS/chaski-5050
  parent_artifact: merged-shard
  base: Qwen/Qwen3.5-0.8B
  seed: 11
  evals: none-this-run
  quality: ROADMAP
  publication_eligible: false
  autonomy_eligible: false
  status: ROADMAP
  hub_put: false
  khipu_lab_pin: false
  tok_s_claim: false
  third_llm: false
  new_train: false
---

# A11OY-MINI

**Later GGUF SKU of live [`SZLHOLDINGS/chaski`](https://huggingface.co/SZLHOLDINGS/chaski).** Not a new train. Not a third LLM. Not [`SZLHOLDINGS/chaski-5050`](https://huggingface.co/SZLHOLDINGS/chaski-5050).

GitHub conversion scripts are in this folder. **ROADMAP until a `.gguf` file exists locally.** This checkout does not Hub PUT GGUF bytes or an empty parent. FORGE pushes after the PR exists.

| | |
|---|---|
| **SKU** | `SZLHOLDINGS/A11OY-MINI` |
| **Parent** | live `SZLHOLDINGS/chaski` **merged shard** (`model.safetensors-00001-of-00001.safetensors`) |
| **Not parent** | `SZLHOLDINGS/chaski-5050` |
| **Base / silhouette** | `Qwen/Qwen3.5-0.8B` (Apache-2.0, Qwen3.5 instruct). Cut is live SZL Chaski. |
| **Convert** | llama.cpp F16 (`convert_hf_to_gguf.py --outtype f16`), then `Q4_K_M` |
| **Banned** | Direct safetensors→Ollama (MEASURED 2026-07-12 `@` spam) |
| **Evals** | none-this-run (inherited from live Chaski). Not 5/5. Not MEASURED. |
| **Quality** | ROADMAP |
| **Publication** | `publication_eligible: false` |
| **Lab** | House CPU lab stays **Khipu GGUF**. Do not retarget inference-lab. |
| **License** | Apache-2.0 |
| **Doctrine** | v11 LOCKED. Seed 11. Λ = Conjecture 1 (advisory, never a theorem). |

> **CHAWPI silhouette.** Silhouette from Qwen3.5 instruct. Cut is original live Chaski. This folder exports that cut to GGUF. We do not republish someone else's tensors and we do not train a third messenger.

## Scripts

```bash
python a11oy_mini/convert_a11oy_mini_gguf.py
python a11oy_mini/eval_a11oy_mini.py
python a11oy_mini/serve_a11oy_mini.py --check
```

Convert, when the live Chaski merge is on disk at `a11oy_mini/chaski-merged/`:

```bash
python a11oy_mini/convert_a11oy_mini_gguf.py --convert --fetch-llama-cpp
```

That is the szl-forge llama.cpp rebirth path in Python, not PowerShell:

1. F16 GGUF via `llama.cpp/convert_hf_to_gguf.py --outtype f16`
2. Then `Q4_K_M` via `llama-quantize` (or `ollama create --quantize q4_K_M` **FROM the F16 GGUF**)

`ollama create` from a safetensors directory is refused.

Default `--status` writes `conversion_receipt.json` with `publication_eligible: false`, `evals: none-this-run`, parent live Chaski, `hub_put: false`. Bytes are labeled MEASURED only when a local `.gguf` hash is written.

Eval without a local GGUF stays honest: `evals=none-this-run`, `quality=ROADMAP`, `gguf=UNAVAILABLE`. A later hash is bytes MEASURED, not an eval.

Serve pins `SZLHOLDINGS/A11OY-MINI` only. It refuses live Chaski overwrite and refuses a 5050 parent. It does not pin `SZLHOLDINGS/SZL-Khipu-1.5B-GGUF` or the inference lab.

## What this is NOT

- Not a new train and not a third LLM
- Not `chaski-5050`
- Not a `base_model_relation: quantized` Hub child (parent is not publication-eligible)
- Not the Khipu lab pin
- Not a tokens/s claim
- Not a Hub PUT from this checkout
