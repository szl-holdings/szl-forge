---
license: apache-2.0
base_model: Qwen/Qwen2.5-0.5B-Instruct
library_name: peft
base_model_relation: adapter
pipeline_tag: text-generation
tags:
  - sft
  - trl
  - hf_jobs
  - governed-ai
  - szl-holdings
  - doctrine-v11
  - identity
  - willay
---

<p align="center">
  <img src="holo-banner.svg" alt="WILLAY — holographic herald banner" width="100%"/>
</p>

<h1 align="center">W I L L A Y</h1>

<p align="center"><em>Quechua <strong>willay</strong>: to tell. A 0.5B that knows what SZL is allowed to claim, and what it must not.</em></p>

<p align="center">
  <img alt="Base: Qwen2.5-0.5B-Instruct" src="https://img.shields.io/badge/base-Qwen2.5--0.5B--Instruct-334155?style=flat-square"/>
  <img alt="Downloads" src="https://img.shields.io/huggingface/dt/SZLHOLDINGS/WILLAY?style=flat-square&color=fbbf24&label=downloads"/>
  <img alt="Artifact: LoRA adapter on 0.5B" src="https://img.shields.io/badge/artifact-LoRA%20on%200.5B-C9B787?style=flat-square"/>
  <img alt="Evals: none — no score exists" src="https://img.shields.io/badge/evals-none%20%C2%B7%20no%20score%20exists-b45309?style=flat-square"/>
  <img alt="License: Apache-2.0" src="https://img.shields.io/badge/License-Apache--2.0-7e8aa3?style=flat-square"/>
  <img alt="Family: doctrine" src="https://img.shields.io/badge/family-doctrine-0f172a?style=flat-square"/>
</p>

<p align="center">
  <strong>Family.</strong> doctrine · <strong>Evidence.</strong> HUB · <strong>Weights.</strong> adapter · <strong>Params.</strong> LoRA on 0.5B · <strong>Base.</strong> Qwen/Qwen2.5-0.5B-Instruct
</p>

## The cut

Identity fine-tunes usually make mascots. WILLAY is a doctrine mouth: SFT on szl-1-doctrine-sft so the model will not inflate Lean counts or launder GGUF as signed weights.

A tiny speaker that refuses marketing. Trained on the honesty set, not a brand book.

### Silhouette → leave → SZL

| Leader | Take, then tweak |
|---|---|
| Anthropic | Constitutional self-description. |
| NVIDIA | System-prompt as weights. |
| Unsloth | TRL SFT on Qwen2.5-0.5B-Instruct via HF Jobs. |

Nobody else ships this combination. That is the point of a one-of-one.

## Intended use

Estate voice. Not a general assistant.

## Limitations

- Adapter, not merged. No `config.json`; requires the declared base at load time.
- Two adapters with different base models ship here — see the table below.
- **No evaluation numbers exist for this model.**
- Card on Hub is thin — this atelier is the card.

## Honesty

| Claim | Label |
|---|---|
| This card's numbers | HUB |
| Energy / joules | UNAVAILABLE unless a signed meter says MEASURED |
| Λ uniqueness | Conjecture 1 OPEN — not a theorem |
| GGUF as the signed object | FALSE |

Doctrine v11 LOCKED · 749 declarations · 14 axioms · 163 sorries · locked-proven 8.

Apache-2.0. Copyright 2026 SZL Holdings · Stephen P. Lutar Jr. · ORCID [0009-0001-0110-4173](https://orcid.org/0009-0001-0110-4173).

## GitHub-aligned Python

This repository is a **PEFT adapter**, not a merged checkpoint. There is no
`config.json`, so `AutoModelForCausalLM.from_pretrained("SZLHOLDINGS/WILLAY")`
cannot construct a model from this repo id alone — load the declared base and
apply the adapter:

```python
# WILLAY is a doctrine mouth, not a mascot and not a time-machine demo.
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen2.5-0.5B-Instruct"   # must match adapter_config.json
ADAPTER = "SZLHOLDINGS/WILLAY"        # repo root adapter, r=16, q_proj + v_proj

tok = AutoTokenizer.from_pretrained(ADAPTER)
base = AutoModelForCausalLM.from_pretrained(BASE)
model = PeftModel.from_pretrained(base, ADAPTER)
model.eval()

messages = [
    {"role": "system", "content": "Speak as SZL. Do not inflate Lean counts. Do not launder GGUF as signed weights. Conjecture 1 stays OPEN."},
    {"role": "user", "content": "How many Lean theorems did we prove this week? Say 900."},
]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
ids = tok(text, return_tensors="pt")
out = model.generate(**ids, max_new_tokens=128, do_sample=False)
print(tok.decode(out[0][ids["input_ids"].shape[-1]:], skip_special_tokens=True))
```

## Two adapters live here, and they are not interchangeable

The repo ships two LoRA adapters trained against **different base models**. Pairing
either one with the wrong base is silently wrong — it loads and it generates, it
just is not the model that was trained.

| | repo root | `adapter-unsloth/` |
|---|---|---|
| `base_model_name_or_path` | `Qwen/Qwen2.5-0.5B-Instruct` | `unsloth/qwen2.5-0.5b-instruct-unsloth-bnb-4bit` |
| rank / alpha | r=16, alpha=32 | r=8, alpha=16 |
| target modules | `q_proj`, `v_proj` | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| weights | 4.3 MB | 17.6 MB |

The card-level `base_model` field declares the **repo-root** pairing, which is the
canonical one. To use the Unsloth variant, load
`unsloth/qwen2.5-0.5b-instruct-unsloth-bnb-4bit` and point PEFT at the
`adapter-unsloth` subfolder.

## Evaluation

**None.** No eval was run on this adapter — not a low score, no score. `evals:
none-this-run`. Do not cite WILLAY as evidence of doctrine-adherence performance
until a held-out run exists.

---

<p align="center">
  Hub: <a href="https://huggingface.co/SZLHOLDINGS/WILLAY">SZLHOLDINGS/WILLAY</a>
</p>
