# SZL Forge

**Train SZL-1 — SZL Holdings' own model — on SZL's own metal.**

SZL Forge is a small, self-contained fine-tuning kit. It turns an open base
model into **SZL-1**: a sovereign model whose identity, doctrine, and weights
live on SZL's own hardware and are served through Ollama like any other estate
model.

## What SZL Forge is

- A **QLoRA fine-tuning kit** built on [Unsloth](https://github.com/unslothai/unsloth).
- It fine-tunes the open base model **`unsloth/Qwen2.5-3B-Instruct`** (4-bit)
  into SZL-1, then merges to a 16-bit safetensors folder Ollama can import.
- Designed to run on **one laptop GPU** — an RTX 5050 (8 GB VRAM). A 3B QLoRA
  run fits in roughly 5–8 GB of VRAM.
- The result is owned end-to-end: owned weights, owned hardware, owned doctrine —
  no rented cloud inference, no vendor lock-in.

## One command (laptop)

The whole pipeline — folder, kit files, Unsloth, the CUDA-build torch the
RTX 5050 needs, training, and the Ollama import — in a single PowerShell
command (run from ANY folder; it puts itself in `%USERPROFILE%\szl-forge`):

```powershell
iwr https://raw.githubusercontent.com/szl-holdings/szl-forge/main/forge.ps1 -OutFile "$env:TEMP\forge.ps1"; powershell -ExecutionPolicy Bypass -File "$env:TEMP\forge.ps1"
```

It prints each step honestly and stops on the real error if one appears.
Prefer step-by-step? [`RUNBOOK.md`](./RUNBOOK.md) is the same pipeline as
one command per step.

## What SZL Forge is NOT

- It is **not from-scratch pretraining**. Training a frontier model from raw
  tokens genuinely requires datacenter-scale metal (many high-end GPUs, weeks of
  compute). Nobody should claim otherwise, and this kit does not.
- It is **fine-tuning an already-open base model** into SZL's own — a real,
  honest, achievable thing on a single laptop, and no more than that.

## Kit contents

| File | What it is |
| --- | --- |
| `forge.ps1` | **One-command bootstrap** — runs the entire pipeline below (downloads kit, fixes CUDA torch, trains, imports into Ollama), stopping honestly on any real failure. |
| [`RUNBOOK.md`](./RUNBOOK.md) | Step-by-step, one-command-per-step runbook for running the whole pipeline on the laptop. |
| `train_szl.py` | Unsloth QLoRA training script: loads the 4-bit base, applies LoRA, trains, merges to `./szl-model` (16-bit safetensors). |
| `szl_dataset.jsonl` | 41 chat-format training examples encoding SZL-1's identity and honesty doctrine. |
| `Modelfile` | Ollama import recipe (`FROM ./szl-model`) with the SZL-1 system prompt and chat template. |
| [`RUNBOOK-NEMO.md`](./RUNBOOK-NEMO.md) | One-command-per-step runbook to put **SZL-Nemo** (doctrine-wrapped NVIDIA Nemotron 3 Nano 4B) on the tower. |
| `Modelfile.nemo` | Ollama recipe for SZL-Nemo (`FROM nemotron-3-nano:4b` + SZL doctrine system prompt — a wrapper, not an SZL fine-tune). |
| `conjecture_machine.py` | **Conjecture Machine** — points the sovereign model at the formula corpus, asking each formula for an *advisory* proof sketch / lemma decomposition / counterexample search. Stdlib-only. NEVER claims proven. |
| [`RUNBOOK-CONJECTURE.md`](./RUNBOOK-CONJECTURE.md) | One-command-per-step runbook to run the Conjecture Machine against the sovereign endpoint. |
| `thesis_formula_index.json` | Local snapshot of the estate's `thesis-formula-index` (80 entries) so the Conjecture Machine runs offline. |

## Pipeline

```
szl_dataset.jsonl
      │  (identity + doctrine examples)
      ▼
Unsloth QLoRA fine-tune  ──  train_szl.py  (base: unsloth/Qwen2.5-3B-Instruct)
      │
      ▼
merged 16-bit safetensors  ──  ./szl-model
      │
      ▼
ollama create szl1 -f Modelfile
      │
      ▼
serve as SOVEREIGN_MODEL=szl1   (Alloy cockpit runs on SZL-1)
```

See **[RUNBOOK.md](./RUNBOOK.md)** for the exact commands, VRAM/disk
requirements, and Windows-specific notes.

## SZL-Nemo (NemoClaw pattern)

The LangChain x NVIDIA **NemoClaw Deep Agents blueprint** (July 2026) pairs an
open model, a tuned agent harness, and a governed runtime — tuned together.
SZL's estate maps onto all three layers: open weights on SZL metal (this kit),
the Alloy backbone as harness, and SZL's receipt/guardrail stack as governance.

**SZL-Nemo** is the estate's open-model slot for that pattern: NVIDIA's open
`nemotron-3-nano:4b` (2.8 GB, 256K context — REPORTED from ollama.com) wrapped
in the SZL honesty-doctrine system prompt via `Modelfile.nemo`. Honest tier:
a **wrapper, not an SZL fine-tune** — SZL has not trained these weights, and
no benchmarks have been measured on SZL hardware yet. See
[`RUNBOOK-NEMO.md`](./RUNBOOK-NEMO.md).

## Conjecture Machine

`conjecture_machine.py` points the sovereign stack at SZL's own formula corpus:
it loads a formula index (a local snapshot of the estate's `thesis-formula-index`
ships in this repo — **80 entries**), iterates each formula, and asks the
sovereign OpenAI-compatible endpoint (default `https://gpu.a-11-oy.com/v1`,
model env-selectable — `llama3-szl-finetuned-q4:latest` or `szl-nemo`) for an
**advisory** proof sketch, lemma decomposition, and counterexample search. Every
attempt is written to `conjecture_runs/<timestamp>/<formula_id>.json`.

Honest tier, hard-coded:

- It **NEVER claims proven.** Model output is an advisory *sketch* only; a
  formula stays a **CONJECTURE** unless a real Lean check passes (done in
  `szl-holdings/lutar-lean`, not here — the built-in `lean_check` is an honest
  stub that never passes). The corpus's own status labels are staged-advisory
  tags, not live proofs, and are not laundered into "proven".
- **Λ uniqueness stays Conjecture-1.** The Λ-uniqueness formula
  (`TH10 — Uniqueness of Lutar Invariant`) is doctrine-locked and can never be
  upgraded by this tool.
- **Endpoint down ⇒ honest UNAVAILABLE**, no fabricated output. Stdlib-only
  (Python 3.8+, no `pip install`).

See **[RUNBOOK-CONJECTURE.md](./RUNBOOK-CONJECTURE.md)** for the one-command-per-step
walk-through.

## Honesty doctrine

SZL-1 is trained to hold to SZL's honesty doctrine: label claims **MEASURED**,
**REPORTED**, or **UNKNOWN**, and let an honest **UNKNOWN** stand rather than
invent an answer. The system prompt baked into `Modelfile` reinforces this at
serving time.

## Benchmarks

**None yet.** No training run has been executed end-to-end on the laptop at the
time of writing, so there are **no measured quality, speed, or accuracy numbers
to report**. Any such figures will be added only once they are genuinely
measured on real hardware — until then, treat performance as UNKNOWN.
