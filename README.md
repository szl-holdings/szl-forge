# SZL Forge

**Train, qualify, and publish bounded SZL model artifacts from explicit source and evidence.**

SZL Forge is the canonical training-recipe, qualification, and publication
control repository for several SZL model artifacts. This GitHub repository is
software, curriculum, schemas, and evidence; it is not itself a weight repo.

## Portfolio truth card

| Surface | Classification | Evidence state | Boundary |
|---|---|---|---|
| Forge bootstrap and runbooks | **Executable software + training recipe** | Source-controlled and locally runnable | A recipe does not prove a run completed or that its output matches a published model. |
| [`agent-forge/`](./agent-forge/) | **Beta local Windows control software** | Portable contracts are source-controlled; Windows enforcement is qualified only by its behavioral self-test | Controls only registered process trees it launches under the same Windows identity. A11oy receives a read-only projection and no process-control authority. |
| [`SZL-Forge-1.5B-ReceiptAgent`](https://huggingface.co/SZLHOLDINGS/SZL-Forge-1.5B-ReceiptAgent) | **Trained fine-tuned weights** | `MEASURED_LIMITED`; expected public weight hashes and owner-signed training/evaluation receipts are bound in `publishing/model-source-bindings.json` | Proposal-only, not promoted, and not independently certified. |
| [`SZL-Khipu-1.5B`](https://huggingface.co/SZLHOLDINGS/SZL-Khipu-1.5B) | **Trained fine-tuned weights** | `MEASURED_RESEARCH_ONLY`; repository-declared key continuity | Held-out abstention is 2/6; autonomous and high-stakes use is prohibited. |
| [`KHIPU-R2`](./khipu_r2/) | **Separate SKU** | Hub job `6a91bf11984507d9db4ea104` COMPLETED; adapter 147.8MB AVAILABLE; abstain MEASURED 3/6 (not a pass; grounding 5/5, plan 11/11). This-kit jobs UNKNOWN; this-SKU evals not-this-run; `publication_eligible: false`. | Does **not** overwrite signed `SZL-Khipu-1.5B`. Signed 1.5B abstain stays MEASURED 2/6. Lab stays signed Khipu GGUF. |
| [`SZL-Khipu-1.5B-GGUF`](https://huggingface.co/SZLHOLDINGS/SZL-Khipu-1.5B-GGUF) | **Quantized derivative** | Exact GGUF bytes are hash-bound | Reproducible quantization and signed runtime outputs are not claimed. |
| [`szl-receiptagent-qwen35-0.8b-v2`](https://huggingface.co/SZLHOLDINGS/szl-receiptagent-qwen35-0.8b-v2) | **Trained LoRA adapter** | Bounded owner-measured acceptance: 5/5 contract drafts and 6/6 adversarial refusals | Small synthetic gate, proposal-only, no broad quality or safety benchmark. |
| `SZL-Khipu-1.5B-BrainNavigator` card source | **Historical/planned card name** | Superseded by the measured `SZL-Khipu-1.5B` binding | Do not treat the old name or model-card template as a separate trained release. |
| Forge Lab / Model Inference Lab | **Presentation and bounded runtime Spaces** | Snapshot verification or constrained inference, depending on the Space | Reachability is transport only; neither Space trains, promotes, or authorizes a model. Forge Lab is SNAPSHOT / `BLUEPRINT_NOT_TRAINED` (ATELIER lock: no Unsloth Studio, no Jobs launcher). Bounded serve is [`szl-model-inference-lab`](https://szlholdings-szl-model-inference-lab.hf.space) only. |
| Fall 2026 original cuts (`SZLHOLDINGS/{chaski,qantu,waman,chakana,tinku}`) | **GitHub recipes + honest job stamps** | Chaski CUTTING (Hub adapter files as of 2026-08-28T17:08Z; evals none-this-run; live `6a91bf10` RUNNING). Qantu/Waman SKIP. Chakana/Tinku Jobs UNKNOWN. | Files on Hub are not an eval. No Qantu/Waman trainers. KILLINCHU-EYE is a Waman alias. Hub README is not recut from this checkout. See [`chaski/README.md`](./chaski/README.md). |

**Investor value.** Forge makes model maturity, public bytes, source lineage,
and promotion boundaries navigable without presenting recipes, derivatives, or
Spaces as equivalent to trained weights.

**Developer/evaluator quickstart.** Inspect the fail-closed portfolio binding
before running or publishing anything:

```bash
python tools/publish_model_source_bindings.py \
  --source-revision "$(git rev-parse HEAD)"
```

This local plan does not publish. Compare its model ID, source files, expected
weight hashes, receipt scope, promotion state, and limitations with the exact
Hub revision under evaluation.

## What SZL Forge is

- A **QLoRA fine-tuning kit** built on [Unsloth](https://github.com/unslothai/unsloth).
- A set of model-specific curricula, schemas, qualification tools, signed
  evidence, and protected publication contracts.
- A way to produce locally controlled model derivatives on supported hardware;
  hardware fit, run completion, and artifact quality must be observed per run.
- A release source that keeps trained weights, quantized derivatives, recipes,
  snapshots, and Spaces in separate evidence lanes.
- The source home for [Owned Agent Control v2](./agent-forge/), a separately
  packaged, fail-closed Windows process-tree supervisor with signed one-shot
  isolation requests and schema-bound context evidence.

## Owned Agent Control v2

`agent-forge/` turns the supplied Owned Agent Control design into an installable
Python package with an operator runbook, explicit threat model, portable contract
tests, a real Windows Server behavioral gate, and a read-only A11oy evidence
projection. It does not grant A11oy process authority, modify remote providers,
or claim to sandbox hostile code running as the same Windows user.

Start with [`agent-forge/README.md`](./agent-forge/README.md). Operational status
must remain `NOT_READY` until the exact Windows runtime passes `doctor` and
`self-test`; a Linux or packaging pass establishes portable contracts only.

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

## If first words are garbage (`@@@@…`)

MEASURED 2026-07-12: Ollama's **direct safetensors import** of the Unsloth
16-bit merge produced corrupted weights — szl1 answered `@` spam at
temperature 0, even in raw mode, and re-quantizing the imported model did not
fix it. Most likely the import path is at fault; the fix re-imports the
already-trained merge at `.\szl-model` properly (no retraining):

```powershell
iwr https://raw.githubusercontent.com/szl-holdings/szl-forge/main/rebirth.ps1 -OutFile "$env:TEMP\rebirth.ps1"; powershell -ExecutionPolicy Bypass -File "$env:TEMP\rebirth.ps1"
```

If rebirth STILL produces `@` spam, the merge itself is suspect — report back
for diagnosis rather than retraining (training is seeded, so an identical re-run
would likely reproduce the same merge; isolating merge vs converter comes first).
Future full `forge.ps1` runs birth via this GGUF path automatically (step 6).

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
| `rebirth.ps1` | **Birth/rebirth into Ollama via GGUF** — converts `./szl-model` to F16 GGUF with llama.cpp's pure-Python converter, then `ollama create --quantize q4_K_M`. Fixes the corrupted-voice import without retraining. |
| `Modelfile.gguf` | Ollama recipe used by `rebirth.ps1` (`FROM ./szl1-f16.gguf`) with the SZL-1 system prompt and chat template. |
| `Modelfile` | Legacy direct-import recipe (`FROM ./szl-model`). **Superseded** — direct safetensors import corrupted SZL-1's voice (MEASURED 2026-07-12: `@` spam at temperature 0). Kept for provenance. |
| [`RUNBOOK-NEMO.md`](./RUNBOOK-NEMO.md) | One-command-per-step runbook to put **SZL-Nemo** (doctrine-wrapped NVIDIA Nemotron 3 Nano 4B) on the tower. |
| `Modelfile.nemo` | Ollama recipe for SZL-Nemo (`FROM nemotron-3-nano:4b` + SZL doctrine system prompt — a wrapper, not an SZL fine-tune). |
| `conjecture_machine.py` | **Conjecture Machine** — points the sovereign model at the formula corpus, asking each formula for an *advisory* proof sketch / lemma decomposition / counterexample search. Stdlib-only. NEVER claims proven. |
| [`RUNBOOK-CONJECTURE.md`](./RUNBOOK-CONJECTURE.md) | One-command-per-step runbook to run the Conjecture Machine against the sovereign endpoint. |
| `thesis_formula_index.json` | Local snapshot of the estate's `thesis-formula-index` (80 entries) so the Conjecture Machine runs offline. |

## Exact-source model publication

`publishing/model-source-bindings.json` is the fail-closed release contract for
the qualified ReceiptAgent and Khipu Hub models. The publisher verifies the
expected public weight hashes, required signed receipts, and every canonical
source file before it writes `publication.json` to either model repository.
It then reads the exact Hub commit back and compares the published bytes.

The binding identifies the current GitHub source snapshot; it deliberately does
not claim that the weight bytes can be reproduced from source alone or that an
independent party certified model quality. Run the same gate locally without
publishing:

```bash
python tools/publish_model_source_bindings.py \
  --source-revision "$(git rev-parse HEAD)"
```

Publication is performed only from protected `main` by the dependent
`publish-bindings` job in `.github/workflows/publish-model-inference-lab.yml`,
after that same workflow verifies the exact live Space revision and using the
repository's encrypted Hugging Face organization credential.

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
llama.cpp convert_hf_to_gguf  ──  szl1-f16.gguf   (rebirth.ps1)
      │
      ▼
ollama create szl1 --quantize q4_K_M -f Modelfile.gguf
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

## Evaluation status

- ReceiptAgent Qwen2.5 and Khipu have repository-bound owner measurements, but
  remain limited or research-only according to `publishing/model-source-bindings.json`.
- ReceiptAgent Qwen3.5 v2 has a small preregistered acceptance gate. Its raw
  counts are evidence for that contract only, not a broad benchmark.
- Any recipe without a qualifying receipt remains **UNAVAILABLE / NOT RUN** for
  performance claims.
- `eval_szl.py` can produce a local result. Label that result **MEASURED** only
  for the exact model, hardware, inputs, and revision actually exercised.
