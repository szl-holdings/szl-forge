# SZL Forge
<!-- szl:header v1 -->
<!-- Native identity: szl-brand@4309f6fce89179331b0b3ed8c99494e76c5be256; Orbit v2 icon. -->
<p>
  <a href="https://github.com/szl-holdings">
    <img src="https://raw.githubusercontent.com/szl-holdings/szl-brand/4309f6fce89179331b0b3ed8c99494e76c5be256/kit/logos/orbit-v2/suite/szl-icon-32.svg" alt="SZL Holdings — Orbit v2" width="32" height="32" />
  </a>
</p>

**Control before action. Evidence after.**

[Product](https://a-11-oy.com) · [Proof](https://a11oy.net) · [GitHub organization](https://github.com/szl-holdings)
<!-- /szl:header -->

## Scope and data boundary

`clinical-gateway/` is outside this statement. It is a separately bounded
integration surface for synthetic and already-deidentified result traffic,
described in the OAC Clinical Gateway section below. The rest of this kit is
not for clinical use, and nothing in it may receive PHI, patient data or
clinical result content.

**Train, qualify, and publish bounded SZL model artifacts from explicit source and evidence.**

SZL Forge is the canonical training-recipe, qualification, and publication
control repository for several SZL model artifacts. This GitHub repository is
software, curriculum, schemas, and evidence; it is not itself a weight repo.

## Portfolio truth card

| Surface | Classification | Evidence state | Boundary |
|---|---|---|---|
| Forge bootstrap and runbooks | **Executable software + training recipe** | Source-controlled and locally runnable | A recipe does not prove a run completed or that its output matches a published model. |
| [`agent-forge/`](./agent-forge/) | **Beta local Windows control software** | Portable contracts are source-controlled; Windows enforcement is qualified only by its behavioral self-test | Controls only registered process trees it launches under the same Windows identity. A11oy receives a read-only projection and no process-control authority. |
| [`clinical-gateway/`](./clinical-gateway/) | **Beta synthetic and pre-deidentified live-shadow integration software** | Source and local contract tests cover the bounded result engine, TLS MLLP listener, file drop, read-only polling, local API, and operator UI | Not site-validated, not authorized for real PHI or clinical use, and not evidence of physical-analyzer interoperability or LIS/EHR delivery. Models and kernels are excluded from result interpretation and release. |
| [`SZL-Forge-1.5B-ReceiptAgent`](https://huggingface.co/SZLHOLDINGS/SZL-Forge-1.5B-ReceiptAgent) | **Trained fine-tuned weights** | `MEASURED_LIMITED`; expected public weight hashes and owner-signed training/evaluation receipts are bound in `publishing/model-source-bindings.json` | Proposal-only, not promoted, and not independently certified. |
| [`SZL-Khipu-1.5B`](https://huggingface.co/SZLHOLDINGS/SZL-Khipu-1.5B) | **Trained fine-tuned weights** | `MEASURED_RESEARCH_ONLY`; repository-declared key continuity | Held-out abstention is 2/6; autonomous and high-stakes use is prohibited. |
| [`KHIPU-R2`](./khipu_r2/) | **Separate SKU** | Hub job `6a91bf11984507d9db4ea104` COMPLETED; adapter 147.8MB AVAILABLE; abstain MEASURED 3/6 (not a pass; grounding 5/5, plan 11/11). Hub receipt `ddf6c50` `publication_eligible: false` is the public claim. This-kit jobs UNKNOWN; this-SKU evals not-this-run; CHAWPI extra lock (no `--run-job`). | Does **not** overwrite signed `SZL-Khipu-1.5B`. Signed 1.5B abstain stays MEASURED 2/6. Lab stays signed Khipu GGUF. |
| [`CHASKI-R2`](./chaski_r2/) | **Separate SKU (CHAWPI silhouette, ATELIER lock)** | Jobs UNAVAILABLE this checkout; evals none-this-run; `publication_eligible: false` | GitHub recipe on disclosed `Qwen/Qwen3.5-0.8B`. Declared Hub id `SZLHOLDINGS/chaski-r2` is not a Hub page. Does **not** overwrite live `SZLHOLDINGS/chaski`. Not 5050. |
| [`SZLHOLDINGS/chakana`](./chakana/) | **Bridge embedding kit (ROADMAP recipe)** | Jobs UNKNOWN this checkout; Hub stub has no weights; nDCG@10 UNKNOWN | NINA (FORGE-class) lane. sentence-transformers. Not a11oy CHAKANA wiring. Does **not** overwrite other Fall 2026 kits. |
| [`SZL-Khipu-1.5B-GGUF`](https://huggingface.co/SZLHOLDINGS/SZL-Khipu-1.5B-GGUF) | **Quantized derivative** | Exact GGUF bytes are hash-bound | Reproducible quantization and signed runtime outputs are not claimed. |
| [`szl-receiptagent-qwen35-0.8b-v2`](https://huggingface.co/SZLHOLDINGS/szl-receiptagent-qwen35-0.8b-v2) | **Trained LoRA adapter** | Bounded owner-measured acceptance: 5/5 contract drafts and 6/6 adversarial refusals | Small synthetic gate, proposal-only, no broad quality or safety benchmark. |
| `SZL-Khipu-1.5B-BrainNavigator` card source | **Historical/planned card name** | Superseded by the measured `SZL-Khipu-1.5B` binding | Do not treat the old name or model-card template as a separate trained release. |
| Forge Lab / Model Inference Lab | **Presentation and bounded runtime Spaces** | Snapshot verification or constrained inference, depending on the Space | Reachability is transport only; neither Space trains, promotes, or authorizes a model. Forge Lab is SNAPSHOT / `BLUEPRINT_NOT_TRAINED` (ATELIER lock: no Unsloth Studio, no Jobs launcher). Bounded serve is [`szl-model-inference-lab`](https://szlholdings-szl-model-inference-lab.hf.space) only. |
| Fall 2026 original cuts (`SZLHOLDINGS/{chaski,qantu,waman,chakana,tinku}`) | **GitHub recipes + honest job stamps** | Chaski CUTTING (attempt 5 `6a91bf10` COMPLETED; weights AVAILABLE; Named-N MEASURED fail 0/5 and 2/6; `publication_eligible: false`). Qantu/Waman SKIP. Chakana/Tinku Jobs UNKNOWN. | Named-N fail is not a pass. Train loss is not an eval. No Qantu/Waman trainers. KILLINCHU-EYE is a Waman alias. Hub README is not recut from this checkout. See [`chaski/README.md`](./chaski/README.md). |

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

## OAC Clinical Gateway v2.5

[`clinical-gateway/`](./clinical-gateway/) is the separately bounded integration
surface for synthetic and already-deidentified cobas liat-shaped result traffic.
It implements a local control engine, an analyzer-initiated TLS MLLP host
listener, deterministic `ACK^R33` responses, bounded file and polling adapters,
a loopback API, and an operator UI. It produces signed, offline FHIR R4
candidate bundles; it does not publish those bundles into a clinical system.

The separate [OAC System Health v1 model](https://huggingface.co/SZLHOLDINGS/oac-system-health-v1)
and [synthetic operational dataset](https://huggingface.co/datasets/SZLHOLDINGS/oac-clinical-transport-observability-synthetic)
provide a reproducible, standard-library baseline for operator-attention
advisories. Its 1,200 generated examples are not gateway event records or
clinical data. The inference kernel verifies artifact receipts and cannot
acknowledge a message, command a device, interpret a result, or authorize release.

The gateway intentionally provides no analyzer command, order download,
calibration, test-start, result reinterpretation, automatic release, or clinical
alert path. Its operational dataset and control-evidence score contain only
local run observations and must never be treated as clinical evidence or a
training corpus. Start with [`clinical-gateway/README.md`](./clinical-gateway/README.md)
and keep every deployment claim at `NOT_SITE_VALIDATED` until the exact device,
site certificate, assay mappings, network, privacy controls, laboratory workflow,
and destination reconciliation have been witnessed and approved.

## Historical one-command bootstrap (held)

The former pipeline downloaded mutable inputs, wrote the fixed `szl-model`
path, and replaced the canonical `szl1` Ollama name. It is retired for current
recovery: `forge.ps1` now stops before training or import. Use
[`local-compute/README.md`](./local-compute/README.md) for bounded new
experiments and [recovery issue #264](https://github.com/szl-holdings/szl-forge/issues/264)
for the held historical Qwen exports. `RUNBOOK.md` preserves the historical
step-by-step recipe, not authorization to train, import, route, or publish `szl1`.

## If first words are garbage (`@@@@…`)

The July 2026 observation of repeated `@` output did not prove whether import,
conversion, merge, or source weights caused it. A later frozen smoke run found
both historical Qwen exports still produced repeated `@`, and the bounded GGUF
forensic run stopped at an unsupported tensor type before full-blob
authentication. The cause and clean source lineage remain unproven.

Do not rerun `rebirth.ps1`, delete or replace either installed model, convert
unknown bytes as a repair, or route either model to production. The script now
fails closed without modifying model state. Preserve the current blobs and
follow [issue #264](https://github.com/szl-holdings/szl-forge/issues/264) for
separately admitted diagnostics and a separately named candidate.

## What SZL Forge is NOT

- It is **not from-scratch pretraining**. Training a frontier model from raw
  tokens genuinely requires datacenter-scale metal (many high-end GPUs, weeks of
  compute). Nobody should claim otherwise, and this kit does not.
- It is **fine-tuning an already-open base model** into SZL's own — a real,
  honest, achievable thing on a single laptop, and no more than that.

## Kit contents

| File | What it is |
| --- | --- |
| `forge.ps1` | **Historical bootstrap, held** — fails closed before training or import. Use the bounded local-compute lane for new experiments. |
| [`RUNBOOK.md`](./RUNBOOK.md) | Step-by-step, one-command-per-step runbook for running the whole pipeline on the laptop. |
| `train_szl.py` | Historical Unsloth QLoRA recipe: refuses to overwrite existing legacy output and labels a new merge experimental, not qualified for canonical import. |
| `szl_dataset.jsonl` | 41 chat-format training examples encoding SZL-1's identity and honesty doctrine. |
| `rebirth.ps1` | **Historical import helper, held** — fails closed without deleting or replacing the canonical `szl1` model. |
| `Modelfile.gguf` | Historical GGUF Ollama recipe (`FROM ./szl1-f16.gguf`); not a qualified recovery path. |
| `Modelfile` | Historical direct-import recipe (`FROM ./szl-model`), retained for provenance and not authorized for canonical recovery. |
| [`RUNBOOK-NEMO.md`](./RUNBOOK-NEMO.md) | One-command-per-step runbook to put **SZL-Nemo** (doctrine-wrapped NVIDIA Nemotron 3 Nano 4B) on the tower. |
| `Modelfile.nemo` | Ollama recipe for SZL-Nemo (`FROM nemotron-3-nano:4b` + SZL doctrine system prompt — a wrapper, not an SZL fine-tune). |
| `conjecture_machine.py` | **Conjecture Machine** — points the owner-controlled model at the formula corpus, asking each formula for an *advisory* proof sketch / lemma decomposition / counterexample search. Stdlib-only. NEVER claims proven. |
| [`RUNBOOK-CONJECTURE.md`](./RUNBOOK-CONJECTURE.md) | One-command-per-step runbook to run the Conjecture Machine against the owner-operated endpoint. |
| `thesis_formula_index.json` | Local snapshot of the estate's `thesis-formula-index` (80 entries) so the Conjecture Machine runs offline. |
| `chaski/bakeoff_named_n.py` | Owner-metal named-N JSON-draft (n=5) + refusal (n=6) bake-off: base `Qwen/Qwen3.5-0.8B` vs local chaski-5050 and chaski-r2 adapters. Writes a MEASURED receipt. Gate files stay held-out. `publication_eligible` stays false. |

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

The [immutable ReceiptAgent v2 byte verifier](docs/published-artifact-byte-verification.md)
replays the repository-declared Ed25519 receipt chain against external OpenSSL
and hashes the actual opaque adapter bytes. It reports source, artifact and
signature integrity separately from signer trust and production authority.

## Kernel Hub ownership

szl-forge does not publish to the converged SZL kernel Hub repositories. Since
2026-10-13 each kernel source repository's own
`.github/workflows/hf-kernel-publish.yml` is the single writer of its
first-class kernel repo (`huggingface.co/kernels/SZLHOLDINGS/<name>`) and of its
same-named model twin, which it keeps as a pointer card.

| Hub target | Owner (single writer) |
| --- | --- |
| `kernels/SZLHOLDINGS/szl-kernels` | `szl-holdings/szl-kernels` `hf-kernel-publish.yml`; the `SZLHOLDINGS/szl-kernels` model repo (SZL-MiniEmbed host, no twin) is written by that repository's own source-binding publisher |
| `szl-invariants`, `szl-govsign`, `szl-maskmod`, `szl-receipt-attn`, `szl-blocked`, `szl-ouroboros`, `szl-provctl`, `szl-block-kv`, `YARQA-ATTN` (kernel repo and model twin) | the same-named `szl-holdings/<name>` repository's `hf-kernel-publish.yml` |
| `szl-formulas` kernel repo + model twin | `szl-holdings/szl-formulas` `hf-kernel-publish.yml` |
| `szl-lambda-gate`, `szl-governed-norm` (kernel repos and model twins) | `szl-holdings/szl-lambda-gate` `hf-kernel-publish.yml` |

The closed registry is [`publishing/kernel-hub-ownership.json`](publishing/kernel-hub-ownership.json).
`tests/test_kernel_hub_ownership.py` fails if a tracked workflow or tool that
can write to the Hub names one of these targets, or if a retired writer
returns, and `tools/publish_kernel_mirror_cards.py` refuses any profile that
targets them. The retired Forge writers were `publish-szl-kernels.yml`,
`publish-szl-invariants.yml`, `reconcile-kernel-provenance.yml`,
`publish-compatibility-kernels.yml`, the six converged kernel profiles of
`publish-kernel-mirror-cards.yml` (szl-formulas last, once its own publisher
was live), and the converged twins in the joblib quarantine workflows,
together with their authorizers, credential helpers, verifiers and tests. Git
history keeps them.

## Doctrine gate (dataset content)

`tools/validate_sft_dataset.py` proves a dataset is well-formed;
`tools/nemo_doctrine_gate.py` proves its **content** conforms to doctrine.
Every training/eval JSONL in this repo is gated through the
[`szl-nemo`](https://github.com/szl-holdings/szl-nemo) kernel (`rule_check`
R1–R5) — fabrication labels, honest unknowns, Λ-is-not-a-theorem, the 0.97
trust ceiling, and persona-correct fine-tune provenance — and each run is
bound into a deterministic, hash-chained `szl.nemo.receipt.v1` ledger.
CI enforces it on every PR that touches doctrine-bearing data
(`.github/workflows/nemo-doctrine-gate.yml`, kernel pinned by commit SHA).
Current state (MEASURED 2026-09-01): all five gated datasets VALID, 75
records, 0 violations.

## Historical SZL-1 pipeline (held; not an operational procedure)

```
szl_dataset.jsonl
      │  (identity + doctrine examples)
      ▼
 Historical QLoRA recipe  ──  train_szl.py  (unqualified output)
      │
      ▼
 experimental 16-bit merge  ──  ./szl-model  (not qualified)
      │
      ▼
 GGUF conversion  ──  HELD (clean source and converter lineage unproven)
      │
      ▼
 canonical Ollama import  ──  BLOCKED (do not run)
      │
      ▼
HOLD: do not route as SOVEREIGN_MODEL=szl1 without source and runtime qualification
```

See **[RUNBOOK.md](./RUNBOOK.md)** for the historical steps, VRAM/disk
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
