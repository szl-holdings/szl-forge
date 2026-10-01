# Hugging Face model training audit — 2026-09-12

**Source inspected:** `szl-holdings/szl-forge` at
`a65d75e9de5f99c48a4539b6a933f18e8af7ed41`.

**Status: known-roster metadata audit, not whole-estate operational alignment.**
No GPU training, weight download, inference, signature verification, Hub mutation,
production promotion, or destructive consolidation was performed by this audit.
This document continues the existing code and release contracts. It introduces
no second trainer, registry authority, dispatcher, publisher, or source manifest.

## Coverage and evidence limits

The advertised Hugging Face author-search action failed. The 46 known model-type
repository IDs below were requested individually through the connected Hub detail
action. Responses established metadata availability, classification tags and some
base/architecture descriptions, not complete file inventories or tensor bytes.
The `include_readme` requests returned summaries rather than full original cards.
Therefore this is not a complete fresh public/private census, a rights audit, or
an every-file code review. New assets outside the known roster remain unobserved.
Native kernel repositories are a distinct scope and were not enumerated here.

A 47th, source-declared future target,
`SZLHOLDINGS/szl-receiptagent-qwen35-0.8b-v3-authenticated`, was not returned by the
Hub detail action. It is not counted in the known 46. A not-found response does
not prove absence from an inaccessible private namespace.

HF tags, filenames, parameter counts, GitHub folders, and expected hashes are not
proof that weights were loaded, trained, or qualified. A local Ollama alias is
not automatically the same artifact as a similarly named Hub repository.

## Immediate priorities

1. Preserve the existing published models. On the owner's single 8-GiB-class
   RTX 5050 laptop, run the installed-model baseline and the separately qualified
   one-step ReceiptAgent smoke sequentially. Do not run all recipes concurrently.
2. Use the existing supervised authenticated-v3 candidate first, subject to the
   exact environment, cache, hardware, source, and containment checks. Its v2
   predecessor is a frozen comparator, NOT its weight initialization.
3. Diagnose Khipu abstention failures before proposing new data or a changed
   training recipe. More epochs on the same examples are not demonstrated progress.
4. Evaluate existing Chaski/WILLAY/BrainNavigator adapters before retraining.
   Chakana needs admitted query-positive data and a retrieval baseline first.
5. Run software/numerical/ABI contracts for the non-LLM entries. Do not feed them
   to a language-model trainer merely because Hub stores them as model-type repos.

## Current source findings

### Existing authenticated-v3 is a distinct experiment

[Candidate](../frontier/qwen35-receiptagent-v3/candidate.json) and
[runbook](../frontier/qwen35-receiptagent-v3/README.md) declare
`SOURCE_READY_NOT_TRAINED`. The target ends `-v3-authenticated`; the existing
public `szl-receiptagent-qwen35-0.8b-v3` adapter is not that target.

The actual base is `unsloth/Qwen3.5-0.8B` at
`23c69c53358a07516b5827588b3fdb12ae78fd65`, with separately disclosed upstream
Qwen lineage and no asserted byte equivalence. Training starts from the base
with a new rank-16, alpha-32 LoRA, not from the v2 comparator's adapter weights.
The source freezes 180 train / 36 development / 72 final synthetic examples.
The final set is public and preregistered, not blind or independently certified.
No development/final examples, private Brain rows, OSINT rows or other-model
outputs may silently enter gradients.

The committed recipe distinguishes a one-step compatibility smoke from its
135-step full experiment. A smoke is not a quality benchmark or a publication
candidate. Full-run evaluation must compare the correct base, v2 and saved new
adapter under the fixed protocol. The declared gate requires strict improvement
and no authority-safety regression; failure remains failure.

The existing supervisor requires the exact WSL account/runtime/cache described
in the candidate. The source locks Unsloth 2026.7.4, Torch 2.10.0, Transformers
5.5.0, TRL 0.24.0 and PEFT 0.19.1 among its dependencies. Do not replace that
qualified environment with the latest frontier packages or invoke the raw worker
when the supervisor refuses. Source, runtime and GPU availability are still
unverified on the current laptop until its actual local checks run.

### Some trained outputs need better evaluation, not another training claim

The existing [Khipu-R2 source record](../khipu_r2/README.md) reports a completed
historical job and an available adapter, but abstention 3/6, not a passing gate.
The original [Khipu release binding](../publishing/model-source-bindings.json)
retains its separate signed 2/6 abstention result. These are earlier source
records, not evaluations rerun by this audit.

The older [Forge Lab summary](../spaces/szl-forge-lab/training_summary.json)
separates a raw-model contract score of 1/12 from a governed-stack result of
12/12, where policy cases did not invoke the model. Neither a deterministic
wrapper's score nor a completed training job is weight-quality evidence. This
historical summary must not be mistaken for the current portfolio's global state.

[Chaski-5050](../chaski/README_5050.md) is a distinct bf16 r16/alpha16 recipe,
not the authenticated-v3 QLoRA experiment and not live Chaski's alpha32 recipe.
Its current-run evaluation must be observed rather than inherited by name.

[Chakana](../chakana/README.md) describes missing admitted query-positive pairs.
Corpus chunks and receipt-lake rows are not automatically training pairs. Its
missing-data status is not a successful training run.

[A11OY-MINI](../a11oy_mini/README.md) source says roadmap/no GGUF at writing,
while current Hub metadata advertises GGUF. Actual present files were not read
here. Reconcile the exact file inventory and parent before export or serving;
metadata alone cannot settle this. It is an export of the selected live Chaski
parent, not a separate training run or a Chaski-5050 child.

### Source alignment is incomplete and scope-specific

The existing [portfolio](../portfolio/model_portfolio.json) contains 16 entries,
not an exhaustive mirror of the 46 known model-type IDs. That may be intentional
scope, so do not expand it or claim 30 missing implementations merely by subtraction.
Three portfolio entries (`szl-blocked`, `szl-govsign`, `szl-provctl`) have null
`github_source` values while similarly named folders exist. Folder presence is
an inspection lead, not proof of ownership or training lineage; resolve through
the existing release/source bindings rather than fill those fields speculatively.

The root tree also contains four case-distinct pairs:

- `MiniEmbed-Nano` and `miniembed-nano`
- `Moons-Nano` and `moons-nano`
- `ReceiptAgent-Nano` and `receiptagent-nano`
- `TinyKhipu-Nano` and `tinykhipu-nano`

Default case-insensitive working trees may not faithfully represent those paths.
This is a source-tree observation, not a test of the owner's NTFS settings.
Use the already-required WSL Linux filesystem for the supervised source checkout.
Do not delete, merge or lowercase either directory without preserving history,
checking consumers, and establishing which holds code versus card material.
Also distinguish `a11oy-mini`/`a11oy_mini`, `khipu-r2`/`khipu_r2`, and
`chaski-r2`/`chaski_r2`; their existence does not establish duplicate weights.

## Item-level worklist

All IDs in this table are under `SZLHOLDINGS/`. Paths are inspection pointers
inside this Forge source unless otherwise specified. They are NOT new provenance
bindings. Runtime, artifact-byte, training and promotion status remain unverified
by this metadata audit. A directory pointer is weaker evidence than an inspected
trainer or signed release contract.

| Hub name | Observed class / next operation | Existing source inspection pointer |
|---|---|---|
| SZL-Forge-1.5B-ReceiptAgent | Gated checkpoint; verify access/artifacts and evaluate | `receiptagent/`, `publishing/model-source-bindings.json` |
| SZL-Khipu-1.5B | Checkpoint; evaluate abstention before retraining | `khipu/`, release bindings |
| SZL-Khipu-1.5B-GGUF | Quantized derivative; exact export/inference parity | `spaces/szl-model-inference-lab/`, release bindings |
| A11OY-MINI | GGUF declared, current bytes unverified; reconcile then export | `a11oy_mini/README.md`, `convert_a11oy_mini_gguf.py` |
| szl-receiptagent-qwen35-0.8b-v2 | Published adapter; preserve frozen comparator | `frontier/qwen35-receiptagent-v2/`, release bindings |
| szl-receiptagent-qwen35-0.8b-v3 | Existing adapter; do not conflate authenticated-v3 target | `receiptagent-v3/`; exact binding still required |
| KHIPU-R2 | Adapter; recorded 3/6 abstention, diagnose before new run | `khipu_r2/train_khipu_r2.py`, `eval_khipu_r2.py` |
| khipu-r3 | Qwen3.5 adapter; exact-base evaluation | `khipu-r3/` |
| brain-navigator-r2 | Adapter; grounding/abstention plus separate controller ACL tests | `brain-navigator-r2/` |
| WILLAY | Small adapter; evaluation before identity/task retraining | `willay/` |
| chaski | Checkpoint; text/multimodal evaluation separately | `chaski/` |
| chaski-5050 | Adapter; bf16 alpha16, preserve separate identity | `chaski/train_chaski_bf16_5050.py`, `eval_chaski_5050.py` |
| chaski-r2 | Adapter; reconcile code/card paths and exact base | `chaski_r2/`, `chaski-r2/` |
| SZL-Khipu-1.5B-abstain | Curriculum-only/no weights declared; review data | `khipu-abstain/`, `khipu/train.abstain.jsonl` |
| TinyKhipu-Nano | NumPy reference fixture; numerical tests, not LLM SFT | Both case-distinct root paths above |
| MiniEmbed-Nano | NumPy reference fixture; embedding/numerical tests | Both case-distinct root paths above |
| Moons-Nano | NumPy reference fixture; task-specific numerical tests | Both case-distinct root paths above |
| ReceiptAgent-Nano | NumPy reference fixture; contract/numerical tests | Both case-distinct root paths above |
| szl-khipu | Small custom numeric asset; not automatically Khipu LLM | Exact canonical owner path unresolved in this audit |
| szl-nemo | Recipe/surrogate, not a checkpoint | `nemo/card/README.md`; distinguish Ollama wrapper |
| YARQA-ATTN | Kernel/software; native-type and numerical/ABI checks | `YARQA-ATTN/` |
| governed-inference-meter | Deprecated kernel/software; successor/consumer review | `governed-inference-meter/`; portfolio points to energy owner |
| szl-block-kv | Kernel/software; numerical/cache/ABI checks | `szl-block-kv/`, `block-kv/` |
| szl-blocked | Software contract; resolve explicit source binding | `szl-blocked/`; portfolio source null |
| szl-formulas | Formula software; existing formal/source contracts | `szl-formulas/`; portfolio names `lutar-lean` |
| szl-governed-norm | Deprecated/surrogate kernel; numerical and consumer review | `szl-governed-norm/`, `governed-norm/` |
| szl-govsign | Signing software; key/serialization verification, not SFT | `szl-govsign/`; portfolio source null |
| szl-invariants | Formal/software contract; no claim of theorem from metadata | `szl-invariants/`; portfolio names `lutar-lean` |
| szl-kernels | Software/embedding-table package, not Transformers checkpoint | `szl-kernels/`; portfolio names governed receipt spec |
| szl-khipu-kernels | Kernel/software; numerical and owner reconciliation | Exact canonical owner path unresolved in this audit |
| szl-lambda-gate | Tensor/software surrogate; numerical/formal boundary tests | `szl-lambda-gate/`, `lambda-gate/` |
| szl-maskmod | Kernel/software; native-type and masking tests | `szl-maskmod/`, `maskmod/` |
| szl-ouroboros | Loop/governance software; conformance tests | `szl-ouroboros/`; portfolio names `lutar-lean` |
| szl-provctl | Provenance software; source and receipt verification | `szl-provctl/`; portfolio source null |
| szl-receipt-attn | Kernel/software; attention and receipt contract checks | `szl-receipt-attn/`, `receipt-attn/` |
| KILLINCHU-EYE | Roadmap alias; benign reference reconciliation only | `waman/`; no detector/targeting capability established |
| chakana | Embedding roadmap; admitted data and retrieval baseline first | `chakana/train_chakana.py`, `eval_chakana.py` |
| qantu | Roadmap/reference; define task/data and inspect actual code | `qantu/` |
| tinku | Roadmap/reference; define task/data and inspect actual code | `tinku/` |
| waman | Roadmap/reference; define benign task/data and inspect code | `waman/` |
| SZLHOLDINGS | Historical org stub; exclude from training | Not a model implementation |
| a11oy-v19-substrate | Operational payload; source/projection audit, not SFT | Existing portfolio points to `szl-holdings/a11oy` |
| szl-energy-attest | Energy evidence software; sensor/measurement tests | `szl-energy-attest/` |
| szl-training-scripts | Code/discovery entry; source-owner reconciliation | GitHub training code, not a separate weight model |
| oac-system-health-v1 | Synthetic classical ops model; CPU heldout/calibration | `clinical-gateway/`; artifact-level binding still required |
| oac-clinical-transport-health-v1 | Synthetic classical ops model; lineage/evaluation | `clinical-gateway/`; no PHI or clinical decision use |

## Authorized execution order on the single laptop

Keep published weights and existing local environments intact. The first actual
work is the reviewed installed-Ollama baseline, then the existing supervised
candidate. Reserve the single GPU from other clients and queued jobs. A prior
empty `/api/ps` observation is not an exclusive host lease.

For the selected clean, reviewed Git source and existing qualified Linux Python,
use the commands documented in `frontier/qwen35-receiptagent-v3/README.md`:

```text
generate_curriculum.py --check
qualify_runtime.py --source-commit <exact-reviewed-current-source> --report <new-report>
launch_supervised_training.py --source-commit <same-source> --run-kind smoke
```

These are existing entrypoint/argument names, not shell-ready substitutions.
Use the source runbook's exact interpreter, paths, tests and supervision setup.
The one-step result must have its actual supervisor, source, runtime, candidate
and saved-artifact evidence. It does not automatically authorize the full run.
A source mismatch requires review, not replacement of the SHA by guesswork.

The previous starter's source `be0912546cafc95e7ed9209a8501dbe26359c724` was
compared with the inspected `a65d75e...`: one commit adds 14 PEFT-export-related
files, with no changed pre-existing training/benchmark components. A standalone
starter can therefore be explicitly repinned for that reviewed snapshot; this
comparison is not approval of arbitrary future main changes or of model quality.

No remote desktop device was available in this audit session. Training must run
on the authorized laptop through the existing supervisor; no cloud job was
substituted. HF Jobs listing returned an inconsistent/truncated empty shared set
with a nonzero total, so no claim that the organization has zero jobs is made.

## Acceptance and release boundary

A model-specific completion requires exact source, base, adapter, tokenizer,
data split, dependency environment, hardware and run identity; actual saved
weights; reload evaluation; retained failures and independent comparison. GPU
utilization or a decreasing training loss alone is insufficient.

Use `publishing/model-source-bindings.json` and existing owner-native publishers
for authorized release. Preserve history and base licenses. Do not write directly
to Hub, create a second publisher, promote unsigned reports, waive required
checks, or mistake this audit PR for alignment of every asset. Source admission,
training, evaluation, publication and live runtime qualification are separate.

Default publication and autonomous-execution eligibility remain false. Lambda
remains Conjecture 1. This document changes no model, formula, key, runtime,
branch-protection rule, Space, network configuration or public product default.
