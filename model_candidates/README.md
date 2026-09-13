# Kernel-derived model candidates — source first

**SOURCE_ONLY_NOT_TRAINED.** Three small, randomly initialized PyTorch modules,
a Python/FastAPI backend and Python-rendered local frontend. This is a source
slice, not a trained release, new estate registry, scheduler, publisher or Space.
The production inference package retains its dependency-free default install.
The CPU validation dependencies are opt-in and isolated.

| Candidate key | Trainable parameters | Interface | Proposed, not reserved, Hub name |
| --- | ---: | --- | --- |
| `router` | 1,377 | normalized `B x R x 8`, external boolean eligibility; route proposal | `SZLHOLDINGS/A11OY-Router` |
| `invariant-risk` | 1,476 | normalized `B x 8`; four advisory risk logits | `SZLHOLDINGS/A11OY-Invariant` |
| `yarqa-causal` | 141,184 | byte/BOS IDs `B x T`; next-token logits | `SZLHOLDINGS/YARQA-1` |

These counts describe the small default research architectures, not useful model
quality. The route scorer is permutation-equivariant over routes. An entirely
ineligible row returns zero probability mass and selection `-1` (ABSTAIN).
Training labels that select an ineligible route are rejected. Feature names,
order, missing values, booleans, nonfinite values and normalization bounds are
checked by `encode_features`; the training-data lane must separately bind their
meaning, provenance and normalization constants. Route and risk feature schemas
are different. Scores and softmax outputs are not calibrated confidence.

The risk model cannot override `szl-invariants`, `szl-router/router_control.app`
or any human approval. The Lambda aggregator remains advisory; Lambda uniqueness
is Conjecture 1, OPEN. Do not train a network merely to approximate that already
available deterministic formula and then claim new learned governance ability.

## YARQA distinction

The existing `szl-holdings/YARQA-ATTN` v0 operator is CPU-only, noncausal attention
inside contiguous compartments. The new module is a **separate causal research
adaptation**, not a wrapper silently claiming the old kernel became a language
model. It uses fixed-width canals plus a causal mask so extending a sequence does
not change past boundaries. Each canal is isolated: no cross-canal context.
It uses dense PyTorch SDPA with a boolean mask, not an optimized sparse kernel.
No speedup, CUDA-kernel support, distributed training, KV cache, GGUF conversion,
Ollama import or frontier-level language quality is claimed. It is a tiny byte
reference to establish correct gradients and causal semantics before scaling.
BOS is 256; ordinary UTF-8 byte IDs are 0 through 255. No tokenizer is downloaded.

## Run the local Python frontend

Use an inspected source checkout and a NEW environment; do not change the working
GPU environment or interrupt an existing Ollama/training process. For the existing
case-distinct Forge directories, prefer a WSL checkout on its Linux filesystem
rather than a case-insensitive Windows source checkout. The new package adds no
case-colliding paths.

```bash
python -m venv .venv-candidate-ui
# Linux / WSL:
. .venv-candidate-ui/bin/activate
# Windows equivalent: .venv-candidate-ui\Scripts\Activate.ps1
python -m pip install fastapi==0.128.2 uvicorn==0.48.0
python -m model_candidates.workbench --port 8765
```

Open `http://127.0.0.1:8765`. This starts only a local recipe inspector, not a
model or training job. It requires no HF, OpenRouter, Tailscale or GitHub token.
`GET /api/candidates`, `/api/candidates/{key}`, `/api/mesh` and `/healthz` expose
source-only declarations. Loopback client/Host checks, same-origin checks,
no-store responses and restrictive CSP are enforced. There are no train, shell,
probe, upload, arbitrary file/URL, provider, checkpoint-load or publish endpoints.
It is NOT a multi-user authenticated service; do not tunnel it or bind it publicly.
The public static `spaces/szl-forge-lab` and existing model-inference Space are
unchanged. Responsive CSS is present; real browser/accessibility acceptance is
separate from ASGI HTTP tests.

## Exercise the model source (CPU test environment only)

In a separate fresh environment, install the bounded CPU validation dependencies:

```bash
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r candidate-tests/requirements.txt
python -m pytest -q candidate-tests
```

These are direct dependency pins for CPU validation, not a complete transitive
lockfile and NOT an instruction to replace the laptop's CUDA Torch build.
The root default test group remains unchanged. The dedicated workflow runs only
synthetic architecture and local HTTP contracts, never a GPU or provider job.

`networks.py` exposes `build_candidate`, `route_loss`, `risk_loss`, and
`CanalLanguageModel.loss` as minibatch interfaces. They are not standalone
trainers and do not save, download or publish weights. Tests include a SafeTensors
round-trip of random initialization entirely in memory, finite backward passes,
an independent masked-matmul attention reference, prefix consistency and gradient
isolation. None of that establishes a trained, calibrated or deployable model.

## Remaining wiring before training and Hugging Face

Keep the existing Forge supervisor/release gates as authority. A successor must
admit versioned data, normalization and label provenance, group/time-held-out
splits, and a fixed evaluation protocol. Then add an explicit bounded candidate
adapter to the existing owner supervisor: source/environment readback, idle-device
lease, resource/time limits, cancellation and non-overwriting checkpoint paths.
Do not claim that adapter exists in this slice. Preserve existing ReceiptAgent,
Khipu and Chaski candidates and every failed/rejected run.

For the router, compare against the existing deterministic router on matched tasks
and real route outcomes, not just whichever route historically received traffic.
Measure selection quality/regret, cost, latency, abstention and hard-policy bypass
attempts; do not send private tasks to OpenRouter without explicit data policy.
For risk, assess per-label false negatives, calibration and adversarial examples;
its output stays advisory even after training. For YARQA, compare a matched dense
causal baseline on held-out loss and context tasks, preserving compute/parameter
budgets. Any later quantization needs its own post-export evaluation.

Only after source admission and explicit repository-purpose review should any HF
model scaffolding be created. No scaffold is a trained model. Actual publication
must use the existing Forge writer with config, SafeTensors/adapter bytes, exact
source/base/data revisions, training/evaluation records, licenses and independent
readback. Then project product status to `a-11-oy.com` and proof to `a11oy.net`.
See [the scoped estate audit](../docs/2026-09-13-kernel-model-audit.md).
