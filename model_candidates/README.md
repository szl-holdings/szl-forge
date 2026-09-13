# Kernel-derived research candidates

**SOURCE_ONLY_NOT_TRAINED.** The admitted [`model-lab/`](../model-lab/) retains
its authenticated operator interface, two 161-parameter baselines, current HF
identities and guarded code-only publisher. This package is a distinct research
slice, not their replacement and not a second estate registry or control plane.

| Candidate | Parameters | Research identity (proposed, not reserved) |
|---|---:|---|
| `router` | 1,377 | `SZLHOLDINGS/A11OY-RouteRank-Research-v1` |
| `invariant-risk` | 1,476 | `SZLHOLDINGS/A11OY-InvariantRisk-Research-v1` |
| `yarqa-causal` | 141,184 | `SZLHOLDINGS/YARQA-Causal-Reference-v1` |

The route scorer consumes normalized `B x R x 8` features and an external boolean
eligibility mask. It is permutation-equivariant, never authorizes or dispatches,
and returns zero probability mass and index -1 when no route is eligible.
The risk model consumes a different named eight-feature contract and emits four
advisory outcome-risk logits. Scores are not calibrated confidence. Neither model
replaces deterministic invariants, router policy or human approval.

These are different feature/target/architecture contracts from Model Lab's binary
baselines. Existing weights must never be loaded or relabeled as these models.
`candidate-tests/test_identity.py` checks the actual Model Lab source catalog,
including deferred tracks, and rejects colliding Hub identities.

## YARQA distinction

The original `YARQA-ATTN` kernel is CPU-only, noncausal compartment attention.
The causal reference here uses fixed-width independent canals and a causal mask
with dense PyTorch SDPA. It is a separate adaptation, not a kernel acceleration.
It has no cross-canal context, KV cache, GGUF conversion or Ollama integration.
Byte IDs are 0..255, BOS is 256. No frontier-quality, throughput or GPU claim.
Lambda uniqueness remains Conjecture 1, OPEN; the Lambda aggregate is advisory.

## Inspect research recipes locally

In a new reviewed environment, not the working owner CUDA environment:

```sh
python -m pip install fastapi==0.128.2 uvicorn==0.48.0
python -m model_candidates.workbench --port 8766
```

Open `http://127.0.0.1:8766`. Model Lab's operator port remains 8765. This local
recipe inspector has no authentication suitable for multi-user deployment and
must not be tunneled or bound publicly. Host, loopback-client and same-origin
checks, no-store headers and CSP remain enforced. It cannot train, probe,
execute, load checkpoints, publish or approve anything. Public Forge Lab and the
existing inference Space remain unchanged.

`GET /api/candidates`, `/api/candidates/{key}`, `/api/blueprints/{key}` and
`/api/mesh` are source declarations, not live inventory or serving qualification.
The compact delivery view shows a sequence, not verified remote completion.

## Test and package source

Use a fresh CPU test environment; never replace the laptop's CUDA Torch build:

```sh
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r candidate-tests/requirements.txt
python -m pytest -q candidate-tests
```

Direct dependency pins are not a complete artifact hash lock or production
qualification. Structural tests use synthetic inputs and random initialization,
not user-data training. The existing CPU workflow also produces three exact-source
code archives for review, never trained releases or automatic HF uploads.

```sh
python -m model_candidates.blueprint --repository . \
  --source-revision FULL_REVIEWED_GIT_COMMIT \
  --candidate router --output NEW-router-source.zip
```

The exporter reads literal Git paths at a full commit and verifies its own running
package matches that source. It does not read arbitrary data or credentials. An
externally supplied manifest hash can establish byte consistency, not signer
identity, source admission, training, model quality or live Hub publication.
Research targets are NOT registered with Model Lab's two-target publisher.

## Remaining gates

Source admission, licensed data and frozen normalization/splits, outcome-label
provenance, bounded owner-supervisor adapters, qualified hardware and matched
baseline evaluation remain separate. Training is not started by source admission.
Model Lab/Forge own operator and publication integration; szl-router owns admission
and dispatch; native Ollama/private Tailscale provide declared transport only.
Research can inform those systems only after explicit compatible integration.

Read [the integration work order](../docs/FRONTIER_INTEGRATION_2026-09-13.md).
GitHub -> Hugging Face -> a-11-oy.com product -> a11oy.net proof. No pipeline
stage is promoted solely because a page responds or an archive is consistent.
