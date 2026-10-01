# SZL frontier integration: preserve authority, distinguish research

## Current source reconciliation

At `cae7b4946b569b6f0b3cd43036c63f2df290877b`, Forge main ALREADY contains
`model-lab/`: an authenticated Python operator interface, two 161-parameter
binary advisory models, and a guarded two-target code-only HF publisher.
**Model Lab is the operator and publication starting point. Do not replace it.**
The earlier archive is not the only copy anymore; apply no blind handoff patch.

PR #271 carries different research: a 1,377-parameter route ranker, a
1,476-parameter four-label risk model, and a 141,184-parameter causal canal
reference. They had proposed the same HF names as Model Lab's current/deferred
catalog. That collision is now repaired in source, before any Hub write:

| Source | Retained or proposed identity | Meaning |
|---|---|---|
| Model Lab router | `SZLHOLDINGS/A11OY-Router` | Existing binary success baseline |
| Model Lab invariant | `SZLHOLDINGS/A11OY-Invariant` | Existing binary violation baseline |
| Model Lab deferred YARQA | `SZLHOLDINGS/YARQA-1` | Noncausal encoder design, not implemented |
| Candidate router | `SZLHOLDINGS/A11OY-RouteRank-Research-v1` | Different route-ranking research contract |
| Candidate risk | `SZLHOLDINGS/A11OY-InvariantRisk-Research-v1` | Different four-label research contract |
| Candidate causal | `SZLHOLDINGS/YARQA-Causal-Reference-v1` | Separate causal adaptation, not the original kernel |

No HF name is reserved by this table. The new research identities are NOT added
to Model Lab's two-target publisher. Weights, features, normalization and labels
are not interchangeable. `candidate-tests/test_identity.py` checks the actual
Model Lab source catalog; the existing workflow now watches both sides.
Historical validation records retain their original scope and hashes.

## Source delivered in this branch

`model_candidates.blueprint` reads six literal regular Git paths at a full
commit: LICENSE and five Python modules. It rejects mutable refs, nonregular
entries, size/blob mismatches and a running exporter different from the projected
source. It emits three code-only research projections with exact recipe/config,
license, source bytes and an unsigned digest manifest. It never uploads or trains.

`GET /api/blueprints/{key}` and the compact four-stage Python-rendered view reuse
those same recipes. Details are expandable. This is an isolated research recipe
inspector on loopback port 8766, NOT the admitted Model Lab operator on 8765, a
public service or a second control plane. Host/origin/client/CSP boundaries remain.
No web route reads arbitrary paths, probes nodes, loads weights or executes jobs.
The static public Forge Lab and existing inference Space remain unchanged.

The existing Linux/Windows CPU workflow builds and verifies all three source
archives and retains them as review artifacts. LF attributes preserve Python
bytes on Windows. Dependency versions are direct test pins, not a complete
artifact hash lock or production qualification. Root runtime dependencies stay
empty; the working owner CUDA environment is untouched.

```sh
python -m model_candidates.blueprint --repository . \
  --source-revision FULL_REVIEWED_GIT_COMMIT \
  --candidate router --output NEW-router-source.zip
```

The output must be new. Partial output is not silently reused. The pure projection
function does not establish Git origin; the byte-map verifier does not authenticate
the hash supplier. Reading an expected hash from the SAME bundle is a structural
self-check only. A PR synthetic-merge SHA is not protected-main admission.
Source, publication, trained weights, model quality and live runtime stay separate.

## Complete the existing alignment chain, not a parallel publisher

**GitHub.** Inspect exact-head checks, source review, threads and repository rules
before normal admission. No force or admin bypass. Do not close owner recovery
#264 or the separate AsyncGRPO GPU HOLD because candidate tests pass.

**Hugging Face.** Model Lab's existing guarded publisher handles the two canonical
binary baselines. Complete its explicit dispatch/readback path; do not replace it
or feed it these incompatible research archives. A reviewed extension is needed
before publishing research identities. Bind admitted source, target/type, exact
file set, hashes and conditional HF parent. Keep credentials in existing protected
injection, never source/chat. Refuse unexpected existing contents. An uncertain
write is UNKNOWN_OUTCOME: read remote state before retry. Verify the immutable HF
commit and every expected byte; keep failures and partial states. GitHub and HF
commit IDs are distinct. No random weights, fabricated metrics or unsupported
Transformers/Ollama tags. Working kernel identities and old releases stay intact.

**a-11-oy.com.** Consume admitted, sanitized capability/runtime status through the
existing product APIs. Do not publish private tailnet addresses, tokens, signing
keys or owner-control endpoints. A reachable page is not a trained/ready model.

**a11oy.net.** Use existing proof schemas and source owners for source SHA, HF SHA,
manifest/digests, evaluation scope, runtime observation and known bounds. Never
substitute an unsigned hash for an approval/signature. Missing observations are
null with an error/scope, not zero. Verify served identity and browser behavior at
320x568, 375x812, 768x1024 and 1440x900: focus, overflow, 44px controls, reduced
motion, high contrast, loading/error/empty states. Neither domain is deployed by
this research branch.

## Pool, mesh, routing and later owner training

Preserve `a11oy/box-scripts/litellm_config.yaml`: native Ollama, private Tailscale
and stateless Docker sidecars. `szl-router/router_control.app` owns policy,
eligibility, dispatch and retry budgets; LiteLLM may be subordinate transport,
not an independent cloud-fallback policy. `szl-mesh` is separate CRDT/UDS work.
Two tailnet GPUs do not combine VRAM. Source declarations are not node observations.

The learned scorer ranks ONLY already-eligible routes. Recheck current model
digest, identity and resource lease at dispatch. Unknown/OOD/uncalibrated scores,
missing weights or timeouts preserve the reviewed deterministic baseline. Consume
`szl.compute-pool/v1` through its actual receipt authority, not arbitrary ready=true.
OpenRouter needs an explicit remote data/provider/model policy and spend cap.
Teacher outputs need rights review and independently checked labels.

Owner training starts later, after licensed data, normalization, provenance,
group/time-separated splits and a frozen evaluation plan are admitted. Register
bounded adapters under existing Owned Agent Control/gpu-bridge; bind source,
environment, attempts, output roots and resource/time limits. Do not unload active
Ollama work, replay an old attempt or initialize from a frozen comparator. No
owner training, model download or paid job was started in this source work.

## Frontier priorities from the thread

Study, adapt with attribution, and measure. These are experiments, not automatic
admissions or novelty claims. Primary sources were consulted on 2026-09-13.

| Priority | Primary source | SZL evaluation |
|---|---|---|
| P0 if used | [DeepSeek V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | Audit actual mutable-alias consumers and served identities; rebaseline those callers, not the entire estate by assumption. |
| P1 | [TRL 1.13.0](https://github.com/huggingface/trl/releases/tag/v1.13.0) | Isolated chunked-loss numerical parity, peak memory and throughput on a fixed workload. Million-token results are not a laptop guarantee. |
| P1 | [vLLM 0.29.0](https://github.com/vllm-project/vllm/releases/tag/v0.29.0) | Continue existing Forge #282/#283: Model Runner V2, rollback, wheel/dependency closure and actual hardware. Preserve older AsyncGRPO bounds. |
| P1 | [Per-tensor GGUF layouts](https://huggingface.co/blog/bartowski/per-tensor-layout-maps-for-gguf-quantization) | Equal-budget quality/KLD/memory comparison for Khipu/A11OY-MINI. Keep old artifacts when not improved; evaluate post-quantization. |
| P1 | [Granite time-series r2](https://huggingface.co/ibm-granite/granite-timeseries-patchtst-fm-r2) | Lyte rolling-origin forecasting, seasonal/naive baselines, leakage-free splits and interval coverage. Not causal evidence or incident authority. |
| P1 research | [OUI-1](https://huggingface.co/thesysdev/OUI-1) | Propose component trees; enforce approved schema/components and evidence handles. No generated executable HTML/JS or action authority. Repeat-run evaluation: published sampler ignores per-request seeds. |
| P2 hardware-gated | [Agnes preview](https://huggingface.co/Agnes-AI/Agnes-3.0-Flash) | Hybrid recurrent/global attention study and document grounding after custom-code/hardware review. Do not inherit production/API checkpoint metrics. |
| P0 verify scope | [Kernel migration](https://huggingface.co/docs/kernels/migration) | Observe both repo types, exact revisions, trust policy and numerical tests. A model listing alone is not failed migration. |

The proposed SZL differentiators are evidence-conditioned routing with matched
counterfactual outcomes; causally gated cross-canal summary exchange evaluated
against dense and isolated-canal baselines; model-proposed layouts whose facts and
actions resolve only through server-validated evidence handles; and Lyte forecasts
that visibly distinguish uncertainty, missing data and observed history.

These hypotheses are NOT implemented by the current candidate models. Preserve
upstream license/NOTICE/lineage, disclose adaptations, predeclare comparisons and
retain negative results. A new name or model card is not measured frontier progress.
