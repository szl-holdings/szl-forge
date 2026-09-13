# Frontier integration: one source path, distinct evidence stages

## Scope and current delivery

This document joins the frontier-release thread to existing Forge PR #271. It
is a work order and source contract, NOT a complete organization census, trained
release, runtime observation, live HF alignment claim or production declaration.
The inspected main was `5cf608e96d9bbd94049d083a46ab93fbc64703b8`; #271's original
head was `7f0bfa454a63a4f02fbc9f9b594814cba0c74d54`. Main's test dependency addition
and the branch's model package registration are reconciled, retaining an empty
default runtime dependency set. Re-read current heads before further mutation.

Use `model_candidates/` as the integration starting point. Do not blindly apply
the earlier standalone `model-lab/` archive: its two 161-parameter models have
DIFFERENT architectures and feature/target contracts. Port useful tests and
adapters individually; never treat their checkpoints as interchangeable.

This addition supplies:

- `model_candidates.blueprint`: exact-Git-object source reader, three code-only
  HF projections, externally anchored byte-map verifier and exclusive local ZIP
  writer. No Hub client, token access, training or promotion operation.
- `GET /api/blueprints/{key}` in the existing loopback workbench, using the same
  recipes as model source and exporter. No user-provided file paths or URLs.
- A compact four-stage delivery view; details are expandable, not a prose wall.
  The view shows a sequence, NOT completion/availability badges for external sites.
- Code-archive build/verification in the existing CPU workflow, Linux and Windows.
  These artifacts are for review, not trained model releases. Package files use LF
  so Windows checkout normalization cannot silently change projected Python bytes.

The three existing architectures remain unchanged: route-utility scorer,
four-label advisory invariant-risk classifier, and fixed-canal causal reference.
No trained weights, dataset, benchmark, owner machine observation or GPU result
is introduced by this source change. Existing static Forge Lab stays read-only.

## Exact-source packaging

Run in a fresh, inspected checkout/environment containing this source. The full
source commit must contain this exact exporter and all projected modules.

```sh
python -m model_candidates.blueprint --repository . \
  --source-revision FULL_REVIEWED_GIT_COMMIT \
  --candidate router --output router-source.zip
```

Candidate choices: `router`, `invariant-risk`, `yarqa-causal`. No automatic upload.
The output path must be new; interrupted partial output is not silently reused.
The CLI prints the manifest hash and explicitly false Hub/training flags.

Source paths are a literal allowlist: LICENSE plus five model-candidate modules.
Datasets, private endpoints, keys, old receipts, arbitrary files and weights are
excluded. Nonregular Git entries, mutable revisions, corrupt blob/size bindings,
and mismatched running/projected source fail. Supplied Git environment overrides
and replacement refs cannot substitute another source. No projected code is
imported by the Git reader.

The pure `project_source` function does NOT establish Git origin. Likewise,
`verify_projection` authenticates neither the hash supplier nor a Hub response;
it compares the complete supplied byte map to an externally provided manifest
hash. A hash read from the SAME bundle is only a structural consistency check.
CI archive verification is intentionally labeled that way. It is not source
admission, a signature, a reproducible training claim or model qualification.

## Remaining GitHub -> HF -> product -> proof work

### 1. Source admission

Read exact-head checks, review threads and repository rules. Reconcile rather
than overwrite concurrent branches. Keep normal branch protections and release
approvals; no admin bypass. Do not close hardware recovery #264 or infer its
runtime state from this unrelated candidate source gate.

### 2. HF projection through the existing release authority

After source admission, extend the existing Forge publisher with a separately
reviewed CODE_ONLY operation for these three proposed IDs only:
`SZLHOLDINGS/A11OY-Router`, `SZLHOLDINGS/A11OY-Invariant`, `SZLHOLDINGS/YARQA-1`.
Do not give a code-only artifact the existing trained-weight admission predicate.
Do not overwrite a newly discovered repository or retire working kernel mirrors.

Bind the admitted source SHA, complete expected file set and manifest SHA before
any write. Reconcile current repo existence/type/access explicitly; a connector
404 is not proof that a private repo cannot exist. A proposed HF name is not a
reservation. No random/test weights, misleading Transformers tags or fabricated
benchmarks. Preserve upstream copyright, license and modification notices.

Use the existing protected publishing environment and credential injection, not
secrets in source/chat. A write timeout is UNKNOWN_OUTCOME: inspect current Hub
commit/files before retry; do not blindly replay. Read back the immutable HF
commit, exact source-manifest hash and ALL declared files. Reject extra weight or
secret files; retain receipts for partial/failed publication. Git and HF SHAs are
different identities. A source ZIP from a PR's synthetic-merge SHA is not evidence
that the SHA was admitted to main.

This writer integration and live readback are NOT implemented by blueprint.py.

### 3. Product and proof projections

Only after HF readback should a-11-oy.com expose the admitted source capability
and a11oy.net expose its source/HF/digest evidence. Use the existing site APIs,
artifact manifests and canonical source owners rather than inventing a second
estate registry. Public pages receive sanitized status, never the private mesh
endpoint list, signing keys, provider credentials or a laptop execution console.

Keep status dimensions separate: source admitted, code published, weights
trained, evaluation passed, runtime qualified, live deployment observed. A page
returning HTTP 200 is not model readiness. Missing observations remain null with
an error/scope, not invented zero counts. The pool adapter must consume the actual
`szl.compute-pool/v1` authority, not trust an arbitrary JSON `ready: true`.

Browser acceptance on BOTH product/proof surfaces: 320x568, 375x812, 768x1024,
1440x900; keyboard focus; no horizontal overflow; 44px targets; reduced motion;
high contrast; error/empty/slow states. Test served source identity, not only HTML.
No new public-domain deployment is claimed by the local workbench render.

### 4. Owner-supervised training remains a separate start

Bind licensed data, group/time-separated train/validation/test splits, frozen
feature normalization, label provenance, environment and explicit job budgets.
Register a bounded adapter under the EXISTING owner supervisor. Check idle-device
leases and attempt identity; never unload an active Ollama workload as a UI side
effect. Do not initialize from a frozen comparator or restart a claimed attempt.
The current candidate minibatch interfaces are not that completed adapter.

## Pool/mesh/provider architecture to preserve

`a11oy/box-scripts/litellm_config.yaml` is the existing two-node intended serving
pool, with native Ollama, private Tailscale transport and stateless Docker
sidecars. `szl-router/router_control.app` is the policy/routing authority to
integrate. `szl-mesh` is separate CRDT/UDS coordination, not combined GPU memory.
The source map is a declaration; no owner-node probe was performed here.

The learned router ranks ONLY already-eligible candidates. Recheck eligibility,
current model digest and resource lease at dispatch. Missing weights, timeout or
uncalibrated/OOD input preserves the reviewed deterministic policy. LiteLLM may
be transport, not an independent fallback engine bypassing the same policy.
Pin actual shared-model digests, not mutable tags, before claiming equivalent
replicas. Two GPUs over a tailnet do not become one combined-VRAM training GPU.

OpenRouter remains an explicit remote lane with approved data/provider/model
policy and spend cap, never silent local-to-cloud fallback. Teacher output needs
rights review and independent label checking. Do not install a new CUDA/PyTorch
stack into a working owner environment merely because upstream released one.

## Frontier choices: evaluate, adapt with attribution, measure

This is an integration backlog, not another alert or automatic model admission.
Primary sources were consulted for this work order on 2026-09-13. Upstream claims
below remain upstream-reported until SZL measures the same condition. Preserve
license/NOTICE/lineage and identify adaptations honestly; no novelty guarantee.

| Priority | Source | SZL experiment / acceptance |
|---|---|---|
| P0, conditional on use | [DeepSeek V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | Audit mutable API aliases and exact served identity; rebaseline any actual caller. Do not switch SZL models merely because the name exists. |
| P1 | [TRL 1.13.0](https://github.com/huggingface/trl/releases/tag/v1.13.0) | Isolated chunked-loss parity/peak-memory/throughput experiment with a frozen workload, before changing existing training recipes. Million-token results are not a laptop guarantee. |
| P1 | [vLLM 0.29.0](https://github.com/vllm-project/vllm/releases/tag/v0.29.0) | Continue existing Forge #282/#283, not a duplicate lane. Model Runner V2 default/rollback, exact wheel bytes, dependency closure and actual hardware need qualification. Keep the older AsyncGRPO bounds intact. |
| P1 | [Per-tensor GGUF layouts](https://huggingface.co/blog/bartowski/per-tensor-layout-maps-for-gguf-quantization) | Compare equal-budget Khipu/A11OY-MINI quant candidates for output quality, KLD and memory; retain the old artifact if not improved. Qualify post-quantization separately. |
| P1 | [Granite time-series r2](https://huggingface.co/ibm-granite/granite-timeseries-patchtst-fm-r2) | Lyte forecasting with rolling-origin evaluation, leakage-free splits, naive/seasonal baselines and coverage tests. Forecasts are not causal explanations or incident approvals. |
| P1 research | [OUI-1](https://huggingface.co/thesysdev/OUI-1) | Generate proposed component trees; validate with an approved schema/library and test task completion/a11y. No raw executable HTML/JS, credential-bearing tools or generated control authority. Capture repeated runs because the published sampler does not honor per-request seeds. |
| P2 hardware-gated | [Agnes-3.0-Flash Preview](https://huggingface.co/Agnes-AI/Agnes-3.0-Flash) | Study hybrid recurrent/global attention and evaluate document grounding only after custom-code and hardware review. Preview weights must not inherit production/API checkpoint benchmarks. |
| P0 verification, not blind migration | [Kernel migration](https://huggingface.co/docs/kernels/migration) | Observe BOTH repo types, immutable revisions, loader/trust policy and numerical tests. Model listing alone does not prove a missing first-class kernel. Preserve mirrors until each consumer is reconciled. |

### Differentiating research hypotheses, not shipped features

1. Evidence-conditioned routing: learn quality/cost tradeoffs on eligible routes
   with matched outcome coverage; measure regret, latency, cost and abstention.
   Historical traffic alone is selection-biased; do not call an unobserved route
   the counterfactual winner.
2. YARQA cross-canal context: the current reference has no cross-canal memory.
   Evaluate a separately named, causally gated summary-exchange adaptation against
   dense causal and isolated-canal baselines with matched data/compute. Prove
   prefix/no-future-leakage contracts and measure privacy-boundary failures before
   any stronger context or efficiency claim. Do not relabel the original kernel.
3. Evidence-bound UI proposals: let a model select a layout, but bind all displayed
   facts/actions to server-validated evidence handles and approved components.
   The model chooses presentation, not data truth, permissions or execution.
4. Lyte evidence-aware forecasting: show forecast interval, actual history,
   missing-data state and evaluation coverage together. The UI must make an
   uncertain forecast distinguishable from an observed metric.

These experiments are not in the candidate implementation and are not approved
for automatic production rollout by this document.
