# SZL evidence-first engineering transfer: public leaders

Status: design research and candidate requirements, **not** implementation or comparative performance proof. Reviewed 2026-11-13. Apply only through canonical owners and independent release gates. Do not reproduce third-party code, marks, visual designs, or proprietary data.

## Public sources and transferable principles

| Organization | Official primary source | Observable practice | SZL-owned adaptation | Required falsifiable acceptance |
| --- | --- | --- | --- | --- |
| OpenAI | https://developers.openai.com/api/docs/guides/evaluation-best-practices | Task-specific eval-driven development, continuous eval, human calibration, failure analysis | Add immutable dataset revision, scorer version, grader agreement, per-task denominator, confidence/uncertainty and failure examples to receipt-backed evaluation | Changing model, grader, or dataset digest invalidates a result; empty denominator displays UNAVAILABLE |
| OpenAI | https://developers.openai.com/api/docs/guides/agent-evals | Trace grading over calls, tools, guardrails and handoffs | Bind governed Hatun MCP tool decisions to trace IDs and signed outcome receipts, keeping actual runtime status distinct from simulated checks | Unauthorized tool use must fail closed; trace ID resolves to exact source/model/config |
| Anthropic | https://www.anthropic.com/research/building-effective-agents | Simple composable agent workflows, bounded tool interface, evaluator-optimizer, parallel review | Keep A11oy routing and agent work partitioned into small bounded lanes with explicit stop, approval and rollback conditions | Adversarial tool-output injection and approval-bypass fixtures fail closed |
| Anthropic | https://www.anthropic.com/research/demystifying-evals-for-ai-agents | Code-based, model-based and human graders; task-specific agent evals | Separate mechanical policy checks from judge-model scoring and independent human approval | Scorecards expose grader identity, disagreement and abstentions; no judge score substitutes for approval |
| NVIDIA | https://developer.nvidia.com/topics/ai/nemotron | Published weights, recipes, training data and technical reports, with deployment variants | Distinguish original checkpoint, LoRA adapter, quantization and deployment image; publish per-variant model/dataset/compute lineage and reproducible hardware-specific performance | No cross-variant result inheritance without artifact-binding and reproduction |
| New Relic | https://docs.newrelic.com/whats-new/2026/08/whats-new-08-26-otel-ai-monitoring/ | OpenTelemetry GenAI spans in distributed traces; content capture off by default | Add vendor-neutral OTLP spans for A11oy route, inference, retrieval, tool and proof publication; no sensitive prompt content by default | PII-redaction fixture, trace parent continuity, latency/error/cost metrics and retention controls |
| True Anomaly | https://www.trueanomaly.space/jackal | Human-machine mission operations, simulated/tested versus live mission capability | A11oy mission-like readiness console: separate simulated preview, verified test, admitted execution and live production states | A simulated agent action can never be labeled as live; safety interlocks and approval provenance displayed |
| Hugging Face | https://huggingface.co/docs/hub/model-cards | Discoverable metadata, intended use, limitations, eval details and reproducibility | Render signed receipt-linked evidence passport on every owned card with metadata and source links | Missing/unsigned/old receipts show HOLD or UNAVAILABLE, not invented graphs |

## Cross-portfolio SZL differentiators (proposals, not deployed claims)

1. **Proof-carrying model passport:** exact GitHub commit -> training dataset digest -> checkpoint/adapter/quantization digest -> evaluation receipt -> runtime build -> observed inference trace. Each edge independently verified and time-bounded.
2. **Honest evidence visualization:** numerator/denominator, held-out vs synthetic, run date, uncertainty, model revision, grader identity and unresolved failures. Training loss is NOT held-out quality; a signed receipt alone does not authenticate its issuer.
3. **Unified OTLP evidence graph:** span correlation from frontend through governed MCP, retrieval, GPU inference and evidence writer; trace contents redacted by default, no private training data on public dashboards.
4. **Mission-control operating posture:** four states with precedence (verified / blocked / unavailable / stale), simulation distinctly labeled, approvals enforced server-side and source-to-runtime drift visible.
5. **NVIDIA runtime profile:** per-GPU VRAM, quantization, CUDA/driver versions, prefill/decode latency, throughput, energy and cost with test harness provenance. No pooled VRAM or extrapolated throughput claims.

## Implementation ownership and sequencing

- Canonical GitHub source owns all edits. Existing `szl-forge#731` owns the six model-card and raw training history work; existing `.github#1008` owns the organization card. Do not introduce a competing Hub writer.
- P0: complete #731 CI and independent review; preserve raw-file and exact-head bindings.
- P0: resolve #1008 **Static control-plane effect boundary** using the documented policy owner; do not change gate rules or forge authority.
- P1: add schema-validated evidence passports and read-only scorecard UI behind existing publication contracts.
- P1: wire OTel traces to the existing telemetry pipeline, with redaction and explicit sampling budgets; verify live spans before claiming operational observability.
- P1: build negative-path tests for stale receipts, revoked signature, incorrect artifact hash, missing denominator, unauthorized tool call, missing execution handoff and synthetic/live confusion.
- P2: run independently reproducible benchmarks on each qualified GPU/variant and publish only the measured data.
- Release: protected-main merge -> canonical Hub publisher -> immutable revision readback -> public card and browser checks -> signed deployment receipt. If any gate fails, retain HOLD and avoid forced publication.

## Competitive integrity

These are **public engineering patterns**, not evidence that SZL matches or outperforms any named company. Distinct SZL implementation requires its own source, original visuals, permissioned datasets, measured results, license checks, and independent review. No third-party private methods or model weights are implied.
