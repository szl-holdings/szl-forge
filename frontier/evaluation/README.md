# Frontier real evaluation runner

This directory turns the model-intake contract into a real, bounded evaluation path. It performs inference against the live Khipu baseline and one Hugging Face Inference Providers route, scores the same synthetic fixtures, exercises an intentional candidate-route failure, verifies Khipu fallback transport, and emits a content-addressed Nemo-shaped receipt.

## Authority and production boundary

The runner is an evaluation instrument, not a deployment controller. Candidate output is proposal-only. It cannot execute tools, mutate infrastructure, route production traffic, or grant production authority. Every run retains `production_disposition: HOLD`; A11oy remains the consequential-action admission layer.

The provider chat API generally does not attest the exact server-side weight revision, runtime build, or hardware. Those fields are therefore recorded as `UNAVAILABLE`, never inferred. A successful run establishes measured request/response behavior for the selected provider route at the run time; it does not prove self-hosted equivalence or authorize promotion.

## Fail-closed fallback

A provider-failure exercise uses an intentionally invalid provider route and then reaches the live Khipu baseline with a dedicated failover fixture. Khipu output is accepted only when it exactly matches the non-executing fallback envelope:

```json
{"decision":"ESCALATE","answer":"PROVIDER_UNAVAILABLE","evidence_ids":["F1"],"tool_calls":[]}
```

If Khipu transport succeeds but its output is malformed or semantically different, a deterministic external safety guard emits that same envelope. The receipt records the baseline attempt, whether its model output passed, which fallback source was selected, the selected-output digest, transport status, semantic-safety status, and zero production authority. This avoids converting transport success into a false semantic pass.

## Execution modes

Pull requests run the two-case smoke suite without publication. Pushes to `main`, the weekly schedule, and approved manual runs execute the four-case suite and publish the receipt bundle to `SZLHOLDINGS/szl-frontier-evaluation-receipts` when a validated write credential is available.

The GitHub workflow independently validates configured Hugging Face credentials. Secret bytes are masked, never placed in artifacts, and never written to the repository. The runner dependency is fixed at `huggingface_hub==1.30.0`, and each receipt records the client software and hardware fingerprint.

## Sealed-run counter

Since merge #193, a provider-unavailable run still seals a local
`HOLD_PROVIDER_UNAVAILABLE` receipt but does **not** publish the payload
bundle. That made public silence ambiguous (no run versus sealed negative).

When `--publish` is set and the candidate provider is unavailable, the runner
updates `runs/SEALED_COUNT.json` on the receipts dataset: UTC dates and
integer counts only. It never uploads the receipt, bundle, summary, provider
name, or run id for that seal. Historical silence from before this counter
existed remains **UNAVAILABLE**, not a filled zero. Genesis evidence:
`frontier/evidence/sealed-run-counter-genesis-2026-09-11.json`.

Public fixtures and their exact `answer_equals` strings are documented in
`FIXTURES.md`. That file does not change the fixture bytes.

## Evidence produced

Each completed run emits:

- `receipt.json`: content-addressed qualification receipt and known bounds;
- `bundle.json`: source checks, baseline records, candidate records, and fixture identity;
- `summary.json`: comparison metrics and the persistent HOLD disposition;
- `publication.json`: Hub dataset commit identity when publication is enabled.

The source attestation pins the GLM-5.3 Flash model revision and the SHA-256 values of its license, card, configuration, generation configuration, and tokenizer configuration. No candidate weight shard is downloaded. No repository Python file is executed.

## Local contract test

```bash
python -m pytest -q \
  tests/test_frontier_evaluation_runner.py \
  tests/test_acquire_hf_inference_token.py
```

A real local run requires a Hugging Face token with Inference Providers access in `HF_INFERENCE_TOKEN`. Publishing additionally requires `HF_TOKEN` with permission to create or update the receipt dataset. Tokens must be supplied through environment variables or a secret manager, never committed.
