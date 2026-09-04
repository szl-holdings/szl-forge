# SZL Forge proof-carrying inference v1

## Decision

Forge is the model lifecycle and inference-qualification authority. It does not
turn every component into a model. The distinctive system is the closed evidence
loop around a replaceable inference engine:

1. **Second Brain** retrieves handles only.
2. A controller checks tenant/principal scope, hydrates authorized content, and
   verifies each content digest.
3. The **formula binding** imports identity and maturity from pinned `lutar-lean`
   and `szl-formulas` revisions. Formula text in a prompt is never authority.
4. The model creates a proposal and emits exact model, adapter, runtime, and
   hardware identity.
5. **SZL Nemo** independently witnesses pre-generation and post-generation state.
6. **A11oy** alone admits consequential tool actions; human approval and a signed
   receipt are required at that boundary.
7. Postconditions are checked after execution.
8. **Living Anatomy** receives a sanitized observation event and cannot modify the
   decision.

The machine contract is `inference/control_plane.v1.json`; its fail-closed
validator is `inference/validate_control_plane.py`.

## Formula law

The current formal binding is exactly:

```text
F1, F4, F7, F11, F12, F18, F19, F22
```

There are 21 callable formulas in the software kernel, but callable is not the
same as locked-proven. The formula package explicitly says the mapping from the
formal F-number corpus to those 21 callable function names is unknown; this
contract preserves that separation instead of inventing a mapping. F23/Lambda
remains `CONJECTURE_1_ADVISORY`; it cannot authorize a consequential action or
serve as the sole basis for `ALLOW`.

A blocking estate inconsistency remains: A11oy's checked-in
`szl_formula_registry.py` still encodes a stale locked-five plus three
experimental interpretation. No formula-dependent model or action path should be
promoted until every consumer resolves to the formal locked-eight source.

## Runtime law: measure, do not crown

There is no universal inference-engine winner. Qualify the engine by workload:

- **Local CPU/GGUF:** llama.cpp lane.
- **Single-node GPU:** vLLM and SGLang bakeoff.
- **Distributed GPU:** NVIDIA Dynamo with a qualified engine only when measured
  concurrency, context length, TTFT/ITL separation, or KV-routing needs justify
  the added system.

Each lane must use exact model, tokenizer, adapter, quantization, image, engine,
and hardware revisions. Selection includes quality, grounding, citations,
abstention, authority behavior, TTFT, inter-token latency, throughput, memory,
energy when genuinely metered, and cost per successful governed request.

Speculative decoding, disaggregated prefill/decode, and aggressive prefix caching
are candidates, not default claims. Keep them only when the declared workload
shows a net gain without governance or quality regression.

## What is implemented in this change

`inference/governed_inference.py` provides an engine-neutral coordinator with:

- adapters for the installed Second Brain and Nemo packages;
- a digest-verifying public JSONL hydrator;
- a constrained OpenAI-compatible adapter usable with llama.cpp, vLLM, or SGLang;
- strict model/adapter/runtime identity checks;
- pre- and post-generation witness stages;
- fail-closed evidence and formula boundaries;
- no tool execution;
- deterministic unsigned local receipts that must be signed before consequential
  action;
- sanitized Anatomy events containing hashes and decisions, never raw prompts or
  private reasoning.

This is an integration contract and executable boundary, not proof that any model
is qualified or that a live deployment has changed.
