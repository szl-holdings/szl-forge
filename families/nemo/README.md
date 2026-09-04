# SZL Nemo witness family

`SZL Nemo` has one meaning in Forge: the deterministic, non-generative doctrine
witness currently sourced from `szl-holdings/szl-nemo`.

It is **not** NVIDIA Nemotron, a Nemotron fine-tune, an LLM, a CUDA kernel, or a
replacement inference engine. Any Nemotron-backed runtime adapter must be named
`nemotron-runtime-adapter` and remain a separately qualified upstream model path.

## Canonical role

Forge owns lifecycle, evaluation, promotion, and publication. The Nemo witness
runs independently around model generation:

```text
Second Brain handles
  -> controller-side authorized hydration + digest verification
  -> Nemo PRE_GENERATION witness
  -> proposal-only model generation
  -> Nemo POST_GENERATION witness
  -> A11oy PRE_TOOL admission + human approval when consequential
  -> execution outside the model
  -> Nemo POST_TOOL/postcondition witness
  -> signed receipt
  -> sanitized Anatomy observation
```

The standalone repository remains the source package during migration. It should
not be archived or redirected until Forge consumes a revision-pinned package,
parity tests pass, downstream users migrate, and a rollback path is recorded.

## Required expansion

The current R1-R5 text checks are a useful baseline. Promotion requires additional
rule families for evidence-handle binding, content-digest verification, exact
formula applicability and maturity, memory/tenant isolation, model and runtime
identity, tool authority, postconditions, and receipt verification.

Nemo may return `ALLOW`, `BLOCK`, or `REVIEW`. `ALLOW` means only that the witness
checks passed for the inspected stage. It never grants tool authority by itself.
