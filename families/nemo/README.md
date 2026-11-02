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

## NVIDIA proposal triad

A Nemotron-backed runtime adapter stays `nemotron-runtime-adapter`. Its
output is a proposal digest, not a verdict and not the `szl-nemo` witness.

The witness package `szl-nemo` exposes `szl_nemo.evaluate_triad`
(`szl.nemo.triad.v1`). A Forge NVIDIA lane that adopts it must pass:

- an authorized caller and a SHA-256 of the request, without the raw request;
- `family: NVIDIA_NEMOTRON` and `authority: PROPOSAL_ONLY`;
- energy as finite NVML joules labeled `MEASURED`, or `null` labeled
  `UNAVAILABLE` with source `NVML_ABSENT`;
- Λ scores in `[0, 1]`. The binder computes the geometric mean and labels
  it modeled. Unconditional uniqueness under A1–A5 is false as stated.
  Theorem U is conditional. The score cannot authorize;
- no DSSE signature. `szl-nemo` holds no signing key, so the record stays
  `UNSIGNED_HONEST`.

The optional test lane is `tests/test_nvidia_triad_lane.py`. It does not
change the production pin. Set `SZL_FORGE_NEMO_TRIAD_LANE=1` to install
immutable `szl-nemo` commit `116630d2274731998124d7af089a9383074ce476`
(package 0.4.2) in a temporary environment and run the triad contract.
Production stays on `810231a531188bb569e3faa17396386eb0a5e260` (package 0.4.0).

This page does not claim a fine-tune, a benchmark, or a live NVML reading.
