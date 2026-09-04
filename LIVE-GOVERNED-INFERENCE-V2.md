# Live governed inference v2 — promotion contract

This document records the exact promotion boundary for the SZL Model Inference
Lab. It is a deployment contract, not evidence that the public runtime is live;
live status is established only by the protected-main publisher and its retained
post-deployment verification report.

## Immutable release identity

- Space: `SZLHOLDINGS/szl-model-inference-lab`
- Release ID: `6165392a-f3b7-4128-8f77-e37074c2e60d`
- Model repository: `SZLHOLDINGS/SZL-Khipu-1.5B-GGUF`
- Model revision: `67d60ec577730747055491640cfb91fc4a4b5d25`
- Model file: `SZL-Khipu-1.5B-Q4_K_M.gguf`
- Model SHA-256: `13c1a1993063e1dff92f7413ccf48eaca6d48efc8801ae9af35961ae3396623a`
- Forge controller source: `943f6ab987bbe120cae32649c46c3a5f0b6f9e9b`
- Second Brain source: `fa3e4605344b13db220a79f9dcd267ee5725c87e`
- Nemo source: `810231a531188bb569e3faa17396386eb0a5e260`

## Runtime path

```text
public request
  -> fixed public-projection ACL
  -> Second Brain hybrid handles
  -> controller-only digest-verified hydration
  -> exact formula applicability binding
  -> bounded llama.cpp proposal
  -> Nemo E1-E10 envelope witness
  -> Nemo R1-R5 output witness
  -> claim/citation validation
  -> deterministic unsigned inference receipt
  -> sanitized local Anatomy observation
```

The public Space is proposal-only. It exposes no tool execution surface and no
A11oy action-admission shortcut. Consequential actions require a separately
signed A11oy continuation and are outside this public runtime.

## Public v2 surfaces

- `GET /api/v2/governed-health`
- `GET /.well-known/szl-governed-inference-contract.json`
- `POST /api/v2/governed-infer`
- `GET /api/v2/anatomy/last`

## Runtime-availability semantics

A credentialless source-binding dry run may execute before the governed Space
exists. A missing Space or required runtime route is recorded explicitly as
`NOT_QUALIFIED_NO_RUNTIME_PROBE` with
`RUNTIME_SERVICE_UNAVAILABLE`; it is not misclassified as source corruption.
This classification never authorizes promotion. Publication and every path that
requires exact runtime evidence continue to fail closed until the Space exists,
the expected routes answer, and their immutable source/model identities verify.

## Promotion gates

The branch may merge only when all repository checks are terminal and green,
the Space source manifest has exact LF-normalized SHA-256 closure, the release
ID agrees across runtime/verifier/model bindings, and no transient repair
workflow remains. After merge, the protected-main publisher must create or
recover the Space repository, publish the exact source set, verify immutable
readback, wait for `RUNNING`, exercise the v2 endpoint, recompute receipt and
citation digests, and retain the sanitized verification report.

## Truth boundary

This release operationalizes governed inference on the existing bounded CPU/GGUF
research runtime. It does not claim a universal inference-engine winner, broad
model quality, independent safety certification, autonomous execution, or a
service-level agreement. Lambda/F23 remains Conjecture 1 and advisory only.
