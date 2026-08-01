---
title: SZL Model Inference Lab
emoji: 🧪
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
license: apache-2.0
short_description: Bounded GGUF API with unsigned execution provenance.
models:
  - SZLHOLDINGS/SZL-Khipu-1.5B-GGUF
tags:
  - gguf
  - llama.cpp
  - cpu
  - provenance
  - bounded-inference
suggested_hardware: cpu-basic
startup_duration_timeout: 30m
---


<div align="center">
<p>

[![governed](https://img.shields.io/badge/governed-SZL%20Holdings-3af4c8?style=flat-square)](https://huggingface.co/SZLHOLDINGS)
[![Λ](https://img.shields.io/badge/Λ-Conjecture%201%20advisory-d7b96b?style=flat-square)](https://a-11-oy.com)
[![license](https://img.shields.io/badge/license-apache--2.0-7e8aa3?style=flat-square)](https://huggingface.co/spaces/SZLHOLDINGS/szl-model-inference-lab)

</p>
</div>
<!-- SZL-ESTATE-CARD:v2:START -->
<p align="center"><a href="https://a-11-oy.com/"><img src="https://huggingface.co/spaces/SZLHOLDINGS/README/resolve/main/assets/estate-banner-v2.svg" alt="SZL Holdings — governed, receipted, verifiable" width="100%"></a></p>
<p align="center">
  <a href="https://github.com/szl-holdings/.github/tree/main/doctrine"><img src="https://img.shields.io/badge/doctrine-v11%20LOCKED-0B1F3A?style=flat-square" alt="doctrine v11"></a>
  <a href="https://a-11-oy.com/"><img src="https://img.shields.io/badge/evidence%20wall-LIVE%20%C2%B7%20verify%20in%20browser-3AF4C8?style=flat-square" alt="live evidence wall"></a>
  <a href="https://huggingface.co/datasets/SZLHOLDINGS/szl-lake"><img src="https://img.shields.io/badge/szl--lake-offline%20verifiable-C9B787?style=flat-square" alt="szl-lake offline verifiable"></a>
  <a href="https://huggingface.co/spaces/SZLHOLDINGS/holographic"><img src="https://img.shields.io/badge/estate%20map-holographic-5B8DEE?style=flat-square" alt="holographic estate map"></a>
</p>
<p align="center"><sub>Part of the <a href="https://huggingface.co/SZLHOLDINGS">SZL Holdings</a> governed estate — claims are designed to carry checkable receipts. Verification proves integrity &amp; origin, never accuracy or performance.</sub></p>
<!-- SZL-ESTATE-CARD:v2:END -->

# SZL Model Inference Lab

A public, zero-secret, bounded CPU demonstration for the exact
`SZLHOLDINGS/SZL-Khipu-1.5B-Q4_K_M.gguf` bytes at immutable model commit
`67d60ec577730747055491640cfb91fc4a4b5d25`.

> **Presentation class: BOUNDED RUNTIME DEMO.** This Space executes one pinned
> quantized derivative under strict limits. It does not train, promote,
> authorize, or certify the model. Runtime outputs and execution records are
> explicitly unsigned; transport availability is not readiness evidence.The canonical application source is
[`szl-holdings/szl-forge/spaces/szl-model-inference-lab`](https://github.com/szl-holdings/szl-forge/tree/main/spaces/szl-model-inference-lab).
The governed deployment binds the exact protected Git revision into the
non-secret `SZL_GITHUB_SOURCE_REVISION` Space variable and verifies it at
[`/api/build-info`](https://szlholdings-szl-model-inference-lab.hf.space/api/build-info).
That endpoint reports `UNKNOWN` rather than inferring a source revision when
the binding is absent or malformed.

The human-facing surface is the **Khipu Loom**: a responsive Formula Genome
instrument that keeps the source thread, immutable model pin, receipt boundary,
runtime state, and unsigned-output limitation visible beside the bounded
inference controls. It uses no external scripts, fonts, trackers, or UI assets,
and exposes a deterministic `data-screenshot-ready` signal only after the
runtime reaches `READY`.

An isolated image-build stage fetches only the exact GGUF and three receipt
files from the immutable model revision, without a token, and verifies them
before the image can finish. It copies only verified regular bytes into the
final image's fixed, root-owned `/opt/szl/model-artifacts` directory; the build
cache does not cross that stage boundary. This avoids depending on a platform
preload cache whose path may differ from the non-root Docker runtime's cache.
The runtime verifies the
986,047,904-byte file against SHA-256
`13c1a1993063e1dff92f7413ccf48eaca6d48efc8801ae9af35961ae3396623a`
before loading it. No mutable or full-repository runtime mount is required. At
startup the app resolves only those bundled immutable regular files from that
fixed directory, with no Hub/cache/network fallback, and verifies
their declared sizes, SHA-256 digests, and receipt signatures, and keeps
runtime Hub access offline.
It requires no provider token or Space secret and is intended for the Hub's free
`cpu-basic` hardware only.

## Boundaries

- One inference at a time; excess concurrent calls receive HTTP 429.
- POST bodies are capped at 8 KiB across ASGI chunks with one absolute
  10-second read deadline; slow/incomplete bodies receive HTTP 408.
- 1,200 input characters, at most 800 formatted prompt tokens, 32 generated
  tokens, and a 45-second best-effort cutoff checked between streamed chunks
  (not a hard wall-clock deadline).
- Greedy decoding (`temperature=0`); outputs are model-generated and may be wrong.
- `/live` and `/healthz` are liveness (`STARTING`/`READY` = 200; `FAILED` =
  503); `/health` and `/readyz` are readiness and return 503 until `READY`.
- `/version` fails closed unless the governed deployment provides one exact
  40-character source revision. `/evidence` fails closed unless that exact
  source identity, source-bundle integrity, and both declared-key receipts are
  simultaneously available. Neither endpoint upgrades unsigned runtime output
  into an attestation.
- `/api/v1/identity` exposes the immutable artifact, runtime limits, source release
  marker, and receipt boundary. Source checksums establish internal bundle
  consistency only; they are not external authorship evidence.
- `GET /v1/models` and `POST /v1/chat/completions` provide a deliberately small
  OpenAI-compatible subset. Chat is non-streaming, one choice, tool-free, and
  deterministic. Requests using streaming, tools, `n > 1`, nonzero temperature,
  or `top_p != 1` are rejected rather than silently changed.
- Chat accepts at most 12 string-only `system`/`user`/`assistant` messages whose
  combined content is at most 1,200 characters. The exact rendered ChatML is
  tokenized and must remain within the same 800-token prompt budget.
- Prompts are not intentionally persisted by this source.

The upstream training and evaluation receipts are checked against the repository's
declared Ed25519 key and chained canonical payload hash. That is **declared-key
continuity**, not independent ownership or authorship evidence. Those receipts do not cover the
GGUF quantization, this Space's source, runtime outputs, independent benchmarking,
or safety certification.

## Bounded OpenAI-compatible API

The public compatibility base URL is
`https://szlholdings-szl-model-inference-lab.hf.space/v1`. The only advertised
model ID is the immutable
`SZLHOLDINGS/SZL-Khipu-1.5B-GGUF@67d60ec577730747055491640cfb91fc4a4b5d25`.
The machine-readable contract is available at
`/.well-known/szl-inference-contract.json`.

This application does not require authentication. If an OpenAI client requires
an API-key string, use a literal dummy such as `not-a-secret`. **Do not send a
real Hugging Face token, OpenAI key, or any other credential.** This is a public,
best-effort demonstration with no provider SLA. Do not submit secrets, regulated
data, personal data, or other sensitive prompts. The source does not intentionally
persist prompts or execution records; platform or network logging outside this
source is not asserted.

Example request:

```bash
curl https://szlholdings-szl-model-inference-lab.hf.space/v1/chat/completions \
  -H "content-type: application/json" \
  -H "authorization: Bearer not-a-secret" \
  -d '{"model":"SZLHOLDINGS/SZL-Khipu-1.5B-GGUF@67d60ec577730747055491640cfb91fc4a4b5d25","messages":[{"role":"user","content":"Explain one limit of cryptographic receipts."}],"max_tokens":24,"stream":false}'
```

Each successful chat response contains standard `chat.completion`, `choices`,
and `usage` fields plus a namespaced `szl_provenance.execution_record`. That
record contains SHA-256 hashes of the canonical request and output, exact
model/source identifiers, tokenizer-derived usage counts, termination state,
timestamp, and request ID. It deliberately contains neither prompt nor output text. Its
`record_sha256` is repeated in `X-SZL-Execution-Record-SHA256`.

The record is **UNSIGNED** and explicitly says authenticity is not established.
It is content-addressed for hash recomputation and self-consistency only. A
separately retained expected hash can reveal later modification, but the record
does not reproduce an execution or resist an attacker who replaces both record
and hash. It is not a signature, attestation, native Hugging Face provider
mapping, or SLA. Recompute its hash without third-party dependencies:

```bash
python verify_execution_record.py response.json
python verify_execution_record.py response.json --request request.json
```

With the full response, the helper independently recomputes both the record and
output hashes. Supplying the original request additionally normalizes the bounded
request subset and checks its canonical SHA-256. It rejects unsupported request
fields, binds the outer response ID, timestamp, model, usage, and finish reason to
the record, and checks the fixed `UNSIGNED` release semantics. Hash and semantic
agreement still establish consistency only, not identity or authenticity; anyone
can construct a new internally consistent unsigned record.

## Attribution and licenses

Space source: Apache-2.0, copyright SZL HOLDINGS LLC.

- Runtime model: [SZL-Khipu-1.5B-GGUF](https://huggingface.co/SZLHOLDINGS/SZL-Khipu-1.5B-GGUF), Apache-2.0.
- Fine-tuned model: [SZL-Khipu-1.5B-BrainNavigator](https://huggingface.co/SZLHOLDINGS/SZL-Khipu-1.5B-BrainNavigator), Apache-2.0.
- Base model: [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), Apache-2.0.
- Inference binding: [llama-cpp-python v0.3.21](https://github.com/abetlen/llama-cpp-python/releases/tag/v0.3.21), MIT; CPU wheel URL and SHA-256 are pinned in `requirements.txt`.

All Python dependency versions are pinned. The llama CPU wheel is hash-pinned;
a complete system-package/SBOM attestation is not claimed.

No independent benchmark, post-quantization evaluation, or safety certification is claimed.

---

<div align="center">

**[SZLHOLDINGS on Hugging Face](https://huggingface.co/SZLHOLDINGS)**   |   **[a-11-oy.com](https://a-11-oy.com)**   |   **[Estate hub](https://szlholdings-szl-estate-live.static.hf.space)**

### Governed AI with inspectable evidence.

<sub>Labels remain explicit: MEASURED / REPORTED / MODELED / SAMPLE / UNKNOWN / UNAVAILABLE. Integrity and origin evidence do not establish model quality, safety, or runtime readiness.</sub>

</div>
