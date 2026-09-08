# Local compute: measured work, zero paid-provider jobs

## Measured September 7-8 batch

The [batch evidence](evidence/local-batch-20260908.json) records five installed
Ollama models, a one-step native smoke, a failed memory-gated attempt, and a
completed 16-step native continuation. The candidate is **HOLD_NOT_QUALIFIED**:
heldout cross-entropy worsened from 1.2760287934 to 1.3987944813, and a separate
saved-weight reload comparison scored **4/6 for both parent and candidate** on
the same six synthetic checks. No existing model was overwritten or promoted.

The completed continuation had 10,822,656 trainable LoRA parameters and saved a
43,346,432-byte safetensors adapter. Its source-revision field in the original
v1 training receipt identifies the committed **curriculum**, not the executable
revision; the executable is separately bound by runner SHA-256. The public
summary names that field `curriculum_source_revision` to avoid conflating them.
The original local receipts retain all failures and parent-integrity checks.

`evaluate_native_smoke.py` requires an externally supplied trusted SHA-256 of
the completed training report, rehashes candidate files, reloads both adapters
onto the same pinned base, and checks normal EOS completion and exact typed
answers. This is a reload/inference check, not an independent safety assessment.

Use existing hardware and installed models. This lane does **not** buy cloud
compute, pull Ollama models, deploy Spaces, publish weights, or execute generated
code. Electricity, hardware depreciation and existing connectivity are not free
and are not measured here.

## Installed-model evaluation

```powershell
python -I -B -m unittest discover -s local-compute -p 'test_*.py' -v
python -I -B local-compute/benchmark_ollama.py --output local-compute/results/ollama-new-run.json
```

The runner admits only an explicit loopback HTTP address, installed GGUF bytes
with immutable digests, and non-cloud model metadata. It disables proxies and
redirects and rejects an already-loaded Ollama workload. Six frozen synthetic
checks use temperature zero, seed 907, 1,024 context tokens and 128 generated
tokens, no tools, and unload the requested model after each request. Digests are
read back after evaluation. No model is created or overwritten.

Version 2 rejects duplicate/nonstandard JSON and requires a normal stop before
an answer can pass. A completed protocol is **not** a passing score. Strict
schema failure, incomplete generation, unavailable runtime, and model drift
remain visible. Do not turn these six smoke tests into a general intelligence,
safety, medical, or production-readiness score. Raw synthetic responses, timing,
model hashes, protocol hashes and runner hashes remain in the local receipt.
Load time is included in wall time; these are not warm-throughput benchmarks.

## Native ReceiptAgent continuation candidate

`train_receiptagent.py` is a bounded native bf16 continuation of the existing
ReceiptAgent-v2 LoRA, not a reimplementation of its original Unsloth 4-bit run.
It preserves `Qwen3_5ForConditionalGeneration` and the exact adapter namespaces.
Only owned, hash-verified committed synthetic curriculum is admitted: 26 unique
training conversations, 37 with refusal oversampling, and 11 historical heldout
conversations. No scraped publications, user secrets, patient data, or existing
benchmark answers are added to training.

Prepare a dedicated environment using your already verified CUDA installation.
Pin PEFT 0.20.0 and Trackio 0.37.0 there; verify the resolver has **not** substituted
a CPU/XPU Torch build. Do not modify a shared working runtime or blindly upgrade
Torch. The verified local execution uses Torch 2.11.0+cu128 and Transformers
5.16.1. The trainer rechecks CUDA/bf16 availability and configuration compatibility.

Download only these exact public snapshots, with `hf download` and explicit
filenames. No `trust_remote_code`, pickle, or executable model artifacts:

| Role | Repository | Immutable revision |
| --- | --- | --- |
| Implementation base | `unsloth/Qwen3.5-0.8B` | `23c69c53358a07516b5827588b3fdb12ae78fd65` |
| Parent adapter | `SZLHOLDINGS/szl-receiptagent-qwen35-0.8b-v2` | `46f54373c6bf8f17a288b4c8799e9fbe4b82ecc1` |

Use `verify_snapshot.py --repo ... --revision ... --directory ... --output ...`
to bind each local file to provider LFS SHA-256 or Git blob hash. The manifest
goes outside the snapshot. The trainer independently rehashes every declared
file and rejects undeclared artifacts before loading weights. The manifest's
own SHA-256 is pinned in reviewed source; changing both a file and the adjacent
manifest cannot rebind arbitrary weights to an approved revision. Regenerate the
manifest with the supplied deterministic writer; edited serialization is refused.

```powershell
python -I -B local-compute/train_receiptagent.py `
  --base-dir local-compute/cache/qwen35-08b `
  --base-manifest local-compute/results/base-manifest.json `
  --adapter-dir local-compute/cache/receiptagent-v2 `
  --adapter-manifest local-compute/results/adapter-manifest.json `
  --source-commit 225751bcf16372360df9ad04d63cba39fc8bc0ca `
  --output local-compute/results/receipt-native-new-run `
  --steps 1 --max-runtime-seconds 1800
```

Run the one-step compatibility smoke first. A subsequent predeclared run can use
16 or 32 optimizer steps, two microbatches per step, learning rate 5e-5 and seed
11. The default masks prompt labels and refuses silent truncation. Full-sequence
loss requires an explicit option and has a different meaning. Baseline and
post-training token-weighted heldout cross-entropy are recorded; lower loss alone
does not establish a better model. This historical split is not a fresh blind
evaluation, and cross-entropy is not a generation acceptance test.

Outputs are new candidate directories, never replacements. The trainer records
failure evidence, uses a local compute lock and cooperative runtime/thermal/VRAM
guards, and writes local-only Trackio metrics. It rejects remote tracking
configuration. A hung driver call still needs external supervision; the
cooperative timer is not a Windows Job Object guarantee. Saved candidates remain
ineligible for publication/promotion pending clean reload, generation gates,
fresh holdout evaluation and exact source/weight provenance.

## Docker, Tailscale and llama.cpp

Offline contracts can run in an existing trusted Python container with no
network, a read-only bind mount, read-only root filesystem, dropped capabilities,
a non-root UID and bounded CPU/memory. Container tests do not prove GPU training
or live device integration.

Tailscale connectivity is inventory evidence, not remote-compute authorization
or proof of a GPU. Verify the peer identity, SSH host key, access policy and
hardware before dispatch. Do not disable host-key verification, enable Funnel,
expose Ollama publicly, or infer a distributed training cluster from a ping.

Ollama provides the current installed GGUF inference path. A standalone
`llama-cli`/`llama-server` installation must be verified separately before claiming
it is available. Quantization is an inference representation, not training.

## Upstream methods and attribution

Reuse licensed methods with attribution: small task-specific adapters, pinned
data provenance, bounded evaluation and measured quantized inference. Relevant
primary sources include [PEFT checkpoint format](https://huggingface.co/docs/peft/developer_guides/checkpoint),
[Qwen3.5 model documentation](https://huggingface.co/docs/transformers/model_doc/qwen3_5),
[DeepSeek-R1 and distilled model lineage](https://github.com/deepseek-ai/DeepSeek-R1),
[llama.cpp](https://github.com/ggml-org/llama.cpp), and
[Ollama chat API](https://docs.ollama.com/api/chat).
Open weights do not imply unrestricted dataset or derivative rights; preserve
each model's base license and attribution. This project claims its implementation
and measured results, not ownership of upstream research or unmeasured novelty.
