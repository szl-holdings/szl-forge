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

Disk admission happens **before** creating output directories or the local lock.
The hard free-space floor is 1.5 GiB, with an additional 256 MiB of conservative
candidate headroom. Missing output paths are probed through their nearest
existing ancestor on the resolved output volume; the lock volume is checked
separately. Probe failure is `DISK_PROBE_UNAVAILABLE`, not a zero or a pass.
Runtime, adapter/processor checkpoint writes, and explicit Trackio writes each
get a fresh probe. These cooperative observations do not atomically reserve
space, bound Trackio background writes, or establish machine-wide GPU ownership.
Use the existing explicit `--no-trackio` option for a pilot that needs no tracking
background writer; it remains recorded as `EXPLICITLY_DISABLED`.

The final receipt is serialized in memory, capped at 1 MiB, and admitted using
its actual UTF-8 byte length plus a 4 KiB filesystem allowance above the hard
floor. A failed capacity check or write returns exit 1, `FAILED_CLOSED`, and
`durable_report_written: false` with the in-memory receipt on stdout. An earlier
execution error remains separately recorded. No blocked/partial write is
claimed as a saved receipt, and neither partial candidates nor another owner's
lock are removed. Receipt bytes are written to a unique pending file, flushed,
synced and closed before the final name is published with a no-overwrite hard
link. Unsupported hard links fail closed; there is no overwrite/copy fallback.
Pending diagnostics are retained and are not final receipts. A failed sync
cannot leave a successful final receipt. Failed owned-lock cleanup retains the
uncertain lock and original execution error, and still attempts bounded failure
evidence. File sync is not a portable directory/crash-durability guarantee. A
saved report is still local evidence, not publication or promotion authority.

### Private archive copy of a completed ReceiptAgent adapter

`archive_receiptagent.py` is a separate, standard-library-only copy step for a
**completed** native continuation. It accepts one immediate child of
`local-compute/results/`, verifies `training-report.json` and the exact
`candidate_files` inventory under its `adapter/` directory, then copies the
report and adapter to a new directory beneath a private local archive root.
It does not rerun the trainer, import a model, read the curriculum, copy Trackio
data, or change a source run. The source run must remain on plain local storage.
The archive root may be in a sync folder: hydrated Microsoft Cloud Files
ancestors are admitted using their native tag and placeholder state; symlinks,
junctions, unknown reparses, offline/recall entries, and unavailable metadata
are refused. A sync-folder copy is still only a local copy.

The operator keeps the request JSON outside the checkout and archive. Its
review digests and statuses are **operator assertions**, not independently
verified signatures or proof of rights. The report hash must be computed from
the completed `training-report.json` bytes, and the source and runner hashes
must agree with that report. A request has these exact keys:

```json
{
  "schema": "szl.native-archive-request/v1",
  "report_sha256": "<64 lowercase hex characters>",
  "source_revision": "<40 lowercase hex characters>",
  "runner_sha256": "<64 lowercase hex characters>",
  "source_provenance_sha256": "<64 lowercase hex characters>",
  "rights_review_sha256": "<64 lowercase hex characters>",
  "leakage_review_sha256": "<64 lowercase hex characters>",
  "owner_release_sha256": "<64 lowercase hex characters>",
  "rights_status": "APPROVED_FOR_PRIVATE_ARCHIVE",
  "leakage_status": "PASS",
  "owner_release_status": "RELEASED_FOR_PRIVATE_ARCHIVE"
}
```

After the reviews and a completed run, a future authorized local operator can
use portable paths without recording an account or private URL in Git:

```powershell
$env:SZL_ARCHIVE_ROOT = '<private local archive root>'
python -I -B local-compute/archive_receiptagent.py `
  --run 'local-compute/results/<completed-run>' `
  --request '<private local review request.json>'
```

The sidecar caps the report at 1 MiB, the adapter at 64 files and 512 MiB
including the report, each file at 384 MiB, and streams one MiB at a time.
The earlier measured adapter weights were 43,346,432 bytes. It checks free
space on both the plain-local temporary staging volume and archive volume.
Each source file is first copied and hash-verified in that temporary stage so
changed source bytes cannot reach the sync directory before validation. It
refuses undeclared files and known executable-serialization extensions,
scans for common private path/email/token patterns, and rehashes each local
destination. Unknown top-level report fields fail closed. The pattern scan is
a guard, not a comprehensive secret review;
the byte copy does not parse safetensors or establish model validity.
The archive is built in a unique `.pending-*` sibling. A failed copy leaves
that pending directory for diagnosis, with no automatic deletion or overwrite;
a retry uses a new sibling. Only after all copied bytes and
`archive-copy-receipt.json` pass local readback is the directory renamed to
`candidate-<report SHA-256>`. The receipt records
`LOCAL_COPY_VERIFIED_REMOTE_UNVERIFIED`, `remote_restore_verified: false`,
`checkpoint_completeness: NOT_RESTARTABLE`, and no promotion claim. Adapter
bytes plus the report do not include optimizer state or a full base model.
Keep the original run until the storage owner independently restores and
rehashes remote bytes. Hugging Face publication and the `a-11-oy.com` and
`a11oy.net` public surfaces have separate release and evidence gates; this
local receipt does not change their status.

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
