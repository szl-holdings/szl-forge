# GGUF tensor forensics for laptop recovery #264

## What this advances

The owner recovery report observed **different provider-reported FROM blob IDs**
for the two repeated-`@` Qwen2-family 3.1B F16 models. That distinguishes their
containers, not necessarily their tensor payloads: GGUF also stores metadata and
padding. Identical underlying tensor bytes in differently packaged files remain
a possible explanation. This tool tests that narrower question without invoking
either model, using the two exact historical FROM SHA-256 identities in #264.
It does not select new model weights or claim an import/conversion root cause.

`gguf_forensics.py` performs a bounded sequential read of each existing local
blob, parses supported tensor descriptors, hashes **every file byte**, and
compares tensor fingerprints that bind name, type, shape and payload but exclude
storage offsets/order/padding. It separately fingerprints serialized tokenizer,
chat-template and architecture metadata. Metadata ordering does not affect those
fingerprints. Missing metadata remains `null`/unknown, not an equal empty result.
Only hashed tensor names, metadata hashes and structural counts are written; no
raw vocabulary, private prompts, raw tensor bytes or local source paths enter the
JSON report.

A full container hash must match the historical provider-reported digest before
its tensor result is eligible for comparison. The tool does **not** query current
Ollama registration: a match binds historical blob bytes, not a currently served
tag. Matching hashes are not a signature, trusted clean-source lineage, numerical
health test, useful output, performance result or production qualification.

## Closed scope

Supported: single-file little-endian GGUF v2/v3, `general.architecture=qwen2`,
unquantized F16/F32 tensors. These are the representations needed for the observed
failures. Quantized/BF16 tensor types, big-endian, split models, unknown versions,
overlapping or out-of-file tensor ranges and oversized structures are refused.
BF16 can only be admitted by the explicit `--allow-bf16` opt-in described in the
forensic v1.3.0 section below. Without the opt-in, it is refused exactly as before.
This is intentionally **not** a general GGUF reader or an inference loader.
Nonzero padding/trailing bytes participate in the whole-file hash but not tensor
fingerprints; accepting a layout is not asserting full format conformance.

Bounds: 8 GiB per file, 64 MiB header, 10,000 tensors, 10,000 metadata entries,
1 MiB per read/string value, 2,000,000 entries per array, 4,000,000 total value
nodes, depth 4, and a cooperative ten-minute read deadline per blob. Storage
reads may still affect another workload's I/O; this is not resource isolation or
an absolute deadline for a hung device. No model-sized temporary copy is made.

Paths are owner-selected local paths. Existing symlink/reparse components and
UNC paths are refused. The open descriptor and path are checked for stable
identity/size/mtime/ctime. This detects normal concurrent mutation, not a hostile
writer able to race and restore metadata. A failed input is not deleted/repaired.
The only writes are a new report directory under the owner's home.

No network/API calls, GPU access, generation, Torch import, downloads, package
installs, model conversion, quantization, training, publication, process stopping,
Ollama service changes, or access to OMEN. Recovery v1.2's code, manifest, tests,
permanent attempt identities and locks remain byte-for-byte unchanged. This tool
is a separate source-controlled diagnostic, not another paired experiment.

## Owner command (only after source admission)

Inspect the exact committed source and tests first. A no-argument invocation is
plan-only: it reads no model blobs and creates no reports.

```powershell
py -3.12 -I -B .\local-compute\recovery\gguf_forensics.py

# Explicit local disk/CPU read; NEVER generation of the broken models.
# Entry point requires Windows BETTERWITHAGE.
py -3.12 -I -B .\local-compute\recovery\gguf_forensics.py --inspect-blobs
```

Default input directory is `$HOME\.ollama\models\blobs`. An owner who deliberately
configured a different model store can explicitly supply `--blob-root` with that
existing local **blobs** directory. Environment variables and Modelfile paths are
not followed automatically. There is no drive scan, reset or force option.

The output is a new `$HOME\szl-gguf-forensics-<uuid>\gguf-forensics.json`.
Exit zero means the admitted forensic comparisons completed, never that a model
is healthy. Incomplete, unsupported or hash-mismatched inputs remain visible;
no model is inferred absent just because the default directory is wrong.

## Interpretation and next decision

- `same_tensor_set_bytes=true`, different containers: tensor payload identity
  matches at names/types/shapes; investigate differences in metadata, registered
  templates, runtime and source provenance. Do not infer healthy weights.
- Different tensor set: inspect changed/shared/missing counts and separate source
  histories. This alone does not tell which artifact is correct.
- Missing or rejected input: retain unknowns; do not download a substitute to make
  this comparison pass.

Regardless of result, a genuine repair still requires a separately named candidate
from verified clean source, correct tokenizer/import lineage, normal completion
and a fresh held-out evaluation. Do not relabel either historical 0/6 score or
close #264 on fingerprints. ReceiptAgent's once-only paired inference remains a
separate gate and is not executed here.

## Validation

`test_gguf_forensics.py` builds tiny GGUF fixtures directly from the documented
byte layout. Coverage includes equal tensors in different containers, tensor
changes, reordered metadata/storage, template separation, unknown missing fields,
all truncation positions of a fixture, unsupported types, bounds/deadlines,
symlinks, full-file hash drift and concurrent mutation. It performs no API/GPU
operations and does not use the owner's model bytes as fixtures. The existing
Windows/Linux recovery workflow discovers these additional tests automatically.

```powershell
py -3.12 -I -B -m unittest discover -s local-compute/recovery -p 'test_gguf_forensics.py' -v
```

## Primary format references

- GGUF specification: `ggml-org/ggml@7840aaba1989c6deeefede1d77d5aaf8f52b947e`,
  `docs/gguf.md` (file layout, value types, tensor offsets, F32/F16 type IDs).
  https://github.com/ggml-org/ggml/blob/7840aaba1989c6deeefede1d77d5aaf8f52b947e/docs/gguf.md
- Ollama model directory/custom storage documentation:
  https://docs.ollama.com/faq#where-are-models-stored

Source reference does not mean upstream authors reviewed this implementation.

## Interrupted reads and durable evidence (forensic v1.2)

This successor preserves #275's Windows path/fstat correction and per-model
error reporting; see `FILE_IDENTITY.md`. The version here belongs to the GGUF
diagnostic, not the separate `szl_recovery.py` v1.2 paired-runtime helper.

The original inspector wrote its only report after both reads. An interruption
during the second read could discard the first completed result. Forensic v1.2
adds exclusive-create checkpoints in the same report directory: one initial
record, then read-intent and result records for each of the two historical blobs
(`checkpoint-000.json` through `checkpoint-004.json`). The directory is printed
before model-byte reading begins. No existing reports are changed or deleted.

Each checkpoint binds the source hash, exact historical blob digests, sequence,
phase, current read intent and completed per-model observations. The envelope is
`szl.gguf-forensics-checkpoint/v1`; `automatic_resume_authorized` is always false.
All checkpoints retain an INCOMPLETE report and null comparison. The existing
`gguf-forensics.json` final report alone records the completed comparison when
both inspected inputs pass. Fingerprint semantics and qualification limits do
not change. A checkpoint is not a model-quality pass or an attempt claim.

Writes are serialized before exclusive creation, then flushed and fsynced before
the next blob is read. A checkpoint write, close or fsync failure propagates and
stops further reads; it is not retried. KeyboardInterrupt and unexpected failures
still terminate the operation, leaving earlier checkpoint files intact. This is
OS-acknowledged file durability, not a guarantee against all power loss, storage
failure or hostile filesystem races. A present file does not by itself prove its
fsync/close succeeded, and a truncated record remains uncertain.

For an interrupted run, inspect the exact existing output directory and all its
checkpoints before planning any further work. Read intent means a read may have
started, not that it completed. A missing model result remains UNKNOWN. Do not
infer successful completion from the largest sequence number, merge unrelated
runs, delete partial evidence, or restart a job to manufacture a final report.
This change has no automatic resume/retry path, no model API calls, and no access
to recovery v1.2's permanent claims. A changed helper version or output directory
never makes a new ReceiptAgent experiment. Read current workload/storage state,
existing recovery reports and claims before any separately admitted device work.

`test_gguf_evidence.py` adds mocked, offline regression cases for interrupted
reads, initial/result/final write failures, exclusive creation, redaction, flush
ordering and unchanged plan/host/quality boundaries. The existing Windows/Linux
recovery workflow discovers it without changing workflow or security controls.

Primary write-durability reference:
https://docs.python.org/3.12/library/os.html#os.fsync

## Opt-in BF16 layouts (forensic v1.3.0)

This is a recovery-only parser increment for #264. It does not load, repair or
qualify a model, and it does not add a new owner observation. Issue #264 stays
OPEN/HOLD.

**Default unchanged.** `TENSOR_TYPES` is still exactly `{0: (F32, 4), 1: (F16, 2)}`.
Without the opt-in, a type-30 descriptor raises the same
`UNSUPPORTED_TENSOR_TYPE:ggml_type_id=30` rejection as v1.2.1. It has the same
bounded observation (`PARTIAL_HEADER_NOT_FULL_FILE_AUTHENTICATED`, blob
verification `NOT_COMPLETED`), and no read goes past the type field. Default
F32/F16 fingerprint results and default rejection records are pinned in the tests
to digests produced by the unmodified v1.2.1 source. Default reads also keep the
parent's exact internal call shape.

**Opt-in.** `--allow-bf16` (or `allow_bf16=True` in code) adds one entry,
`OPT_IN_BF16 = {30: (BF16, 2)}`. This follows the pinned primary definitions at
`ggml-org/ggml@7840aaba1989c6deeefede1d77d5aaf8f52b947e`:
- `include/ggml.h` defines `GGML_TYPE_BF16 = 30` and declares `ggml_bf16_t` as a
  single `uint16_t`.
- `src/ggml.c` type traits give `blck_size = 1` and
  `type_size = sizeof(ggml_bf16_t)`.

The flag must be a real boolean. Otherwise the call is refused with
`OPT_IN_FLAG_BOUND` before any file access. Opting in changes only the admitted
width table. All other types, including quantized, integer, F64, removed and
unknown IDs, still reject, and the rejection records the admitted IDs `[0, 1, 30]`.

Opted-in results and reports record `opt_in_tensor_types: ["BF16"]`. The final
report and checkpoints always carry the field (empty without the opt-in). A BF16
tensor's declared type is bound into its fingerprint. The same bytes declared as
F16 and as BF16 are different tensors.

**No values decoded.** The parser still only hashes payload bytes. It never
converts BF16 to F32, checks finiteness or evaluates weights. The full-file
SHA-256 gate against the historical FROM digest, the stable-identity checks, the
bounds, the checkpoints and the BETTERWITHAGE host gate are unchanged and apply
in opt-in mode too.

**Independent fixtures.** `test_gguf_bf16_opt_in.py` assembles its own GGUF bytes
and does not reuse the other test builders. BF16 payloads are hand-written hex
literals. Before any parser call, their decoded values are checked two
independent ways:
- bf16 to f32 as a 16-bit left shift;
- equality with the upper two little-endian bytes of the exactly representable
  binary32 value.

The tests cover a pure BF16 file and a mixed F32/F16/BF16 file. They also cover
tight packing that proves the 2-byte width (16 elements fit, 17 overlap), every
truncation point, trailing bytes, default rejection, unknown types under the
opt-in, non-boolean flags, full-file authentication and CLI threading.

**Not claimed.** The following remain open:
- authentication of either complete historical blob, and the current artifact
  location on the owner device;
- normal load or generation;
- frozen held-out behavioral qualification.

Numeric type 30 in a partial header does not authenticate a full file, and it
does not diagnose the repeated-`@` output. This change never runs a model,
downloads or converts weights, touches Hugging Face, trains or replaces anything,
or changes `szl_recovery.py`, its manifest or the once-only ReceiptAgent
experiment. ReceiptAgent failures remain a separate matter. A new owner
inspection, if one is admitted, still needs the existing pre-run reads of
reports, claims, locks and workload.

```powershell
py -3.12 -I -B -m unittest discover -s local-compute/recovery -p 'test_gguf_bf16_opt_in.py' -v
```
