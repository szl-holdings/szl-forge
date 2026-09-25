# Unsupported GGUF tensor-type evidence (forensic v1.2.1)

This is a recovery-only observation correction for #264, not a new model loader,
training recipe, model repair or runtime-acceptance closure. It builds on #276
and preserves the #275 Windows identity fix and #270 local-drive checks.

## Owner evidence and its exact limit

The owner-produced forensic v1.1.0 report started at
`2026-09-13T13:54:55.849791+00:00`. Both named models were rejected with
`UNSUPPORTED_TENSOR_TYPE`. No comparison, inference, training, weight changes or
paired-attempt changes were reported. The report declares source SHA-256
`79af616edf5b9e4e1a1fe7a9f9c3bf96ff9d44df16c6f36239f06f953ba625f4`.
Its uploaded bytes have SHA-256
`e2fafbb503f6c33848b7769c605f536eb602a885914bce8f6a8e4918a2ff2a9f`.
This records a received file, not independent attestation of the laptop.

That error occurs after opening-identity admission and during tensor-descriptor
parsing. It precedes the completed full-file hash and final stability checks.
The previous report omitted the numeric GGML type ID. Therefore it cannot prove
BF16, any particular quantization, corrupt weights, tensor equality, or healthy
model behavior. Do not infer a per-tensor type from a model-level F16 label.

## Minimal change

The admitted layout table stays exactly `{0: (F32, 4), 1: (F16, 2)}`. No rejected
format is newly accepted. Instead, a rejection carries:

- the actual unsigned 32-bit `ggml_type_id` read from the descriptor;
- its zero-based tensor index and the byte offset of that type field;
- the unchanged list of supported IDs;
- `scope = PARTIAL_HEADER_NOT_FULL_FILE_AUTHENTICATED` and
  `historical_blob_sha256_verification = NOT_COMPLETED`.

The exception's code also includes the numeric ID so it is visible immediately
in the console. The fixed, bounded numeric fields are retained in the per-model
record, final report, and existing result checkpoints. No raw tensor name,
vocabulary, prompt, local source path, tensor value or header byte is exported.

A rejected descriptor stops parsing before its tensor offset/payload is read;
no extent is guessed and no fallback conversion or acceptance exists. A partial
header cannot authenticate itself. The pair stays INCOMPLETE with null comparison
unless both whole-file inspections independently succeed. The error remains
UNAVAILABLE_OR_REJECTED, not 'model broken'.

This patch does not change existing fingerprints, whole-file SHA-256 gates,
byte/layout/resource bounds, local-drive/reparse/stable-file checks, default
plan-only behavior, exclusive/fsynced checkpoints, or no-resume behavior.
`szl_recovery.py` and its manifest, request identity, permanent claims, model pins,
historical scores and once-only ReceiptAgent experiment are untouched.

## Interpreting the next observation

The numeric code must first be observed on the owner file. Consult the exact
pinned upstream definition and review/test any needed byte-layout support only
then. A known upstream enum is not automatic admission by this inspector.
Examples used by tests are synthetic and are NOT observations of owner files.
Unknown or malformed IDs remain explicit rejection, never a download request.

Primary specification references, pinned independently of the owner's model:

- https://github.com/ggml-org/ggml/blob/7840aaba1989c6deeefede1d77d5aaf8f52b947e/docs/gguf.md
- https://github.com/ggml-org/ggml/blob/7840aaba1989c6deeefede1d77d5aaf8f52b947e/include/ggml.h

## Tests and execution boundary

15 new offline tests assemble small GGUF fixtures, retaining numeric codes,
including an unsupported type despite a model-level F16 metadata label. They
cover index/offset exactness, no payload reads after rejection, missing partial
type fields, no authentication claims, no private names in output, checkpoint
and console persistence, unchanged accepted F16/F32 fingerprints, ordinary
error redaction, and default no-I/O behavior. The existing Windows/Linux
workflow discovers the tests without changes to CI permissions or gates.

The tool still requires explicit `--inspect-blobs` on BETTERWITHAGE after exact
source admission and review of existing reports/workload. It writes a new local
report/checkpoints only; it never contacts OMEN, calls model APIs, imports Torch,
downloads weights, changes model files, starts training, or resets claims/locks.
An expected unsupported-type exit code 1 is successful *evidence collection*,
not a passing model or completed tensor comparison. Preserve it; do not retry
until that numeric evidence has been reviewed.
