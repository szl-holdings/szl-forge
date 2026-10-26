# Legacy publication artifact handoff — transport only

This closes the missing producer/consumer transport portion of issue #266.
It does **not** complete or authorize the legacy generic publisher. Its
production disposition remains **BLOCKED** until the independent boundaries
below are implemented and demonstrated in a non-publishing end-to-end rehearsal.
No workflow dispatch, model training or Hub mutation is part of this repair.

## Source and run binding

The workflow checks out the exact `github.sha`. Producers create new directories
outside the checkout, beneath `RUNNER_TEMP`. Committed `release/` and `bom/`
snapshots are not consumed or overwritten. Each producer seals its exact small
file set before upload; absent files are errors. Each downstream job receives
the producer's immutable `artifact-id` through a declared `needs` dependency,
rejects blank/malformed IDs, downloads that ID only, and verifies the envelope
before reading any evidence. It never falls back to an artifact name or selects
all artifacts. The upstream dataset admission gate remains separate.

| Producer | Exact evidence carried | Consumers |
| --- | --- | --- |
| heldout-eval | `heldout-eval.json` | model-bom, sign, publish |
| model-bom | `model-bom.cdx.json` | sign, publish |
| sign | Manifest, weights digest, and manifest/BOM/digest signature-certificate pairs | conformance, publish |

`scripts/release_artifact_handoff.py` binds the fixed profile, canonical repository,
full source SHA, workflow name, run ID and attempt to the file names, byte lengths
and SHA-256 values. It rejects missing/extra files, symlinks/reparse points,
non-regular or empty files, files above 16 MiB, malformed/duplicate-key receipts,
and an existing stage or envelope. The envelope has a 16 KiB limit. It contains
no weights, tokens or private keys. These bounds are deliberate; larger evidence
requires a reviewed contract change, not silent truncation.

The byte hashes are **MEASURED** by this helper. The envelope remains unsigned;
`signature_verification` is `UNAVAILABLE` and `publication_eligible` is always
false. Its identity values originate in the hosted workflow environment and its
source SHA must match checkout HEAD. This is not independent source attestation,
proof of evaluation correctness, or evidence that a signing identity is trusted.
A malicious producer or compromised runner can forge an envelope; same-user
filesystem races and runner isolation require separate controls.

## Reruns and failures

Re-run **all jobs** after a failed attempt. Re-running failed jobs only can retain
successful upstream artifacts from the prior attempt; those must fail the exact
attempt comparison. Do not edit the envelope or relax the comparison to continue.
Existing staging directories are not deleted or reused. A failed verification
does not rewrite evidence, generate approval, or fall back to tracked snapshots.

## Independent boundaries still BLOCKED / UNKNOWN

- The evaluation/refusal evidence aggregation and policy-qualified held-out
  results must be implemented and verified; transport does not create them.
- Final immutable manifest bytes, BOM and weights must be cryptographically
  verified against the permitted signing identity, with actual conformance
  evidence. Existing false signature/conformance declarations stay false.
- Installed evaluation/signing/scanning tool versions and their runtime must be
  established. Hosted runner defaults are not installation evidence.
- The generic publisher still needs model-family ownership, exact target/source
  allowlists, complete input validation, provider CAS, a single coherent release,
  immutable Hub revision/readback, mutation-uncertainty reconciliation and rollback.
- `hub-production` approval, admitted data rights, source-bound model quality
  gates and independent witnesses are not supplied by these software tests.

Tests use **SIMULATED** tiny files, never model weights or held-out corpus rows.
They establish the software transport contract only. Successful CI is not a
deployment, a model-quality result, or permission to publish.

## Scoped local checks

```text
python -m pytest -q tests/test_release_artifact_handoff.py tests/test_release_handoff_workflow.py
python -O -m pytest -q tests/test_release_artifact_handoff.py tests/test_release_handoff_workflow.py
python -m ruff check scripts/release_artifact_handoff.py tests/test_release_artifact_handoff.py tests/test_release_handoff_workflow.py --select E9,F
```

Provider behavior is documented by the pinned official
[upload-artifact](https://github.com/actions/upload-artifact/tree/cf430e030ddbb5b0abf93d22962f4752f3646cd9)
and [download-artifact](https://github.com/actions/download-artifact/tree/9000827ccba6bdab643e8b6fd33ac0654aef8333)
implementations. An artifact ID selects transport bytes; it is not a signature
or release authority.
