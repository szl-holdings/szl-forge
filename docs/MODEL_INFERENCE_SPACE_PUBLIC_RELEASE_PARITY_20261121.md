# Model Inference Lab: immutable public source-integrity incident (2026-11-21)

Observed: public SZLHOLDINGS/szl-model-inference-lab returns HTTP 503 on GET /readyz with failure_code=SOURCE_INTEGRITY_MISMATCH. The Space is RUNNING with root GET 200, but model_sha256_verified=false and receipt_status=NOT_CHECKED. This is source-integrity refusal, not measured model execution.

- GitHub szl-holdings/szl-forge main: be775d0ae4418860eeddab6d2cbe8800cd624c81.
- Immutable public HF Space revision: e028aa7fdf3668bf739bb844a053b10285884a76.
- Separately served *self-declared* source: c94e9218093e0815745cbd5dccaddea1797379a3.
- GitHub and HF release.json byte-identical; SHA-256 557e4c1463e46d02ef15fd92366b3106e274a7dc5edd3785288ef416f82e25ac.
- 25/26 manifest-listed files match; README.md alone differs. GitHub README: 14,033 bytes SHA-256 89d7b1c755cbff0609e7cf659fbfc190c168949903655d265c7b2bfcd30e39ae. HF README: 14,900 bytes SHA-256 df42d196d75a4c8f8551223aad2067cc81034b05c744d6d9b3d6dd7d406e74a8.
- The appended 867-byte showcase card identifies the estate-cards weaver, generated 2026-11-20T06:12:54Z. This is not evidence of compromised weights or a training failure.

## Read-only reproduction

From the exact GitHub source checkout, run:

    python tools/check_public_space_release_parity.py \
      --source-dir spaces/szl-model-inference-lab \
      --space-id SZLHOLDINGS/szl-model-inference-lab \
      --hf-revision e028aa7fdf3668bf739bb844a053b10285884a76 \
      --output /tmp/szl-model-lab-parity.json

This returned exit 1 (DRIFT), 25 matches, mismatches=["README.md"], unavailable=[]. Anonymous public GETs only, byte-capped, four workers; never a provider write. Invalid revision, symlink, duplicate JSON key, private/inaccessible source, HTTP 429/503, partial evidence or unknown status never becomes green. Even 26/26 establishes repository file-parity only, NOT deployed-image identity, signature validity, owner approval or runtime inference.

## Owner-controlled path — HOLD

1. Stop generic direct Space README rewrites. The overlapping estate-cards draft PRs #3 and #4 need independent review and policy convergence. A SOURCE.json-only guard is insufficient for release.json-sealed Spaces; PR #4's blanket no-direct-Space-writes policy covers both.
2. Publish the exact GitHub-owned README and release through the existing governed szl-forge publisher with approved access and maintenance window. Do not rehash contaminated HF README bytes or bypass the source-integrity check.
3. At a new immutable HF revision, independently verify 26/26 exact files, actual served-image revision and provenance, receipts, GET /readyz and one bounded real inference. All remain unverified by this source-only observation.

No secrets, provider state, model weights, training jobs or deployments were changed by the diagnostic.
