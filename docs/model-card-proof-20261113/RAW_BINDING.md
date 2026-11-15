# Training-curve raw-evidence binding

## Reproduced defect and repair

The original #731 CLI checked the projected curve structure and reported a newly computed projection digest, but it did not read the raw trainer-state record on each invocation. Missing and whitespace-tampered raw inputs were accepted. A separate historical comparison was not a permanent CLI invariant.

The CLI now requires the four retained, digest-named public trainer-state JSON files in `training-raw/`, or an explicit `--evidence-dir`. It verifies raw SHA-256, strict JSON, checkpoint identity, and the exact ordered loss/step projection before importing plotting dependencies or creating output. Duplicate keys, non-finite values, booleans, missing or changed records, invalid ordering, symlinks, non-regular files and oversized input are rejected. Linux opens are non-blocking so a substituted FIFO cannot stall the regular-file check.

The original 139 observations and all four reviewed input digests remain unchanged. The new raw files were independently fetched from the manifest's immutable public URLs and matched byte-for-byte. `.gitattributes` preserves their raw bytes on Windows. This change does not establish training truth, current-weight equivalence, held-out performance, independent signer identity or production readiness.

## Reproduce offline

```bash
python -B -m tools.render_model_training_curves \
  --input docs/model-card-proof-20261113/training-curves.json
python -B -m pytest -q tests/test_training_curve_raw_binding.py \
  tests/test_model_training_curves.py tests/test_model_card_evidence_passport.py
```

Plotting additionally requires the existing optional matplotlib dependency and a new `--output-dir`. Validation alone uses the standard library. There is no network client, credential acquisition, provider write, model loader or training call in this renderer.

## Scope and validation record

Three new CLI regressions failed on the unchanged renderer: missing raw evidence, altered raw bytes, and absent raw-binding attestation. The repaired source passed the affected card/publisher/regression selection on Python 3.12.10: 347 passed, nine explicitly skipped. Eight skips are the pre-existing Khipu-specific cases inapplicable to Chaski; one is a real-symlink fixture where Windows does not grant symlink creation. Linux validation is recorded separately when performed.

Earlier local setup failures are retained in the task evidence: a sparse checkout omitted the existing `khipu_r2/README.md`; Git's Windows checkout converted card line endings; and an oversized fixture generated an overlong pytest parameter ID. The original Git bytes were materialized locally and the fixture received a short test ID. No acceptance assertion was weakened.

No existing card, license, negative result, publisher workflow, closed publication registry, authority pin or model artifact changes in this repair. Review of expected manifest digests remains essential: a caller who changes both the raw data and reviewed expected digest is proposing different evidence, not obtaining independent verification. The fix extends the existing #731 source path; it does not publish a separate Hub writer or grant release approval.
