# KHIPU-R2 dirty evals — 2026-09-13

During the 2026-09-13 audit of the standalone-checkpoint publish, KHIPU-R2 intake was found with stale/dirty eval records: the Friday (2026-09-11) eval set did not correspond to the published artifact state, and the expected evals_dir was missing from the intake bundle.

## SUPPRESSION_2026-09-13

- Event: intake suppression, evals_dir absent
- Action: quarantined per SSEF 4.2 — re-verify provenance and completion
- Attest: 89a0b01e-cbd3-4fd1-8e8b-b3e55f96b2ca
- Classification: intake failure — separate class from the empty-merge incident. KHIPU-R2 weights themselves were a genuine merge (reproduced published eval numbers).

## Last clean gate on record

- KHIPU-R2 abstain gate: 3/6 (published as-is; abstention is expected behavior, not a failure)
