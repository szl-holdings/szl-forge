# KHIPU-R2 salvage plan

1. Rebuild evals_dir from the held-out gate fixtures (do not regenerate from the training split).
2. Re-run the abstain gate on the exact published artifact bytes; record the score.
3. Re-verify the provenance chain: adapter SHA, merge receipt, published checkpoint SHA.
4. Re-attest and clear the SUPPRESSION_2026-09-13 quarantine per SSEF 4.2.
5. Update the model card eval section with re-run numbers only if they differ from the published 3/6; if they differ, annotate the delta and cause.

**Acceptance:** evals_dir present, gate re-run on published bytes, provenance verified, new attestation recorded. No retraining. No republish unless numbers change.
