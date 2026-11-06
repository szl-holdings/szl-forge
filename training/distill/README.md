# Governed Distillation

Teacher models generate freely; a deterministic, fail-closed grounding gate
decides which outputs become student training data. Every teacher output is
receipted (accepted or rejected, with reasons). The acceptance rate is
published verbatim - a low rate is a finding about the teachers, not a failure
to hide.

This is MOPD-style multi-teacher consolidation with the SZL twist: the filter
between teacher and student is our admission math, not a human preference label.

## Pipeline
1. distill_wave.py - teachers -> generations -> grounding gate -> corpus
   (pushed as a dataset) + per-output receipts (pushed to model-eval-receipts)
2. train_wave.py - student SFT on the gate-accepted corpus

## Gate contract (deterministic, unit-tested)
ACCEPT requires: non-empty, not a repetition loop, no fabricated identifiers
(every identifier in the response must appear in the source context or be
whitelisted), and the response must either answer or explicitly abstain.
Code-fenced identifiers are not fabrication. REJECT receipts carry the exact
reasons.

## Honesty contract
- No capability claim from distillation alone; the student needs its own eval
- The gate is deliberately strict; a measured low acceptance rate is expected
  for 0.8B-scale teachers (the measured confabulation floor) and is published
  as-is
