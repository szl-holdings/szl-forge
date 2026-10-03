# OAC synthetic transport-health model

This directory contains the reproducible source artifacts for a small,
standard-library logistic-regression model. It estimates whether synthetic
operational transport telemetry resembles conditions that should attract an
operator's attention.

It is **not a medical model**. It does not accept patient, order, specimen,
assay, observation, diagnostic, or result fields. It is not calibrated on a
health-care site or physical analyzer. Its score cannot drive an HL7
acknowledgement, device command, result interpretation, result release, or
clinical decision.

## Reproduce and verify

From `clinical-gateway/` with Python 3.11-3.13:

```powershell
python -I -B tools/train_operational_health_model.py
python -I -B tools/train_operational_health_model.py --verify
python -I -B src/oac_operational_health.py `
  --model operational-model/artifacts/model.json `
  --receipt operational-model/artifacts/model-receipt.json `
  --input operational-model/example-input.json
```

The first command deterministically regenerates the fixed-seed dataset, model,
metrics, receipts, and Hugging Face upload staging trees. The second regenerates
them in a temporary directory and byte-compares every generated file. The
inference kernel refuses an artifact whose SHA-256 does not match its receipt.
The local receipt detects an accidental or partial mismatch; it is not a
signature, external transparency log, or trust root, so it does not prevent an
attacker from replacing both the artifact and receipt.

## Files

- `data/{train,validation,test}.jsonl`: 1,200 fixed-seed synthetic rows.
- `schema.json`: closed JSON Schema for every dataset row.
- `artifacts/model.json`: eight-feature logistic-regression weights.
- `artifacts/model-receipt.json`: model/source hashes and synthetic metrics.
- `artifacts/dataset-receipt.json`: generator/schema/data hashes and row counts.
- `example-input.json`: operational-only inference example.

The validation split selects an advisory threshold by balanced accuracy and F1.
The test split is held out from training and threshold selection, but it is
still synthetic. Metrics therefore establish deterministic implementation
behavior only; they are not evidence of production performance, clinical
validity, device compatibility, or site acceptance.

## Descriptive risk–coverage evaluation

From the repository root:

```powershell
python -I -B clinical-gateway/tools/evaluate_operational_health_model.py --fresh-seed-lanes
```

The fixed test report and each synthetic shift lane include `risk_coverage`.
Seven predeclared cutoffs describe retaining examples by normalized distance
from the **unchanged 0.16 decision boundary**. Distance is not calibrated
confidence, a correctness probability, or a trust score. No best cutoff is
selected, no training runs, and no abstention policy is installed in the kernel.

Each point preserves accepted/abstained counts, errors, confusion, class support,
and class-specific coverage. Error risk divides accepted errors by accepted
rows, never by the entire corpus. Equal-distance ties are retained together.
Comparisons use exact rationals of the exported numbers' shortest decimal
representations, preserving decimal ties without a broad numerical tolerance.
An explicit abstain-all control has zero coverage and **null risk**, not a
perfect result. Numeric cutoff 1 still retains scores at 0 or 1. Small subsets
with no observed errors retain a nonzero Wilson upper bound; pointwise intervals
are not simultaneous risk guarantees or qualification gates. Risk may increase
when coverage decreases. Public test/shift examples remain reused synthetic
evidence and must not be used to tune a subsequent policy or model release.

This is an original OAC-specific descriptive implementation of established
selective-prediction ideas, not a claim of a new method or inherited guarantees.
Prior art: [Selective classification for deep neural networks](https://arxiv.org/abs/1705.08500)
and [SelectiveNet](https://arxiv.org/abs/1901.09192).
