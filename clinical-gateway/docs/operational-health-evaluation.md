# Operational health evaluation

This offline evaluator adds descriptive evidence for the existing OAC System
Health v1 advisory model. It does **not** train a v2 model, change coefficients,
retune the 0.16 decision threshold, ingest external observations, or qualify
clinical or production use. It needs Python 3.11-3.13 and the source checkout;
it makes no network requests and needs no GPU or paid service.

## Run from the repository root

```powershell
python -I -B clinical-gateway/tools/evaluate_operational_health_model.py --output oac-operational-evaluation.json
python -m unittest discover -s clinical-gateway/tests -p test_oac_operational_evaluation.py -v
```

Choose a new output filename each time. Existing files are never overwritten.
Success exits 0 and produces `complete: true` with status
`SYNTHETIC_EVALUATION_COMPLETE`. A validation failure exits 2 and emits a
`complete: false` JSON receipt; no input contents or paths are echoed in its
fixed error code. If the output cannot be created, the failure receipt goes to
standard error. A successful run still leaves every promotion, clinical-use,
training, weight-update, threshold-tuning, and signature-verification claim false.

## What is checked

- The exact committed model and test-data SHA-256 values are fixed in the tool.
  A changed baseline requires a separately reviewed evaluator change, not an
  implicit re-baseline.
- The model receipt binds model bytes, the inference kernel, the trainer, and
  the dataset receipt. The dataset receipt binds its generator, schema, and all
  three data splits. Every consumed source file is hashed in the output.
- All 768 training, 192 validation, and 240 test rows must satisfy the closed
  synthetic operational-only schema, expected sequence IDs, feature ranges,
  and boolean labels. Unknown fields, duplicate JSON keys, nonfinite numbers,
  oversized input, and nonregular input files fail closed.
- The existing inference kernel scores only the test split. Its exported
  rounded scores must agree with its decisions at the original threshold.
  No advisory may acquire acknowledgement, release, device-control, or clinical
  authority.
- Recomputed test counts, confusion matrix, accuracy, precision, recall,
  specificity, balanced accuracy, F1, and AUROC must agree with the published
  receipt. Rates allow an absolute tolerance of `1e-10` for the kernel's
  12-decimal score export. Published training/validation metrics and log loss
  are not independently re-evaluated by this tool.
- Source bytes are read again after scoring and must still match the original
  snapshots. This detects ordinary concurrent edits; it is not a filesystem
  sandbox or a defense against a privileged process racing and restoring files.

Each ordinary input is limited to 2 MiB, each JSONL row to 4,096 bytes, and the
optional admission manifest to 16 KiB. JSON container nesting is capped at 64
levels, independent of the Python runtime's recursion limit. The evaluator rejects symlinks and Windows
reparse points at the file itself; run it only from a trusted checkout with
trusted ancestor directories. No external corpus path or URL is accepted.

## How to read the new metrics

The unchanged baseline has 40 true positives, 132 true negatives, 57 false
positives, and 11 false negatives on its 240 public synthetic test rows.
That is a reproducibility check, not new evidence of real-world performance.

The report adds two-sided 95% Wilson intervals for supported proportions,
Brier score, a constant-score Brier baseline using **training** prevalence,
Brier skill relative to that baseline, and ten equal-width calibration bins.
Expected calibration error weights each bin's absolute mean-score/observed-rate
gap by its row count. Bins are lower-inclusive and upper-exclusive, except the
last bin includes score 1. AUROC uses average ranks for tied scores. Unsupported
rates, empty-bin rates, and one-class AUROC are `null`, never invented zeros.

Wilson intervals assume independent Bernoulli observations. They do not account
for synthetic generator bias, correlated devices, site shift, or repeated use of
these public examples. Brier and bin summaries are descriptive, not a calibration
certificate. Feature-overlap counts detect only exact canonical JSON feature
matches, not semantic, device-family, or temporal leakage.

The output also includes 240 per-example records with sample ID, source-line
hash, canonical feature hash, label, score, and decision. These records contain
only the committed public synthetic test examples. They support an auditable
recalculation and deterministic repeated output on the same source/runtime;
they are not signed attestations. Source signature verification and canonical
GitHub/Hugging Face parity are separate release checks.

## Preparing a future approved-data evaluation

An optional `--admission-manifest <file.json>` checks only a **declaration
contract**. It never opens the referenced corpus, verifies permission, or admits
data for evaluation. The manifest must contain exactly these fields:

```json
{
  "schema": "szl-oac/operational-corpus-admission/v1",
  "data_class": "nonsensitive_operational_only",
  "contains_phi": false,
  "contains_clinical_results": false,
  "authorized_for_evaluation": true,
  "corpus_sha256": "<actual lowercase 64-character SHA-256>",
  "permission_evidence_sha256": "<actual lowercase 64-character SHA-256>",
  "privacy_review_sha256": "<actual lowercase 64-character SHA-256>",
  "label_protocol_sha256": "<actual lowercase 64-character SHA-256>",
  "split_policy_sha256": "<actual lowercase 64-character SHA-256>",
  "deduplication_report_sha256": "<actual lowercase 64-character SHA-256>",
  "heldout_isolation_evidence_sha256": "<actual lowercase 64-character SHA-256>"
}
```

The angle-bracket values above deliberately fail validation: replace them only
with hashes of actual reviewed materials. Even syntactically valid hashes produce
`MANIFEST_VALID_EVIDENCE_UNVERIFIED`, `evidence_verified: false`, and
`external_corpus_admitted: false`. Hash-shaped strings alone establish nothing
about permission or privacy.

Before a real-data successor can be considered, the corpus bytes and referenced
evidence need substantive review, lawful permission, nonsensitive operational
scope, a defined labeling protocol, device/site/time-aware isolation, duplicate
checks, frozen acceptance criteria, and an untouched evaluation set. Collecting
those materials, training a successor, and running a prospective pilot are
separate work. No real patient records or clinical results belong in this path.
