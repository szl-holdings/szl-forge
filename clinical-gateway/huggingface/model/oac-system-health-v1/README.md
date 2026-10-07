---
license: apache-2.0
tags:
  - logistic-regression
  - operations
  - observability
  - synthetic-data
  - standard-library
---

# OAC System Health v1

OAC System Health v1 is a tiny, dependency-free logistic-regression model
for **synthetic operational transport telemetry**. It emits a non-authoritative
operator-attention advisory. The artifact files reproduce byte-for-byte from
szl-forge commit
[`0e9812f989b3548ddcff7e24443da3cb5764cf50`](https://github.com/szl-holdings/szl-forge/commit/0e9812f989b3548ddcff7e24443da3cb5764cf50)
(see Provenance).

Companion data: [synthetic transport observability dataset, revision `f7ab6170`](https://huggingface.co/datasets/SZLHOLDINGS/oac-clinical-transport-observability-synthetic/tree/f7ab6170bf78138b187b8cb707d374a25ad85375).
Source at the pinned commit: [szl-forge `clinical-gateway/` at `0e9812f9`](https://github.com/szl-holdings/szl-forge/tree/0e9812f989b3548ddcff7e24443da3cb5764cf50/clinical-gateway).
That directory also hosts a separate integration stack. This model consists
only of `src/oac_operational_health.py`, `tools/train_operational_health_model.py`,
`operational-model/` and `huggingface/` in that directory, and it is bounded
by the section below.

Evidence labels used on this card: **MEASURED** (re-run and checked by SZL
Holdings on 2026-09-27 to 2026-09-29 against the files at the revisions named
here), **REPORTED** (taken from the named source), **MODELED** (follows from
reading the code), **UNKNOWN** (not established). **SYNTHETIC** marks every
number computed on generated data.

## Critical boundary

This is not a medical, diagnostic, prognostic, triage, treatment, or clinical
decision model. It was not trained or validated on patients, laboratory
results, specimens, orders, assays, physical devices, or a health-care site.
It must never receive PHI or clinical/result content.

The model cannot:

- accept, reject, or acknowledge an HL7 message;
- command or identify a medical device;
- interpret, validate, autoverify, route, or release a result;
- authorize clinical use; or
- establish regulatory, privacy, security, or site acceptance.

## Provenance

- **Artifact revision.** `model.json`, `artifact_receipt.json`,
  `oac_operational_health.py`, `example_input.json` and `LICENSE` are the files
  published at Hub revision
  [`dd7d109813abcd90c5250106dc2eabd2804e7ac3`](https://huggingface.co/SZLHOLDINGS/oac-system-health-v1/tree/dd7d109813abcd90c5250106dc2eabd2804e7ac3)
  (2026-09-24). This card revision changes `README.md` only. The kernel never
  reads `README.md` and no receipt hashes it, so `model_sha256` and every other
  receipt hash are unchanged. [MEASURED for the file hashes; MODELED from the
  kernel and trainer code for the README independence]
- **Source commit.** szl-forge `0e9812f989b3548ddcff7e24443da3cb5764cf50`
  (merge of pull request #351, 2026-09-24). How it was verified:
  1. The Hub commit `dd7d1098` names this commit in its message. [REPORTED:
     Hub commit history of this repository]
  2. Each of the six model files at Hub revision `dd7d1098` has the same
     SHA-256 as its copy under
     `clinical-gateway/huggingface/model/oac-system-health-v1/` at this
     commit. The eight dataset files at dataset revision `f7ab6170` also match,
     and the four hashes in `artifact_receipt.json` reproduce. [MEASURED]
  3. szl-forge's own `tools/verify_hub_alignment.py` (`verify_alignment()`),
     run offline against the downloaded Hub files for this commit, reports
     `complete: true`. [MEASURED]
  4. `tools/train_operational_health_model.py --verify` rebuilds the data,
     model and receipts from seed 2500 in a temporary directory and
     byte-compares them with the 22 committed paths: no mismatch. [MEASURED
     at `0e9812f9` and at its descendant `9a6fdd2c`]
- **What the pin does not prove.** The pin is unsigned
  (`source_signature_verified: false`): no signature binds these files to the
  commit. Every szl-forge commit on `main` from `eb871c9b` (2026-09-08) through
  `9a6fdd2c` reproduces the same bytes, so which commit actually ran the
  trainer is UNKNOWN. [MEASURED; UNKNOWN as stated]
- **This card's own text** comes from a later szl-forge commit, which the Hub
  commit that publishes this card names in its message. A file cannot contain
  the hash of the commit that contains it.

## Inputs

The kernel requires exactly eight operational fields:

| Field | Range | Meaning |
| --- | ---: | --- |
| `listener_running` | 0/1 | Whether the local listener process reports running |
| `tls_enabled` | 0/1 | Whether transport TLS is configured |
| `peer_allowlist_configured` | 0/1 | Whether a peer IP allowlist is configured |
| `queue_utilization` | 0-1 | Fraction of the bounded work queue in use |
| `consecutive_failures` | 0-20 | Bounded consecutive operational failures |
| `seconds_since_last_success` | 0-86400 | Bounded age of last operational success |
| `ledger_integrity_ok` | 0/1 | Whether the local operational ledger check passed |
| `configuration_valid` | 0/1 | Whether local configuration validation passed |

Binary fields are used as is. The three continuous fields are min-max scaled
with the bounds above (`feature_manifest` in `model.json`).

When every JSON key appears once, unknown, missing, non-finite, out-of-range,
identity-like, HL7, FHIR, patient, specimen, order, and result fields fail
closed. [MEASURED] **Known gap:** the v1 kernel parses duplicate JSON keys
last-wins, so an earlier duplicate (for example a first `features` object that
carries a prohibited field name) is dropped before these checks run, and the
input is accepted. Reject duplicate keys before calling the kernel until a
kernel fix ships. [MEASURED]

## Run without third-party packages

```bash
python -I -B oac_operational_health.py \
  --model model.json \
  --receipt artifact_receipt.json \
  --input example_input.json
```

The kernel verifies the model SHA-256 from `artifact_receipt.json` before
inference (`model_sha256`
`f111b7fc65db561c80763b25e538366a19bed18526ca881915777487935d9557`). Output
includes `operator_attention_score`, the validation-selected threshold, the
boolean advisory, per-feature contributions, and an explicit all-false
authority map.

The receipt is a reproducibility/mismatch control, not a signature or external
trust root. A trusted deployment must pin the Hub commit and verify it through
its own software-supply-chain policy; replacing both the model and receipt can
otherwise bypass this local comparison.

## Training and evaluation

The model is batch-gradient-descent logistic regression implemented using only
the Python standard library (2400 epochs, learning rate 0.42, L2 penalty 0.01).
Fixed seed 2500 generates 768 training, 192 validation, and 240 test examples.
The receipt records exact split metrics and hashes. Validation chooses the
decision threshold by balanced accuracy, then F1; the chosen threshold is 0.16.
The test split was not used for fitting or for threshold selection. It is a
public split that was already used during development, so it is not a new
blind evaluation.

Synthetic labels are sampled from a logistic function of the same eight
normalized inputs (`_synthetic_label_probability` in the dataset's
`training_source_snapshot.py`). The model family therefore matches the
generator by construction, and the metrics measure how well v1 recovers that
generator under label noise, not signal in real telemetry. [MODELED]

All reported metrics are from generated synthetic examples. The score is not
production-calibrated and the metrics must not be generalized to a real
transport, analyzer, laboratory, patient population, or clinical workflow.

## Synthetic test results, with intervals and baselines

Every number in this section is **SYNTHETIC**. Real-world performance,
calibration and alert burden are UNKNOWN.

The test split has 240 rows, 51 of them positive. At threshold 0.16 the confusion counts are 40 true positives, 57 false positives,
132 true negatives and 11 false negatives. Rescoring all 240 rows with this
kernel reproduces the receipt's counts exactly. [MEASURED, SYNTHETIC]

| Metric | v1 | Majority baseline | Rule baseline |
| --- | --- | --- | --- |
| Balanced accuracy | 0.7414 [0.6735, 0.8044] | 0.5000 [0.5000, 0.5000] | 0.5174 [0.4480, 0.5911] |
| Precision | 0.4124 [0.3131, 0.5104] | undefined (no alerts) | 0.2344 [0.1333, 0.3438] |
| Recall (sensitivity) | 0.7843 [0.6604, 0.8913] | 0.0000 [0.0000, 0.0000] | 0.2941 [0.1739, 0.4348] |
| Specificity | 0.6984 [0.6344, 0.7641] | 1.0000 [1.0000, 1.0000] | 0.7407 [0.6778, 0.8040] |
| F1 | 0.5405 [0.4380, 0.6329] | 0.0000 [0.0000, 0.0000] | 0.2609 [0.1538, 0.3697] |
| False-alert share, FP/(TP+FP) | 0.5876 [0.4896, 0.6869] | undefined (no alerts) | 0.7656 [0.6562, 0.8667] |
| ROC AUC | 0.8302 [0.7655, 0.8874] | 0.5000 [0.5000, 0.5000] | 0.5174 [0.4480, 0.5911] |
| Brier score (lower is better) | 0.1362 [0.1083, 0.1657] | 0.1676 [0.1373, 0.1979] | 0.3542 [0.2917, 0.4126] |
| ECE, 10 equal-width bins (lower is better) | 0.0480 [0.0288, 0.0955] | 0.0159 [0.0008, 0.0659] | 0.3542 [0.2917, 0.4126] |

Each cell is the point estimate followed by a 95% percentile bootstrap
interval: 2,000 paired resamples of the 240 test rows, drawn with Python
`random.Random(20260926)`, percentiles by linear interpolation. All three
classifiers are scored on the same resamples. [MEASURED, SYNTHETIC]

- **Majority baseline:** always predicts the training majority class (no
  attention) and scores every row with the training prevalence 151/768. It
  never alerts, so its precision and false-alert share are undefined.
- **Rule baseline:** alerts when `consecutive_failures > 0` and scores the row
  0 or 1. For a 0/1 score, the Brier score and ECE both equal the error rate.

Paired differences on the same resamples, v1 minus baseline:

| Difference | v1 minus rule | v1 minus majority |
| --- | --- | --- |
| Balanced accuracy | +0.2239 [+0.1226, +0.3270] | +0.2414 [+0.1735, +0.3044] |
| ROC AUC | +0.3127 [+0.2132, +0.4052] | +0.3302 [+0.2655, +0.3874] |
| Brier score | -0.2180 [-0.2766, -0.1589] | -0.0314 [-0.0456, -0.0184] |
| F1 | +0.2797 [+0.1396, +0.4107] | not computed |
| Recall | +0.4902 [+0.3077, +0.6558] | not computed |
| Specificity | -0.0423 [-0.1371, +0.0564] | not computed |

What this shows, on this synthetic split only:

- **Most alerts are false.** The false-alert share is 0.5876 [0.4896, 0.6869].
- v1 beats both baselines on balanced accuracy, ROC AUC and Brier score; those
  paired intervals exclude 0.
- v1's specificity is not distinguishable from the rule's.

## Known limitations

- Duplicate JSON keys bypass the fail-closed checks (see Inputs). [MEASURED]
- At inference the kernel checks only `model_sha256`. The receipt's
  `kernel_sha256`, `generator_sha256` and `dataset_receipt_sha256` are not
  checked. [MEASURED]
- A few extreme inputs (very large integer literals, very deeply nested
  arrays) are refused through an uncaught exception (exit 1 with a traceback)
  instead of the documented exit-2 JSON error. [MEASURED]
- The receipt is not a signature (see Run without third-party packages).
- The generator and the model share a functional form (see Training and
  evaluation). No evidence beyond synthetic data exists. [MODELED; UNKNOWN for
  real telemetry]

## REFORMS checklist

The 32 items of the REFORMS reporting checklist for ML-based science
(arXiv:2308.07832v2), with this card's status. **Source id** is the checklist's
own item id. The item text is SZL's paraphrase, not a quotation. Totals:
12 satisfied, 12 partial, 4 not applicable, 4 missing.

| Source id | Checklist item (paraphrase) | Status | Where, or why |
| --- | --- | --- | --- |
| 1a | State the population or distribution the scientific claim is meant to generalize to. | Satisfied | Claims are limited to the fixed-seed synthetic generator's distribution (Training and evaluation) |
| 1b | Explain why that particular population or distribution was chosen. | Partial | The synthetic-only scope is stated; no reason is given for the generator's particular distributions |
| 1c | Explain why machine learning is an appropriate tool for the study's question. | Partial | v1 beats a majority and a one-feature rule on synthetic data; no case is made that ML is needed for real telemetry |
| 2a | Share the training and evaluation data with a persistent link or identifier pinning the exact dataset. | Satisfied | Dataset at Hub revision `f7ab6170`, with per-split SHA-256 in `dataset_receipt.json` |
| 2b | Share training, evaluation and results code with a link or identifier fixing the exact code version. | Satisfied | Kernel in this repository; generator and trainer at szl-forge `0e9812f9` (Provenance) |
| 2c | Describe the hardware and software environment the computations ran on. | Partial | Python standard library only; audited with CPython 3.12.10. Hardware is not recorded |
| 2d | Provide a README explaining how to regenerate the reported results from the shared data and code. | Satisfied | Run command here; rebuild command (`--verify`) on the dataset card |
| 2e | Provide one script that regenerates every reported result end to end. | Partial | `--verify` regenerates data, model, receipts and receipt metrics; the intervals and baselines on this card come from an SZL audit script that is not published |
| 3a | Document where training and evaluation data came from, when collected, and how ground-truth labels were produced. | Satisfied | Fixed-seed generator; labels are sampled, not annotated (dataset card; Training and evaluation) |
| 3b | Describe the sampling frame, i.e. the distribution or set the dataset was drawn from. | Partial | The sampling distributions are defined only in code (`_generate_features`, `_synthetic_label_probability`) |
| 3c | Argue concretely why this dataset suits the modeling task. | Missing | No argument is made that the generator resembles real transport telemetry |
| 3d | Define the target the model predicts and summarise it statistically, per class if the target is categorical. | Partial | Outcome `operator_attention_required` is defined and positives per split are given; no per-class feature statistics |
| 3e | Report the sample size and how often each outcome occurs. | Satisfied | 768 / 192 / 240 rows; 151 / 33 / 51 positives (train / validation / test) |
| 3f | Report the share of missing values, split by class for categorical outcomes. | Not applicable | Generated rows are complete by construction and the closed schema requires all eight fields |
| 3g | Argue that the sampling frame represents the population targeted by the claim. | Not applicable | No claim is made beyond the generator's own distribution; representativeness of real telemetry is not claimed (UNKNOWN) |
| 4a | State whether any samples were excluded and give the reason. | Not applicable | No rows are excluded; the generator's output is used as is |
| 4b | Explain how impossible or corrupted samples were handled. | Partial | Generated values are in range by construction; at inference invalid inputs are refused, except for the duplicate-key gap (Inputs) |
| 4c | List every transformation from raw data to model input, such as missing-value handling and scaling, ideally diagrammed. | Satisfied | Binary fields as is; continuous fields min-max scaled with the bounds in Inputs |
| 5a | Describe every trained model in detail. | Satisfied | Logistic regression, eight weights plus intercept; algorithm and hyperparameters in Training and evaluation |
| 5b | Justify the chosen model families. | Partial | Chosen for size, standard-library execution and per-feature contributions; no other family was compared |
| 5c | Explain the evaluation procedure, e.g. how data were partitioned for training versus testing or how folds were formed. | Satisfied | Fixed train / validation / test split; threshold on validation; metrics on test |
| 5d | Describe how the reported model or models were selected among candidates. | Partial | Only the threshold is selected (balanced accuracy, then F1); no candidate models are recorded |
| 5e | Give hyperparameter tuning details for the reported model or models. | Missing | Values are given; how they were chosen is not recorded (UNKNOWN) |
| 5f | Justify that the baselines used for comparison are appropriate. | Partial | A no-skill majority baseline and a one-feature operator rule are reported; no stronger baseline |
| 6a | Show that preprocessing and modeling used only training data, never test data. | Partial | Weights fit on train, threshold on validation, scaling bounds are fixed constants; but the test split was used during development |
| 6b | Explain how dependencies or duplicates across train and test partitions were handled, e.g. grouping related samples together. | Missing | Rows are independent draws from one seeded generator; cross-split duplicates were not checked |
| 6c | Argue why every input used by the model is legitimately available and cannot reveal the label. | Satisfied | Labels are generated from these eight inputs by design, and the model reads nothing else |
| 7a | Report every performance metric and justify the one used to pick the final model. | Partial | All metrics above are reported; the threshold criterion is stated but not justified |
| 7b | Report uncertainty estimates such as confidence intervals and explain how they were computed. | Satisfied | Paired percentile bootstrap, 2,000 resamples, seed 20260926 |
| 7c | Justify any statistical tests used and check that their assumptions hold. | Not applicable | No significance tests are used; only bootstrap intervals over rows that are independent draws by construction |
| 8a | Provide evidence that the results hold beyond the study data (external validity). | Missing | No evaluation beyond synthetic data; real-world performance is UNKNOWN |
| 8b | State the settings in which the findings are not expected to hold. | Satisfied | Critical boundary; Training and evaluation |

## License

Apache-2.0. See `LICENSE`.
