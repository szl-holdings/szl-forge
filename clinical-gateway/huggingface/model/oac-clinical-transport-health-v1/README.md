---
license: apache-2.0
tags:
  - logistic-regression
  - operations
  - observability
  - synthetic-data
  - standard-library
---

# OAC Transport Health v1

OAC Transport Health v1 is a tiny, dependency-free logistic-regression model
for **synthetic operational transport telemetry**. It emits a non-authoritative
operator-attention advisory. The repository is staged from the canonical source
at [szl-holdings/szl-forge](https://github.com/szl-holdings/szl-forge).

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

Unknown, missing, non-finite, out-of-range, identity-like, HL7, FHIR, patient,
specimen, order, and result fields fail closed.

## Run without third-party packages

```bash
python -I -B oac_operational_health.py \
  --model model.json \
  --receipt artifact_receipt.json \
  --input example_input.json
```

The kernel verifies the model SHA-256 from `artifact_receipt.json` before
inference. Output includes `operator_attention_score`, the validation-selected
threshold, the boolean advisory, per-feature contributions, and an explicit
all-false authority map.

The receipt is a reproducibility/mismatch control, not a signature or external
trust root. A trusted deployment must pin the Hub commit and verify it through
its own software-supply-chain policy; replacing both the model and receipt can
otherwise bypass this local comparison.

## Training and evaluation

The model is batch-gradient-descent logistic regression implemented using only
the Python standard library. A fixed seed generates 768 training, 192
validation, and 240 test examples. The receipt records exact split metrics and
hashes. Validation chooses the decision threshold; the test split is otherwise
held out.

All reported metrics are from generated synthetic examples. The score is not
production-calibrated and the metrics must not be generalized to a real
transport, analyzer, laboratory, patient population, or clinical workflow.

## License

Apache-2.0. See `LICENSE`.
