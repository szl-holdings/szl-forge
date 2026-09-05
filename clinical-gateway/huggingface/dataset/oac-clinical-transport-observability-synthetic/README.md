---
license: apache-2.0
task_categories:
  - tabular-classification
tags:
  - operations
  - observability
  - synthetic
  - transport-health
size_categories:
  - 1K<n<10K
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train.jsonl
      - split: validation
        path: data/validation.jsonl
      - split: test
        path: data/test.jsonl
---

# OAC Clinical Transport Observability — Synthetic

This dataset contains 1,200 fixed-seed, **entirely synthetic operational
transport-health examples** for the companion OAC Transport Health v1 model.
It contains no records collected from a patient, laboratory, analyzer,
instrument, LIS, EHR, network, or health-care site.

Companion model: [OAC Transport Health v1](https://huggingface.co/SZLHOLDINGS/oac-clinical-transport-health-v1).
Canonical source: [szl-forge clinical gateway](https://github.com/szl-holdings/szl-forge/tree/main/clinical-gateway).

## Data boundary

The closed schema contains only eight bounded operational counters/flags and a
synthetic `operator_attention_required` label. It contains no PHI, personal
identifiers, patient/order/specimen fields, assay data, observations, diagnostic
content, result values, raw HL7, or FHIR resources.

This dataset is unsuitable for medicine, diagnosis, prognosis, treatment,
triage, result interpretation, autoverification, result release, device control,
or claims about real-world performance. It must not be joined with patient or
clinical data.

## Splits

| Split | Rows | Role |
| --- | ---: | --- |
| train | 768 | Fit logistic-regression weights |
| validation | 192 | Select the advisory threshold |
| test | 240 | Synthetic held-out implementation check |

Each JSONL row has this shape:

```json
{
  "features": {
    "configuration_valid": 1,
    "consecutive_failures": 0,
    "ledger_integrity_ok": 1,
    "listener_running": 1,
    "peer_allowlist_configured": 1,
    "queue_utilization": 0.12,
    "seconds_since_last_success": 14.0,
    "tls_enabled": 1
  },
  "label": {"operator_attention_required": false},
  "sample_id": "train-000000",
  "schema": "szl-oac/transport-health-observation/v1",
  "synthetic": true
}
```

`schema.json` is a closed JSON Schema. `dataset_receipt.json` records the fixed
seed, row counts, generator hash, schema hash, and SHA-256 for every split.

## Reproduction

`training_source_snapshot.py` is the exact generator/trainer source snapshot
used for the published artifacts. It is included for inspection and hashing;
its canonical source-tree layout also requires `src/oac_operational_health.py`.
Clone the source repository and run:

```bash
git clone https://github.com/szl-holdings/szl-forge.git
cd szl-forge/clinical-gateway
python -I -B tools/train_operational_health_model.py --verify
```

The generator uses Python's standard library only. Synthetic labels are sampled
from a documented operational risk function under a fixed pseudorandom seed;
they are generated targets, not human annotations or ground truth about a real
system.

## License

Apache-2.0. See `LICENSE`.
