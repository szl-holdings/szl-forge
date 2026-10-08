---
license: apache-2.0
tags:
  - synthetic-data
  - operations
  - observability
  - logistic-regression
  - standard-library
---

# OAC Ops Health v2 — synthetic operational advisory

This package is a deterministic Python standard-library kernel and a JSON table
of coefficients fitted on generated transport counters. It is not a transformer
checkpoint or adapter; it is not a clinical model or result service. It accepts eight
bounded operational fields and proposes `ALERT`, `NO_ALERT`, or `ABSTAIN` for
operator attention. Every authority flag is false: it cannot acknowledge a
transport message, control a device, interpret a result, release a result, or
make a care decision. Do not submit patient information, laboratory results,
specimens, orders, or device identifiers.

## Source and files

The four executable/data files in this repository are byte copies of
[`ops-health/v2/ops-health/` at the protected szl-forge merge
`56a00821858825f529c40c7322c2f1584608d6e5`](https://github.com/szl-holdings/szl-forge/tree/56a00821858825f529c40c7322c2f1584608d6e5/ops-health/v2/ops-health).
`LICENSE` is the same Apache-2.0 license byte copy staged with OAC v1. This
card is publication documentation and is not part of the frozen receipt.

| File | SHA-256 |
| --- | --- |
| `ops_health.py` | `b0a64ff3f26ea284b0588de351ed7795a6089203b35fded0e7d82cbd4871aed9` |
| `model.json` | `b830a5edca271d667ab09b378dd5d3ab505d71a7e3451de8d9ca2bacfeed977c` |
| `artifact_receipt.json` | `442486a3b451f0ad765aac253830cdc5455bad3a34186b4cf73388cf207056f2` |
| `example_input.json` | `afa7c8e5081c682877ad489259592951f854ba2558d409fdcf071ba845d4fb63` |

The receipt records a research-origin commit
`92872f88193242b3fc0c1f604321d889d6988358` from an unpublished local
repository. The public source verifies the landed bytes and the kernel's
self-hash, but seven origin-layer tests remain unavailable to an independent
public checkout. A receipt attests to integrity and origin under its stated
boundary, not real-world accuracy or authorization.

## Evidence and limits

The preregistered comparison and single sealed test opening are recorded in
[`FINAL_RESULTS.md`](https://github.com/szl-holdings/szl-forge/blob/56a00821858825f529c40c7322c2f1584608d6e5/ops-health/v2/results/FINAL_RESULTS.md)
and [`final_evaluation.json`](https://github.com/szl-holdings/szl-forge/blob/56a00821858825f529c40c7322c2f1584608d6e5/ops-health/v2/results/final_evaluation.json).
Those results are **REPORTED for synthetic generated data only** in the source
project; they do not establish performance on real transports,
devices, sites, populations, or workflows. The public source also contains the
preregistration, amendments, provenance inventory, and bounded verification
tools. This Hub package is a proposal-only research artifact, not a clinical
release or a live service.

Run only with authored synthetic input, using the files above:

```bash
python -I -B ops_health.py --model model.json --receipt artifact_receipt.json --input example_input.json
```

The command emits an advisory for local inspection. It has no network,
device-control, result-delivery, or authorization effect.
