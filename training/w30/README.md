# Wave 30 draft — blocked

`distill-r2.blocked.json` is a draft of the reported W30 target. It is not a
launch config. `training/launch_contract.py` rejects it.

Reported ids, not recovered pins:

- teacher `SZLHOLDINGS/SZL-Nemotron-Mini-4B-Estate-v1`
- student `Qwen/Qwen3.5-0.8B`
- data `SZLHOLDINGS/szl-frontier-sft-mix-v3`, split `train`

`distill-r2.json` is not in this repository. Commit `2e5f0d2` was not
recovered. No revision in the draft is an immutable pin, and none was invented.

Public Hub metadata checked on 2026-11-06 for
`SZLHOLDINGS/governed-distill-sft-v1` and
`SZLHOLDINGS/governed-distill-4b-sft-v1`: each repo listed
`.gitattributes` and `distill-corpus.jsonl` only. `README.md` was absent.
The public overview tags did not include a license. This draft does not treat
organization ownership as admission and does not assign a license. Row counts
and acceptance rates were not measured.

These corpora are not substitutes for the missing mix:

- `SZLHOLDINGS/szl-quant-sft-v1` (HELD_COUNSEL)
- `SZLHOLDINGS/killinchu-osint-corpus` (not admitted)
- `SZLHOLDINGS/SZLHOLDINGS` (metadata only)
- `SZLHOLDINGS/why-we-lead` (historical metadata)

Operator prerequisites before any launch: an immutable teacher revision, an
immutable student revision, an immutable dataset revision, a local train JSONL
and a disjoint holdout JSONL, and an operator admission receipt whose corpus
sha256 matches the train file. The receipt has to record a license and
`commercial_use: ADMITTED`. Thirty-two accepted train examples are required
and do not waive any other gate. PR 703 remains a separate proposal; this
draft does not start its teacher.
