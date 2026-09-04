# Hugging Face mirror gaps

Observed 2026-09-04 from GitHub source + public Hub cards. No fabricated benches.

## `SZLHOLDINGS/szl-energy-attest` — empty Hub shell

- GitHub canonical: https://github.com/szl-holdings/szl-energy-attest (complete package)
- Hub card: README + bom only. No code sync, no `TRAINING_RECEIPT.json`, no `BENCH.laptop-blackwell.json`
- Decision: **do not invent a bench**. Republish from GitHub with an owner HF token, then run the real harness on owner metal.
- Until republish, estate managers must treat the Hub repo as `MIRROR_EMPTY`, not OPERATIONAL.
- Tracker: [szl-forge#91](https://github.com/szl-holdings/szl-forge/issues/91) remains `estate:blocked-external` for the publish step.

## `SZLHOLDINGS/governed-inference-meter` — retired

- GitHub: archived. Description already points at `szl-energy-attest`.
- Decision: **Option A — retire honestly.** Do not finish receipt/bench blocks on a superseded package.
- Remaining Hub work: promote `DEPRECATED.md` to the top of the Hub README. That write is owner-token only.
- Tracker: [szl-forge#93](https://github.com/szl-holdings/szl-forge/issues/93).

## What this file is not

- Not a measured energy number
- Not a promotion certificate
- Not a substitute for `forge.ps1` publish
