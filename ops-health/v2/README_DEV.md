# OAC Ops Health v2: developer notes (stage 1)

Everything under `v2/` is SYNTHETIC. The protocol is `v2/PREREGISTRATION.md`, which is frozen
(sha256 `145497789c7ad78a80a94cfd7f2b031ca9a20df49c06e0b3412af9070fa1ef4b`). Never edit it: file
problems in `v2/AMENDMENT_REQUESTS.md`. For the stage-1 status and its evidence, see
`v2/STAGE1_REPORT.md`.

## Layout

```
v2/
  PREREGISTRATION.md, PREREGISTRATION.receipt.json   frozen protocol + registration record (bca9225)
  AMENDMENT_REQUESTS.md                              open requests AR-1..AR-3 (Stephen decides)
  STAGE1_REPORT.md, README_DEV.md                    this stage's report and these notes
  ops-health/ops_health.py      the kernel: stdlib only, fail-closed, advisory-only CLI; self-hash
                                receipt binding; ALERT / NO_ALERT / ABSTAIN (Mondrian conformal)
  research/                     stdlib-only research library (imported as v2.research.*)
    registry.py                 fixed paths, split table, registered digests
    generator.py                ops-health-generator/2.0.0 (DO NOT MODIFY; bound by MANIFEST)
    make_data.py                one-time split writer / hash-only verifier (DO NOT MODIFY)
    legacy_v1_source.py         provenance record of the v1 public splits
    dataio.py                   row loading; refuses any path under a 'sealed' directory
    sealed_guard.py             the only gate to registered splits; single FINAL_TEST_OPENING
    kernel_bridge.py            loads the kernel file as a module (single arithmetic source)
    lr.py                       Newton/IRLS L2 logistic regression (deterministic order)
    calibration.py              none / Platt / isotonic (exact PAV, ties pooled)
    conformal.py                Mondrian split conformal, exact rank, q_hat frozen upward
    pipeline.py                 build ONE candidate from given rows + hyperparameters (no search)
    freeze.py                   canonical model.json + receipt; clean-tree rule; verify_origin
    metrics.py                  BA, Brier, ECE, AUROC, coverage, INVALID_BASELINE gating
    bootstrap.py                paired percentile bootstrap (count vectors, strata)
    gates.py                    fail-closed probe suite, metamorphic grid, behavioral gates
    truth.py                    generator-truth weights and recovery measures
    v1_gd.py                    v1's exact GD trainer (ABL-optimizer) + 12-dp reproduction
    bench_bootstrap.py          timing benchmark on 50,000 DEV rows
  tests/                        unittest suite (DEV data only; _support.py holds the DEV layout)
  data/                         MANIFEST.json, train/calibration/conformal/validation.jsonl,
                                legacy_v1/, sealed/ (test + 4 shift splits; DO NOT OPEN)
logs/v2/                        raw outputs cited by the reports; logs/v2/review/ = stage-1 review
```

## Rules that the code enforces

- **Sealed splits.** Files under `v2/data/sealed/` are opened only by
  `sealed_guard.open_split(<sealed>, "FINAL_TEST_OPENING", contenders=...)`. The opening happens
  once and writes `runs/TEST_OPENED.json`. `make_data.py --verify` only hashes their bytes.
- **Registered validation split.** It is opened only through `sealed_guard.open_split("validation",
  purpose)`, with purpose `REGISTERED_SEARCH`, `ABLATION` or `GATES`. Stage 1 never opened it.
- **Freezing.** `freeze.freeze` refuses to run unless
  `git status --porcelain -- v2/research v2/ops-health/ops_health.py` is empty. It records HEAD
  as `source.commit` and never overwrites its outputs.
- **Kernel refusals.** The kernel refuses any receipt whose preregistration digest is not the
  registered one. It also refuses any artifact with alpha ≠ 0.10 or with l2 off
  {0, 1e-4, 1e-3, 1e-2}.

## Commands (from the repository root; Git Bash)

```bash
cd /c/Users/steph/szl-work/oac-frontier

# full suite (Python 3.12.10); expected: "Ran 119 tests ... OK"
PYTHONUTF8=1 py -3.12 -B -m unittest discover -s v2/tests -t .
# CONTRACT section 6 form (Python 3.11.9)
PYTHONUTF8=1 python -B -m unittest discover -s v2/tests

# kernel CLI (advisory only; exit 0 = advisory on stdout, exit 2 = refusal on stderr as one
# ASCII JSON line of at most 1,024 bytes; values are never echoed)
py -3.12 -I -B v2/ops-health/ops_health.py --model <dir>/model.json \
    --receipt <dir>/artifact_receipt.json --input <input.json | ->

# v1 GD reproduction (12 decimals vs the published v1 model; writes nothing)
PYTHONUTF8=1 py -3.12 -B -m v2.research.v1_gd --reproduce

# bootstrap timing benchmark (DEV rows; the intervals are NOT results)
PYTHONUTF8=1 py -3.12 -B -m v2.research.bench_bootstrap

# data reproducibility, hash-only (regenerates in memory, hashes the files on disk)
PYTHONUTF8=1 py -3.12 -B -m v2.research.make_data --verify

# stage-1 review harnesses (their outputs live in logs/v2/review/)
PYTHONUTF8=1 py -3.12 -B logs/v2/review/kernel_attack.py
PYTHONUTF8=1 python -B logs/v2/review/stats_reference.py     # needs numpy + scipy (3.11)
PYTHONUTF8=1 py -3.12 -B logs/v2/review/sealing_audit.py
```

Never pass `--write` to `make_data` or `legacy_v1_source`: the data is already written and sealed,
and both refuse to overwrite.

## Stage 1: done (details and evidence in `v2/STAGE1_REPORT.md`)

- Generator 2.0.0, with byte-exact parity against the v1 public splits (M1, M2). The splits were
  generated, hashed and sealed before any training (M3).
- The kernel, the research library and the frozen-resolution candidate builder. Determinism: the
  same seed gives a byte-identical `model.json`.
- A 119-test suite: fail-closed probes, receipt and self-hash integrity, the broken baselines
  (constant-ALERT and fail-open kernel give INVALID_BASELINE), metric recomputation against v1's
  receipt, calibration and conformal, IRLS, gates, the sealed guard, and the review regressions.
- The adversarial review (`logs/v2/review/STAGE1_REVIEW.md`), with fixes in `b75785c`, and a
  fixer round in `efde7ff` that bounds every refusal echo (completes A5).

## Stage 2: pending (nothing below has been run)

1. **Registered search** (section 6). Run exactly 12 trials: l2 ∈ {0, 1e-4, 1e-3, 1e-2} ×
   calibrator ∈ {none, platt, isotonic}.
   - Use `pipeline.build_candidate`, with the registered train, calibration and conformal splits
     and the validation split (as the threshold rows), all opened via `sealed_guard`.
   - Run the bounded loop following `math/ouroboros_adapter.py`: cap 12, stop reasons
     `SEARCH_SPACE_EXHAUSTED` or `CAP_REACHED`.
   - Write one governed receipt per trial to `runs/receipts/`.
   - Selection is by validation BA, with the registered tie rules.
   - Report any `CAP_REACHED` or `LINE_SEARCH_FAILED` fit (AR-2).
2. **Freeze the selected v2.** Commit first, then call `freeze.freeze`, then
   `freeze.verify_origin`.
3. **v1 as-is contender.** Score through the Hub kernel file (`hub/oac-v1/oac_operational_health.py`,
   revision `dd7d1098…`, threshold 0.16). Test its parity with `v1_gd.v1_committed_score` before
   using it.
4. **Retrains.** Train five retrains of the selected recipe on train splits from seeds M+1..M+5
   (the known-good fixtures, section 11).
5. **Mutation suite and aggregation** (section 10).
   - Run the full mutation suite on the frozen artifact, including the integrity-bypassed second
     pass. The receipt-corruption mutants must go through `freeze.verify_origin`. "Tampered
     metrics" is realized as AR-3 describes.
   - Compute the behavioral gates, then run Λ versus AND, min and mean (section 11,
     `math/lambda_stdlib.py`).
6. **Ablations** (section 9): ABL-data, ABL-optimizer (`v1_gd.train_v1_gd` on the full v2 train
   split), ABL-calibration and ABL-conformal (validation coverage ≥ 0.88 marginal and ≥ 0.85 per
   class).
7. **The single final opening.** Call `sealed_guard.open_split("sealed", "FINAL_TEST_OPENING",
   contenders=...)`. Then compute:
   - primary BA with the paired bootstrap (2,000 resamples, seed 20260927), stratified by regime
     for the pooled shift suite;
   - W1–W3;
   - the section 8 secondaries. Baseline per-class rates go only through
     `metrics.binary_report`.
8. **REPORT.md.** Say plainly whether v2 wins, and disclose AR-1..AR-3 as decided.
