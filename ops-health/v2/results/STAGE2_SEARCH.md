# OAC Ops Health v2: stage-2 search, freeze, contenders and known-good retrains

**Status.** The registered search ran once. v2 is frozen, the section 9 ablation contenders are
frozen, and the five AM-4 known-good retrains are frozen. **No sealed split has been opened**:
`runs/TEST_OPENED.json` does not exist (MEASURED, `logs/v2/stage2_tests_post_search.txt`). This
document therefore holds **no test-split or shift-suite result and no statement about v2 versus v1**.
Every number below comes from the registered **validation** split, which the protocol reserves for
selection, thresholds and gates. All of it is **SYNTHETIC** and says nothing about any real system.
Nothing was pushed, published or posted.

Labels: **MEASURED** (the command ran here, and its output is in the cited log or results file),
**REPORTED** (read from the cited file), **MODELED** (reasoning without a measurement behind it).

## Commits (local, unpublished)

| commit | content |
| --- | --- |
| `5ff1802` | `v2/research/search.py`, `v2/research/stage2.py`, `v2/tests/test_search.py`. Kernel `receipt.selection` gains 4 provenance keys; `freeze.verify_origin` checks the amendments digest. Committed **before** the search |
| `064481f` | pre-search suite run (147 OK) and sealing audit (0 sealed opens, 0 validation opens) |
| `92872f8` | the registered search: trace, 13 receipts, 12 candidate models, run and verify logs |
| `bd47a2a` | v2 frozen into `v2/ops-health/`, plus the kernel CLI log |
| `8c9ecb3` | ablation contenders and `contenders/INDEX.json` |
| `dff5ae5` | five known-good retrains and `known_good/INDEX.json` |

## Protocol state at search time (MEASURED, `v2/results/search_trace.json` `protocol`)

| item | value |
| --- | --- |
| git HEAD / tree clean | `064481f0d603416c851073f0aad1f93ab9da16c2` / true (v2/research, kernel, v2/tests, prereg, amendments, v2/data) |
| PREREGISTRATION.md sha256 | `145497789c7ad78a80a94cfd7f2b031ca9a20df49c06e0b3412af9070fa1ef4b` (registered; unchanged) |
| AMENDMENTS.md sha256 | `5152cd7f816c41e810861bbcd6489dee1d2598da89a36f6b418ef4a9867f181d` (AM-1..AM-6; the search refuses any other digest) |
| data MANIFEST sha256 | `c0d2fa0a924162e42c4d50c8eaf7d155f3e991f8f47ca518b1b87c3927ad7252` |
| generator sha256 | `a502d3644534ec599663d8678b8066b390e612396b25350f33fa987194b44384` |
| kernel sha256 | `b0a64ff3f26ea284b0588de351ed7795a6089203b35fded0e7d82cbd4871aed9` |
| math adapters (sha256) | `ouroboros_adapter.py` `67ccb2a0…c43`, `receipt_adapter.py` `db75a7e7…55ce` |
| python | 3.12.10 |

The registered splits were opened only through `sealed_guard.open_split`, each with its
registered purpose: train/TRAIN, calibration/CALIBRATION, conformal/CONFORMAL,
validation/REGISTERED_SEARCH. Their row sha256 values equal the manifest's (MEASURED,
`logs/v2/stage2_search_run.txt`).

## The search (MEASURED, `logs/v2/stage2_search_run.txt`, `v2/results/search_trace.json`)

- Loop: `math/ouroboros_adapter.run_bounded_search` with hard cap 12 over the fixed 12-trial
  space, in registered order (l2 outer, calibrator inner). It stopped with
  **`SEARCH_SPACE_EXHAUSTED`** after 12 iterations.
- Receipts: 12 trial receipts plus 1 loop-summary receipt (seq 0..12, one hash chain), in
  `runs/receipts/v2-registered-search/`, next to the adapter's `trace.json`. Checks run on them:
  - `verify_run_dir`: **0 errors**;
  - the chain: 0 errors;
  - schema, digest and envelope against the governed-receipt-spec schema file: 0 errors.

  (MEASURED, `logs/v2/stage2_receipts_verify.txt`.) The receipts attest the **integrity and
  origin** of each trial record only, never its quality. They are unsigned (the DSSE envelope
  has `signed: false`), and their `ts` values are wall-clock UTC.
- Receipts-directory digest: `7bc24c31dfb7f7c18b972718411f1799a533e97f74e90069d738549567181555`.
  Method: sha256 over sorted `"<sha256>  <name>\n"` lines covering all 14 files.
- `search_trace.json` sha256: `cc6d6809a5ae76083e42e97bb2eeca9024512fe1120da98dfa97a3f3c7d47eee`.
- AM-2: **all 12 IRLS fits and all 4 Platt fits converged**, so **no trial is ineligible**.

### All 12 trials: validation metrics at each trial's own threshold (MEASURED, SYNTHETIC)

Validation: 4,096 rows, of which 768 are positive. Column notes:

- **t**: the §6 threshold.
- **BA**: exact fraction first, then decimal.
- **ECE**: 10 equal-width bins.
- **IRLS** and **Platt**: stop reason / iterations.
- **rank**: position under the registered selection rule.

| trial | l2 | calibrator | t | BA (exact) | BA | Brier | ECE | F1 | IRLS | Platt | eligible | rank |
| --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- | ---: |
| T01_l2_0_none | 0 | none | 0.19 | 3829/4992 | 0.767027 | 0.101309 | 0.009819 | 0.587640 | CONVERGED/7 | n/a | True | 5 |
| T02_l2_0_platt | 0 | platt | 0.19 | 7649/9984 | 0.766126 | 0.101234 | 0.009375 | 0.585666 | CONVERGED/7 | CONVERGED/7 | True | 6 |
| T03_l2_0_isotonic | 0 | isotonic | 0.28 | 15289/19968 | 0.765675 | 0.101476 | 0.012656 | 0.598688 | CONVERGED/7 | n/a | True | 9 |
| T04_l2_1e-4_none | 0.0001 | none | 0.19 | 5107/6656 | 0.767278 | 0.101378 | 0.015547 | 0.588832 | CONVERGED/7 | n/a | True | 3 |
| T05_l2_1e-4_platt | 0.0001 | platt | 0.19 | 2553/3328 | 0.767127 | 0.101172 | 0.010826 | 0.588501 | CONVERGED/7 | CONVERGED/7 | True | 4 |
| T06_l2_1e-4_isotonic | 0.0001 | isotonic | 0.22 | 15289/19968 | 0.765675 | 0.101333 | 0.012843 | 0.582825 | CONVERGED/7 | n/a | True | 8 |
| **T07_l2_1e-3_none** | 0.001 | none | 0.20 | 15349/19968 | 0.768680 | 0.103556 | 0.034658 | 0.601539 | CONVERGED/7 | n/a | True | **1** |
| T08_l2_1e-3_platt | 0.001 | platt | 0.20 | 5113/6656 | 0.768179 | 0.101575 | 0.016405 | 0.601067 | CONVERGED/7 | CONVERGED/7 | True | 2 |
| T09_l2_1e-3_isotonic | 0.001 | isotonic | 0.17 | 3823/4992 | 0.765825 | 0.101545 | 0.010452 | 0.577754 | CONVERGED/7 | n/a | True | 7 |
| T10_l2_1e-2_none | 0.01 | none | 0.17 | 581/768 | 0.756510 | 0.122499 | 0.088117 | 0.544841 | CONVERGED/6 | n/a | True | 12 |
| T11_l2_1e-2_platt | 0.01 | platt | 0.15 | 15181/19968 | 0.760266 | 0.106628 | 0.033731 | 0.557179 | CONVERGED/6 | CONVERGED/7 | True | 10 |
| T12_l2_1e-2_isotonic | 0.01 | isotonic | 0.18 | 15113/19968 | 0.756861 | 0.105018 | 0.011163 | 0.543717 | CONVERGED/6 | n/a | True | 11 |

- **Selected (MEASURED): `T07_l2_1e-3_none`**: l2 = 1e-3, no calibrator, threshold 0.20. It has
  the highest validation BA, 15349/19968 = 0.768680, so the first key decided the selection. The
  only exact BA tie in the table is T03 against T06 (15289/19968). The registered Brier
  tie-break resolved it: T06 at 0.101333 ranks above T03 at 0.101476.
- **Ineligible trials: none** (AM-2).
- **ABL-calibration, as the search decided it (MEASURED).** At l2 = 1e-3, validation BA ranks
  none (0.768680) above platt (0.768179) above isotonic (0.765825), and the BA-first rule
  therefore selected `none`. Platt calibrates better on validation: Brier 0.101575 against
  0.103556, and ECE 0.016405 against 0.034658. The registered rule does not weigh calibration
  unless BA ties.
- **MODELED.** The top two trials differ by 0.0005 in validation BA. That is far inside the
  sampling noise of a 4,096-row split with 768 positives. The selection follows the registered
  rule, and it does not show that T07 generalizes better than T08. The selected model's
  validation ECE, 0.0347, is under the §10 `g_calibration` tolerance of 0.05. The gate itself
  was not evaluated here.

### ABL-conformal decision on validation (MEASURED, `search_trace.json` `conformal`)

The conformal layer is Mondrian split conformal at α = 0.10, fit on the conformal split with
n₀ = 3,346 and n₁ = 750. The frozen quantiles are q̂₀ = 0.223684755436 and
q̂₁ = 0.911152623366.

| validation (4,096 rows) | value |
| --- | ---: |
| marginal coverage | 0.900635 (rule ≥ 0.88) |
| class 0 coverage | 0.901142 (rule ≥ 0.85) |
| class 1 coverage | 0.898438 (rule ≥ 0.85) |
| abstention rate (all BOTH, 0 EMPTY) | 0.415283 (1,701 rows) |
| selective BA on 2,395 non-abstained rows | 0.839850 |
| BA without abstention (threshold decision, all rows) | 0.768680 |

**Decision: KEEP the conformal layer.** All three coverage conditions hold. The decision is
recorded as `conformal_keep: true` in every stage-2 receipt.

## v2 frozen (MEASURED, `logs/v2/stage2_freeze_v2.txt`, `logs/v2/stage2_kernel_cli.txt`)

| file | sha256 |
| --- | --- |
| `v2/ops-health/model.json` | `b830a5edca271d667ab09b378dd5d3ab505d71a7e3451de8d9ca2bacfeed977c` |
| `v2/ops-health/artifact_receipt.json` | `442486a3b451f0ad765aac253830cdc5455bad3a34186b4cf73388cf207056f2` |
| `v2/ops-health/example_input.json` | `afa7c8e5081c682877ad489259592951f854ba2558d409fdcf071ba845d4fb63` |

- **Where `model.json` came from.** It is the searched T07 candidate, byte-identical to the
  receipted `model_sha256`. It is also byte-identical to a fresh rebuild from the registered
  splits at HEAD.
- **Freeze conditions.** Frozen by `freeze.freeze` at source commit `92872f8`, with a clean tree.
- **What `receipt.selection` holds:**
  - `selected_trial` `T07_l2_1e-3_none`, `trials` 12, `stop_reason` `SEARCH_SPACE_EXHAUSTED`;
  - `search_trace_sha256` `cc6d6809…7eee`;
  - `receipts_dir_sha256` `7bc24c31…1555`;
  - `amendments_sha256` `5152cd7f…181d`;
  - `conformal_keep` true.
- **`freeze.verify_origin`.** It checks the kernel, generator, manifest, preregistration and
  amendments blobs at `92872f8` against the receipt's digests. It passes.
- **Kernel CLI** (`py -3.12 -I -B v2/ops-health/ops_health.py …`):
  - `example_input.json` gives `NO_ALERT` with score 0.069311, exit 0;
  - a degraded input gives `ALERT` with score 0.994705, exit 0;
  - a probe with a prohibited key and a dummy numeric value is refused with exit 2.

## Ablation contenders (MEASURED; `v2/results/contenders/INDEX.json` sha256 `086dbca90c69b9a2026bdedeb9fa9657f51af036884d79a839707c64a54b5718`)

All contenders were fit and selected without sealed data. Their splits were opened with purpose
ABLATION. Validation metrics are at each contender's own threshold, on the registered validation
split.

| contender | status | t | BA | Brier | ECE | F1 | files (sha256) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| v2 (T07, reference) | FROZEN | 0.20 | 0.768680 | 0.103556 | 0.034658 | 0.601539 | `v2/ops-health/model.json` `b830a5ed…977c` |
| ABL-data | FROZEN | 0.16 | 0.761418 | 0.107061 | 0.039262 | 0.577951 | `contenders/ABL-data/model.json` `0cbe0e4a4d76184c6a53945a7ddfe8ab1ec16318e959296caafd5a0415d5d598`, `artifact_receipt.json` `876bde74b7289e0d7b0edcd4f720c4d1766777c857d14356a88bd3d43c67fa23` |
| ABL-optimizer | FROZEN (params file) | 0.17 | 0.759265 | 0.122883 | 0.089898 | 0.547906 | `contenders/ABL-optimizer/params.json` `3a89941b066db919b10b608b3537ce4044872b2314761a732b7e828a975e4a1f` |
| ABL-calibration | NOT_APPLICABLE | – | – | – | – | – | – |
| ABL-conformal | DECIDED_ON_VALIDATION: keep | – | – | – | – | – | (search trace) |

- **ABL-data.** The selected recipe on train rows 0..767. The 768-row subset has sha256
  `087260fa…b244`. IRLS converged in 7 iterations. The artifact passes `verify_origin`.
- **ABL-optimizer.**
  - Training: `v1_gd.train_v1_gd` (v1's exact GD: 2,400 epochs, lr 0.42, l2 0.01, zero start)
    on the full 16,384-row train split.
  - Threshold: chosen by v1's own rule on validation, over t = 0.05..0.95. The rule maximizes
    rounded BA, then rounded F1, then closeness to 0.5. The result is t = 0.17, and 12-decimal
    parameters give the same threshold.
  - Why it is a params file: the v2 kernel pins the IRLS training algorithm, so a GD model cannot
    be a truthful v2 artifact. It is frozen as a documented params file and bound by sha256.
- **ABL-calibration.** NOT_APPLICABLE, because the selected calibrator is `none` and there is no
  calibrator to remove. The search itself compared none, platt and isotonic at every l2, as
  section 9 anticipates.
- **Decisions on validation (MEASURED numbers, MODELED reading).**
  - Full data (BA 0.768680) against 768 rows (0.761418): full data kept.
  - IRLS with the selected l2 (0.768680) against v1 GD (0.759265): IRLS kept.
  - The v2 choice also has the lower Brier and the lower ECE in both pairs.
  - These are validation comparisons only. The test values come at the single final opening.

## Known-good retrains, AM-4 (MEASURED, `logs/v2/stage2_known_good.txt`; `v2/results/known_good/INDEX.json` sha256 `1b5f0c6e7091edbe1127e800e7493cbe72061f8b4b42ed811bb00c4f7496df9c`)

Each retrain re-runs the selected recipe end to end:

- IRLS with l2 = 1e-3;
- calibrator none;
- threshold re-selected on validation (opened with purpose GATES);
- Mondrian conformal on the conformal split.

Only the train split changes. Each train split is 16,384 nominal rows from seed material
`oac-ops-health-v2:{20260926+k}:train`, using the §3 derivation. These splits are not sealed.

| k | seed material | train sha256 | model sha256 | receipt sha256 | t | IRLS | AM-2 converged |
| --- | --- | --- | --- | --- | ---: | --- | --- |
| 1 | `oac-ops-health-v2:20260927:train` | `68860affab3b74d8e19cbb4a88cb5ea3a862f3a7e5342ce05581f88e95134d4e` | `58cbfe92905075607ed1f5677f5792cd7297054076b3058a2080ebadd896f273` | `d0fbc3d954b1b49be7f0c47a6c7368d26e5af967efdea3b8c82fcf064cb58b8d` | 0.19 | CONVERGED/7 | True |
| 2 | `oac-ops-health-v2:20260928:train` | `1be3d6898edac4ab412b5c2675543006a79e83132aae3382ba7b7b78105913df` | `504c8a1d03240d7ff04f7f850a7f4c29c1f4120ebb6c9e49054ae36694109147` | `d36a9318a8bebdead47bd517801ca22041bfd36992ef2e5ea077bfd825c24079` | 0.18 | CONVERGED/7 | True |
| 3 | `oac-ops-health-v2:20260929:train` | `a21ddeff822a6be1e34a071678b68db84d6eb27d06dc5dfb23a1b641472c5725` | `49a56b3612bdaa22bb2bd63f9fc92ea8d96078b9c0ce3f601db96672199d884a` | `06288cbb2a8f765d1c2c0068a7c9dc0d158a716801c66c07816218e756d2aa33` | 0.19 | CONVERGED/7 | True |
| 4 | `oac-ops-health-v2:20260930:train` | `49630c78c510f31d8838eef57970990150d257bd9f9cf431cee0e50c0dfa95d8` | `7734a817f202017189027dd1b689dc72b4ab81167a226a20250d6c77e999e6d3` | `96060e7388b4a7848736e30548617dd0ad9d7f2587ec4697e935e3442a08ad64` | 0.19 | CONVERGED/7 | True |
| 5 | `oac-ops-health-v2:20260931:train` | `33744d8e6cae2bff7262b3f2a5b963280febd5fec5086a84e84178e75883b477` | `7773d9f4f2e360738a1f96915101dcb086fa8f256857936398702146468e7e0c` | `3204c9f1ff02d6090a1ae04fb8e8897eaf862498f7295966d8d9c4d82b833967` | 0.20 | CONVERGED/7 | True |

- Each retrain passes `verify_origin` at source commit `8c9ecb3`, and the kernel loads each
  model/receipt pair.
- As a check on the derivation, k = 0 regenerates the registered train split byte for byte
  (sha256 `5211e7ec…0adb`). The five retrain splits are distinct from each other and from the
  registered split.
- The hashes were recorded before any section 10 or 11 gate was evaluated. No gate was evaluated
  in this workstream.

## Tests and sealing (MEASURED)

- **Before the search** (`logs/v2/stage2_tests_pre_search.txt`, clean tree at `5ff1802`): **147
  tests OK**, made up of the stage-1 119 plus 28 in `test_search.py`, which use DEV data only.
  The sealing audit (`logs/v2/stage2_sealing_audit_pre_search.txt`) recorded 0 opens under
  `v2/data/sealed/` and 0 opens of the registered validation split during the suite.
- **After all stage-2 steps** (`logs/v2/stage2_tests_post_search.txt`, clean tree at `dff5ae5`):
  **147 tests OK**. `make_data --verify` reports ok, and PREREGISTRATION.md and AMENDMENTS.md
  hash unchanged. The first post-search attempt hit one `MemoryError`: it failed to allocate a
  1 MB test string while the host had about 0.5 GB of free physical memory out of 33 GB. That
  log is kept at `logs/v2/stage2_tests_post_search_memoryerror.txt`. A re-run a minute later
  passed. Its log also contains a failed shell memory probe line, kept as is.

## Disclosures

1. **Kernel change before the search (MEASURED, `5ff1802`).** `receipt.selection` gained four
   required keys: `search_trace_sha256`, `receipts_dir_sha256` and `amendments_sha256`, each
   null or 64-hex, and `conformal_keep`, which is true, false or null. The kernel checks their
   form only. `freeze.verify_origin` checks `amendments_sha256` against the committed blob. This
   changed the kernel sha256 after the stage-1 review. The frozen receipt binds the new one,
   `b0a64ff3…aed9`. Scoring arithmetic did not change.
2. **Untracked adapters (REPORTED).** `math/ouroboros_adapter.py` and `math/receipt_adapter.py`
   are not tracked by git here. Their sha256 values are recorded in the trace, but git cannot
   reproduce their bytes.
3. **Per-trial wall times** (1.4–75 s) reflect shared host load (MEASURED in the run log). They
   are not a property of the trials.
4. `v2/README_DEV.md` still says "Stage 2: pending". This workstream did not edit it.
5. **No amendment requests** were filed by this workstream.

## What this does not show

- Nothing here compares v2 with v1, the majority baseline or the rule baseline on the test split
  or the shift suite. W1–W3 are **UNKNOWN** until the single `FINAL_TEST_OPENING`.
- v2 is not described as better, and it must not be until W1–W3 all hold.
