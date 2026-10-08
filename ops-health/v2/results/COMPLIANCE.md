# OAC Ops Health v2: preregistration compliance of the final evaluation

**Verdict: COMPLIANT, with the deviations disclosed below (MEASURED, SYNTHETIC).** The sealed
splits were opened exactly once. Selection used the validation split only. W1–W3 were evaluated
exactly as `v2/PREREGISTRATION.md` section 7 registers them. No deviation changes the question,
the primary metric, the win rule, the splits, the seeds, the search space or the shift regimes.

Everything here is **SYNTHETIC** and says nothing about any real transport, device, site,
population or workflow. Labels: **MEASURED** (computed in this audit, log cited), **REPORTED**
(read from a cited file or commit message), **MODELED** (reasoning without a measurement),
**UNKNOWN** (not verifiable here).

## How this was checked

- Script: `logs/verify/v2-compliance/compliance_audit.py` (standard library only, no
  `v2.research` import). Log: `logs/verify/v2-compliance/compliance_audit.txt`. Data:
  `logs/verify/v2-compliance/compliance_audit.json`.
- It reads git history, the preregistration and amendment files, `v2/data/MANIFEST.json`,
  `runs/TEST_OPENED.json`, the score manifest, `v2/results/search_trace.json`,
  `v2/results/contenders/INDEX.json`, `v2/results/final_evaluation.json`,
  `v2/results/final_robustness.json` and `logs/v2/final_opening_run.txt`. It reads **no sealed
  split**; sealed digests are compared through the manifests and the opening marker.
- Result (MEASURED): 10 of 10 checks pass. The audit ran after the opening and after the final
  evaluation, so it can only find problems, not prevent them.

## Commit order (MEASURED, check A)

Each commit is an ancestor of the next, and commit times do not decrease.

| step | commit | committed (UTC) | what |
| ---: | --- | --- | --- |
| 1 | `bca9225` | 2026-09-26 21:24:00 | Preregistration committed before any v2 code or data |
| – | `852381b`, `00e9d19`, `7c7226a` | 2026-09-26 21:58–22:07 | Legacy v1 splits, generator with v1 parity, v2 splits generated and sealed (before any training) |
| – | `974159c` … `db7a5b4` | 2026-09-29 09:59–10:46 | Stage 1: library, fail-closed kernel, bootstrap (type-7 percentiles), review hardening |
| 2 | `ed3fc88` | 2026-09-29 14:54:22 | Amendments AM-1..AM-6, before any search |
| 3 | `5ff1802` | 2026-09-29 15:10:14 | Registered search loop and freezing code, before any search |
| 4 | `92872f8` | 2026-09-29 15:17:51 | The single registered search run: 12/12 trials, T07 (l2 1e-3, none) selected |
| 5 | `bd47a2a` | 2026-09-29 15:19:29 | v2 frozen into `v2/ops-health` |
| 6 | `8c9ecb3` | 2026-09-29 15:21:56 | Section 9 ablation contenders frozen, without sealed data |
| 7 | `dff5ae5` | 2026-09-29 15:22:39 | Five AM-4 known-good retrains |
| 8 | `bb8d56f` | 2026-09-29 15:59:28 | Section 10/11 mutation suite and readiness code, before the run |
| 9 | `c46a42c`, `929677d` | 2026-09-29 16:08–16:09 | Mutation run (11 BLIND_SPOT, Λ REJECTED) and post-mutation suite |
| 10 | `f9c5588`, `cbac497`, `7a3ea7c` | 2026-09-29 20:27–21:01 | Final scorer, final analysis, full-disk write retry, all before the opening |
| 11 | `3b0912a` | 2026-09-29 21:15:20 | The opening recorded (opened 21:13:47Z at HEAD `30d50b5`) with the per-row scores |
| 12 | `d07b314` | 2026-09-29 21:40:50 | Final evaluation committed, verdict WIN |

## Checks (MEASURED unless marked)

| id | question | result | evidence |
| --- | --- | --- | --- |
| A | Commit order as above | PASS | 14 ancestor pairs, times non-decreasing, HEAD contains all |
| B | PREREGISTRATION.md frozen | PASS | One commit (`bca9225`); sha256 `14549778…` at every step, in the working tree and in the registration receipt |
| C | Amendments before any search | PASS | `v2/AMENDMENTS.md` has one commit (`ed3fc88`), strictly before `5ff1802`, `92872f8` and `3b0912a`; its sha256 `5152cd7f…` equals the value the opening preflight recorded |
| D | Sealed data untouched | PASS | `v2/data/sealed/` has one commit (`7c7226a`, before any training). The marker's five sealed digests equal `v2/data/MANIFEST.json`, whose sha256 `c0d2fa0a…` equals the marker's |
| E | Opened exactly once | PASS | `runs/TEST_OPENED.json` added by one commit (`3b0912a`, all refs), never modified, purpose `FINAL_TEST_OPENING`. The opening log shows 1 opening, 1 marker written and 1 pre-opening refusal (attempt 1 refused on the memory check before anything was opened). No other log has an opening line. Only `v2/research/final_score.py` passes the sealed purpose (`sealed_guard.py` defines it; `dataio.py` and `make_data.py` name it only in messages) |
| F | Contenders frozen | PASS | All 11 contender files in the marker equal the bytes at the opening HEAD and now; no commit touched them between the freeze and the opening or after it. The three `hub/oac-v1/` files are not tracked in git, so they were compared on disk only |
| G | Selection on validation only | PASS | The search used train (TRAIN), calibration (CALIBRATION), conformal (CONFORMAL) and validation (REGISTERED_SEARCH) only. Re-deriving the ranking from the trace with the section 6 rule and AM-2 gives the recorded ranking: T07 (validation BA 0.768680) over T08 (0.768179). All 12 IRLS fits converged; the loop stopped with SEARCH_SPACE_EXHAUSTED. The frozen v2 threshold (0.20) and calibrator (none) are T07's. ABL-conformal was kept on validation coverage (0.9006 marginal; 0.9011 and 0.8984 per class) |
| H | W1–W3 exactly as section 7 | PASS | 2,000 resamples, seed 20260927, 95% percentile intervals, paired; pooled suite = the four regimes, 12,500 rows each, resampled within regime. W1/W2 compare v2 with exactly v1, majority and rule, strictly; W3 uses point estimates and ≤ over 10 equal-width ECE bins (last closed). Re-derived verdict WIN equals the recorded one; no undefined resample in any W1/W2 interval. The independent numpy recomputation (REPORTED from `v2/results/final_robustness.json`) also gives WIN, max abs difference 5.6e-17 |
| I | Post-opening changes listed | listed | See deviations 11, 14, 15 and 16 |
| J | Frozen artifacts unchanged since the opening | PASS | No change in commits or in the working tree to `v2/PREREGISTRATION.md`, `v2/AMENDMENTS.md`, `v2/data`, `runs/TEST_OPENED.json` or `v2/results/final` |

## Deviations and disclosures

"Before the opening" means before `runs/TEST_OPENED.json` was written (2026-09-29T21:13:47Z).

| # | deviation or disclosure | section | when | before the opening | effect on W1–W3 | label |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | Amendments live in the append-only `v2/AMENDMENTS.md` instead of section 12, because the kernel and the registry pin the preregistration's sha256. Form only. | 12 | `ed3fc88` | yes, and before any search | none | MEASURED (C) |
| 2 | AM-1: the registration receipt keeps v1's upstream generator path, which contains a word banned for v2 names; it is a provenance pointer, not a v2 name. | receipt | `ed3fc88` | yes | none | REPORTED (`v2/AMENDMENTS.md`) |
| 3 | AM-2: a fit that stops with `LINE_SEARCH_FAILED` counts as non-converged and cannot be selected (section 6 names only convergence and the cap). All 12 fits converged, so the rule excluded nothing. | 6 | `ed3fc88` | yes | none | MEASURED (G) |
| 4 | AM-3: the "tampered metrics" receipt mutant is realized as an added metrics block, because the receipt carries no metrics. | 10 | `ed3fc88` | yes | none (not part of the win rule) | REPORTED |
| 5 | AM-4: a known-good retrain re-runs the selected recipe on a new nominal train split (seeds M+1..M+5). | 11 | `ed3fc88` | yes | none | REPORTED |
| 6 | AM-5: W3 compares against v1's `operator_attention_score` exactly as the Hub kernel emits it, without recalibration. This fixes what "v1 as-is" means for Brier and ECE. | 5, 7 | `ed3fc88` | yes, and before any search | defines v1's side of W3 | REPORTED |
| 7 | AM-6: v1 accepts duplicate JSON keys and v2 refuses them, so the probe suites differ by construction; `g_failclosed` uses v2's suite and v1's probe results are descriptive. v1 refused 65 of 66 probes (`unwrapped_features` escaped), so it is `INVALID_BASELINE` for per-class rates, as section 10 requires; its BA is still reported and used in W1/W2. | 10 | `ed3fc88` | yes | none | REPORTED (opening log, `FINAL_RESULTS.md`) |
| 8 | Section 7 does not say how to interpolate a percentile. The implementation uses Hyndman–Fan type 7 (linear interpolation between order statistics) and draws row indices as `int(random() * m)` from one `random.Random(20260927)` stream, turned into per-resample count vectors shared by every contender. | 7 | `974159c` (stage 1) | yes, and before any search | fixes the interval definition | MEASURED (H), REPORTED (`v2/research/bootstrap.py`) |
| 9 | Resamples on which a statistic is undefined are excluded and counted (`n_undefined`); section 7 is silent on this. For every W1/W2 interval, `n_undefined` = 0. | 7 | `974159c` | yes | none | MEASURED (H) |
| 10 | The opening needed two attempts. Attempt 1 was refused by a fail-closed pre-opening memory check before anything was opened or written; attempt 2 was the single opening. | 4, 7 | `3b0912a` log | – | none | MEASURED (E) |
| 11 | At the opening, `math/ouroboros_adapter.py` and `math/receipt_adapter.py` (loaded by `search.py`, which the scorer imports through `stage2.py`) and `math/lambda_stdlib.py` were on disk but not tracked in git. Their current bytes equal the digests recorded before the opening in `search_trace.json` and `mutation_readiness.json`; they were committed unchanged afterwards (`d9c5fb3`). | 6, 11 | `d9c5fb3` | tracked after, bytes from before | none | MEASURED (sha256 this session), REPORTED (`d9c5fb3` message) |
| 12 | The W1 comparison against v1 is decided by 0.000319 BA (v2 lower 0.759159, v1 upper 0.758840; 0.7592 vs 0.7588 at 4 decimals). The rule is met as registered. A post-hoc check that is **not registered** repeated it with 100 other resample streams: it held in all 100, margin min 0.000081, median 0.000522, max 0.000940. | 7 | after the opening | – | none; honesty note | MEASURED (`FINAL_RESULTS.md`, `v2/results/final_robustness.json`) |
| 13 | On `shift_clock_skew` and `legacy_v1_test`, the paired v2 − v1 BA interval includes zero (clock skew 0.0066 [-0.0013, 0.0146]): v2 and v1 are not distinguishable there. W2 is registered on the pooled suite only, so this is a section 8 secondary finding, stated rather than hidden. | 7, 8 | after the opening | – | none | MEASURED (`final_evaluation.json`) |
| 14 | Post-opening changes on the final evaluation's path: `v2/research/final_analyze.py` gained rendering-only additions (deciding numbers, not-distinguishable line, robustness and corrections sections, `--render-only`, which restores the analysis-time row order that the sorted JSON keys lose) and went from CRLF to LF line endings; its tests grew; two post-hoc scripts were added (`logs/v2/final_verify_independent.py`, `logs/v2/final_reanalyze_check.py`). Every analysis-path function is AST-identical to the analysis-time version (`cbac497`), and a full re-analysis reproduced every numeric field of the committed `final_evaluation.json`. | 7 | this change | – | none | MEASURED (`logs/v2/final_reanalyze_check.txt`, `v2/results/final_recheck.json`) |
| 15 | Post-opening change off the final path, by another workstream: `969f561` corrected the wording, rendering and tests of the section 10/11 mutation report. `v2/results/mutation_readiness.json` is unchanged since `c46a42c`. | 10, 11 | `969f561` | – | none (not read by the scorer or the analysis beyond a presence check) | MEASURED (`git diff c46a42c HEAD` of the JSON is empty), REPORTED (commit message) |
| 16 | Post-hoc analyses that are **not registered**: the independent recomputation and the Monte Carlo sensitivity check in `v2/results/final_robustness.json`. They are labeled post-hoc wherever they appear and cannot change the registered verdict. | 7 | this change | – | none | MEASURED |

## Section 10 and 11 outcomes (not deviations)

- The mutation suite recorded 11 BLIND_SPOT mutants, and the readiness rule REJECTED Λ
  (REPORTED from `c46a42c` and `v2/results/MUTATION_READINESS.md`). These are registered
  outcomes, reported as they came out.
- The pre-search sealing audit of the test suite found 0 opens under `v2/data/sealed/` and 0
  opens of the validation split during the in-process suite; subprocesses spawned by tests are
  not covered by that hook (REPORTED, `logs/v2/stage2_sealing_audit_pre_search.txt`). An
  independent check of the registered search refits reported PASS with no sealed split parsed
  (REPORTED, `logs/verify/v2-search/verdict.json`).

## What this audit cannot show

- That no process outside `sealed_guard` ever read the sealed files before the opening. Git
  shows they were never changed, and the guard and the audits above show the code paths refuse
  them, but a read by some other program on this machine would leave no trace here (UNKNOWN).
- That the untracked `hub/oac-v1/` files were the same at every moment before the opening. Their
  digests at the opening are in the marker and equal the files now (MEASURED); earlier states
  rely on the pinned Hub revision `dd7d1098…` (REPORTED).
