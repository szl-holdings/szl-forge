# OAC Ops Health v2: final evaluation on the sealed splits

## Verdict: v2 **WIN** (MEASURED, SYNTHETIC)

All three registered conditions hold. On this synthetic benchmark only, v2's balanced accuracy interval lies strictly above v1's, the majority baseline's and the rule baseline's, on nominal test and on the pooled shift suite, and its Brier score and ECE on nominal test are no worse than v1's. This says nothing about any real system.

- **W1 (nominal test) HOLDS.** v2 BA 0.7644 [0.7592, 0.7695]; v2 lower bound against each baseline's upper bound: v1: 0.7592 vs upper 0.7588 (holds); majority: 0.7592 vs upper 0.5000 (holds); rule: 0.7592 vs upper 0.5370 (holds).
- **W2 (pooled shift suite) HOLDS.** v2 BA 0.7581 [0.7542, 0.7618]; v2 lower bound against each baseline's upper bound: v1: 0.7542 vs upper 0.7087 (holds); majority: 0.7542 vs upper 0.5000 (holds); rule: 0.7542 vs upper 0.5483 (holds).
- **W3 (calibration, nominal test) HOLDS.** Brier v2 0.101330 vs v1 0.115765 (<=); ECE v2 0.030599 vs v1 0.084775 (<=).

Deciding numbers (MEASURED): the narrowest interval comparison is W1 against v1: v2 lower bound 0.759159 minus v1 upper bound 0.758840 = 0.000319 BA (holds, strict). At the 4 decimals shown above the two bounds read 0.7592 and 0.7588, a gap of 0.0004. A change of 0.000319 BA in either bound would reverse this comparison (MODELED). The paired v2 − v1 BA difference on nominal test is 0.010380 [0.006559, 0.014191] (section 8 secondary; not part of the win rule).

Not distinguishable (MEASURED, section 8 secondary): in balanced accuracy, v2 and v1 cannot be told apart on shift_clock_skew (paired v2 − v1 BA 0.0066 [-0.0013, 0.0146]; v2 0.7564 [0.7455, 0.7665], v1 0.7498 [0.7399, 0.7595]); legacy_v1_test (paired v2 − v1 BA 0.0251 [-0.0237, 0.0708]; v2 0.7664 [0.6970, 0.8371], v1 0.7414 [0.6728, 0.8047]). On each of these sets the paired v2 − v1 interval includes zero. W2 is registered on the pooled shift suite only, so this does not change the verdict, and no BA advantage of v2 over v1 is claimed on these sets.

Everything here is **SYNTHETIC** and says nothing about any real transport, device, site, population or workflow. Labels: **MEASURED** (computed here from the saved score files, cited below), **REPORTED** (read from a cited file), **MODELED** (reasoning without a measurement). Unlabeled numbers in tables are MEASURED.

## Robustness of the verdict (post-hoc, NOT registered)

Written after the opening. It cannot change the registered verdict above, which is defined by the single registered resample stream (seed 20260927).

- **Independent recomputation (MEASURED).** `logs/v2/final_verify_independent.py` (numpy 2.4.6, Python 3.11.9, no v2.research import) recomputed the W1–W3 point estimates (BA of every contender on nominal test and on the pooled shift suite; Brier and ECE of v2 and v1) and re-implemented the registered bootstrap stream from its written definition (the W1/W2 intervals and the paired v2 − v1 intervals). The largest absolute difference from `final_evaluation.json` is 5.6e-17 (tolerance 1e-12), and its verdict is **WIN**.
- **Monte Carlo sensitivity of W1 against v1 (MEASURED, not registered).** The same rule (2,000 resamples, type-7 percentiles) was repeated on nominal test with 100 other resample streams (numpy.random.default_rng(seed) PCG64, seeds 1..100). v2's lower bound exceeded v1's upper bound in 100 of 100 streams. Margin: min 0.000081, median 0.000522, max 0.000940 (registered stream 0.000319).
- **Reading (MODELED).** The deciding comparison (W1 against v1) sits 0.000319 BA from the registered non-overlap threshold; it held in all 100 other resample streams, so on this check its sign did not depend on the resample stream, although the margin stays small. The next-narrowest W1/W2 comparison has margin 0.045445 (W2 against v1). W3's gaps (v1 − v2) are Brier 0.014435 and ECE 0.054176. The rule requires two separate marginal intervals not to overlap, which is stricter than the paired difference interval excluding zero.

Source: `v2/results/final_robustness.json`, computed from score manifest `a8dd49bfb74b44aa…` and `final_evaluation.json` sha256 `36b494bd043ae097…`; log `logs/v2/final_verify_independent.txt`.

## Post-opening corrections

Every change made after the single opening to the code or tracked files on the final evaluation's path. None of them changes a number or the verdict; the proof follows the table.

| # | change | reference | reason | effect on the numbers |
| ---: | --- | --- | --- | --- |
| 1 | Tracked `math/lambda_stdlib.py`, `math/ouroboros_adapter.py`, `math/receipt_adapter.py` and `math/lambda_reference.md` in git, bytes unchanged. | d9c5fb3 | Committed v2 code loads them by path, so a clean checkout could not run those tests. | None on any number. Each file's sha256 equals the digest recorded before the opening (`v2/results/mutation_readiness.json`, `v2/results/search_trace.json`). |
| 2 | `v2/research/final_analyze.py`, rendering only: the deciding-numbers line (6-decimal W1/W2 margin, 4-decimal gap, paired v2 − v1 BA interval), the not-distinguishable line, the post-hoc robustness section, this section, and the `--render-only` mode (which restores the analysis-time row order that the sorted JSON keys lose). | this change | Make the thin W1 margin against v1 and the sets where v2 and v1 cannot be told apart explicit, and re-render the page without recomputing anything. | None on any number. `analyze()` and every function it calls are unchanged; the diff touches rendering, loading of the two post-hoc JSON files and argument parsing. The full re-analysis below reproduced every numeric field. |
| 3 | `v2/research/final_analyze.py` line endings CRLF → LF. | this change | Repository rule: committed files use LF. | None on any number. The script's sha256 is no longer the analysis-time `a5325246…` (commit cbac497), which is the version that produced `final_evaluation.json`. |
| 4 | `v2/tests/test_final_analyze.py`: tests for the rendering additions and `--render-only` (DEV data only). | this change | Cover the new code. | None on any number. |
| 5 | `logs/v2/final_verify_independent.py` (new, post-hoc): independent numpy recomputation (Parts A, B) and the Monte Carlo sensitivity check (Part C). Runs 1 and 2 stopped on MemoryError with the host disk nearly full, so Part C was made memory-light, checkpointed per stream, cut to 100 streams and given a per-stream MemoryError retry. Run 3 reused the 15 streams checkpointed by run 2. | this change | Check the verdict with code that shares nothing with `v2.research`; survive the host's memory shortage. | None on the registered analysis. It reads only the score files and `final_evaluation.json` and writes `v2/results/final_robustness.json`. A stream is a pure function of its seed, so a retried or reused stream gives the same margin. |
| 6 | `logs/v2/final_reanalyze_check.py` (new): re-runs the registered analysis from the saved score files into `scratch/` and compares every field with the committed `final_evaluation.json`. | this change | Prove that the changes above left every number unchanged. | None on any number. It writes `v2/results/final_recheck.json` and its log. |

- **Proof that no number changed (MEASURED).** `logs/v2/final_reanalyze_check.py` re-ran the registered analysis (`analyze()`, 2,000 resamples, seed 20260927, no saved parts reused) with `v2/research/final_analyze.py` sha256 `afe518d439309816…`, Python 3.12.10, from score manifest `a8dd49bfb74b44aa…`, and compared every field with the committed `final_evaluation.json` (sha256 `36b494bd043ae097…`). Numeric fields compared: 3884; identical: 3884; largest absolute difference: 0.0e+00. Verdict WIN against the committed WIN (same). Wall-clock `seconds` fields excluded from the comparison: 7. Other fields that differ, all run metadata: `generated_utc`, `inputs.analysis_script.git_head`, `inputs.analysis_script.sha256`. Unexpected differences: 0.
- Log: `logs/v2/final_reanalyze_check.txt`; data: `v2/results/final_recheck.json`.

## Provenance (MEASURED)

- Opening: `TEST_OPENED.json` sha256 `273c43b6afa86bc5f82fcbe99c3f7815fd76115eb097c2f2d679adc3ffae849a`, opened 2026-09-29T21:13:47Z at git HEAD `30d50b5b4537e4262489c160351feaaed08e33aa`, purpose `FINAL_TEST_OPENING`. The opening happened once, inside `v2/research/final_score.py`.
- Scorer: `v2/research/final_score.py` sha256 `8372f8818defc1224d1906529f7a6c806b0367a53c6b5ce25d3969857476636d` at `30d50b5b4537e4262489c160351feaaed08e33aa`, Python 3.12.10. Score manifest sha256 `a8dd49bfb74b44aa8d038d0f583c1eb65d36cf1577ee1d8bf3215340f5131299`.
- Analysis: `v2/research/final_analyze.py` sha256 `a53252460d8210ca5bf9227868e27298e6300627231d01d7108e0a0a9d26bedf`, reading only the score files below.
- Rendering: this page was re-rendered from the saved `final_evaluation.json` (sha256 `36b494bd043ae0973f4cfce98be2218e33c72c74fc840a790d32b3236688bebe`) by `v2/research/final_analyze.py --render-only` (sha256 `afe518d439309816c8ec2d2eb76850d369d5d7e3a06a9a5c19e9d59bc0d5c130`) after the opening. The render-only changes add the deciding-numbers and not-distinguishable lines, the post-hoc robustness section and the post-opening corrections section; no number was recomputed.

| split | rows | positives | score file sha256 | source split sha256 |
| --- | ---: | ---: | --- | --- |
| test | 50000 | 9096 | `922690d0769ac6c9…` | `6479d0bffab20771…` |
| shift_queue_saturation | 12500 | 6378 | `9e9d5a7207a304bd…` | `c647bb89be02205f…` |
| shift_failure_bursts | 12500 | 3178 | `aa5b9843e4d23bbb…` | `4199085505a0e3ea…` |
| shift_clock_skew | 12500 | 2279 | `52d325bbb2ed7f12…` | `d98e19e8fd1e9d3c…` |
| shift_missing_tls | 12500 | 3378 | `fdd2dc950bab681e…` | `a4a96189c864c939…` |
| legacy_v1_test | 240 | 51 | `0bb6345095f739ce…` | legacy_v1/test.jsonl (not sealed) |

Bootstrap: 2000 resamples, seed 20260927, percentile, type-7 (2.5th, 97.5th) 95% interval; one count-vector stream per evaluation set, shared by every contender and statistic; the pooled shift suite is resampled within each regime (shift_queue_saturation, shift_failure_bursts, shift_clock_skew, shift_missing_tls).

## Primary metric: balanced accuracy, point [95% interval] (MEASURED)

| contender | nominal test | pooled shift suite |
| --- | --- | --- |
| v2 | 0.7644 [0.7592, 0.7695] | 0.7581 [0.7542, 0.7618] |
| v1 | 0.7540 [0.7494, 0.7588] | 0.7053 [0.7019, 0.7087] |
| majority | 0.5000 [0.5000, 0.5000] | 0.5000 [0.5000, 0.5000] |
| rule | 0.5317 [0.5262, 0.5370] | 0.5439 [0.5392, 0.5483] |
| ABL-data | 0.7588 [0.7535, 0.7639] | 0.7523 [0.7483, 0.7560] |
| ABL-optimizer | 0.7552 [0.7500, 0.7602] | 0.7113 [0.7075, 0.7147] |

W1 and W2 compare v2 with v1, majority and rule only. ABL-data and ABL-optimizer are section 9 ablations (confirmation only).

## Paired differences v2 − v1 (point [95% paired interval], MEASURED)

| set | BA | Brier | ECE |
| --- | --- | --- | --- |
| test | 0.0104 [0.0066, 0.0142] | -0.0144 [-0.0151, -0.0138] | -0.0542 [-0.0560, -0.0524] |
| pooled_shift | 0.0528 [0.0498, 0.0557] | -0.0313 [-0.0322, -0.0304] | -0.0702 [-0.0724, -0.0670] |
| shift_queue_saturation | 0.0617 [0.0566, 0.0668] | -0.0692 [-0.0718, -0.0667] | -0.1344 [-0.1359, -0.1299] |
| shift_failure_bursts | 0.0381 [0.0309, 0.0454] | -0.0191 [-0.0205, -0.0178] | -0.0583 [-0.0645, -0.0510] |
| shift_clock_skew | 0.0066 [-0.0013, 0.0146] | -0.0137 [-0.0150, -0.0125] | -0.0546 [-0.0576, -0.0478] |
| shift_missing_tls | 0.0645 [0.0570, 0.0713] | -0.0232 [-0.0247, -0.0216] | -0.0498 [-0.0552, -0.0442] |
| legacy_v1_test | 0.0251 [-0.0237, 0.0708] | -0.0132 [-0.0216, -0.0048] | 0.0321 [-0.0182, 0.0760] |

## Score metrics of the scored contenders (MEASURED)

| set | contender | Brier | ECE | log loss | ROC AUC |
| --- | --- | ---: | ---: | ---: | ---: |
| test | v2 | 0.1013 | 0.0306 | 0.3411 | 0.8387 |
| test | v1 | 0.1158 | 0.0848 | 0.3837 | 0.8345 |
| test | ABL-data | 0.1046 | 0.0337 | 0.3482 | 0.8345 |
| test | ABL-optimizer | 0.1200 | 0.0914 | 0.3930 | 0.8326 |
| pooled_shift | v2 | 0.1445 | 0.0549 | 0.4472 | 0.8451 |
| pooled_shift | v1 | 0.1758 | 0.1251 | 0.5254 | 0.8389 |
| pooled_shift | ABL-data | 0.1523 | 0.0777 | 0.4658 | 0.8396 |
| pooled_shift | ABL-optimizer | 0.1826 | 0.1249 | 0.5409 | 0.8322 |
| shift_queue_saturation | v2 | 0.2088 | 0.1161 | 0.5986 | 0.7784 |
| shift_queue_saturation | v1 | 0.2780 | 0.2506 | 0.7626 | 0.7766 |
| shift_queue_saturation | ABL-data | 0.2275 | 0.1691 | 0.6423 | 0.7761 |
| shift_queue_saturation | ABL-optimizer | 0.2881 | 0.2641 | 0.7859 | 0.7733 |
| shift_failure_bursts | v2 | 0.1308 | 0.0433 | 0.4145 | 0.8393 |
| shift_failure_bursts | v1 | 0.1499 | 0.1016 | 0.4648 | 0.8361 |
| shift_failure_bursts | ABL-data | 0.1317 | 0.0405 | 0.4164 | 0.8362 |
| shift_failure_bursts | ABL-optimizer | 0.1601 | 0.0852 | 0.4886 | 0.8312 |
| shift_clock_skew | v2 | 0.1027 | 0.0248 | 0.3447 | 0.8348 |
| shift_clock_skew | v1 | 0.1164 | 0.0794 | 0.3856 | 0.8305 |
| shift_clock_skew | ABL-data | 0.1059 | 0.0322 | 0.3518 | 0.8306 |
| shift_clock_skew | ABL-optimizer | 0.1205 | 0.0863 | 0.3945 | 0.8290 |
| shift_missing_tls | v2 | 0.1358 | 0.0412 | 0.4309 | 0.8265 |
| shift_missing_tls | v1 | 0.1590 | 0.0910 | 0.4888 | 0.8233 |
| shift_missing_tls | ABL-data | 0.1441 | 0.0688 | 0.4526 | 0.8216 |
| shift_missing_tls | ABL-optimizer | 0.1616 | 0.1139 | 0.4945 | 0.8202 |
| legacy_v1_test | v2 | 0.1230 | 0.0801 | 0.3922 | 0.8392 |
| legacy_v1_test | v1 | 0.1362 | 0.0480 | 0.4335 | 0.8302 |
| legacy_v1_test | ABL-data | 0.1273 | 0.0634 | 0.4051 | 0.8340 |
| legacy_v1_test | ABL-optimizer | 0.1396 | 0.0563 | 0.4415 | 0.8196 |

v1's score is the Hub kernel's `operator_attention_score` as emitted, without recalibration (AM-5); v1's own card calls it not production calibrated (REPORTED, AM-5).

## Decision metrics and section 10 validity (MEASURED)

Per-class rates are shown only where the four section 10 preconditions hold; otherwise the contender is `INVALID_BASELINE` and only BA and prevalence are reported.

| set | contender | BA | status | precision | recall | specificity | F1 | false-alert share |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| test | v2 | 0.7644 | VALID | 0.5361 | 0.6548 | 0.8740 | 0.5895 | 0.4639 |
| test | v1 | 0.7540 | INVALID_BASELINE (iv_refused_all_probes) | – | – | – | – | – |
| test | majority | 0.5000 | INVALID_BASELINE (i_receipt_verifies, iii_decisions_not_constant, iv_refused_all_probes) | – | – | – | – | – |
| test | rule | 0.5317 | INVALID_BASELINE (i_receipt_verifies, iv_refused_all_probes) | – | – | – | – | – |
| test | ABL-data | 0.7588 | VALID | 0.4882 | 0.6750 | 0.8427 | 0.5666 | 0.5118 |
| test | ABL-optimizer | 0.7552 | INVALID_BASELINE (i_receipt_verifies) | – | – | – | – | – |
| pooled_shift | v2 | 0.7581 | VALID | 0.5411 | 0.8205 | 0.6957 | 0.6521 | 0.4589 |
| pooled_shift | v1 | 0.7053 | INVALID_BASELINE (iv_refused_all_probes) | – | – | – | – | – |
| pooled_shift | majority | 0.5000 | INVALID_BASELINE (i_receipt_verifies, iii_decisions_not_constant, iv_refused_all_probes) | – | – | – | – | – |
| pooled_shift | rule | 0.5439 | INVALID_BASELINE (i_receipt_verifies, iv_refused_all_probes) | – | – | – | – | – |
| pooled_shift | ABL-data | 0.7523 | VALID | 0.5282 | 0.8280 | 0.6766 | 0.6450 | 0.4718 |
| pooled_shift | ABL-optimizer | 0.7113 | INVALID_BASELINE (i_receipt_verifies) | – | – | – | – | – |
| shift_queue_saturation | v2 | 0.5928 | VALID | 0.5637 | 0.9588 | 0.2269 | 0.7100 | 0.4363 |
| shift_queue_saturation | v1 | 0.5311 | INVALID_BASELINE (iv_refused_all_probes) | – | – | – | – | – |
| shift_queue_saturation | majority | 0.5000 | INVALID_BASELINE (i_receipt_verifies, iii_decisions_not_constant, iv_refused_all_probes) | – | – | – | – | – |
| shift_queue_saturation | rule | 0.5373 | INVALID_BASELINE (i_receipt_verifies, iv_refused_all_probes) | – | – | – | – | – |
| shift_queue_saturation | ABL-data | 0.5842 | VALID | 0.5578 | 0.9666 | 0.2017 | 0.7074 | 0.4422 |
| shift_queue_saturation | ABL-optimizer | 0.5693 | INVALID_BASELINE (i_receipt_verifies) | – | – | – | – | – |
| shift_failure_bursts | v2 | 0.7592 | VALID | 0.5254 | 0.7492 | 0.7693 | 0.6176 | 0.4746 |
| shift_failure_bursts | v1 | 0.7211 | INVALID_BASELINE (iv_refused_all_probes) | – | – | – | – | – |
| shift_failure_bursts | majority | 0.5000 | INVALID_BASELINE (i_receipt_verifies, iii_decisions_not_constant, iv_refused_all_probes) | – | – | – | – | – |
| shift_failure_bursts | rule | 0.6053 | INVALID_BASELINE (i_receipt_verifies, iv_refused_all_probes) | – | – | – | – | – |
| shift_failure_bursts | ABL-data | 0.7497 | VALID | 0.4730 | 0.8052 | 0.6942 | 0.5959 | 0.5270 |
| shift_failure_bursts | ABL-optimizer | 0.7485 | INVALID_BASELINE (i_receipt_verifies) | – | – | – | – | – |
| shift_clock_skew | v2 | 0.7564 | VALID | 0.5265 | 0.6415 | 0.8713 | 0.5783 | 0.4735 |
| shift_clock_skew | v1 | 0.7498 | INVALID_BASELINE (iv_refused_all_probes) | – | – | – | – | – |
| shift_clock_skew | majority | 0.5000 | INVALID_BASELINE (i_receipt_verifies, iii_decisions_not_constant, iv_refused_all_probes) | – | – | – | – | – |
| shift_clock_skew | rule | 0.5336 | INVALID_BASELINE (i_receipt_verifies, iv_refused_all_probes) | – | – | – | – | – |
| shift_clock_skew | ABL-data | 0.7525 | VALID | 0.4860 | 0.6608 | 0.8441 | 0.5601 | 0.5140 |
| shift_clock_skew | ABL-optimizer | 0.7511 | INVALID_BASELINE (i_receipt_verifies) | – | – | – | – | – |
| shift_missing_tls | v2 | 0.7428 | VALID | 0.5139 | 0.7475 | 0.7382 | 0.6091 | 0.4861 |
| shift_missing_tls | v1 | 0.6783 | INVALID_BASELINE (iv_refused_all_probes) | – | – | – | – | – |
| shift_missing_tls | majority | 0.5000 | INVALID_BASELINE (i_receipt_verifies, iii_decisions_not_constant, iv_refused_all_probes) | – | – | – | – | – |
| shift_missing_tls | rule | 0.5385 | INVALID_BASELINE (i_receipt_verifies, iv_refused_all_probes) | – | – | – | – | – |
| shift_missing_tls | ABL-data | 0.7450 | VALID | 0.5520 | 0.7004 | 0.7895 | 0.6174 | 0.4480 |
| shift_missing_tls | ABL-optimizer | 0.6295 | INVALID_BASELINE (i_receipt_verifies) | – | – | – | – | – |
| legacy_v1_test | v2 | 0.7664 | VALID | 0.5469 | 0.6863 | 0.8466 | 0.6087 | 0.4531 |
| legacy_v1_test | v1 | 0.7414 | INVALID_BASELINE (iv_refused_all_probes) | – | – | – | – | – |
| legacy_v1_test | majority | 0.5000 | INVALID_BASELINE (i_receipt_verifies, iii_decisions_not_constant, iv_refused_all_probes) | – | – | – | – | – |
| legacy_v1_test | rule | 0.5174 | INVALID_BASELINE (i_receipt_verifies, iv_refused_all_probes) | – | – | – | – | – |
| legacy_v1_test | ABL-data | 0.7211 | VALID | 0.4776 | 0.6275 | 0.8148 | 0.5424 | 0.5224 |
| legacy_v1_test | ABL-optimizer | 0.7323 | INVALID_BASELINE (i_receipt_verifies) | – | – | – | – | – |

### Validity inputs per contender (MEASURED before the opening, in the scoring process)

| contender | receipt verified | fail-closed probes refused | escaped probes |
| --- | --- | ---: | --- |
| v2 | True | 66/66 | none |
| v1 | True | 65/66 | unwrapped_features |
| majority | False | 0/66 | missing_field, extra_field, nan_continuous, inf_continuous, neg_inf_continuous, nan_binary … |
| rule | False | 3/66 | missing_field, extra_field, nan_continuous, inf_continuous, neg_inf_continuous, nan_binary … |
| ABL-data | True | 66/66 | none |
| ABL-optimizer | False | 66/66 | none |

The probe suite is `gates.FAILCLOSED_PROBES` (v2's own suite). v1 was probed through its own CLI input path; AM-6 records that the two contenders' probe behavior differs by construction, and the v1 audit separately MEASURED that v1 accepts duplicate JSON keys (REPORTED from `audit/oac_v1_audit.md`, finding B-7a). Receipt and probe results do not enter W1–W3.

## v2 conformal layer (Mondrian, alpha 0.10; MEASURED)

| set | marginal coverage | class 0 | class 1 | abstention | BOTH | EMPTY | selective BA | BA without abstention |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test | 0.9013 | 0.9003 | 0.9061 | 0.4143 | 0.4143 | 0.0000 | 0.8447 | 0.7644 |
| pooled_shift | 0.8121 | 0.7426 | 0.9711 | 0.3820 | 0.3820 | 0.0000 | 0.7402 | 0.7581 |
| shift_queue_saturation | 0.6567 | 0.2994 | 0.9997 | 0.1761 | 0.1761 | 0.0000 | 0.5031 | 0.5928 |
| shift_failure_bursts | 0.8470 | 0.8118 | 0.9503 | 0.4124 | 0.4124 | 0.0000 | 0.7899 | 0.7592 |
| shift_clock_skew | 0.8967 | 0.8953 | 0.9030 | 0.4089 | 0.4089 | 0.0000 | 0.8387 | 0.7564 |
| shift_missing_tls | 0.8480 | 0.7982 | 0.9825 | 0.5308 | 0.5308 | 0.0000 | 0.7206 | 0.7428 |
| legacy_v1_test | 0.9000 | 0.8889 | 0.9412 | 0.4167 | 0.4167 | 0.0000 | 0.8564 | 0.7664 |

The class-conditional coverage guarantee (at least 0.90 per class) holds only under exchangeability with the nominal conformal split (MODELED, section 6). Under shift and on legacy_v1_test the coverage figures are descriptive.

## Disaggregation (MEASURED)

BA by regime, point [95% interval]:

| set | v2 | v1 | majority | rule | ABL-data | ABL-optimizer |
| --- | --- | --- | --- | --- | --- | --- |
| test | 0.7644 [0.7592, 0.7695] | 0.7540 [0.7494, 0.7588] | 0.5000 [0.5000, 0.5000] | 0.5317 [0.5262, 0.5370] | 0.7588 [0.7535, 0.7639] | 0.7552 [0.7500, 0.7602] |
| pooled_shift | 0.7581 [0.7542, 0.7618] | 0.7053 [0.7019, 0.7087] | 0.5000 [0.5000, 0.5000] | 0.5439 [0.5392, 0.5483] | 0.7523 [0.7483, 0.7560] | 0.7113 [0.7075, 0.7147] |
| shift_queue_saturation | 0.5928 [0.5871, 0.5985] | 0.5311 [0.5277, 0.5344] | 0.5000 [0.5000, 0.5000] | 0.5373 [0.5294, 0.5456] | 0.5842 [0.5788, 0.5896] | 0.5693 [0.5644, 0.5742] |
| shift_failure_bursts | 0.7592 [0.7511, 0.7674] | 0.7211 [0.7136, 0.7285] | 0.5000 [0.5000, 0.5000] | 0.6053 [0.5953, 0.6150] | 0.7497 [0.7418, 0.7576] | 0.7485 [0.7405, 0.7570] |
| shift_clock_skew | 0.7564 [0.7455, 0.7665] | 0.7498 [0.7399, 0.7595] | 0.5000 [0.5000, 0.5000] | 0.5336 [0.5228, 0.5443] | 0.7525 [0.7420, 0.7627] | 0.7511 [0.7409, 0.7614] |
| shift_missing_tls | 0.7428 [0.7344, 0.7508] | 0.6783 [0.6712, 0.6854] | 0.5000 [0.5000, 0.5000] | 0.5385 [0.5296, 0.5481] | 0.7450 [0.7361, 0.7530] | 0.6295 [0.6230, 0.6353] |
| legacy_v1_test | 0.7664 [0.6970, 0.8371] | 0.7414 [0.6728, 0.8047] | 0.5000 [0.5000, 0.5000] | 0.5174 [0.4483, 0.5900] | 0.7211 [0.6522, 0.7932] | 0.7323 [0.6613, 0.7995] |

Nominal test, BA by feature value (point estimates):

| feature = value | rows | positives | v2 | v1 | majority | rule | ABL-data | ABL-optimizer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tls_enabled = 0 | 4984 | 1502 | 0.7161 | 0.6008 | 0.5000 | 0.5315 | 0.7269 | 0.5162 |
| tls_enabled = 1 | 45016 | 7594 | 0.7631 | 0.7616 | 0.5000 | 0.5321 | 0.7610 | 0.7637 |
| listener_running = 0 | 3022 | 2044 | 0.5000 | 0.5000 | 0.5000 | 0.5259 | 0.5000 | 0.5000 |
| listener_running = 1 | 46978 | 7052 | 0.7251 | 0.7319 | 0.5000 | 0.5357 | 0.7221 | 0.7259 |

## legacy_v1_test (v1's public 240-row test split; secondary, REPORTED comparability)

- v2: BA 0.7664 [0.6970, 0.8371], Brier 0.1230, ECE 0.0801 (VALID)
- v1: BA 0.7414 [0.6728, 0.8047], Brier 0.1362, ECE 0.0480 (INVALID_BASELINE)
- majority: BA 0.5000 [0.5000, 0.5000] (INVALID_BASELINE)
- rule: BA 0.5174 [0.4483, 0.5900] (INVALID_BASELINE)
- ABL-data: BA 0.7211 [0.6522, 0.7932], Brier 0.1273, ECE 0.0634 (VALID)
- ABL-optimizer: BA 0.7323 [0.6613, 0.7995], Brier 0.1396, ECE 0.0563 (INVALID_BASELINE)

Before the opening, the scorer checked that v1 reproduces its published confusion on this split (tp 40, fp 57, tn 132, fn 11; published values REPORTED from the v1 receipt, reproduction MEASURED).

## Generator-truth recovery (MEASURED from the frozen parameters)

| model | sign agreement | Spearman vs truth | relative L2 (intercept, weights) | disagreeing |
| --- | ---: | ---: | ---: | --- |
| v2 | 8/8 | 1.0000 | 0.3282 | none |
| v1 | 8/8 | 0.9048 | 0.7024 | none |
| ABL-data | 7/8 | 0.9286 | 0.4011 | seconds_since_last_success |
| ABL-optimizer | 8/8 | 0.9048 | 0.7608 | none |

## Ablations (section 9; test values confirm the validation decisions)

| ablation | set | ablation BA | v2 − ablation BA | ablation Brier | v2 Brier | ablation ECE | v2 ECE |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| ABL-data | test | 0.7588 [0.7535, 0.7639] | 0.0056 [0.0030, 0.0080] | 0.1046 | 0.1013 | 0.0337 | 0.0306 |
| ABL-data | pooled_shift | 0.7523 [0.7483, 0.7560] | 0.0058 [0.0038, 0.0078] | 0.1523 | 0.1445 | 0.0777 | 0.0549 |
| ABL-optimizer | test | 0.7552 [0.7500, 0.7602] | 0.0092 [0.0060, 0.0122] | 0.1200 | 0.1013 | 0.0914 | 0.0306 |
| ABL-optimizer | pooled_shift | 0.7113 [0.7075, 0.7147] | 0.0469 [0.0440, 0.0500] | 0.1826 | 0.1445 | 0.1249 | 0.0549 |

- ABL-calibration: NOT_APPLICABLE (the selected calibrator is none, so the selected model has no calibrator to remove; the search itself compared none, platt and isotonic (section 9)).
- ABL-conformal (kept on validation: True): on nominal test, selective BA 0.8447 on the non-abstained rows at abstention rate 0.4143, against BA 0.7644 without abstention. Selective BA is computed on a different, easier subset of rows and is not comparable to the primary metric.

## Reliability (nominal test, v2 and v1; MEASURED)

![Reliability diagrams](reliability.svg)

| bin | v2 n | v2 mean score | v2 fraction positive | v1 n | v1 mean score | v1 fraction positive |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [0.0, 0.1) | 23210 | 0.0727 | 0.0467 | 0 | – | – |
| [0.1, 0.2) | 15680 | 0.1388 | 0.1312 | 40320 | 0.1443 | 0.0889 |
| [0.2, 0.3) | 3811 | 0.2396 | 0.2839 | 3944 | 0.2309 | 0.3600 |
| [0.3, 0.4) | 1116 | 0.3415 | 0.4355 | 1407 | 0.3590 | 0.6866 |
| [0.4, 0.5) | 1327 | 0.4537 | 0.5305 | 3164 | 0.4424 | 0.6564 |
| [0.5, 0.6) | 1754 | 0.5485 | 0.6345 | 781 | 0.5370 | 0.8681 |
| [0.6, 0.7) | 1313 | 0.6474 | 0.7281 | 206 | 0.6410 | 0.9563 |
| [0.7, 0.8) | 1006 | 0.7476 | 0.8678 | 146 | 0.7543 | 0.9589 |
| [0.8, 0.9) | 491 | 0.8418 | 0.9389 | 31 | 0.8302 | 1.0000 |
| [0.9, 1.0] | 292 | 0.9460 | 0.9623 | 1 | 0.9502 | 1.0000 |

Every other reliability table (each set, each scored contender) is in `v2/results/final_evaluation.json`.

## What this does not show

- Nothing about any real system: the data, labels and shifts are synthetic, generated from a logistic rule that the v2 model class matches exactly (MODELED, section 2).
- The shift regimes are four pre-registered synthetic perturbations. Other shifts were not tested (MODELED).
- Receipts attest integrity and origin, not quality.

Generated 2026-09-29T21:21:59Z by `v2/research/final_analyze.py` from `v2/results/final/` (re-runnable; it opens no data split).
