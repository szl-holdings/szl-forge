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
checkpoint or adapter; it is not a health-care model or a result service. It accepts eight
bounded operational fields and proposes `ALERT`, `NO_ALERT`, or `ABSTAIN` for
operator attention. Every authority flag is false: it cannot acknowledge a
transport message, control a device, interpret a result, release a result, or
make a care decision. Do not submit patient information, laboratory results,
specimens, orders, or device identifiers.

<!-- SZL-NATIVE-PRESENTATION:v1 -->
<p><a href="https://huggingface.co/SZLHOLDINGS"><img src="https://raw.githubusercontent.com/szl-holdings/szl-brand/27d12e77fc0f9f1a0e5fae8b34adf2380d51d070/kit/logos/orbit-v2/suite/szl-icon-32.svg" alt="SZL Holdings" width="48" height="48" /></a></p>

[Command Lab](https://huggingface.co/spaces/SZLHOLDINGS/szl-command-lab) · [Canonical source](https://github.com/szl-holdings/szl-forge/tree/main/ops-health/v2) · [Evidence and limits](#evidence-and-limits)
<!-- /SZL-NATIVE-PRESENTATION:v1 -->

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
public checkout. This unsigned receipt records file digests and a declared
local research origin. Matching digests support byte integrity against this
record; the receipt does not authenticate that origin or establish accuracy
or authorization.

## Evidence and limits

The preregistered comparison and single sealed test opening are recorded in
[`FINAL_RESULTS.md`](https://github.com/szl-holdings/szl-forge/blob/56a00821858825f529c40c7322c2f1584608d6e5/ops-health/v2/results/FINAL_RESULTS.md)
and [`final_evaluation.json`](https://github.com/szl-holdings/szl-forge/blob/56a00821858825f529c40c7322c2f1584608d6e5/ops-health/v2/results/final_evaluation.json).
Those results are **REPORTED for synthetic generated data only** in the source
project; they do not establish performance on real transports,
devices, sites, populations, or workflows. The public source also contains the
preregistration, amendments, provenance inventory, and bounded verification
tools. This Hub package is a proposal-only research artifact, not a health-care
release or a live service.

**Labels.** Unless marked otherwise, every number below is MEASURED in the source
project and SYNTHETIC: it was computed on generated data, and the link next to it opens
the evidence file at `56a00821`. REPORTED means read from the named file, MODELED means
reasoning from the code or the protocol without a measurement, and UNKNOWN means not
established.

## Preregistered verdict: WIN on a synthetic benchmark

Under a protocol registered before any v2 code or data existed
([preregistration][prereg], [amendments][amend]), v2 met all three win conditions against
v1 as shipped and two trivial baselines. **v2 is a WIN on its preregistered synthetic
benchmark, and only there.** One condition held by 0.000319 in balanced accuracy.

The primary metric is the balanced accuracy (BA) of each contender's binary decision: v2's
selected threshold 0.20, v1's shipped threshold 0.16, and the fixed decisions of the majority
(always no alert) and rule (`consecutive_failures > 0`) baselines. It is computed on 50,000 sealed nominal test rows and on a pooled suite of four
sealed synthetic shift regimes (4 × 12,500 rows), opened once
([opening record][opened]). Intervals are 95% percentile intervals from 2,000 paired
bootstrap resamples (seed 20260927). v2 wins only if all three conditions hold:

- **W1 and W2.** v2's lower bound is strictly above the upper bound of v1, majority and
  rule, on nominal test (W1) and on the pooled shift suite (W2).
- **W3.** v2's Brier score and ECE on nominal test are no worse than v1's.

| contender | nominal test BA [95% CI] | pooled shift BA [95% CI] |
| --- | --- | --- |
| **v2** | **0.7644 [0.7592, 0.7695]** | **0.7581 [0.7542, 0.7618]** |
| v1 as shipped (Hub revision `dd7d1098`) | 0.7540 [0.7494, 0.7588] | 0.7053 [0.7019, 0.7087] |
| majority (always `NO_ALERT`) | 0.5000 [0.5000, 0.5000] | 0.5000 [0.5000, 0.5000] |
| rule (`consecutive_failures > 0`) | 0.5317 [0.5262, 0.5370] | 0.5439 [0.5392, 0.5483] |

Source: [`FINAL_RESULTS.md`][results] and [`final_evaluation.json`][eval].

- **W1 was decided by 0.000319 BA.** v2's lower bound is 0.759159 and v1's upper bound is
  0.758840 (0.7592 against 0.7588 at four decimals). A change of 0.000319 in either bound
  would reverse the comparison (MODELED). A post-hoc check, not registered, repeated W1
  with 100 other resample streams. It held in 100 of 100, with margins from 0.000081 to
  0.000940 ([`final_robustness.json`][robust]). An independent NumPy recomputation
  matched every quantity behind the verdict (BA of every contender, v2 and v1 Brier and
  ECE, and their bootstrap and paired v2 − v1 intervals, on nominal test and the pooled
  shift suite) to within 5.6e-17 and gave the same verdict ([log][verify]). The secondary
  metrics, including the conformal coverage and abstention below, were not part of it.
- **The effect is small.** The paired v2 − v1 BA difference is **+0.0104 [+0.0066,
  +0.0142]** on nominal test and **+0.0528 [+0.0498, +0.0557]** on the pooled shift suite.
- **W2 holds** with a margin of 0.0454 over v1. **W3 holds**: Brier 0.1013 against 0.1158,
  and ECE 0.0306 against 0.0848.
- **Not distinguishable from v1.** Under clock skew (paired +0.0066 [−0.0013, +0.0146]) and
  on v1's own 240-row public test split (paired +0.0251 [−0.0237, +0.0708]), the paired
  interval includes zero. No BA advantage over v1 is claimed on either.
- **Queue saturation degrades every contender.** No pass/fail rule is registered for this
  single regime. The measured BAs there are v2 0.5928, v1 0.5311, rule 0.5373 and majority
  0.5000. v2's specificity there is 0.2269: it alerts on most rows that need no attention
  when queues run hot.
- **Many alerts are false.** On nominal test, v2's precision is 0.5361, its recall 0.6548
  and its specificity 0.8740, so 46% of its threshold-decision alerts are false.
- **Per-class rates are withheld for v1, majority and rule** (`INVALID_BASELINE`): each
  fails at least one registered validity precondition. v1 accepts one input form that v2's
  probe suite requires refusing (65 of 66 probes refused; v2 refused 66 of 66). Their BA
  still enters W1 and W2.

## Calibration

| set | v2 Brier | v1 Brier | v2 ECE | v1 ECE |
| --- | ---: | ---: | ---: | ---: |
| nominal test | 0.1013 | 0.1158 | 0.0306 | 0.0848 |
| pooled shift | 0.1445 | 0.1758 | 0.0549 | 0.1251 |
| queue saturation | 0.2088 | 0.2780 | 0.1161 | 0.2506 |

Source: [`FINAL_RESULTS.md`][results]. ECE uses 10 equal-width bins weighted by count. v1 is
scored on its score exactly as emitted (amendment AM-5).

- **Where v2 misses.** Above a score of 0.3 it under-predicts: in the bins from 0.3 to 0.9,
  the observed positive fraction exceeds the mean score by about 0.08 to 0.12. Its ECE stays
  low because most rows score below 0.2.
- **No calibrator was selected.** At the chosen penalty, a sigmoid recalibrator had a better
  validation ECE (0.0164 against 0.0347) but trailed by 0.0005 BA, and the registered rule
  ranks by BA first ([`STAGE2_SEARCH.md`][search]).
- **Not production calibrated.** The kernel names its score
  `synthetic_attention_probability_estimate_not_production_calibrated`. None of these
  figures carries over to real data (MODELED).

## Conformal abstention

`ABSTAIN` comes from label-conditional split conformal prediction (alpha 0.10), fit on a
separate 4,096-row nominal split. The layer was kept because validation met the registered
rule: marginal coverage 0.9006 (at least 0.88) and per-class coverage 0.9011 and 0.8984 (at
least 0.85) ([`STAGE2_SEARCH.md`][search]).

| set | coverage | class 0 | class 1 | abstention | selective BA | BA, all rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| nominal test | 0.9013 | 0.9003 | 0.9061 | 0.4143 | 0.8447 | 0.7644 |
| pooled shift | 0.8121 | 0.7426 | 0.9711 | 0.3820 | 0.7402 | 0.7581 |
| queue saturation | 0.6567 | 0.2994 | 0.9997 | 0.1761 | 0.5031 | 0.5928 |

Source: [`FINAL_RESULTS.md`][results].

- **v2 abstains on about 41% of nominal rows**, each time because both labels stay in the
  prediction set. A definite `ALERT` or `NO_ALERT` comes back on about 59% of rows.
- **Selective BA counts only the rows where v2 did not abstain.** Those are the easier rows,
  so selective BA is not comparable with the primary metric.
- **Under shift, abstention stopped helping.** On the pooled suite, selective BA (0.7402) is
  below plain BA (0.7581).
- **The coverage guarantee is nominal only.** Per-class coverage of at least 0.90 is
  guaranteed only for data exchangeable with the nominal conformal split (MODELED). Under
  queue saturation, class-0 coverage is 0.2994.

## Underspecification: one weight is weakly identified

`seconds_since_last_success` has a generator weight of +2.20 on its normalized scale
([`final_evaluation.json`][eval]). In nominal data its values sit near zero on the
0–86,400 s scale, so the data barely constrain it. The preregistration predicted this before
any fit (§2).

- v2's own weight is +0.0365: positive, but about 60 times smaller than the truth
  (REPORTED: `model.json`).
- Five known-good retrains of the selected recipe on fresh training draws put this weight
  below zero in **3 of 5** (−0.0035, −0.0094, −0.0134). Those three fail the registered sign
  gate and the monotonicity gate (3,200 of 19,600 checks), while their validation BA stays
  within 0.004 of v2's ([`MUTATION_READINESS.md`][mutation]).
- v2's 8 of 8 sign agreement with the generator (rank correlation 1.0000,
  [`FINAL_RESULTS.md`][results]) is therefore not a property of the recipe. Do not read
  meaning into the sign or size of this weight (MODELED).

## Mutation testing and readiness aggregation

All mutants ran on the validation split, before the test opening
([`MUTATION_READINESS.md`][mutation]).

- **36 artifact mutants**: a weight scaled ×0.5 or ×1.5, a sign flip, a dropped feature, the
  intercept ±1 or the threshold ±0.1. The hash check caught 36 of 36. With the receipt
  regenerated so that the hash check passes, the behavioral gates caught 26 of 36.
- **Receipt mutants:** 4 of 5 caught. **Kernel-source mutants:** integrity caught 4 of 5,
  and the tests caught 5 of 5.
- **11 `BLIND_SPOT` mutants.** Ten are artifact mutants that no behavioral gate caught once
  the hash check was bypassed: ×0.5 and ×1.5 scaling of `tls_enabled`,
  `queue_utilization`, `consecutive_failures` and `seconds_since_last_success`, and the
  threshold ±0.1. The eleventh, `R-source-commit-other`, is a receipt re-pointed to a parent
  commit whose bound files are identical, and the integrity layer accepted it. With the
  gates as registered, only the hash check detects a modified model (MODELED).
- **Broken baselines.** A constant-`ALERT` model, a fail-open kernel and majority all return
  `INVALID_BASELINE`, with no per-class rates.
- **Λ REJECTED.** The registered comparison set SZL's Λ readiness aggregator against AND,
  min and mean over 6 known-good and 36 known-broken fixtures. Λ's margin (−0.0436) was not
  better than the mean's (−0.0412), and Λ misclassified 14 fixtures at the cut, against 13
  for AND and min. No aggregator separated the two groups: every margin was negative. No
  property of Λ is claimed.

## Compliance with the preregistration

- **Order.** The preregistration was committed on 2026-09-26, before any v2 code or data.
  Amendments AM-1 to AM-6 were committed before the registered search, every fit on the
  registered splits, and the opening (stage-1 unit and smoke fits on DEV-only data came
  earlier), and none changes
  the question, metric, win rule, splits, seeds, search space or shifts
  ([`AMENDMENTS.md`][amend]).
- **Selection on validation only.** A registered 12-trial search (penalty × calibrator)
  stopped with `SEARCH_SPACE_EXHAUSTED` and selected `T07_l2_1e-3_none` (penalty 1e-3, no
  calibrator, threshold 0.20) by validation BA ([`STAGE2_SEARCH.md`][search]).
- **A single opening.** The sealed splits were opened exactly once, at 2026-09-29T21:13:47Z
  ([`TEST_OPENED.json`][opened]).
- **Audit.** A compliance audit of the final evaluation passed 10 of 10 checks and
  discloses **16 deviations**. None changes the question, the primary metric, the win rule,
  the splits, the seeds, the search space or the shift regimes. One example: the
  preregistration did not say how to interpolate a percentile, and the implementation uses
  type-7 percentiles ([`COMPLIANCE.md`][compliance]).
- **After the opening.** Every later change to code or tracked files on the final
  evaluation's path is listed, and a full re-analysis from the saved
  score files reproduced all 3,884 numeric fields of `final_evaluation.json` exactly
  ([`FINAL_RESULTS.md`][results]).

## REFORMS checklist

The 32 items of the REFORMS reporting checklist for ML-based science
([arXiv:2308.07832v2](https://arxiv.org/abs/2308.07832v2); published version
[doi:10.1126/sciadv.adk3452](https://doi.org/10.1126/sciadv.adk3452), whose checklist
wording was not compared: UNKNOWN), with this card's status for each. **Source id** is the
checklist's own item id. The item text is SZL's paraphrase, not a quotation.

Totals: 16 satisfied, 10 partial, 3 not applicable, 3 missing.

| Source id | Checklist item (paraphrase) | Status | Where, or why |
| --- | --- | --- | --- |
| 1a | State the population or distribution the scientific claim is meant to generalize to. | Satisfied | Only `ops-health-generator/2.0.0`'s nominal regime and its four preregistered shift regimes |
| 1b | Explain why that particular population or distribution was chosen. | Partial | The nominal regime reproduces v1's generator so the versions are comparable; each shift stresses one input family; no reason is given for their parameters |
| 1c | Explain why machine learning is an appropriate tool for the study's question. | Partial | v2 beats a majority and a one-feature rule, but the labels come from a logistic rule, so the model class is correct by construction; no case is made for real telemetry |
| 2a | Share the training and evaluation data with a persistent link or identifier pinning the exact dataset. | Partial | Every split is pinned by SHA-256 in [`MANIFEST.json`][manifest] and regenerates byte for byte; the split files are not hosted as a dataset |
| 2b | Share training, evaluation and results code with a link or identifier fixing the exact code version. | Partial | Code at szl-forge `56a00821`; the receipt's `source.commit` names the unpublished research repository (Source and files) |
| 2c | Describe the hardware and software environment the computations ran on. | Partial | Standard library only, CPython 3.12.10 on Windows 11; the independent check used CPython 3.11.9 with NumPy; CPU and memory are not recorded |
| 2d | Provide a README explaining how to regenerate the reported results from the shared data and code. | Satisfied | "How to check it" in the [source README][source] |
| 2e | Provide one script that regenerates every reported result end to end. | Partial | Several commands, not one; the sealed opening cannot be repeated by design; `final_analyze` recomputes every test number from the saved score files |
| 3a | Document where training and evaluation data came from, when collected, and how ground-truth labels were produced. | Satisfied | Fixed-seed generator; labels are random draws from the logistic rule in [preregistration][prereg] §2, not annotations |
| 3b | Describe the sampling frame, i.e. the distribution or set the dataset was drawn from. | Partial | Shift regimes are fully specified in preregistration §3; the nominal per-field draws are defined in code (v1's draws) and only partly stated in the preregistration |
| 3c | Argue concretely why this dataset suits the modeling task. | Missing | No argument is made that the generator resembles real transport telemetry |
| 3d | Define the target the model predicts and summarise it statistically, per class if the target is categorical. | Partial | `operator_attention_required`, with positives per split (3e); no per-class feature statistics |
| 3e | Report the sample size and how often each outcome occurs. | Satisfied | Positives / rows: train 2,940/16,384, calibration 744/4,096, conformal 750/4,096 ([counts][counts]); validation 768/4,096 ([mutation report][mutation]); test 9,096/50,000 and each shift regime ([results][results]) |
| 3f | Report the share of missing values, split by class for categorical outcomes. | Not applicable | Rows are complete by construction, and the closed schema requires all eight fields |
| 3g | Argue that the sampling frame represents the population targeted by the claim. | Not applicable | No claim is made beyond the generator's own distributions; representativeness of real telemetry is UNKNOWN |
| 4a | State whether any samples were excluded and give the reason. | Not applicable | No rows are excluded |
| 4b | Explain how impossible or corrupted samples were handled. | Satisfied | Generated values are in range by construction (clock skew is capped at 86,400 s); at inference, invalid input is refused, including duplicate keys |
| 4c | List every transformation from raw data to model input, such as missing-value handling and scaling, ideally diagrammed. | Satisfied | Binary fields as is; the three continuous fields are min-max scaled with fixed bounds |
| 5a | Describe every trained model in detail. | Satisfied | Algorithm, penalty, convergence, weights, threshold and conformal quantiles are in `model.json` |
| 5b | Justify the chosen model families. | Partial | Logistic regression matches the known generator and runs on the standard library; no other model family was tried |
| 5c | Explain the evaluation procedure, e.g. how data were partitioned for training versus testing or how folds were formed. | Satisfied | Separate train, calibration, conformal and validation splits, plus sealed test and shift splits opened once |
| 5d | Describe how the reported model or models were selected among candidates. | Satisfied | A registered 12-trial search with a fixed selection and tie rule; every trial is in [`STAGE2_SEARCH.md`][search] |
| 5e | Give hyperparameter tuning details for the reported model or models. | Satisfied | The grid, per-trial validation metrics and stop reason, recorded in hash-chained receipts |
| 5f | Justify that the baselines used for comparison are appropriate. | Partial | Majority, a one-line operator rule and v1 as shipped were preregistered, with two ablations; no stronger nonlinear baseline |
| 6a | Show that preprocessing and modeling used only training data, never test data. | Satisfied | Scaling bounds are constants; training and selection code refuses sealed paths; the sealed splits were opened once, after the freeze |
| 6b | Explain how dependencies or duplicates across train and test partitions were handled, e.g. grouping related samples together. | Missing | No duplicate check was registered or run before the opening. A post-hoc check of the saved score files found 0 exact matches between scored and development rows ([log][dupes]); near-duplicates were not checked |
| 6c | Argue why every input used by the model is legitimately available and cannot reveal the label. | Satisfied | Labels are generated from these eight inputs by design; under clock skew the model sees the skewed value while the label uses the true one |
| 7a | Report every performance metric and justify the one used to pick the final model. | Satisfied | All preregistered metrics are reported; BA was fixed in advance and does not move with the 18% prevalence (MODELED) |
| 7b | Report uncertainty estimates such as confidence intervals and explain how they were computed. | Satisfied | Paired percentile bootstrap, 2,000 resamples, seed 20260927, type-7 percentiles, resampled within regime for the pooled suite |
| 7c | Justify any statistical tests used and check that their assumptions hold. | Satisfied | No significance test is used; the interval rule assumes independent rows, which holds by construction; resample-stream sensitivity was checked post hoc (100 of 100) |
| 8a | Provide evidence that the results hold beyond the study data (external validity). | Missing | No evaluation beyond synthetic data; real-world performance is UNKNOWN |
| 8b | State the settings in which the findings are not expected to hold. | Satisfied | The boundary statements at the top of this card, queue saturation, clock skew, abstention and the blind spots above |

[prereg]: https://github.com/szl-holdings/szl-forge/blob/56a00821858825f529c40c7322c2f1584608d6e5/ops-health/v2/PREREGISTRATION.md
[amend]: https://github.com/szl-holdings/szl-forge/blob/56a00821858825f529c40c7322c2f1584608d6e5/ops-health/v2/AMENDMENTS.md
[opened]: https://github.com/szl-holdings/szl-forge/blob/56a00821858825f529c40c7322c2f1584608d6e5/ops-health/runs/TEST_OPENED.json
[results]: https://github.com/szl-holdings/szl-forge/blob/56a00821858825f529c40c7322c2f1584608d6e5/ops-health/v2/results/FINAL_RESULTS.md
[eval]: https://github.com/szl-holdings/szl-forge/blob/56a00821858825f529c40c7322c2f1584608d6e5/ops-health/v2/results/final_evaluation.json
[robust]: https://github.com/szl-holdings/szl-forge/blob/56a00821858825f529c40c7322c2f1584608d6e5/ops-health/v2/results/final_robustness.json
[verify]: https://github.com/szl-holdings/szl-forge/blob/56a00821858825f529c40c7322c2f1584608d6e5/ops-health/logs/v2/final_verify_independent.txt
[search]: https://github.com/szl-holdings/szl-forge/blob/56a00821858825f529c40c7322c2f1584608d6e5/ops-health/v2/results/STAGE2_SEARCH.md
[mutation]: https://github.com/szl-holdings/szl-forge/blob/56a00821858825f529c40c7322c2f1584608d6e5/ops-health/v2/results/MUTATION_READINESS.md
[compliance]: https://github.com/szl-holdings/szl-forge/blob/56a00821858825f529c40c7322c2f1584608d6e5/ops-health/v2/results/COMPLIANCE.md
[manifest]: https://github.com/szl-holdings/szl-forge/blob/56a00821858825f529c40c7322c2f1584608d6e5/ops-health/v2/data/MANIFEST.json
[source]: https://github.com/szl-holdings/szl-forge/blob/56a00821858825f529c40c7322c2f1584608d6e5/ops-health/README.md
[counts]: https://github.com/szl-holdings/szl-forge/blob/56a00821858825f529c40c7322c2f1584608d6e5/ops-health/logs/drafts/split_positive_counts.txt
[dupes]: https://github.com/szl-holdings/szl-forge/blob/56a00821858825f529c40c7322c2f1584608d6e5/ops-health/logs/drafts/dup_check_from_scores.txt

## How to run

Run only with authored synthetic input, using the files above:

```bash
python -I -B ops_health.py --model model.json --receipt artifact_receipt.json --input example_input.json
```

The command emits an advisory for local inspection. It has no network,
device-control, result-delivery, or authorization effect.

<!-- SZL-NATIVE-ARTWORK:v1 -->
SZL Orbit artwork: [source](https://github.com/szl-holdings/szl-brand/blob/27d12e77fc0f9f1a0e5fae8b34adf2380d51d070/kit/logos/orbit-v2/suite/szl-icon-32.svg) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Software and artifact licensing remain declared above.
<!-- /SZL-NATIVE-ARTWORK:v1 -->
