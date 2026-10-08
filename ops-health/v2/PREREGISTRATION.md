# OAC Ops Health v2 — Preregistration

Status: REGISTERED before any v2 training code, v2 data, or v2 evaluation exists.
Registration evidence: the local git commit that first adds this file (see
`v2/PREREGISTRATION.receipt.json`). Any later change is an explicit, dated amendment in
section 12, made before the sealed splits are opened, or it is a protocol deviation that
REPORT.md must disclose.

No pilot was run. No v2 candidate has been trained or scored on anything. The only data seen
before registration is the public v1 artifact (Hub revision
`dd7d109813abcd90c5250106dc2eabd2804e7ac3`), its receipt, and the v1 generator source
(`szl-forge` commit `9a6fdd2c`, `tools/train_operational_health_model.py`).

Everything here is SYNTHETIC. No result from this protocol may be generalized to any real
transport, device, site, population, or workflow.

## 1. Question

Does a v2 operator-attention advisory, trained and selected on synthetic operational transport
telemetry, beat v1 as shipped and two trivial baselines on balanced accuracy? The comparison
covers held-out nominal data and a pooled suite of pre-registered distribution shifts, without
worse calibration.

## 2. Frozen facts about the data-generating process (generator truth)

Read from the v1 generator source. The eight fields are normalized exactly as in the v1 kernel:
binary fields as 0/1, and continuous fields min-max over the declared ranges
queue_utilization [0,1], consecutive_failures [0,20], and seconds_since_last_success [0,86400].
Labels are Bernoulli draws from p(x) = sigmoid(z), where

```
z = -3.6 + 3.1(1-listener_running) + 1.15(1-tls_enabled) + 1.35(1-peer_allowlist_configured)
    + 4.2 queue_utilization + 2.7 consecutive_failures + 2.2 seconds_since_last_success
    + 4.4(1-ledger_integrity_ok) + 3.4(1-configuration_valid)
```

Equivalent parameterization, directly comparable to a fitted model:
intercept 9.8, with weights

| feature | true weight |
| --- | ---: |
| listener_running | -3.10 |
| tls_enabled | -1.15 |
| peer_allowlist_configured | -1.35 |
| queue_utilization | +4.20 |
| consecutive_failures | +2.70 |
| seconds_since_last_success | +2.20 |
| ledger_integrity_ok | -4.40 |
| configuration_valid | -3.40 |

Consequence, recorded in advance (MODELED): the model class, logistic regression on these
features, is correctly specified. Nominal seconds_since_last_success follows Exp(mean 900 s) on a
0–86400 s scale, so its normalized values are near 0 and its true weight is weakly identifiable.
We expect underspecification on that coordinate.

## 3. Generator v2 (`ops-health-generator/2.0.0`)

- The nominal regime uses the same per-row feature draws, in the same order, as the v1 generator,
  and the label rule above. A parity test must show that the v2 generator, fed v1's seed (2500)
  and sequential split sizes (768/192/240), reproduces the v1 public rows byte for byte. Any
  mismatch blocks training.
- Seeds. The master seed is M = 20260926. Each split s gets its own `random.Random`, seeded with
  `int.from_bytes(sha256(f"oac-ops-health-v2:{M}:{s}".encode()).digest()[:8], "big")`.
- Shift regimes are never used for training, calibration, conformal calibration, or selection.
  Labels are always drawn from the truth rule applied to the true (latent) features.
  1. `queue_saturation`: queue_utilization ~ Beta(4.0, 1.5), capped at 1.0. Everything else nominal.
  2. `failure_bursts`: with probability 0.35, consecutive_failures ~ UniformInt(3, 20); otherwise
     the nominal draw.
  3. `clock_skew`: true seconds come from the nominal draw. The observed value given to the model
     is min(86400, true + skew), with skew ~ Uniform(0, 7200) s (monitoring clock ahead). The label
     uses the true value.
  4. `missing_tls`: tls_enabled ~ Bernoulli(0.30), instead of the nominal 0.90.
- The generator source hash and every data file's SHA-256 are written to `v2/data/MANIFEST.json`
  and committed before any training run. Test and shift files live under `v2/data/sealed/`. The
  training, selection, and calibration code must refuse any path under `sealed/`.

## 4. Splits (fixed)

| split | regime | rows | use |
| --- | --- | ---: | --- |
| train | nominal | 16,384 | fit weights |
| calibration | nominal | 4,096 | fit the Platt/isotonic calibrator only |
| conformal | nominal | 4,096 | split-conformal quantiles only |
| validation | nominal | 4,096 | hyperparameter, calibrator, and threshold selection only |
| test | nominal | 50,000 | SEALED; opened once |
| shift_queue_saturation | shift 1 | 12,500 | SEALED |
| shift_failure_bursts | shift 2 | 12,500 | SEALED |
| shift_clock_skew | shift 3 | 12,500 | SEALED |
| shift_missing_tls | shift 4 | 12,500 | SEALED |
| legacy_v1_test | v1 public test split | 240 | secondary only (REPORTED comparability) |

The pooled shift suite is the concatenation of the four shift splits, 50,000 rows, with equal
regime weights.

## 5. Contenders

- **v2**: the model chosen by section 6, with its calibrator, validation threshold, and conformal
  layer, frozen before the test split is opened.
- **v1 as-is**: the Hub kernel and model at revision `dd7d1098…`, threshold 0.16, scores exactly as
  the kernel emits them.
- **majority**: always NO_ALERT. Balanced accuracy is 0.5 by construction.
- **rule**: ALERT iff consecutive_failures > 0.

## 6. v2 training and selection (validation only)

- The model is L2-regularized binary logistic regression on the eight normalized features, with
  an unpenalized intercept. It is fit by Newton/IRLS in the Python standard library with
  deterministic operation order. Convergence: max |Δθ| < 1e-10 or 100 iterations; hitting the cap
  must be reported.
- The fixed search space has 12 trials: l2 ∈ {0, 1e-4, 1e-3, 1e-2} (penalty λ/2·‖w‖² on the mean
  log-loss) × calibrator ∈ {none, platt, isotonic}. Platt is 2-parameter logistic regression on the
  logit of the score, fit by Newton. Isotonic is pool-adjacent-violators with ties pooled, using
  interpolation-free step prediction.
- The loop is bounded, following the ouroboros contract as mirrored in `math/ouroboros_adapter.py`.
  Hard cap: 12 iterations. The only stop reasons are SEARCH_SPACE_EXHAUSTED and CAP_REACHED. Each
  trial emits one governed receipt to `runs/receipts/` (governed-receipt-spec format). Receipts
  attest integrity and origin, never quality.
- The decision threshold for each trial is chosen on validation over t ∈ {0.01, 0.02, …, 0.99}.
  It maximizes balanced accuracy; ties go to higher F1, then to t closer to 0.5.
- Trial selection maximizes validation BA at the trial's own threshold. Ties go, in order, to
  lower validation Brier, lower validation ECE, larger l2, then calibrator order
  none < platt < isotonic.
- Conformal layer: Mondrian (label-conditional) split conformal on the conformal split, α = 0.10.
  The nonconformity score is s(x,y) = 1 − p̂_y(x). For each class y, q̂_y is the
  ⌈(n_y+1)(1−α)⌉-th smallest score among conformal rows of class y. The prediction set is
  C(x) = {y : s(x,y) ≤ q̂_y}. The operational advisory is ALERT iff C = {1}, NO_ALERT iff C = {0},
  and ABSTAIN otherwise; the reason is BOTH or EMPTY. The guarantee is class-conditional coverage
  of at least 1−α under exchangeability, which holds on nominal data only. Coverage under shift is
  descriptive.

## 7. Primary metric and win condition

The primary metric is balanced accuracy (BA) of each contender's binary decision. For v2 that is
the validation-selected threshold decision, not the three-way advisory. It is computed on
(a) the nominal test split and (b) the pooled shift suite.

Uncertainty: 2,000 bootstrap resamples, seed 20260927, percentile 95% intervals (2.5th and 97.5th
percentiles). Row indices are resampled with replacement, and the same resample indices are shared
across contenders (paired). The pooled shift suite is resampled within each regime.

v2 **wins** only if all of these hold:
- **W1**: on nominal test, v2's BA interval lower bound is strictly greater than the BA interval
  upper bound of v1, majority, and rule.
- **W2**: the same holds on the pooled shift suite.
- **W3**: calibration is not worse. On nominal test, Brier(v2) ≤ Brier(v1) and ECE(v2) ≤ ECE(v1),
  as point estimates, with ECE over 10 equal-width bins (last bin closed) weighted by count.

If any of W1–W3 fails, v2 **does not win**. REPORT.md must then say so plainly, and nothing may
describe v2 as better.

## 8. Secondary metrics (reported, never part of the win condition)

On nominal test, each shift regime, the pooled shift suite, and legacy_v1_test:
- Brier, ECE with a reliability table, log loss, and ROC AUC.
- Precision, recall/sensitivity, specificity, F1, and false-alert share = FP / (TP + FP).
- Conformal marginal and per-class coverage, abstention rate (split into BOTH and EMPTY), and
  selective BA on non-abstained rows.
- Paired-bootstrap intervals of the differences v2 − v1 in BA, Brier, and ECE.
- Disaggregated BA by regime, and on nominal test by tls_enabled and by listener_running.
- Generator-truth recovery for v1 and v2: sign agreement (k/8), Spearman correlation of weights
  against the true weights, and relative L2 error of (intercept, weights).

## 9. Ablations

All ablations use identical splits and seeds. Keep/drop decisions are made on validation; the
test values are reported as confirmation.
- ABL-data: the v2 method trained on the first 768 rows of the v2 train split versus the full
  16,384 rows.
- ABL-optimizer: v1's exact algorithm (batch gradient descent, lr 0.42, 2,400 epochs, l2 0.01)
  retrained on the full v2 train split, versus v2's IRLS with the selected l2.
- ABL-calibration: the selected model with and without its calibrator. This is decided by the
  search itself, since `none` is in the grid.
- ABL-conformal: selective metrics with and without abstention. Keep the conformal layer only if
  validation coverage is ≥ 0.88 marginal and ≥ 0.85 for each class.

## 10. Validity gates, mutation suite, and broken baseline

- **Baseline validity (INVALID_BASELINE)**: sensitivity, specificity, precision, or false-alert
  share are never reported for a fixture unless (i) its receipt verifies, (ii) the evaluation set
  contains at least 30 rows of each class, (iii) its decisions are not constant on that set, and
  (iv) it refused every fail-closed probe. Otherwise it gets `INVALID_BASELINE` with the failed
  precondition(s). Two deliberately broken accepting fixtures must trigger this: a constant-ALERT
  model, and a fail-open kernel that accepts prohibited or malformed input. The majority baseline
  is also constant, so it reports BA = 0.5 (by construction) and INVALID_BASELINE for per-class
  rates.
- **Mutation suite**, applied to the frozen v2 artifact:
  - weight scaling ×0.5 and ×1.5 per feature
  - a sign flip per feature
  - dropping each feature (weight set to 0)
  - intercept ±1
  - threshold ±0.1
  - receipt corruption: a model-hash byte flip, a wrong schema, an altered source commit, and
    tampered metrics
  - kernel-source mutants: a removed finiteness check, a removed range check, removed
    prohibited-key screening, and an inverted hash comparison
  - A mutant is CAUGHT if the integrity layer or the test suite fails. Artifact mutants are also
    evaluated a second time with a consistently regenerated receipt (integrity bypassed); they must
    then be caught by a behavioral gate. Any mutant that no layer catches is reported as
    BLIND_SPOT.
- **Behavioral gates** (known-good reference = v2), each with a value in [0,1]:
  - `g_integrity`: receipt verifies
  - `g_failclosed`: fraction of probes refused
  - `g_truth_sign`: sign agreement with generator truth, k/8
  - `g_monotone`: fraction of pre-declared metamorphic monotonicity checks passed. Raising
    consecutive_failures, queue_utilization, or seconds_since_last_success must not lower the
    score. Switching a health flag from 1 to 0 must not lower the score.
  - `g_validation`: min(1, BA_val / 0.70)
  - `g_calibration`: 1 if validation ECE ≤ 0.05, else max(0, 1 − (ECE − 0.05)/0.10)
  - `g_baseline_valid`: the section 10 preconditions hold

## 11. Readiness aggregation: Λ versus baselines

Known-good fixtures are v2 plus five retrains of the selected recipe on train splits drawn from
seeds `M+1..M+5`, generated with the same regime and sizes. Known-broken fixtures are every
integrity-bypassed artifact mutant. Four aggregators combine the gate vector: AND (all gates = 1),
min, the equal-weight mean, and Λ. Λ is exactly as defined in `math/lambda_reference.md`, frozen
at szl-lambda-gate commit `99bc2cf1` and implemented in `math/lambda_stdlib.py`. The readiness cut
is aggregate ≥ 0.99 (AND: all gates = 1). Separation is measured by:
1. misclassifications at the cut (broken fixtures marked READY, plus good fixtures marked
   NOT_READY)
2. the margin min(good) − max(broken)
3. the AUROC of good versus broken

Λ is kept only if it is strictly better than every baseline on (2), with no more misclassifications
on (1). Otherwise it is recorded as REJECTED, with the numbers. No Λ property is called "proven"
unless it is a closed lutar-lean theorem with zero sorries in its dependency cone and a local
`lake build` output is attached.

## 12. Amendments

None yet.
