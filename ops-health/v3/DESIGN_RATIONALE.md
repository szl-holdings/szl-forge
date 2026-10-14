# OAC Ops Health v3: design rationale

Status: DRAFT companion to `v3/PREREGISTRATION.DRAFT.md`, which is NOT registered. Both files may
change before registration, and every change is a commit. Stephen gave the green light on
2026-10-13; the decisions are recorded in `v3/DECISIONS_FOR_GREEN_LIGHT.md` and prereg §18.

This file explains each choice in `v3/PREREGISTRATION.DRAFT.md`: why it was made, which v2
evidence it answers, and which literature anchor supports it. It also says what v3 does not
attempt. It holds **no v3 result**: no v3 code, data or fit exists. Every number about v3 is a
prediction or an approximation, labeled MODELED or CONJECTURE. Everything is SYNTHETIC.

This revision answers two adversarial reviews of the first draft (evidence:
`logs/verify/v3-prereg-review-a/modeled_arithmetic_20261013.txt`, MODELED arithmetic; and
`logs/verify/v3-prereg-review-b/checks_20261013.txt` and `ledger_20261013.txt`, MEASURED greps
and MODELED reasoning). `v3/REVIEW_DISPUTES.md` records where a fix differs from the one a
reviewer suggested. A later revision of 2026-10-13 records the green-light decisions and replaces
the Stephen-held salt with a future drand beacon round (D1, D7, D15). A revision of 2026-10-14
answers the adversarial review of that amendment: the timing proof now rests on durable records
(an OpenTimestamps proof anchored in Bitcoin, and a public archive record), not on GitHub's
expiring event records (D13, D15; `v3/REVIEW_DISPUTES.md`, second part).

## 0. Conventions

- **Labels.** A claim about v2 carries the label it has in the cited file:
  - MEASURED: computed there from saved outputs;
  - REPORTED: read from the cited file;
  - MODELED: reasoning;
  - CONJECTURE: reasoning with weak support;
  - UNKNOWN.

  The arithmetic in this file is MODELED unless marked otherwise.
- **Paths.** Paths are relative to the research root, which is `ops-health/` in the public home.
  Of the drafting-only files cited here, Stephen decided on 2026-10-13 that `literature/*`,
  `drafts/forum_thread.md` and the review evidence under `logs/verify/v3-prereg-review-a/` and
  `logs/verify/v3-prereg-review-b/` are published under `ops-health/v3/evidence/`. `REPORT.md`,
  `COORDINATION.md` and `LANDING_PLAN.md` stay internal; on 2026-10-14 each citation of them was
  re-cited to a public source, or relabeled UNKNOWN (prereg preamble and §14.2). So were the
  other non-public paths that prereg §14.2 lists, `CONTRACT.md` among them. The raw drand
  responses under `logs/v3/register/`, and the review and fix evidence under
  `logs/verify/v3-register-review/` and `logs/verify/v3-register-fix/`, are drafting copies; each
  claim also names the relay URL or public service it was read from.
- **Literature.** Anchors come from `literature/anchors.json`, and verdicts from
  `literature/evidence_matrix.md`, cited as "EM-n" for row n. Those verdicts are MODELED in that
  file.
  - An anchor whose `key` names a topic is cited by key, for example `anchor:underspecification`.
  - An anchor whose `key` contains a person's name is cited by its EM row and the DOI or arXiv id
    from the same anchor record. This program reproduces no personal names outside the literature
    folder (REPORTED: `literature/leaders.md`, header: nobody listed there is mentioned in any
    artifact outside that folder).
- **What an anchor does here.** It supports a choice; it does not certify it. The matrix cells
  marked [M] were not checked against full texts (UNKNOWN: `literature/VERIFICATION.md` §11).

## 1. From each v2 weakness to the v3 response

| v2 weakness (label, file) | root cause (MODELED) | v3 response | where |
| --- | --- | --- | --- |
| W1 against v1 won by 0.000319 BA (MEASURED, `v2/results/FINAL_RESULTS.md`) | two near-ceiling models compared by non-overlap of marginal intervals, which is a strict rule | paired non-inferiority to v2, plus paired superiority to v1, majority and rule | prereg §8; D10 |
| Clock skew: paired v2 − v1 BA +0.0066 [−0.0013, +0.0146] (MEASURED, same file) | both models nearly ignore seconds, so the shift barely moves either; and 12,500 rows give wide intervals | a dedicated 100,000-pair twin set; non-inferiority on the skew cost of identifying the weight | H3; D8 |
| seconds_since_last_success weight +0.036548 against truth +2.2; wrong sign in 3 of 5 known-good retrains (weights REPORTED, gate failures MEASURED, `v2/results/MUTATION_READINESS.md`) | two causes: the sign is an information limit at 16,384 nominal rows; the magnitude comes from the L2 penalty biting on a [0, 86400] min-max scale | training design D (sign) and rescale R (magnitude), in a 2×2 ablation | H1; D3, D4 |
| 11 BLIND_SPOT mutants; every readiness margin negative (MEASURED, same file) | gates with absolute tolerances cannot see moderate changes; the origin check binds only blobs that a parent commit shares | reference-relative parameter and decision gates with tolerances from 100 band retrains; an origin rule bound to a tag and the run-time HEAD; a registered kernel contract | H2; D5, D6, D7, D14 |
| Λ REJECTED (MEASURED, same file) | with uniform weights Λ lies between min and mean | Λ excluded | prereg §12; D12 |
| About 41% abstention (MEASURED, `v2/results/FINAL_RESULTS.md`) | class overlap of the generator; Mondrian sets do not change under recalibration | abstention at fixed per-class coverage, against a TRUTH floor; non-gating | H4; D9 |
| Queue saturation is the worst regime for v2 (0.5928), v1 (0.5311), ABL-data (0.5842) and ABL-optimizer (0.5693); not for rule or majority (MEASURED, `v2/results/FINAL_RESULTS.md`) | the nominal-chosen threshold is misplaced under a large covariate and prevalence shift | measured again; not attempted | §5 |
| Origin layer unavailable publicly (source commit absent from szl-forge, MEASURED by the public API on 2026-10-14; 7 skip sites, REPORTED from the public v2 tests; prereg §11.1, §14.5) | v2's receipt names an unpublished research commit | public branch, tags, and a rule that a skipped origin test fails | prereg §14; D7 |

## 2. Decisions

### D1. Same truth, fresh seeds, beacon-salted seal

- **Decision.**
  - Reuse v2's generator truth and regimes unchanged.
  - Use a new namespace and master seed (M3 = 20261013).
  - Generate sealed splits only at the opening, from a salt that nobody holds: the randomness of
    drand round R = 6578966 of the default mainnet chain, scheduled 2026-10-24T00:00:00Z, which
    is fixed at registration (prereg §3.4; reasons and limits in D15).
- **v2 evidence.**
  - v2's sealed results are public (MEASURED: `v2/results/FINAL_RESULTS.md`), so reusing v2's
    sealed splits would test v3 on data whose outcomes informed its design.
  - v2's compliance audit could not show that no program outside the guard read a sealed file
    early (UNKNOWN: `v2/results/COMPLIANCE.md`).
- **Literature.**
  - EM-7 (doi:10.1016/j.patter.2023.100804, ADOPT): split disjointness and test-free choices.
  - EM-8 (doi:10.1145/2382577.2382579, ADOPT): say what the label is a function of. Here it is
    the unchanged truth of prereg §2.
  - `anchor:reforms` (EM-5, ADAPT): audit checklist.
- **Reasoning (MODELED).**
  - Fresh seeds remove reuse of the test sample. They do not remove tuning to the distribution,
    which is v2's (prereg §17).
  - The salt turns "the code refuses sealed files" into "the sealed rows do not exist yet". That
    holds against the research process itself only if nobody in it can know SALT before the
    frozen state is public. The first drafts met this with a secret that Stephen would generate
    and hold, which made the guarantee only as good as one person's custody, and nobody else
    could audit it. Stephen supplied no salt. A future beacon round meets the same need with no
    custody: SALT does not exist until the round's scheduled time, and the frozen state must be
    fixed and public before then, as durable public records show (prereg §3.4, §3.5; D15).
  - Keeping the truth fixed keeps v3 and v2 on the same task, so every v3 − v2 difference is about
    the model, not the data-generating process.
- **Rejected.**
  - A new truth or new fields: v3 is defined to add no new truth (prereg §2), and a new truth
    would make v3 against v2 a comparison across tasks.
  - Reusing v2's non-sealed splits: they are v2's development data, and v3 decisions use v3 data
    only.

### D2. Fixed hyperparameters, no search

- **Decision.** Inherit v2's selected l2 = 1e-3 and calibrator `none`. Run no search.
- **v2 evidence.**
  - The 12 v2 trials span 0.756510 to 0.768680 validation BA.
  - The top two differ by 0.0005, which v2 calls "far inside the sampling noise".
  - Platt calibrated better on validation, but the BA-first rule picked `none`.
  - The lowest validation Brier was T05 (l2 1e-4, Platt; 0.101172).

  (MEASURED values and MODELED reading: `v2/results/STAGE2_SEARCH.md`.)
- **Literature.** `anchor:underspecification` (EM-10, ADAPT): near-equal predictors can behave
  differently, so report spread across seeds and the grid.
- **Reasoning (MODELED).**
  - A new search on v3 data would choose among near-ties by noise. v3 − v2 differences would then
    mix the targeted changes with a random hyperparameter change.
  - Fixing l2 and the calibrator isolates D and R.
  - It keeps v2's penalty, but not exactly v2's shrinkage of the other weights. Those weights are
    0.64 to 0.83 of truth in v2 (REPORTED weights: `v2/results/MUTATION_READINESS.md`; truth:
    `v2/PREREGISTRATION.md` §2). Under D, the rows with wide seconds have higher scores and so a
    higher information E[p(1−p)]; the penalty then shrinks the other weights a few percent less
    than in v2 (MODELED, rough).
  - That small systematic difference matters most under queue saturation. There, v2's nominal
    threshold sits far from that regime's BA optimum (v2: BA 0.5928, specificity 0.2269;
    MEASURED: `v2/results/FINAL_RESULTS.md`), so BA reacts at first order to small offsets of the
    scores or the threshold. A v3 − v2 shift there is a named P2 risk (D10).
  - Whether v2's shrinkage acts as accidental robustness under queue saturation is not known
    (CONJECTURE).
- **Rejected.** Selection by validation log loss, a strictly proper score (EM-11,
  doi:10.1198/016214506000001437, ADOPT). It is attractive, but it would likely move l2 and
  confound H1 to H3. It is a candidate for a later version.

### D3. Training design D: make the data constrain the staleness weight

- **Decision.** In the training split only, a quarter of the rows (π = 0.25) draw their true
  seconds from U(0, 86400), the field's declared range. Labels follow the unchanged truth.

**v2 evidence.**

- Nominal seconds follow Exp(mean 900 s) on a 0 to 86400 scale, so v2 §2 predicted weak
  identifiability (REPORTED, MODELED there: `v2/PREREGISTRATION.md` §2).
- v2's weight is +0.036548 (REPORTED: `v2/results/MUTATION_READINESS.md`).
- k1 to k3 have negative weights (weights REPORTED from each `model.json`; the gate failures they
  cause MEASURED: same file).
- The generator draws each field independently (REPORTED: `v2/research/generator.py`).

**Why the sign flips: an information limit (MODELED, a rough approximation).**

- The true effect of seconds, per standard deviation of the nominal draw, is
  β = 2.2 × 900 / 86400 ≈ 0.0229 logits.
- For an independent feature, the standard error of the estimate per standard deviation is
  about 1/√(n·w̄), where w̄ = E[p(1−p)].
- w̄ is at most the Brier score of a near-truth model. v2's nominal-test Brier is 0.1013
  (MEASURED: `v2/results/FINAL_RESULTS.md`), so w̄ ≈ 0.10.
- At n = 16,384: SE ≈ 1/√1638 ≈ 0.0247, and β/SE ≈ 0.93. P(wrong sign) ≈ Φ(−0.93) ≈ 0.18 per
  retrain, which predicts about 0.9 wrong signs in 5.
- v2 observed 3 of 5 (weights REPORTED: `v2/results/MUTATION_READINESS.md`). That count lies in
  the upper tail of this approximation (P(≥ 3 | 0.18) ≈ 0.04). The approximation treats w̄ as
  constant, so it is rough. Either way, the sign is not reliably identifiable at this size.
- To get 20 of 20 correct signs with probability 0.95 from nominal rows alone, each retrain needs
  β/SE ≥ 2.80. That is n ≈ (2.80/0.0229)²/0.10 ≈ 149,000 rows, about 9 times v2's train split.

**Why D works (MODELED).**

- With π = 0.25 the seconds variance is 0.75·E_nom[s²] + 0.25·E_U[s²] − E[s]². That gives
  σ ≈ 22,170 s, so β ≈ 2.2 × 22,170 / 86400 ≈ 0.565 logits per standard deviation.
- At n = 32,768, β/SE ≈ 32 (at 16,384 it is about 23). A wrong sign is then negligible.
- The relative standard error of the magnitude is about 3%. On the R scale (D4) that is
  SD(r) ≈ 0.03, against H1's limit of 0.10.
- Even π = 0.05 would give β/SE ≈ 11 at 16,384 rows. π = 0.25 is chosen for margin and was not
  tuned on any data. It is frozen at registration (prereg §15.3).

**Why D is valid and is not training on a shift (MODELED).**

- The model class is correctly specified, and labels come from the truth. Changing the covariate
  design of the training rows therefore does not bias the fitted conditional model.
- The threshold and the conformal quantiles are still fit on nominal splits.
- The four sealed shifts are covariate shifts in queue utilization, failures and TLS, plus a
  measurement error in seconds (REPORTED: `v2/PREREGISTRATION.md` §3). D changes true seconds
  with matching labels, so it is none of them.
- If anything, D works against v3 under clock skew: it teaches "more seconds, more attention",
  which a skewed clock triggers falsely. H3 measures exactly that cost.

**Literature.**

- `anchor:underspecification` (EM-10).
- `anchor:shortcut_learning` (EM-9, ADAPT): test with per-feature ablations and pre-registered
  regimes, not new machinery. The D and R 2×2 is such an ablation.

**Rejected.**

- **About 149,000 nominal rows.** This identifies the weight too, but at 4.5 times v3's train
  size, with more disk, and it tests no design idea.
- **A sign constraint.** It makes `g_truth_sign` and H1 pass by construction, which is circular.
- **Dropping the field.** The truth weight is 2.2, so dropping it misspecifies the model wherever
  seconds are large. On nominal rows it costs nothing measurable, which is why it is tempting and
  why it would test nothing.
- **log1p(seconds).** The truth is linear in seconds, so log1p misspecifies the model, worst at
  large seconds, which D now covers. It would also break the truth comparisons.
- **Robust scaling with median and IQR estimated from data.** It is linear, so it acts only
  through the penalty (D4). It also adds a fitted preprocessing step, and EM-7 hygiene prefers
  fixed constants.

### D4. Representation R: seconds divided by 900

- **Decision.** Map seconds to s/900, with 900 s the generator's declared nominal mean. Keep v1's
  min-max scaling for the other continuous fields.

**Reasoning (MODELED).**

- **A linear rescale cannot change the sign.** The unpenalized estimate is equivariant under
  linear rescaling: x → a·x maps ŵ → ŵ/a and leaves every fitted probability unchanged. So
  rescaling cannot fix the sign flips. This is the core reason D, not R, carries H1's sign
  criterion.
- **The rescale does change the penalty's bite.**
  - On the min-max scale, the per-row curvature of the mean log loss along the seconds weight is
    about w̄·Var(x) = 0.10 × 0.0104² ≈ 1.1e-5, against λ = 1e-3.
  - The penalty therefore shrinks the weight to about 1.1e-5 / (1.1e-5 + 1e-3) ≈ 1% of its
    unpenalized value, so the expected weight is about 0.024 ± 0.025.
  - v2's +0.0366 fits this account (REPORTED weight: `v2/results/MUTATION_READINESS.md`).
- **On s/900 the curvature is about 0.10, so the penalty removes only about 1% of the weight.**
  With D the curvature is about 60, and the shrinkage vanishes.
- **So the "learned about 0.037 against 2.2" gap is a scaling-and-penalty effect, which R fixes.
  The sign flips are an information effect, which D fixes.** The 2×2 ablation of prereg §10
  separates the two.

**Why 900, and why seconds only (MODELED).**

- 900 s is a constant of the generator (REPORTED: `v2/research/generator.py`), not an estimate.
  It puts nominal values on the order of 1, like the other features.
- Changing only seconds keeps one change per weakness. The other weights keep v2's penalty, and
  their shrinkage changes only slightly under D (D2).

### D5. Training size 32,768

- **Decision.** Double v2's train size. ABL-n measures what 16,384 would have given.
- **Reasoning (MODELED).**
  - The coefficient band of H2 must separate a ×0.5 or ×1.5 mutant from the difference between
    two known-good fits.
  - The detection ratio is ρ_j = 0.5·|w_j| / (√2·SD_j), and it must exceed c = 4.
  - SD_j uses the penalized sandwich approximation SD_j ≈ √(I_j / (I_j + λ)² / n), with
    I_j ≈ w̄·Var(x_j).
  - The variances come from the generator (REPORTED: `v2/research/generator.py`):
    - listener_running 0.94 × 0.06 = 0.0564; tls 0.9 × 0.1 = 0.09; peer 0.93 × 0.07 = 0.0651;
      ledger 0.985 × 0.015 = 0.0148; configuration 0.96 × 0.04 = 0.0384;
    - queue Beta(1.25, 4.5) ≈ 0.0252;
    - consecutive_failures/20 ≈ 0.0151, from v1's weighted draw;
    - seconds under D on the R scale ≈ 607.
  - w_j are v2's weights, because the l2 is the same (REPORTED:
    `v2/results/MUTATION_READINESS.md`).
  - The four binary rows other than tls are new in this revision. They use the same
    independent-feature approximation, which is rougher for rare binary values, because
    p(1−p) differs on those rows.

| weight | ρ at n = 16,384 | ρ at n = 32,768 |
| --- | ---: | ---: |
| listener_running (w ≈ −2.54) | 10.2 | 14.4 |
| tls_enabled (w ≈ −0.80) | 3.8 | 5.4 |
| peer_allowlist_configured (w ≈ −1.04) | 4.4 | 6.2 |
| queue_utilization (w ≈ 2.71) | 8.6 | 12.1 |
| consecutive_failures (w ≈ 1.72) | 5.0 | 7.1 |
| seconds, with D (w ≈ 0.0229 per 900 s) | 8.1 | 11.4 |
| seconds, without D | 0.33 | 0.47 |
| ledger_integrity_ok (w ≈ −2.85) | 8.3 | 11.8 |
| configuration_valid (w ≈ −2.82) | 10.0 | 14.1 |

- **Why 16,384 is not enough.**
  - At 16,384, tls has ρ ≈ 3.8, below c = 4, so the band misses tls scaling unless the SD
    estimate happens to be low; with 20 band retrains the miss probability is about 0.58
    (MODELED: review A evidence §1). The failures margin is thin there too.
  - At 32,768, every registered ×0.5 / ×1.5 class has ρ ≥ 5.4.
  - Without D, no training size in this range lets the band see seconds scaling.
- **Cost.**
  - The train split is about 11 MB, at about 330 bytes per row (CONJECTURE).
  - Retrain splits are regenerated from seeds and not stored.
  - IRLS on 32,768 × 9 is cheap; the 227 declared fits are the larger compute item (D10).
- **Rejected.** Keeping 16,384 rows with c = 3.5. With 11 statistics per fixture and 20 known-good
  fixtures, the chance of a false alarm would be about 10% even with the SD known, and far more
  with it estimated (P(|t₁₉| > 3.5) ≈ 2.4e-3 per statistic; MODELED: review A evidence §1), so
  H2's zero-misclassification criterion would fail by chance too often.

### D6. Gate set and tolerance rule

- **Decision.**
  - Keep v2's seven gates.
  - Add `g_coef_band` and `g_threshold_band` (parameter space) and `g_ba_rel` (outputs), all
    relative to the frozen v3.
  - Set tol = max(floor, 4·√2·SD) from **100 band retrains**, which are separate from the 20
    known-good retrains used to count false alarms. The threshold tolerance is an integer number
    of grid steps.
  - Freeze the rule, c, the floors and the band count at registration, and commit the values
    before any mutant exists.

**v2 evidence.**

- The blind spots keep every sign and monotonicity. `g_validation` and `g_calibration` are
  absolute (BA ≥ 0.70, ECE ≤ 0.05). The largest blind-spot BA loss was 0.060146 (MEASURED:
  `v2/results/MUTATION_READINESS.md`).
- Known-good thresholds were 0.18 to 0.20 (MEASURED: `v2/results/STAGE2_SEARCH.md`).
- Known-good validation BA had a standard deviation of about 0.00128, computed from the five
  values in `v2/results/MUTATION_READINESS.md` (MEASURED values; arithmetic MODELED).
- Known-good validation ECE was 0.026614 to 0.030538, and v2's own 0.034658 (MEASURED:
  `v2/results/MUTATION_READINESS.md`, `v2/results/STAGE2_SEARCH.md`).

**Literature.**

- EM-28 (doi:10.1109/tse.2010.62, ADOPT): mutation testing.
- `anchor:deepmutation` (EM-29, ADAPT): model-level mutants to check that receipts and evaluation
  gates catch altered models.
- `anchor:deepcrime` (EM-30, ADAPT): training-level operators. They inform the descriptive
  recipe-substitution mutants.
- EM-31 (doi:10.1016/j.jss.2010.11.920, ADOPT): metamorphic relations. `g_monotone` is kept.
- `anchor:ml_testing_survey` (EM-32).
- `anchor:underspecification`: spread across seeds as the scale of "normal".

**Reasoning (MODELED).**

- **Why reference-relative.** A modified artifact is modified relative to the frozen one. An
  absolute floor lets anything above BA 0.70 through.
- **Why tolerances from band retrains.** The recipe's own seed-to-seed spread is the only
  principled scale for "not what this recipe produces", and it can be measured before any mutant
  exists. This answers the forum draft's third question (§3).
- **Why 100 band retrains, not 20 (the first draft's error).** Each tolerance uses an SD
  estimated from the band retrains, and all 20 known-good fixtures share that SD and the same
  reference. With 20 band retrains, (known-good − reference)/(√2·SD) follows t with 19 degrees of
  freedom: P(|t₁₉| > 4) ≈ 7.7e-4, twelve times the normal value of 6.3e-5. The first draft
  quoted only the normal value, so it put the known-good false-alarm chance at 1–2%. With the
  shared reference and SD, the chance of at least one false alarm on the nine parameter bands
  over 20 known-good fixtures is about 0.087 with 20 band retrains, about 0.0185 with 100, and
  about 0.0105 with the SD known (MODELED: review A evidence §1; review B's independence
  approximation gave 10–20% for 20 retrains, ledger §C).
- **The false-alarm budget.** The target is at most 0.05 over all fixtures and statistics (prereg
  §11.3). At c = 4 with 100 band retrains:
  - nine parameter bands: about 0.0185;
  - threshold band, in integer steps with a floor of 2: about 0.001–0.01 (rough);
  - BA band, with its 0.005 floor: small;
  - retained absolute gates: `g_integrity`, `g_failclosed` and `g_baseline_valid` do not vary
    across retrains of one kernel; `g_truth_sign` and `g_monotone` fail only with a wrong sign,
    which D makes negligible; `g_validation` (BA ≥ 0.70 against about 0.766) is about 0;
    `g_calibration` (ECE ≤ 0.05) is about 0 if v3's ECE lies near v2's known-good range, but v3's
    level is UNKNOWN before the fit.

  In all, about 0.02–0.03. This, not detection, is H2's main risk.
- **Why integer threshold steps.** On the float grid i/100, 67 of 96 two-step pairs give
  |Δt| > 0.02 and 29 give ≤ 0.02, for example 0.21 against 0.19 gives 0.01999999999999999
  (MEASURED by a calculator: review A evidence §6). A tolerance in integer steps removes that
  dependence on where a difference sits on the grid.
- **Why θ on the registered scale.** `g_coef_band` maps θ to representation R with the
  artifact's declared table. On the model's own scale, an artifact that changes only the declared
  seconds scale, with a consistent receipt, would leave θ unchanged and pass (review B ledger §D).
  The kernel also refuses any table other than the registered one (D14).
- **Detection with 100 band retrains.** A ×0.5 tls mutant escapes the band only if the SD
  estimate exceeds the true SD by 1.35 times. With 99 degrees of freedom that chance is about 0
  (with 19 it was about 0.016; review A evidence §1).
- **Expected catch per blind-spot class.** This draws on v2's measured validation BA drops
  (MEASURED: `v2/results/MUTATION_READINESS.md`) and the ρ table of D5.

| class | v2 BA drop (×0.5 / ×1.5, or +0.1 / −0.1) | expected catching gate(s) |
| --- | --- | --- |
| tls_enabled scaling | 0.0139 / 0.0232 | `g_coef_band` (ρ ≈ 5.4); `g_ba_rel` (tolerance about 0.005 to 0.008, CONJECTURE) |
| queue_utilization scaling | 0.0315 / 0.0149 | both |
| consecutive_failures scaling | 0.0063 / 0.0024 | `g_coef_band` (ρ ≈ 7.1); ×0.5 possibly also `g_ba_rel` |
| seconds scaling | 0.0007 / 0.0000 | `g_coef_band` only (ρ ≈ 11; needs D) |
| threshold ±0.1 | 0.0302 / 0.0601 | `g_threshold_band` (10 steps against a predicted tolerance of 3 to 5 steps); `g_ba_rel` |

- **Detection floor per feature.** At c = 4 the band catches a scaling ×k of feature j when
  |1 − k| > c·√2·SD_j/|w_j| = 2/ρ_j. With the ρ values of D5 at 32,768 rows:

| feature | ρ_j | smallest caught \|1 − k\| | ×0.75 / ×1.25 |
| --- | ---: | ---: | --- |
| listener_running | 14.4 | 0.14 | caught |
| tls_enabled | 5.4 | 0.37 | missed (band miss ≈ 0.98 with 20 band retrains, higher with 100: review A evidence §1) |
| peer_allowlist_configured | 6.2 | 0.32 | missed |
| queue_utilization | 12.1 | 0.17 | caught |
| consecutive_failures | 7.1 | 0.28 | missed (ratio ≈ 3.55 against 4) |
| seconds (with D) | 11.4 | 0.18 | caught |
| ledger_integrity_ok | 11.8 | 0.17 | caught |
| configuration_valid | 14.1 | 0.14 | caught |

- **What H2 can and cannot show.** c, the band count and the training size were sized against
  the ±50% and ±0.1 classes that H2 then tests. So H2 shows what reference-relative parameter and
  decision gates do on those classes, not on smaller changes. The descriptive ×0.75 / ×1.25 and
  ±0.05 mutants carry registered predictions (prereg §7 H2) and are reported, but stay outside the
  verdict. Seconds and failures scaling are caught mainly or only by parameter-space gates, so
  the claim is about parameter and decision gates, not about behavior in general.
- **Threshold ±0.05.** Five grid steps. v2's known-good threshold SD was 0.0071 (0.71 steps,
  MEASURED values: `v2/results/STAGE2_SEARCH.md`), which would give a tolerance of
  ⌈4·√2·0.71⌉ = 5 steps and miss it. With twice the training rows, the SD should be about 0.5
  steps and the tolerance 3 steps, which catches it (MODELED, rough: threshold noise taken to scale
  as 1/√n).

**Rejected.**

- **A joint parameter test (a quadratic form against a chi-square cut with 9 degrees of
  freedom).** At the same false-alarm rate, its cut is about 29, against ρ² for a
  one-coordinate shift. At 16,384, failures (ρ² ≈ 25) would escape it, while the per-coordinate
  band catches it. At 32,768, tls (ρ² ≈ 29) would be borderline. Per-coordinate bands are more
  sensitive to the registered one-coordinate mutants.
- **c from the t distribution with 20 band retrains.** It would meet the false-alarm target, but
  only with c of about 4.3–4.6 for 19 degrees of freedom. Against tls's ρ ≈ 5.4, and with the SD
  still estimated from 20 retrains, the tls band miss chance would rise from about 0.016 to
  roughly 0.07–0.12 (MODELED, rough). More band retrains cost no disk, because they are
  regenerated from seeds.
- **Bands centered on generator truth.** They exist only in synthetic settings, and the penalty
  biases them: v2's weights are 0.64 to 0.83 of truth. They are reported as truth recovery, not
  used as a gate.
- **A relative ECE gate.** ECE estimators are biased and noisy (EM-16, arXiv:1904.01685 and
  arXiv:1909.10155, ADAPT), so such a gate would raise false alarms.
- **Λ** (D12).

### D7. Origin binding, public provenance and the seal

- **Decision.**
  - Develop v3 on a public szl-forge branch, and tag every receipt-named commit before any merge.
  - Each receipt names a commit C and a step tag T. `verify_origin` requires that T's target is
    exactly C, that C introduced the bound trace, and that C's first parent is the git HEAD that
    the run recorded in its trace.
  - In a public clone, a skipped origin test is a failure.
  - The frozen state is tagged `oac-v3-frozen` in public, and durable records show it fixed and
    public before drand round R's scheduled time; the opening uses that round's randomness as the
    salt (D15).
  - Tags matching `oac-v3-*` are protected by a szl-forge ruleset that blocks update and deletion
    (approved by Stephen on 2026-10-13).
- **v2 evidence.**
  - Re-pointing v2's receipt to the parent commit `064481f`, whose five bound blobs are identical,
    passed both layers (MEASURED: `v2/results/MUTATION_READINESS.md`).
  - v2's source commit `92872f88…` is absent from szl-forge: the public API answered 422 for it on
    2026-10-14 (MEASURED). The public v2 tests skip 7 origin test sites on `ORIGIN_SKIP_REASON`
    (REPORTED: `v2/tests/_support.py` and three test files at szl-forge main `c9a03e46`; prereg
    §14.5).
  - Three `math/` modules were untracked at v2's opening (MEASURED: `v2/results/COMPLIANCE.md`).
  - The OAC landing pull requests szl-forge#570 (6 commits), #569 (4) and #596 (5) each landed
    as one commit with one parent, that is, squashed (MEASURED: the public pull request and
    commit records, 2026-10-14; the reading "squashed" is MODELED).
- **Literature.**
  - `anchor:factsheets` (EM-2, ADAPT): lineage and testing history.
  - `anchor:model_cards` (EM-1, ADOPT).
  - `anchor:datasheets` (EM-3, ADOPT). v2 did not apply it: a case-insensitive search for
    "datasheet" in the drafting copy of `v2/` (data excluded) and GitHub code search of
    `ops-health/v2` in szl-forge both return 0 hits (MEASURED 2026-10-14). v3 adds a
    datasheet-style description of its synthetic splits.
- **Reasoning (MODELED).**
  - Binding blobs that a parent also has cannot single out a commit.
  - The first draft bound the trace and required that C's first parent lack it. That excludes a
    parent and a descendant, but not a duplicate: a squash commit of the branch onto `main`, a
    merge commit, a cherry-pick or a rebase can hold the same trace bytes while its own first
    parent lacks them (review B ledger §E). The first draft's claim that C was "unique" was
    therefore too strong.
  - Recording the run-time HEAD P in the trace, and requiring first-parent(C) = P, ties C to the
    code that ran. Requiring that the named tag's target is exactly C ties the receipt to one
    commit hash. A duplicate with the same tree and the same first parent still has another hash,
    so it fails. This is why the registered receipt mutants now include a squash-style duplicate.
  - Tag protection, approved by Stephen on 2026-10-13, blocks updates and deletions by anyone
    without a bypass, but a repository administrator can change the ruleset, so it does not make
    a tag immutable. The guarantee therefore does not rest on the live tag: a receipt names C
    itself, each tag's target is recorded in the next step's commit, and the anchored freeze
    manifest records every earlier tag's target and tag object (prereg §3.5, §14.4). A moved tag
    is detected, not trusted.
  - Public tags make the check runnable by anyone, and they keep the commits reachable even if a
    later squash merge leaves them out of `main`.
  - Commits named in records are public szl-forge commits; drafting-repository commits are named
    only as information. Otherwise v3 would repeat v2's unpublished-origin problem.
- **Deferred.**
  - Signing receipts. It needs key custody and a verification policy, and it is a founder
    decision.
  - Binding the whole git tree. It would fail on any later, unrelated commit, and the tag and
    HEAD rule reaches the same goal.

### D8. H3: a dedicated twin set, no skew-shaped training

**v2 evidence.**

- On skew, v2 has 0.7564 [0.7455, 0.7665] against its nominal 0.7644 [0.7592, 0.7695]. The two
  intervals overlap.
- The paired v2 − v1 half-width is about 0.0080 on 12,500 skew rows, against about 0.0038 on
  50,000 nominal rows.

(MEASURED: `v2/results/FINAL_RESULTS.md`.)

**What the skew does (MODELED).**

- v2's clock skew adds U(0, 7200) s to the observed seconds, while the label uses the true value
  (REPORTED: `v2/PREREGISTRATION.md` §3; `v2/research/generator.py`). For a model that carries the
  true weight, it adds Δz = 2.2·skew/86400, which is between 0 and 0.183 logits, with
  E[Δz] = 0.0917 and E[Δz²] = 0.0112.
- **BA moves mainly to second order.** At a BA-maximizing threshold, the class-conditional score
  densities (each scaled by its class share) balance, so a small shift of the scores changes BA
  only to second order: about 0.25·g·E[Δz²] ≈ 0.0007–0.0011, with g = 0.25–0.40 per logit read
  from v2's reliability table (MEASURED table: `v2/results/FINAL_RESULTS.md`). If the selected
  threshold sits about 0.1 logit from the BA optimum, a first-order term of up to ±0.0012–0.0018
  adds to it. So ΔBA_skew(v3) is about −0.0011 to +0.0029 (MODELED: review A evidence §3).
- **Brier moves by about E[Δp²].** For a calibrated model, a perturbation Δp of the
  probabilities raises Brier by about E[Δp²], here E[(p(1−p)·Δz)²] ≈ 0.012 × 0.011 ≈ 1–2e-4.
- **For v2, both effects are negligible.** Its weight of 0.0366 (REPORTED:
  `v2/results/MUTATION_READINESS.md`) scales Δz by about 0.017.
- **Class-0 conformal coverage of v3 should fall by about one point.** Scores near q̂₀ ≈ 0.22 rise
  by about 0.016, and the class-0 density there is about 0.67 per unit of probability. That
  density is read from v2's reliability table, `v2/results/FINAL_RESULTS.md` (MEASURED table;
  arithmetic MODELED, rough). This is descriptive: the coverage guarantee holds only under
  exchangeability with the nominal conformal split (REPORTED: `v2/PREREGISTRATION.md` §6).

**The standard error of the H3 statistic (MODELED; the first draft gave none).**

- For a pair i of class y, D_i(m) = 1[m correct on the latent twin] − 1[m correct on the skewed
  twin], which is −1, 0 or +1, and nonzero only when m's decision flips between the twins.
- ΔBA_skew(m) = Σ_y Σ_{i in y} D_i(m)/(2 n_y). For the difference v3 − v2, with v2's flips rare:
  Var ≈ Σ_y φ_y/(4 n_y), where φ_y is v3's flip rate in class y.
- With a flip rate of 2.75% overall and 20% of flips in class 1, on 25,000 pairs
  (n₁ ≈ 4,550, n₀ ≈ 20,450): φ₁ ≈ 0.030, φ₀ ≈ 0.027, and SE ≈ 0.0014. Over flip rates of
  1.8–3.5% the SE is 0.00113–0.00156, so the 95% upper bound sits 0.0022–0.0031 above the true
  value (review A evidence §3 gives the same values).
- On 25,000 pairs, that left only about 0.002 of the 0.005 margin, and power at a true 0.002 was
  only 0.48–0.76. The first draft's "≈ 0.99" had no derivation and was wrong.
- On 100,000 pairs the SE halves, to 0.00057–0.00078. Power, Φ((0.005 − true)/SE − 1.96):
  0.97–1.00 at a true 0.002, and 0.77–0.96 at the modeled worst case of 0.0029.
- Brier arm: mean about 1.5e-4, SE about 4.9e-5 on 25,000 pairs and about 2.5e-5 on 100,000,
  against a margin of 0.002 (review A evidence §3).

**Design (MODELED).**

- The twin files hold the same latent rows and labels. Paired per-row differences therefore
  isolate the skew and give power without adding any truth.
- The H3 set is separate from the pooled suite's `shift_clock_skew`. That keeps P2's pool
  unchanged and gives H3 its own bootstrap stream (D11). It costs about 66 MB of sealed rows,
  generated only at the opening (CONJECTURE: about 330 bytes per row).
- H3 is non-inferiority because the question is a trade-off: does identifying the weight cost
  robustness? A gain under skew is not expected.

**Literature.**

- `anchor:wilds` (EM-37, ADAPT): report per-regime and worst-regime gaps, and do not assume that
  robustness to one shift transfers to another.
- EM-7 (doi:10.1016/j.patter.2023.100804): no training on evaluation data or its look-alikes.
- `anchor:conformal_covariate_shift` (EM-21, ADAPT): weighted conformal under covariate shift
  needs a likelihood ratio from unlabeled shift covariates, and v3 forbids any use of shift data.

**Rejected.**

- **A larger H3 margin on 25,000 pairs.** No minimum relevant cost has been stated that would
  justify more than 0.005 BA, and more data answers the question directly.
- **Skew-shaped augmentation.** One-sided jitter shaped like the sealed shift is training on the
  evaluation shift.
- **Symmetric jitter.** It cannot remove a one-sided offset. Under D its variance,
  1800²/3 ≈ 1.1e6 s², is about 0.2% of the seconds variance, so the attenuation is negligible. It
  is kept only as the exploratory ABL-jitter.
- **A skew-aware feature.** The eight fields carry no per-row information about skew, so such a
  feature would need a new field, and with it a new truth.

### D9. H4: why calibration cannot reduce abstention, and a statistic that holds coverage fixed

**The invariance (MODELED, exact algebra).**

- For the Mondrian rule, 1 ∈ C(x) iff p̂(x) ≥ 1 − q̂₁, and 0 ∈ C(x) iff p̂(x) ≤ q̂₀. Each q̂_y is
  an order statistic of the class-y scores on the conformal split.
- A strictly increasing map g of p̂ preserves every order statistic and every comparison. Every
  prediction set, and hence the abstention rate, is therefore unchanged.
- With v2's quantiles (MEASURED: `v2/results/STAGE2_SEARCH.md`), the BOTH band is
  p̂ ∈ [0.0888, 0.2237]. v2's reliability table has 15,680 of 50,000 test rows in [0.1, 0.2) alone
  (MEASURED: `v2/results/FINAL_RESULTS.md`).

**Why the first draft's H4 statistic did not answer its question (MODELED).**

- The question is "abstains less at the same per-class coverage". The first draft compared each
  model's operational abstention, with v2 on its frozen quantiles (n₁ = 750) and v3 and TRUTH on
  v3's `conformal` split (n₁ ≈ 1,490). Each lands on a different realized coverage.
- The realized coverage of one calibration draw has an SD of about √(0.09/(n_y + 2)): 0.0109
  (class 1) and 0.0052 (class 0) for v2; 0.0078 and 0.0037 for v3 (n MEASURED:
  `v2/results/STAGE2_SEARCH.md`; review A evidence §4).
- Abstention moves about 2.4–3 points per point of class-1 coverage. So the calibration draw
  alone gives the v3 − v2 abstention gap an SD of about 3.3–4.1 points, more than the 0.02 margin,
  and the paired bootstrap, which holds the quantiles fixed, does not see it. H4a could then come
  out SUPPORTED by luck about 27–31% of the time (review A evidence §4).

**The registered statistic A\* (MODELED).**

- For each model, cut-offs are placed on the evaluation rows themselves so that each class has
  exactly 0.90 empirical coverage, and they are recomputed inside every bootstrap resample.
  Abstention at those cut-offs depends only on how the model ranks the rows, and every model is
  compared at the same per-class coverage. It uses test labels to place the cut-offs, so it is a
  measure of ranking, not a deployable rule.
- So the only lever is ranking, and the TRUTH reference bounds it, up to finite-sample effects.
- v3 keeps v2's model class and l2, and the nominal seconds signal is about 0.02 logits. v3 should
  therefore not change the ranking materially, and H4a is predicted NOT SUPPORTED.
- H4b asks whether v2 is already near the floor (CONJECTURE: yes). If it is, then 41% is what 90%
  per-class coverage costs on this generator, and it is not a model defect.
- The operational Mondrian abstention is still reported, as a descriptive number with the
  calibration-split noise stated next to it.
- ABL-calibration (Platt, strictly increasing for a positive slope) doubles as an empirical check
  of the invariance. Its Mondrian sets and its A\* should equal v3's exactly.

**Literature.**

- `anchor:mondrian_conditional_validity` (EM-20, ADOPT).
- `anchor:gentle_conformal` (EM-19, ADOPT).
- EM-25 (doi:10.1109/tit.1970.1054406; arXiv:1610.02136, ADOPT): report the reject-option
  baseline next to conformal ABSTAIN.
- EM-26 (arXiv:1705.08500, ADAPT): risk–coverage curve.

**Why non-gating.** v3 deliberately leaves the only effective lever, ranking, almost unchanged.
Gating on a predicted null would hang the verdict on a non-target.

**Rejected for v3.**

- Recalibrating v2 and TRUTH on v3's `conformal` split, so that all three share calibration rows.
  It removes the split mismatch but keeps one calibration draw's noise inside the comparison.
- `anchor:conformal_risk_control` and `anchor:learn_then_test` (EM-23, EM-24, ADAPT). They would
  change the decision rule together with the levers.
- Isotonic two-probability predictors (EM-18, arXiv:1211.0025, ADAPT), which the matrix lists as
  a later-amendment candidate.

### D10. Win condition and power

**Options considered.**

- (a) v2's non-overlap of marginal intervals.
- (b) Paired superiority on BA.
- (c) Paired non-inferiority on BA and calibration, plus superiority on the targeted properties
  (H1 to H3), plus paired superiority against v1, majority and rule (P0).

Chosen: **(c)**.

**Why not (a) (MODELED).**

- v2's W1 margin was 0.000319 (MEASURED: `v2/results/FINAL_RESULTS.md`).
- Between two correctly specified, near-ceiling models, non-overlap would almost never happen.
- v2's own record notes that the rule is stricter than a paired interval excluding zero (MODELED
  there: `v2/results/FINAL_RESULTS.md`).

**Why not (b) (MODELED).** The expected nominal BA difference between v3 and v2 is close to 0, so
a superiority test on BA would have no power, and BA is not what v3 is for.

**Margins.** Every margin below was set with v2's sealed results in view (prereg §17).

| margin | value | reasoning |
| --- | --- | --- |
| δ_BA | 0.005 | Below the lower bound of v2's nominal gain over v1, +0.0104 [0.0066, 0.0142] (MEASURED: `v2/results/FINAL_RESULTS.md`). A v3 exactly at the margin keeps 52% of v2's point gain over v1 and 24% of its lower bound (arithmetic); P0 checks v3 against v1 directly. |
| Brier | +0.002 | About 14% of v2's paired Brier gain over v1, −0.0144 (MEASURED: `v2/results/FINAL_RESULTS.md`). |
| ECE | +0.01, point estimates | ECE estimators are noisy (EM-16). The point-estimate condition keeps the spirit of v2's W3, while the margin stops estimator noise from deciding. |
| H3 | 0.005 BA, 0.002 Brier | The same margins as P1 and P3, applied to the skew cost. |
| H4a | 0.02 abstention | About 5% of 0.41: a minimum relevant reduction. This answers the forum draft's first question. |

**Three readings of a non-inferiority result.** A failed non-inferiority test means only that the
interval reached the margin. With power below 1, that can happen by chance. So each
non-inferiority condition has three registered readings (prereg §16): shown, inconclusive, and
cost beyond the margin. Only the last may be called a cost or a trade-off.

**Power (MODELED).**

- For paired BA, Var(ΔBA) ≈ (d₁/n₁ + d₀/n₀)/4, where d_y is the rate at which the two decisions
  disagree in class y.
- **Calibration of d from v2.**
  - v2 − v1 on nominal test is 0.010380 [0.006559, 0.014191] (MEASURED:
    `v2/results/FINAL_RESULTS.md`), which implies a standard error of about 0.00195.
  - With n₁ = 9,096 and n₀ = 40,904 (MEASURED: same file), that implies d ≈ 0.11. In words, v2
    and v1 make different decisions on about 11% of nominal rows.
- **Standard error on 100,000 rows** (prevalence 0.182): about 0.00092 at d = 0.05, 0.00136 at
  d = 0.11, and 0.00159 at d = 0.15.
- **Power** = Φ(δ/SE − 1.96) at a true difference of 0: 1.00, 0.96 and 0.88 respectively. At a
  true difference of −0.002 with d = 0.11, it is 0.60.
- **Expected disagreement between v3 and v2: about 2 to 6% (CONJECTURE).** They share the model
  class and l2. They differ in:
  - the seconds weight, a nominal logit difference of about 0.02 ± 0.02;
  - sampling noise in the other weights, a per-row logit SD of about 0.09 (CONJECTURE);
  - slightly less shrinkage of the other weights under D (D2);
  - possibly the threshold, by 0.01 to 0.02.
- **The pooled shift suite (P2).**
  - The v2 − v1 standard error was about 0.00150 on 50,000 rows (MEASURED interval
    [0.0498, 0.0557]: `v2/results/FINAL_RESULTS.md`). On 100,000 rows it is about 0.00106. At a
    v3–v2 disagreement level the SE is likely nearer 0.0007 (CONJECTURE).
  - Power at a true pooled difference of 0: 0.997–1.00; of −0.002: 0.81–0.99; of −0.003:
    0.47–0.82 (SE 0.00106 to 0.0007; review A evidence §2).
  - **The named P2 risk** is a systematic v3 − v2 shift under queue saturation, where BA reacts
    at first order (D2). A pooled difference of −0.003 could come from roughly −0.01 in that
    regime alone, which holds a quarter of the pooled rows (MODELED, rough).
- **Sizes.**
  - Test is 100,000 rows, twice v2's. That gives P1 a power of at least 0.88 across the whole
    disagreement range considered (5% to 15%). At 50,000 rows, the power at the v2–v1 level
    would be about 0.74.
  - Each shift split is 25,000 rows, twice v2's, for P2.
  - The H3 twin set is 100,000 pairs (D8).
  - Band retrains: 100 (D6). Declared fits: 227 (prereg §6.6).
- **Overall.** The verdict is an intersection of conditions, so no multiplicity correction is
  needed: if any one condition is truly false, a false WIN is no more likely than that
  condition's own false pass.
  - If every lever behaves as modeled, the chance of WIN is about 0.6–0.85 (CONJECTURE). For the
    first draft's design it was about 0.55–0.8, not the 0.7–0.85 that draft stated (CONJECTURE:
    review A evidence §9).
  - The largest risks are: a systematic v3 − v2 shift under queue saturation (P2); a known-good
    false alarm in H2, about 0.02–0.03 (D6); and a selected threshold off the BA optimum (H3).
- **Storage and compute.**
  - 400,000 sealed rows, roughly 132 MB at about 330 bytes per row (CONJECTURE), generated only
    at the opening; per-row score files in addition.
  - The opening preflight requires at least 2 GB free.
  - This host's C: drive had 12 GB free of 952 GB on 2026-10-13 (MEASURED:
    `logs/verify/v3-prereg-review-b/checks_20261013.txt` §13). The first draft's figure of
    5.2 GB had no saved output and is withdrawn.
  - v2's runs met MemoryError on this host (REPORTED: `v2/results/FINAL_RESULTS.md`,
    post-opening correction 5; `v2/results/STAGE2_SEARCH.md`). The FRAGILE check therefore has
    a budget and a NOT_COMPLETED outcome (D11).

### D11. Bootstrap, the stream check and BCa

- **Decision.**
  - Register v2's bootstrap implementation, which v2 had only disclosed, with its stream layout
    stated in full: one `random.Random(20261014)` per evaluation set; strata in a listed order;
    the H3 twin set resampled by pair index on its own stream.
  - Register v2's post-hoc resample-stream check as a FRAGILE label, with the same per-set layout
    for each of the 100 check seeds, a 72-hour budget and a NOT_COMPLETED outcome.
  - Add BCa intervals as a sensitivity check.
- **v2 evidence.**
  - Type-7 percentiles, undefined resamples excluded and counted (REPORTED:
    `v2/results/COMPLIANCE.md` deviations 8 and 9).
  - "One count-vector stream per evaluation set", with the pooled suite resampled within regime
    (REPORTED: `v2/results/FINAL_RESULTS.md`); one `random.Random(seed)` drives all resamples of
    a set, stratum by stratum in order (REPORTED: `v2/research/bootstrap.py`). COMPLIANCE
    deviation 8 says "one `random.Random(20260927)` stream" without naming the per-set layout
    (REPORTED: `v2/results/COMPLIANCE.md`); the first v3 draft copied that ambiguity.
  - W1 held in 100 of 100 other streams, in a check that was not registered (MEASURED:
    `v2/results/FINAL_RESULTS.md`). That check stopped twice on MemoryError before it finished
    (REPORTED: same file, post-opening correction 5).
- **Reasoning (MODELED).** Near a margin, the stream layout decides the result, so it must be
  fixed in advance. With 2,000 resamples, the Monte Carlo SD of a 2.5th-percentile endpoint is
  about 0.06 of the statistic's SE, so FRAGILE fires only within about 1e-4 to 2e-4 of a boundary
  (review A evidence §7). Sets of the same size share index sequences under one seed; this does
  not affect any single interval.
- **Literature.** EM-27 (doi:10.1080/01621459.1987.10478410; doi:10.1214/ss/1032280214, ADAPT):
  keep the registered percentile interval, and add BCa as sensitivity.

### D12. Readiness aggregation: AND, then min; Λ excluded

- **v2 evidence.**
  - Every aggregator had a negative margin.
  - On every fixture, Λ lay between min and mean.
  - Λ was REJECTED under the registered rule.

  (MEASURED: `v2/results/MUTATION_READINESS.md`.)
- **Reasoning (MODELED).**
  - Λ was bounded by min and mean on every v2 fixture, and under the registered rule it did not
    beat both (MEASURED: same file). Nothing says it would do better on v3's gate vector.
  - With v3's binary new gates, AND and min agree at the cut.
  - No new argument for Λ exists, so it is not computed. Nothing about Λ is claimed. Adding it
    later would change a frozen item and so would need a new registration.

### D13. Amendments, DEV data, re-runs and endings: stricter than v2

- **v2 evidence.**
  - v2's six amendments were committed in one commit, before any search (MEASURED:
    `v2/results/COMPLIANCE.md` check C, which measures commit order).
  - Their content is REPORTED from `v2/AMENDMENTS.md`. The file states that none of them changes
    the question, the primary metric, the win condition, the splits, the seeds, the search space
    or the shift regimes. AM-2 added an eligibility rule for non-converged fits.
  - v2's post-opening corrections had to show that no number changed: 3,884 of 3,884 numeric
    fields identical (MEASURED: `v2/results/FINAL_RESULTS.md`).
- **Reasoning (MODELED).**
  - **Freeze everything numeric at registration.** The first draft froze its list only after the
    first data commit and left the recipe out (π, the train size, l2, the 900 s constant, the
    calibrator, the band and known-good counts, the bootstrap), while saying here that π was
    frozen. That was a contradiction and a forking path. Now the whole recipe, every count and the
    bootstrap are frozen at registration (prereg §15.3). A later change is either a
    re-registration, allowed only before any v3 data exists, or a PROTOCOL_DEVIATION.
  - **DEV data is not a pilot.** v2's generator exposes DEV rows for every regime (REPORTED:
    `v2/research/generator.py`, `dev_rows`), so code run on DEV data runs on the test
    distribution. DEV runs are therefore capped at 2,048 rows per regime, compute no P, H,
    tolerance, power or WIN-chance quantity, and are committed (prereg §15.5). The registration
    receipt's "pilot runs: 0" then stays true.
  - **The first run decides.** If a defect could be found and fixed after a result is seen, and
    the corrected run then decided, the choice of which defects to look for would depend on the
    result. So the first committed run of each registered step, and the official opening run
    (the one recorded in the opening commit), decide; a corrected run can only confirm (prereg
    §15.6). The kernel-relevant test list that
    defines a caught kernel mutant is fixed before any mutant exists.
  - **Endings.** The opening proceeds whatever H1 and H2 show, so a pre-opening failure cannot
    become a reason to stop and redesign with the same seal. ABANDONED covers only what was not
    yet evaluated, and committed outcomes are always reported. A non-converged retrain is not
    replaced, because a spare seed would reopen a choice. A FRAGILE check that cannot finish is
    reported as NOT_COMPLETED and never abandons the verdict. No re-attempt happens without a new
    registration (prereg §16).
  - **Seal endings.** SEAL_EXPOSED ends the seal, not v3: nothing was opened, so a disclosed seal
    re-registration with a new future round is not a re-attempt. But it must not be a lever. It
    is decided once, by frozen code, only from records fixed before R (prereg §3.5), so a record
    that expires after R cannot create it. After R, a new round is allowed only after EXPOSED or
    if the chain never produced R (prereg §3.6); host failures, refused preflights, relay
    problems and a missed opening window never allow one. A choice to drop a round is allowed
    only before its scheduled time, because afterwards its randomness, and so its sealed rows,
    are public (D15). SEAL_COMPROMISED, evidence that round R was known early, that a sealed row
    came from anything other than the verified round, or an unresolvable relay conflict, rules out
    WIN (prereg §16).
  - **The fix window closes at the first registered round.** Under the earlier text, a seal
    re-registration moved R, and with it the end of the window for frozen-state fixes. A late
    freeze would then have let the process compute round R's whole sealed sample from public
    data, run the frozen contenders on it, and "fix" what looked bad before the new round, at the
    cost of a disclosure only (review issue 3). Now no change to the frozen state is allowed
    after the first registered round's scheduled time, under any round, and a state sealed after
    any registered round was public carries a label in every statement of the verdict
    (prereg §15.3, §15.6). A new master seed for that case was considered and not taken: it would
    discard H1, H2 and every committed fit, while the label and the closed window already remove
    the gain (MODELED).
  - **The official run, pinned.** "The first opening run decides" was not well defined once round
    R is public, because anyone can run the frozen code privately first, and an unpinned
    interpreter could let a process try environments and choose (review issue 7). The runtime is
    now pinned (CPython 3.12.10, `python -I`, `PYTHONHASHSEED=0`), the official run is the one
    recorded in the opening commit, and the verdict waits for a byte-identical independent re-run
    (prereg §13 step 9, §15.6).

### D14. The kernel contract and the transform table

- **Decision.** Register the v3 kernel's fail-closed contract (prereg §6.7): standard library
  only; exactly v2's eight input fields; prohibited-key screening before the schema check;
  duplicate-key refusal; v2's all-false authority map, by reference to `AUTHORITY_MAP` in
  `v2/ops-health/ops_health.py`; ABSTAIN; the source commit pinned in the receipt; refusal of any
  transform table other than the registered one; determinism. Add ten transform-table probes, a
  transform-table artifact mutant, a squash-style receipt mutant and two kernel-source mutants
  (duplicate-key refusal removed; an authority value set true).
- **v2 evidence.**
  - v2's kernel imports only standard-library modules, takes the exact eight input fields,
    refuses malformed input, carries the all-false authority map `AUTHORITY_MAP` and an ABSTAIN
    outcome, and its receipt pins a source commit; v2 has a determinism test (REPORTED:
    `v2/ops-health/ops_health.py`, `v2/ops-health/artifact_receipt.json`,
    `v2/tests/test_determinism.py`, all public in szl-forge under `ops-health/`).
  - v2's kernel refuses duplicate JSON keys, while v1's accepts them (REPORTED:
    `v2/AMENDMENTS.md` AM-6).
  - The probe classes of v2's 66 fail-closed probes include no duplicate-key class and no
    authority-map check (REPORTED: `v2/research/gates.py` docstring). v2 checks both in separate
    tests (REPORTED: `v2/tests/test_failclosed_probes.py`, `v2/tests/test_review_hardening.py`).
  - The first v3 draft named none of these as v3 rules (MEASURED: review B checks §7).
- **Reasoning (MODELED).**
  - v3 must meet v2's standard, so the contract is registered rather than assumed.
  - v3 adds one new surface: the artifact declares an (offset, scale) table that the kernel
    applies. If the kernel accepted any table, an artifact that changed only the seconds scale,
    with a consistent receipt, would change every seconds contribution about 96-fold while
    `g_coef_band` on the model's own scale saw nothing. Two independent defenses close it: the
    kernel refuses any table but the registered one, and `g_coef_band` compares θ on the
    registered scale (D6). The transform-table mutant must be caught by both.
  - The authority map is cited by reference to the v2 constant, so v3 text never spells its keys.

### D15. The beacon seal: a future public randomness round instead of a held secret

- **Decision** (orchestrating session, under Stephen's 2026-10-13 reply; prereg §3.4–§3.6, §18
  items 5 and 9–12).
  - SALT is the randomness of drand round R = 6578966 of the default mainnet chain (chain hash
    `8990e7a9…b2ce`, period 30 s, genesis_time 1595431050), the first round scheduled at or
    after 2026-10-24T00:00:00Z. Its scheduled time is exactly that instant (MEASURED from the
    relays' `/info` responses; drafting copies: `logs/v3/register/round_R_20261013.txt`, and
    `logs/v3/register/recheck_info_*` for the byte-identical re-fetch of 2026-10-14).
  - Seal rule: the frozen state is tagged `oac-v3-frozen` in public szl-forge, and durable public
    records show it fixed and public before R's scheduled time (prereg §3.5):
    - T1: an OpenTimestamps proof of M, the sha256 of the freeze manifest, upgraded to a Bitcoin
      block attestation at height H, with the median header time of blocks H to H + 10 at least
      2 hours before R's scheduled time;
    - T2: a public archive record (a Software Heritage visit or Internet Archive captures), dated
      by the archive at least 1 hour before R's scheduled time, that shows the frozen tag and the
      seal tag with their targets.
  - The seal check is decided once, by frozen code, from those records only. EXPOSED means no
    opening on round R and a disclosed seal re-registration; the frozen state does not change.
  - The opening fetches rounds R and R + 1 from all three relays and needs `drand.cloudflare.com`
    and a `drand.sh` relay to answer and agree, randomness = sha256(signature), and round R + 1's
    previous signature equal to round R's signature. A standard-library BLS check is optional;
    the level reached is reported either way.
  - The opening runs within 7 days after R. A new round after R is allowed only after EXPOSED, or
    if the chain objectively never produced R.
- **Why not the held secret.**
  - The earlier drafts asked Stephen to generate SALT outside every session and repository and
    to give only `sha256(SALT)` at registration. That guarantee rested on one person's custody,
    which nobody else could audit, and Stephen supplied no salt.
  - A commit-reveal secret also leaves open whether the holder could have seen the sealed rows
    early. A future beacon round removes the holder: before the round's time the value does not
    exist for anyone, the research process included (MODELED, under the trust model below).
- **v2 evidence.** For v2, whether any process outside the guard read a sealed file before the
  opening is UNKNOWN (UNKNOWN: `v2/results/COMPLIANCE.md`). For the sealed sample, the v3 seal
  turns that question into an ordering that anyone can check from durable public records.
- **Literature.** The literature matrix has no anchor on public randomness beacons, timestamping
  or archives (MEASURED: a case-insensitive search of `literature/anchors.json` and
  `literature/evidence_matrix.md` for "beacon", "drand", "timestamp", "randomness",
  "opentimestamps", "bitcoin", "software heritage" and "internet archive" finds 0 hits, on
  2026-10-14 UTC). EM-7 (doi:10.1016/j.patter.2023.100804, ADOPT) supports the goal, test-free
  choices; it does not cover the mechanism. The mechanism's support is the reasoning below and the
  public sources it cites, not an anchor.
- **Reasoning (MODELED).**
  - **What it proves.** If one frozen state was fixed and public before round R's scheduled time,
    and round R could not be computed before then, every choice that decides the verdict was
    fixed before any sealed row could exist. After the opening, anyone can regenerate every
    sealed row from the public code and the public round, recompute every score, and check the
    timing proof.
  - **The target is fixed, not chosen.** R is derived from a fixed target instant by a stated
    rule. Choosing among future instants carries no information about their randomness, so the
    choice of target cannot favor any outcome.
  - **Why the first timing proof failed.** The amendment of 2026-10-13 used GitHub's record of
    the push that created the tag, from the activity or events API. The review showed that the
    activity API lists no tags, that an events-API tag event names no target commit, and that the
    events API keeps only the last 300 events, which covered 227.4 hours of szl-forge on
    2026-10-13 (MEASURED by the review; drafting copies: `logs/verify/v3-register-review/gh_*`).
    SEAL_EXPOSED was to be decided by a fresh fetch at the preflight, after R. So the process
    could compute the sealed rows from the public round, look, wait about ten days for the record
    to expire, and force SEAL_EXPOSED and a new round. Claims that the ordering was "checkable by
    anyone" were therefore false as written.
  - **Why OpenTimestamps (orchestrator decision).** A proof anchored in a Bitcoin block is made
    by parties other than the author, needs no account, sends only a 32-byte digest, and once
    upgraded can be checked offline against any copy of the Bitcoin headers, with no retention
    window. The pending proofs are public before R, so anyone can upgrade them. On 2026-10-14
    four public calendars answered, and their status pages showed a Bitcoin transaction every
    0.71 to 8.84 hours on average over the previous week (MEASURED; drafting copies:
    `logs/verify/v3-register-fix/ots_calendar_*`). Submitting 48 hours before R leaves a wide
    margin (MODELED).
  - **Why a forward median, not the attesting block's median-time-past (fixing-run
    adjustment).** The orchestrator's rule compared the attesting block's median-time-past with
    R. That median covers block H and the 10 blocks before it, all mined no later than H, so it
    lags the moment H was mined by about an hour at the usual 10-minute spacing. A digest made up
    to about an hour after R could then pass. Header times are the miners' own: a header must be
    later than the median of the previous 11 blocks, and nodes reject headers more than two hours
    ahead of their clock (REPORTED: `https://developer.bitcoin.org/reference/block_chain.html`).
    The median of blocks H to H + 10 is at or after the moment H was mined if most of those
    blocks carry honest times, because blocks H + 1 to H + 10 were mined after H. So the rule
    uses that forward median, with a 2-hour margin for clock error and small misreporting.
  - **Why an archive record as well (fixing-run adjustment).** T1 shows that M existed before R,
    but not that M was the only manifest timestamped. A process could timestamp several frozen
    variants privately before R, compute round R's outcome for each after R, then create the tag
    on the best one and publish its proof. It could also keep a valid proof private and claim
    EXPOSED to get a new round. T2 closes both: a public archive's own dated record shows that the
    name `oac-v3-frozen` pointed at F, and that the commit holding the pending proofs was public,
    before R. The research process cannot remove an archive's record, and anyone can upgrade the
    archived proofs. A record that shows a tag with two targets disqualifies that tag. Two archive
    services are accepted, so one archive's delay does not decide the seal alone. szl-forge had
    no Software Heritage visit on 2026-10-14, and the registration step requests one, which also
    tests the path before the freeze (prereg §13 step 1).
  - **The frozen identity.** F, the tag object id and M are recorded, and every check uses them,
    never a tag's current target. Listing every file's sha256 in the manifest keeps the identity
    independent of git's SHA-1 ids.
  - **No re-roll.** After R's scheduled time its randomness is public. If the protocol could then
    drop round R by choice, it could generate the sealed rows, look, and take another round. The
    earlier text said that its forced cases did not depend on the round's value, but two of them,
    "no verified round R within 14 days" and a refused preflight, were under the opening host's
    control after R (review issue 2). Now availability is a fact about the relays (prereg §3.6);
    the seal check uses only records fixed before R; host failures, refused preflights, relay
    problems and a missed opening window never allow a new round; and a new round needs either
    EXPOSED or daily evidence over 14 days that every relay's latest round stayed below R. A
    seal re-registration by choice is allowed only before R's scheduled time.
  - **The opening window.** A late opening gains nothing once outputs are pinned and the seal is
    decided from durable records, but a time limit keeps the record clean. Its time is shown the
    same way, by an OpenTimestamps proof of `TEST_OPENED.json`.
  - **Clock independence.** The opening decides that round R exists from the relays' answers,
    never from the host clock, which ran about 12 s behind the relays' `Date` headers on
    2026-10-13 (MEASURED; drafting copies: `logs/v3/register/probe_latest_*_headers_20261013.txt`).
  - **Verification levels.** randomness = sha256(signature) held on live rounds at all three
    relays, and round 6551372 carried round 6551371's signature as its previous signature
    (MEASURED; drafting copies: `logs/v3/register/round_R_20261013.txt`,
    `logs/verify/v3-register-fix/drand_chain_link_check_20261014.txt`). The hash check shows only
    that the randomness follows from the signature it came with. The chain link adds consistency
    across rounds, not authenticity: a relay that forged round R could forge round R + 1 to match.
    Only a BLS pairing check shows that the signature is valid for the registered public key. A
    standard-library BLS12-381 pairing is feasible but slow and new, with its own defect risk
    (CONJECTURE), so it is optional: the level is fixed in the frozen tree before R and reported,
    and anyone holding the public key can run the check with drand's own client.
- **Limits: what the seal rests on.**
  - **The drand threshold network.** The randomness is the sha256 of a threshold BLS signature
    over the previous signature and the round number. If a threshold of the network's operators
    colluded, they could compute round R before its scheduled time. Nothing in this protocol can
    detect that. A single relay, or operators below the threshold, cannot (MODELED, from the
    threshold design).
  - **Bitcoin's header times.** T1 trusts that most mining power reports honest times. A miner
    majority that misreported times by more than the 2-hour margin could make a late digest look
    early. The calendars cannot forge an attestation; they can only fail to make one, which
    leaves the seal EXPOSED if no other calendar attested in time.
  - **The archive's own date.** T2 trusts the archive's clock and records. Neither archive is
    controlled by the research process.
  - **Tag protection.** The ruleset on `refs/tags/oac-v3-*` blocks moving or deleting a tag by
    anyone without a bypass, but a repository administrator can change a ruleset. The seal and the
    origin rule do not rely on it: they use the anchored identities, so a moved tag is detected,
    not trusted.
  - **The relays.** Without a BLS check, a relay that served a forged signature together with its
    matching sha256 would pass the hash check. The relay rule makes that need agreement between
    `drand.cloudflare.com` and a `drand.sh` relay. The two `drand.sh` relays send the same
    `Server` header and may share an operator, so their agreement alone does not count
    (MEASURED headers; operators UNKNOWN).
  - **Self-generated samples and tuning to the distribution.** The generator is public, so
    anyone can draw rows of the same distribution at any time. The DEV rule forbids doing so at
    scale, but no seal can detect it. The seal protects the sample, not the design: v3 was
    designed with v2's sealed results in view (prereg §17).
- **Timing risk.** R's scheduled time is about 10 days after the green light, and the freeze
  digest is planned for submission 48 hours before it, by 2026-10-22T00:00:00Z. Registration,
  the v3 code and tests, 227 fits, the mutation suite, H1, H2 and the freeze must all be public
  before that (CONJECTURE that the schedule is tight; the fits' wall-clock time on this host is
  UNKNOWN before any v3 code exists). Missing it is no longer free: if any registered round
  passes before the freeze is sealed, the verdict carries the label "frozen after round R's
  randomness, a full sealed-equivalent sample, was public", and the fix window is already closed
  (prereg §15.3, §15.6). Before registration, the draft's target instant can still be moved at no
  cost; that is the only clean way to buy time.
- **Rejected.**
  - A secret held by Stephen or by any session: custody that nobody can audit, and none was
    supplied.
  - A past or current round: its value is already public, so it seals nothing.
  - A beacon round chosen after the freeze, or "the next round after the freeze push": the round
    would then depend on when the research process pushed, which it controls. A fixed target is
    simpler and leaves no choice.
  - A hash of a future blockchain block or a market price as the salt: other trust models,
    harder to verify in the standard library, and not part of the decision.
  - GitHub's events or activity records as the timing proof: they expire, name no tag target and
    cannot be read back later (review issue 1).
  - A GitHub commit comment on F, or a comment on a locked issue, as a required anchor: its
    server time is real, but its author or an administrator can delete it without a public
    trace, so it can neither show uniqueness nor stop withholding. It may be added as context.
  - A Sigstore transparency-log attestation or an RFC 3161 timestamp token as the required
    anchor: both are workable, but the first needs a signing workflow in szl-forge and the second
    trusts one authority's clock and key; OpenTimestamps needs neither (MODELED).
  - Hugging Face commit dates: a push sets them, and v3 makes no Hugging Face write before a
    verdict (prereg §14.9).
  - The attesting block's own median-time-past as the cutoff: it can admit a digest made after R
    (above).

## 3. The forum draft's three questions

The draft forum post `drafts/forum_thread.md` is never posted. It asks three questions (REPORTED:
that file). v3 answers each.

1. **Should a card lead with the preregistered pass/fail, or with the effect size? Should a
   minimum practically relevant difference be preregistered?**
   - v3 preregisters minimum relevant differences: 0.005 BA, 0.002 Brier and 0.02 abstention.
   - It tests every one as a paired difference with its interval, and it registers three
     readings (shown, inconclusive, cost beyond the margin), so a failed test is never reported
     as a measured cost unless the whole interval lies beyond the margin.
   - The verdict is pass/fail, and each condition carries its paired interval, so the two lead
     together.
2. **Report the weakly identified weight, constrain its sign, or drop the field?**
   - v3 takes a fourth path: change the training design so that the data constrain the weight
     (D3).
   - It keeps reporting the spread across retrains, as H1's statistic 3.
   - A sign constraint is circular, and dropping the field misspecifies the model (D3).
3. **Which behavioral gate catches ±50% weight scaling and ±0.1 threshold shifts, and how should
   its tolerance be set before any mutant is seen?**
   - Reference-relative parameter and decision gates: per-parameter bands, a threshold band in
     integer grid steps, and a relative BA gate. Seconds and failures scaling are caught mainly
     or only in parameter space, so these are not purely behavioral gates.
   - Tolerance = 4·√2 × the standard deviation across 100 separate known-good band retrains, with
     c = 4 and the band count fixed at registration against a stated false-alarm target (D6).
   - Smaller changes (×0.75 / ×1.25) escape the band for tls, peer and failures; that limit is
     predicted and reported, not hidden.

## 4. Predictions recorded now, and what would falsify them

| item | prediction (MODELED unless marked) | falsified if |
| --- | --- | --- |
| Fits | all 227 declared fits converge | any non-converged fit |
| H1, v3 arm | 21/21 positive; median r ≈ 1; SD(r) ≈ 0.03 | any negative weight, or SD(r) > 0.10 |
| H1, ABL-D | about 2/21 negative; SD(r) ≈ 0.8 | ≥ 19/21 positive with SD(r) ≤ 0.10: rescaling alone identifies the weight, and D3 is wrong |
| H1, ABL-R | 21/21 positive; median r ≈ 0.87 | median r outside [0.75, 0.95] |
| H1, ABL-DR | about 2/21 negative; median r ≈ 0.01 | median r > 0.1 |
| H2 | all 37 artifact mutants (36 by the gates; the table mutant by the kernel and `g_coef_band`), 7 receipt and 8 kernel mutants caught; 0 known-good false alarms (chance of ≥ 1 about 0.02–0.03) | any escape or false alarm |
| H2 gate pattern | tls caught by `g_coef_band` and `g_ba_rel`; seconds and failures ×1.5 only by parameter-space gates | a different pattern (reported, not a failure) |
| Descriptive ×0.75 / ×1.25 | band catches listener, queue, seconds, ledger and configuration; misses tls, peer and failures, which `g_ba_rel` also misses (CONJECTURE for `g_ba_rel`) | a different pattern (reported, not a failure) |
| Descriptive threshold ±0.05 | tolerance 3 steps; both caught by `g_threshold_band`; −0.05 also by `g_ba_rel` | tolerance ≥ 5 steps, or a miss |
| Descriptive recipe substitution | ABL-R and ABL-DR refused by the kernel's table check; ABL-D caught by `g_coef_band` with probability about 0.8 | ABL-R or ABL-DR accepted by the kernel |
| H3 | ΔBA_skew(v3) between about −0.0011 and +0.0029; ΔBrier_skew ≈ 0.0002; class-0 coverage −0.01 (descriptive) | bounds exceed 0.005 and 0.002 |
| H4a | NOT SUPPORTED (A\*) | upper bound < −0.02 |
| H4b | SUPPORTED (CONJECTURE) | gap > 0.02 |
| ABL-calibration | Mondrian sets and A\* identical to v3's | any set differs (a tie or implementation issue to explain) |
| P1 | v3 − v2 BA ≈ 0 ± 0.002 | lower bound ≤ −0.005 |
| Seal timing | the seal check is SEALED: the freeze digest attested with a forward median at or before 2026-10-23T22:00:00Z, and an archive record of the tags by 2026-10-23T23:00:00Z (CONJECTURE; the schedule is tight, D15) | EXPOSED: a seal re-registration, with the label of prereg §15.3 and no fix |
| Verification level | BLS_VERIFIED is feasible in the standard library (CONJECTURE); HASH_AND_RELAY_AGREEMENT is the registered minimum | the frozen tree ships HASH_AND_RELAY_AGREEMENT, which is reported as such |
| Official run and re-run | byte-identical score files on the opening host and a GitHub-hosted Windows runner with the same pins (MODELED) | any score file differs: a defect under prereg §15.6 |

## 5. What v3 does not attempt

- **No new truth, field, regime or model class.** It stays L2 logistic regression.
- **No claim about real telemetry.** Everything stays SYNTHETIC.
- **No change to any v2 file, artifact or card.** `v2/PREREGISTRATION.md` and `v2/AMENDMENTS.md`
  stay byte-identical.
- **No fix for queue saturation.**
  - It is the worst regime for v2 (BA 0.5928, class-0 coverage 0.2994), v1 (0.5311), ABL-data
    (0.5842) and ABL-optimizer (0.5693). It is not the worst for the rule, whose lowest shift
    regime is clock skew (0.5336) and whose nominal BA (0.5317) is lower still, nor for majority,
    which is 0.5000 everywhere (MEASURED: `v2/results/FINAL_RESULTS.md`).
  - A fix needs shift-time data, for example threshold re-selection or weighted conformal
    (`anchor:conformal_covariate_shift`, EM-21), and the protocol forbids any use of shift data.
  - It is measured again, and P2 must still hold with it in the pool.
- **No skew-shaped training and no skew-aware feature** (D8).
- **No sign-constrained fit, no dropped field, no log1p, no data-derived scaling** (D3).
- **No recalibration aimed at abstention.** It is inert for Mondrian sets (D9).
- **No hyperparameter search, and no proper-score selection rule** (D2).
- **No conformal risk control, Learn then Test, or isotonic two-probability predictors** (D9).
  They are candidates for a later version.
- **No Λ** (D12).
- **No receipt signing** (D7; a founder decision).
- **No live monitoring or change detection** (EM-33, ADOPT for live data). There is no live data.
- **No Hugging Face card or publish** until a verdict exists and a separate change is approved.
- **No reference to any external person or organization.** No outreach, and no remote write
  from this draft.

## 6. Risks to running the plan

- **Known-good false alarms in H2** (about 0.02–0.03; D6). This is H2's main risk, not the
  detection margins. No tolerance may move after the fact.
- **A systematic v3 − v2 shift under queue saturation** (D2, D10). It is P2's main risk.
- **A selected threshold off the BA optimum** (D8). It lowers H3's power to about 0.77–0.96.
- **Disk and memory.** The host disk is nearly full: 12 GB free of 952 GB (MEASURED:
  `logs/verify/v3-prereg-review-b/checks_20261013.txt` §13). v2 met MemoryError (REPORTED:
  `v2/results/FINAL_RESULTS.md`, `v2/results/STAGE2_SEARCH.md`). Mitigations: retrain splits are
  not stored; the opening preflight fails closed; the FRAGILE check has a budget and a
  NOT_COMPLETED outcome. A run that cannot finish is ABANDONED for what was not yet evaluated,
  and committed outcomes are still reported (prereg §16).
- **Seal timing.** Everything up to the freeze must be public, and the freeze digest submitted,
  by about 2026-10-22T00:00:00Z, 48 hours before R (D15). Missing the seal now costs a label on
  every statement of the verdict and closes the fix window (prereg §15.3, §15.6). The fits'
  wall-clock time on this host is UNKNOWN before any v3 code exists. Before registration, the
  target instant can still be moved at no cost.
- **Seal trust.** The seal rests on the drand threshold network, Bitcoin's header times under an
  honest majority, and the public archive's own record date (D15). A threshold collusion that
  computed round R early could not be detected here. If the BLS verifier is not ready at the
  freeze, the level is HASH_AND_RELAY_AGREEMENT and the report says so.
- **Timing-proof services.** If every calendar fails to attest before the cutoff, or no archive
  records the tags in time, the seal check is EXPOSED (prereg §3.5). Mitigations: at least two
  calendars on different domains, two accepted archives, a submission 48 hours before R, and a
  rehearsal of both paths at registration.
- **Beacon availability.** If round R is not available from the relays, or they disagree, the
  opening waits; the window extends while the cause lasts. A new round needs daily evidence over
  14 days that the chain never produced R (prereg §3.6). drand could also retire the default
  chain; that case is handled the same way (CONJECTURE that it will not happen before R).
- **Other lanes in szl-forge.** The per-repo write lane applies. Another session's open change on
  `ops-health/` paths blocks pushes until it lands.
- **CI on the public branch.** v3 tests must pass there with 0 skipped origin tests. Any
  dependency on untracked local files fails that check by design.

## 7. Decisions at the green light

On 2026-10-13 Stephen replied "do all your recommendations" to `v3/DECISIONS_FOR_GREEN_LIGHT.md`
(REPORTED: the orchestrating session's record of his reply). All eight items there are DECIDED,
and prereg §18 maps each to the sections it governs:

- register, with the public pushes and annotated tags of prereg §14;
- the recommended win condition, with H2 strict: zero known-good misclassification;
- 100 band retrains; an H3 twin set of 100,000 pairs; `test` 100,000 rows; four shift splits of
  25,000 rows;
- 227 declared fits; the FRAGILE check capped at 72 hours; at least 2 GB free at the opening
  preflight;
- the seal: Stephen gave no salt, so the orchestrating session replaced the held salt with drand
  round R = 6578966 (D15);
- tag protection for `oac-v3-*` in szl-forge, which hardens the tags; nothing relies on it alone
  (D7, D15);
- published evidence: `literature/*`, the two v3 review evidence folders and
  `drafts/forum_thread.md` only; `REPORT.md`, `COORDINATION.md` and `LANDING_PLAN.md` stay
  internal, and their citations were re-cited to public sources on 2026-10-14;
- per-row score files: at most 50 MB per file is committed; above that, digests only.

On 2026-10-14 the orchestrating session decided, under the same reply, the timing proof, the
opening window and the relay rule, and the fixing run adjusted the cutoff and added the archive
record with the reasons in D15 (prereg §18 items 9–12).

The draft is still NOT registered. Before registration, the check of cited paths in prereg §14.2
is repeated.
