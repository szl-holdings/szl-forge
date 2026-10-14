<!--
DRAFT. NEVER POSTED. Posting needs Stephen's explicit approval of this exact post.
Where: AI for Science community forum, new topic, category "Open Discussion".
Before posting:
- Replace {PINNED_COMMIT} with the szl-forge main commit that adds ops-health/.
- Check that https://huggingface.co/SZLHOLDINGS/oac-ops-health-v2 (the proposed id) is live and
  was built from that commit by CI.
- Post only the text between the two horizontal rules below.
Labels: the posted text labels each claim in square brackets (MEASURED, REPORTED or MODELED, and
SYNTHETIC on every number). The numbers come from v2/results/FINAL_RESULTS.md,
v2/results/MUTATION_READINESS.md and v2/results/final_robustness.json.
-->

**Title:** Evidence-bound model cards: how should a card report a preregistered win that clears its bar by 0.0004?

**Category:** Open Discussion

---

We would like critique of how one model card documents its evidence. The model is deliberately
tiny, and what we want reviewed is the documentation, not the model.

Labels in square brackets: **MEASURED** means computed from the files cited on the card, with the
output saved; **REPORTED** means read from a named file; **MODELED** means it follows from the
design or the code, with no measurement behind it. **Every number is SYNTHETIC.** File paths are
relative to `ops-health/` at the pinned commit linked below.

**Limitations first.**

1. **Everything is synthetic.** The labels are drawn from a known logistic rule over the same
   eight inputs the model reads, so the model class is correct by construction. [MODELED] Nothing
   here says anything about any real system.
2. **The headline win is narrow.** Our preregistered rule required v2's lower 95% bootstrap bound
   on balanced accuracy (BA) to sit strictly above v1's upper bound. On the nominal test split it
   did, by 0.0004 as printed (0.000319 unrounded). The paired difference v2 − v1 is +0.0104
   [+0.0066, +0.0142]: about one point of BA. [MEASURED, SYNTHETIC]
3. **One synthetic shift breaks every contender.** Under queue saturation, v2's BA is 0.5928, its
   specificity 0.2269, and the conformal layer's coverage of the negative class 0.2994.
   [MEASURED, SYNTHETIC]
4. **The advisory is often silent or wrong.** The conformal layer abstains on about 41% of nominal
   rows, and 46% of the alerts from v2's threshold decision are false. [MEASURED, SYNTHETIC]
5. **One weight is weakly identified.** In 3 of 5 known-good retrains of the selected recipe, the
   weight on `seconds_since_last_success` came out with the opposite sign. [MEASURED, SYNTHETIC]
6. **Our checks have holes.** Our preregistered behavioral gates missed 10 of 36 model mutants:
   ±50% rescaling of four weights, and ±0.1 threshold shifts. Only the hash check caught them. The
   readiness aggregator we built (Λ) was rejected under our own preregistered rule.
   [MEASURED, SYNTHETIC: `v2/results/MUTATION_READINESS.md`]
7. **Receipts are unsigned.** They attest the integrity and origin of files, not their quality.
   [REPORTED: the 13 search receipts record `"signed": false`, and the model's receipt has no
   signature field]

**What the card does.**

- The protocol was registered before any v2 code or data existed. Six amendments were committed
  before any search, fit or opening. [REPORTED: `v2/PREREGISTRATION.receipt.json`,
  `v2/AMENDMENTS.md` and the git history]
- The 50,000-row test split and four 12,500-row shift splits were sealed and then opened exactly
  once, after the model was frozen. [REPORTED: `runs/TEST_OPENED.json`]
- The card maps all 32 REFORMS checklist items: 16 satisfied, 10 partial, 3 not applicable and 3
  missing. These are our own assessments, which is one of the things we would like checked.
- Every claim on the card carries a label: MEASURED, REPORTED, MODELED or UNKNOWN.
- The "cannot do / no authority" section is a first-class field. The model's output carries five
  authority flags, and all five are always false: the kernel copies them from a constant
  [MODELED from the kernel source], and a rerun of the command-line kernel shows them all false
  [MEASURED].

**Numbers.** Balanced accuracy, point [95% percentile bootstrap interval], 2,000 paired resamples.
[MEASURED, SYNTHETIC: `v2/results/FINAL_RESULTS.md`]

| contender | nominal test (50,000 rows) | pooled shift suite (50,000 rows) |
| --- | --- | --- |
| v2 | 0.7644 [0.7592, 0.7695] | 0.7581 [0.7542, 0.7618] |
| v1 as shipped | 0.7540 [0.7494, 0.7588] | 0.7053 [0.7019, 0.7087] |
| majority baseline | 0.5000 [0.5000, 0.5000] | 0.5000 [0.5000, 0.5000] |
| rule baseline (`consecutive_failures > 0`) | 0.5317 [0.5262, 0.5370] | 0.5439 [0.5392, 0.5483] |

- **Calibration on nominal test (the third preregistered condition):** Brier 0.1013 for v2 against
  0.1158 for v1, and ECE 0.0306 against 0.0848. [MEASURED, SYNTHETIC]
- **Paired differences v2 − v1** [MEASURED, SYNTHETIC]:

  | metric | nominal test | pooled shift |
  | --- | --- | --- |
  | Brier | −0.0144 [−0.0151, −0.0138] | −0.0313 [−0.0322, −0.0304] |
  | BA | +0.0104 [+0.0066, +0.0142] | +0.0528 [+0.0498, +0.0557] |

- **Where v2 cannot be told apart from v1:** under the clock-skew shift the paired BA difference is
  +0.0066 [−0.0013, +0.0146]. [MEASURED, SYNTHETIC]
- **Resample-stream sensitivity:** after the opening, we repeated the first condition with 100 other
  resample streams. This check was not registered. The condition held in 100 of 100, with a
  margin between 0.00008 and 0.00094. [MEASURED, post hoc, not registered, SYNTHETIC:
  `v2/results/final_robustness.json`]

**Links.**

- Model card (v2): https://huggingface.co/SZLHOLDINGS/oac-ops-health-v2
- Model card (v1, for comparison): https://huggingface.co/SZLHOLDINGS/oac-system-health-v1
- Source at the pinned commit: https://github.com/szl-holdings/szl-forge/tree/{PINNED_COMMIT}/ops-health
- Preregistration: https://github.com/szl-holdings/szl-forge/blob/{PINNED_COMMIT}/ops-health/v2/PREREGISTRATION.md
- Amendments: https://github.com/szl-holdings/szl-forge/blob/{PINNED_COMMIT}/ops-health/v2/AMENDMENTS.md

**Three questions.**

1. **Which result should lead?** The interval-separation rule passed by 0.0004 BA, while the
   paired difference is +0.0104 [+0.0066, +0.0142]. Should the card lead with the preregistered
   pass/fail or with the effect size? Would you have preregistered a minimum practically relevant
   difference instead of interval separation, or alongside it?
2. **How should a card document a coefficient the data barely constrain?** Our weight on
   `seconds_since_last_success` changed sign in 3 of 5 retrains. The choices we see are:
   - report it as weakly identified, which is what we do now;
   - constrain its sign in advance;
   - drop the feature in a preregistered v3.

   Which would you choose, and what would you require the card to show?
3. **Which behavioral gate would you register?** Our gates use absolute tolerances (BA ≥ 0.70,
   ECE ≤ 0.05), and they miss ±50% rescaling of four weights and ±0.1 threshold shifts. What
   gate would you register for a small tabular model so that a behavioral check, not only a hash,
   catches these? How would you set its tolerance before seeing any mutant?

Critique of anything above is welcome, including "this card is too long to be useful".

Stephen Lutar, CTO, SZL Holdings

---
