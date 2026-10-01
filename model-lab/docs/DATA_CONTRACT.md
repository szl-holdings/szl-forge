# Data and evaluation contract — first research slice

## Router model

Target: boolean `task_succeeded`, determined by an independent, predeclared task
rubric after the request. A model's self-reported confidence is not the label.
The current default router must be retained as a comparator; do not quietly
replace it because a neural model exists.

Eight inputs: `input_load`, `output_budget`, `queue_utilization`,
`memory_utilization`, `prior_quality_rate`, `latency_pressure`, `unit_cost`,
`domain_match`. All require actual pre-outcome observations and a versioned
normalization recipe mapping them into [0,1]. Zero cost must reflect a real priced
observation, not a missing field or promotional provider tier. Historical quality
features must be generated without observing the current or future outcomes.

These features are a deliberately small tabular baseline, not a representation
of arbitrary text, tool permissions or full business context. The model is not
integrated into the production router. No output here supplies route eligibility.

## Invariant predictor

Target: boolean `violation_observed`, determined by the reviewed violation rubric
or a qualified deterministic verifier. Keep the verifier authoritative. Train a
predictor only where it helps prioritize inspection, predict downstream failures,
or flag proposals; do not learn an approximation of an inexpensive exact check
and then replace that check with the approximation.

Eight inputs: `action_scope`, `evidence_gap`, `policy_distance`, `privilege_level`,
`data_sensitivity`, `dependency_churn`, `verification_age`, `rollback_difficulty`.
These are normalized numeric observations or explicitly annotated rubric values.
A subjective rubric must be disclosed and audited, not mislabeled a measurement.

Lambda remains an advisory weighted geometric mean, not a proven trust score or
a mathematical justification for these chosen features. No lambda result grants
action authority. No clinical/PHI data or clinical-device actions enter this kit.

## Required JSONL fields

`example_id` is globally unique. `group_id` groups related requests/cases that
must remain in one split. `case_sha256` is a stable commitment to the underlying
case, supplied by the corpus builder so paraphrases/duplicates do not enter a
second split. `split` is exactly train, validation or test. `features` is the
exact named mapping for the chosen track. `label` is a JSON boolean.

`provenance` has exactly: `source_id`, `rights_basis`, `synthetic`, `feature_time`,
`outcome_time`, `normalization_id`. Rights basis is one of owner-created, licensed,
consented, test-fixture. Both timestamps are ISO8601 with timezone and the feature
time must precede the outcome. The validator checks these assertions' consistency;
it does not independently certify rights, identity, chronology or semantic truth.

No dataset generator runs by default. The only included examples are transient
pytest fixtures with `rights_basis=test-fixture` and `synthetic=true`. They are not
a publishable training corpus, not evidence of operational outcomes, and not an
independent benchmark. The fixture's simple separable pattern only exercises the
optimizer, gradient path, serialization and reporting contract.

## Split and feature freeze

Admit data and feature normalization before fitting. Keep test cases separately
access-controlled by the existing experiment authority; this local file format
checks split membership but cannot stop the owner from reading the file. Related
entities and neighboring time windows should be grouped before hashing. Record
both raw-corpus and derived-dataset commits outside this package's local manifest
when creating a real publication contract.

All three splits must contain both label classes. This deliberately rejects
single-class folds rather than silently returning misleading rates. The current
validator does not build a time-series cross-validation scheme. Choose temporal
holdouts where the use case demands them, and add a reviewable splitter upstream.

## Evaluation, not just loss

The implemented bounded metrics are n, positive count, four confusion-matrix
counts at fixed threshold 0.5, Brier score and binary log loss. Calibration remains
unvalidated even though Brier/log-loss values are computed. There is no inferred
confidence interval, power claim, statistical superiority, or promotion rule.

The next integration must compare router decisions with the current deterministic
policy, including task quality, request cost, tail latency, abstention, and regret
on a frozen evaluation plan. Respect domain, model-family, request-length and
provider strata. A small class-balanced fixture is not evidence for a naturally
imbalanced production distribution.

For invariant prediction, report false-negative counts and rates with denominators
and reviewed uncertainty bounds. Evaluate adversarial and out-of-distribution
cases. Even a good measured rate does not authorize a model to override a deny.

Only after an independent calibration split is admitted should calibrated scores
or threshold optimization be introduced. Do not optimize on the final test set.
The explicit evaluate-test command is not a global one-shot oracle; enforce
experiment-level access, immutable attempts, and finality in existing Forge tools.
