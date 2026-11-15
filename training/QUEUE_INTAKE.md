# Mass-training intake and exact-response smoke characterization

This source adds two offline commands, not another trainer, signer or publisher.
The authority remains `training/launch_contract.py`, `training/preflight/preflight.py`,
the existing admitted GPU execution path, and the separately approved model publisher.
Existing evaluation receipts and producer scores are not rewritten.

## Import a queue without executing it

```text
python training/queue_intake.py intake --queue queue.json --snapshot hub-metadata-summary.json --out new-intake-report.json
```

The queue schema is `szl.training-queue-import/v1`, with an explicit integer
`declared_total` and an `entries` array containing `name`, `base`, and `dataset`.
`name` is a bare model name in SZLHOLDINGS. Entries must be unique, including case
aliases. Downloads can be retained as an attributed historical snapshot; they are
not training permission, evaluation quality, or a stable priority signal.

The recorded metadata input is an array of `models`/`datasets` observations, each
with `id`, `status`, immutable `sha` when observed, and optional `pipeline_tag`,
`cardData`, `files`, and an `adapter_config` readback. It is offline evidence, not
an authenticated attestation. Metadata does not qualify weights or grant rights.
Missing, unauthorized and failed reads remain unavailable rather than absent.

For each entry the importer preserves supplied identities, proposes observed
immutable revisions without approving them, detects incompatible target types
and declared lineage differences, and invokes the existing launch contract.
It does not fabricate an operator-admission receipt, train/holdout corpus, split,
training budget, exclusive GPU lease, or hyperparameter qualification. Every
imported entry remains `HOLD_NOT_LAUNCHABLE`; an exit code of zero means the
intake record was written, not that training was authorized.

The November 13 supplied script declared 59 targets but included 24, with the
rest delegated to an unavailable local path. Missing entries must be recovered
from their original record, not invented. Its generic causal-LM trainer must not
overwrite embedding models, mixed software/weight repositories or existing
published checkpoints. Public license metadata does not substitute for the
source-rights admission required by Forge. The original contained a credential;
that credential is not retained in this source or used for any read or write.

The proposed 200-step/rank-16/alpha-32 profile is a research hypothesis, not a
proven universal optimum. The existing trainer's supported configuration and
installed environment must be reconciled before translating such a proposal.
This importer does not assert that all proposed settings would be honored.

## Characterize complete captured responses

```text
python training/queue_intake.py characterize --fixtures training/fixtures/exact-response-smoke-v2.json --fixture-sha256 ACTUAL_SHA256 --responses responses.json --out new-characterization.json
```

Compute `ACTUAL_SHA256` from the exact fixture bytes before collecting outputs.
`responses.json` uses `szl.raw-smoke-responses/v2`, that same `fixture_sha256`,
`evidence_class` (`SIMULATED` or `LIVE_SMOKE`), `model_binding`, and `responses`.
Each response has a unique fixture `id`, `status`, complete `response` string,
and `finish_reason`. Every fixture identity must be present exactly once; empty,
missing, duplicate, unexpected or equal-count/wrong-ID populations are rejected.

This is a NEW protocol, not a revised score for the earlier estate battery.
The six supplied convenience questions have a frozen response instruction and
explicit accepted short answers. Normalization casefolds and collapses whitespace
and removes terminal periods/exclamation marks. It does not search a response for
an abstention substring. A quoted/negated refusal is not accepted. Correct exact
answers and exact fixture refusals are counted separately from wrong numeric
answers, abstention on answerable questions, review-required text and unavailable
or truncated generations. Full content and content hashes are retained, not a
200-character sample. A semantically correct explanation can remain review-required
under this deliberately narrow exact-response contract.

The three context-unsupported fixtures are declared as such for this smoke
protocol, not proven universally unanswerable. No score establishes model honesty,
generalization, causal scaling, an optimal training dose or deployment readiness.
A serving alias, model-list metadata or transport success is not an immutable
weight binding. LIVE_SMOKE is provenance supplied by the capture, not authenticated
by this offline module. Signatures and independent evaluation remain unverified.

## Evidence and validation

Outputs are exclusively created and fsynced. Reusing an existing output path fails;
no historical receipt is overwritten. The embedded digest covers the object before
adding `receipt_sha256` and `digest_scope`, serialized with sorted keys, ASCII,
compact separators and nonfinite values forbidden. It is not a digital signature.
JSON duplicate members, nonfinite values, malformed text and credential-like
strings are rejected without printing their values.

The new `tests/test_queue_intake.py` suite is collected by the existing base Python
workflow's `pytest -q tests`; no workflow or privilege changes are required. It
uses synthetic fixtures and the real non-executing launch contract, prohibits
model-stack imports, reproduces the legacy wrong-answer/uppercase-ABSTAIN defects,
and checks exact populations, response retention and immutable output creation.

```text
python -m unittest discover -s tests -p test_queue_intake.py -v
python -m pytest -q tests/test_launch_contract.py tests/test_forge_preflight.py tests/test_audit_training_portfolio.py
```

For actual training, first qualify a task-specific admitted config and resource
window through the existing owner-controlled path. Run existing preflight, use
`local_only: true`, keep new candidate outputs separate from published identities,
and qualify held-out results before requesting publication. This module exposes
no train, upload, restart, token, credential-rotation or approval command.

## Serving-catalog readback boundary

The optional advertised-settings projection removes only per-response model
creation timestamps and permission identifiers/timestamps observed varying in
vLLM catalogs. It retains model ID, root, context limits, actual permission flags
and every other field, including unrecognized additions. Tests prove changes in
those retained fields remain visible. Equal projections are not immutable-weight
verification. Raw before/after hashes and initial rejected analyses should be
retained; corrected interpretation is a new record naming the earlier analysis.
