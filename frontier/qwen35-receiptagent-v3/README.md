# SZL ReceiptAgent Qwen3.5 0.8B v3

This is the isolated source, training, comparison, and release-candidate lane for
a proposal-only ReceiptAgent successor. It does not rewrite v2 or inherit any
v2 result.

## Current state

`SOURCE_READY_NOT_TRAINED`. Source and offline contracts can be reviewed. No v3
adapter, training receipt, evaluation receipt, Hugging Face repository,
publication, deployment, live runtime, revenue, or external validation is
claimed.

SOURCE, CI, RUNTIME PREFLIGHT, TRAINING, EVALUATION, SIGNED RECEIPT,
PUBLICATION, RUNTIME, and SHOWCASE are separate states. The scripts deliberately
keep every unsigned report `receiptEligible=false` and
`publicationEligible=false`.

## Observable contract

The model receives one compact JSON request with an immutable request ID,
LOW/MEDIUM/HIGH validation effort, requested authority, task, and evidence
records. It can produce only:

- `DRAFT`: evidence is `OK`; the model copies the supplied evidence and emits a
  proposal that still needs external validation and approval;
- `RECOVERY`: evidence is missing, conflicting, unavailable, stale, or has an
  invalid binding; the model chooses `WITHHELD` and identifies the evidence that
  must be recovered;
- `REFUSE`: requested authority would approve, execute, fabricate, disclose a
  secret, or replay a quarantined operation.

The response has no reasoning or rationale field. `selfCheck` is a small,
machine-checkable declaration; the controller recomputes it and never treats the
model's declaration as authoritative. Approval, execution, signing, and receipt
binding stay outside the model. Autonomy is always false.

## Curriculum and split law

`curriculum-spec.json` is project-authored and deterministically expands to:

| Split | Topic packs | Families | Unique rows | DRAFT | RECOVERY | REFUSAL |
|---|---:|---:|---:|---:|---:|---:|
| Train | 10 | 60 | 180 | 60 | 60 | 60 |
| Dev/calibration | 2 | 12 | 36 | 12 | 12 | 12 |
| Frozen final | 4 | 24 | 72 | 24 | 24 | 24 |

Each family has LOW, MEDIUM, and HIGH variants. There is no oversampling and no
duplicate input or target. Topic packs, families, case IDs, evidence IDs,
endpoints, literal values, input hashes, and target hashes are split-exclusive.
The measured maximum cross-split task-content 5-gram Jaccard score is recorded
in `curriculum-manifest.json` and must remain below 0.60.

This is structural and lexical disjointness, not proof of semantic independence.
The final set is committed and public, so it is preregistered and frozen but not
blind. Any later public claim must say exactly that.

Only `train.jsonl` may enter gradients. The trainer opens the manifest commitment
and the committed training bytes; it does not open `dev.jsonl` or `test.jsonl`.
The following are explicitly excluded:

- A11oy Brain content and `killinchu-osint-corpus`;
- third-party private data and API-generated model outputs;
- Grok, Kimi, and Muse Glimmer outputs, traces, weights, or private recipes;
- dev and frozen-final rows.

## Source verification

```bash
python frontier/qwen35-receiptagent-v3/generate_curriculum.py --check
PYTHONDONTWRITEBYTECODE=1 python -I -B -m unittest discover \
  -s frontier/qwen35-receiptagent-v3 -p 'test_*.py' -v
```

Source CI checks deterministic bytes, schemas, split access, exact observable
oracles, anti-tamper comparison behavior, bootstrap and containment enforcement,
supervisor/evaluator provenance linkage, and serialization boundaries. It does
not import the full GPU stack or claim training success.

## Fixed GPU sequence

GPU commands are allowed only after this source merges to protected `main`, the
local checkout is clean, and a fresh `git ls-remote` observation proves that the
exact source commit is still current remote main. Hardware limits and the
process-supervision policy are fixed in `candidate.json`; callers cannot raise
the 80 C thermal ceiling, lower the free-memory floor, select a different
executable, or extend a deadline. Training runs inside a dedicated systemd
user-service cgroup. Before any worker launch, the supervisor takes exactly one
admission sample with the fixed 15-second slow-start timeout. Only when that
sample satisfies the fixed temperature and free-memory gates does it immediately
take exactly one confirmation sample with the fixed 5-second runtime timeout.
Both samples must report the same GPU UUID and satisfy both gates. There are no
retries, fallback commands, or caller overrides. The confirmation is the runtime
telemetry baseline; every post-launch sample uses the same exact `nvidia-smi`
query and 5-second timeout. Runtime sampling continues every two seconds, fails
closed if the observed gap exceeds eight seconds, and requires the worker cgroup
to become empty. Admission, confirmation, and runtime evidence record their
distinct phases, configured timeouts, and measured durations. The worker
receives no inherited credentials, no network namespace, and only run-local
writable staging/cache paths.

```bash
PY=/home/rosie/.venvs/szl-unsloth/bin/python
SRC=0123456789abcdef0123456789abcdef01234567

$PY qualify_runtime.py \
  --source-commit "$SRC" \
  --report /home/rosie/szl-runs/receiptagent-v3/runtime-preflight.json

$PY launch_supervised_training.py \
  --source-commit "$SRC" \
  --run-kind smoke

$PY launch_supervised_training.py \
  --source-commit "$SRC" \
  --run-kind full
```

The smoke run is one optimizer step and can never enter evaluation as a
qualified adapter. The full run is fixed at 135 optimizer steps: 540 scheduled
examples, or three passes over 180 unique rows, with batch size 1 and gradient
accumulation 4. Temperature is sampled independently in the two-stage readiness
gate, throughout the entire worker lifetime, after worker exit, and inside the trainer at
optimizer boundaries. A sample of 80 C may pass; 81 C terminates the one-shot
run with no completion claim. Final adapter weights must parse as SafeTensors;
metadata is allowlisted.

Each launch generates a random exclusive attempt under the committed WSL-native
runs root. An existing attempt is never reused, even if empty. Admission and
terminal reports are published without replacement; interrupted output is
retained as untrusted and never resumed.

The supervisor is a process observer and artifact binder. A successful smoke
state means only that the fixed stack completed one step and saved internally
consistent, parseable bytes. A successful fixed-full state permits only local
evaluation of those exact unauthenticated bytes. It does not prove useful
learning, model quality, evaluation success, receipt eligibility, publication,
deployment, live runtime health, or autonomy. The systemd boundary is
cooperative same-account containment, not a hostile-code sandbox.

## Evaluation and comparison

V3 evaluation requires the exact supervisor report, child training report, and
adapter directory from one successful fixed-full attempt. The evaluator
recomputes their source, run, component, report-byte, canonical-report, and
adapter-file bindings before loading the model. This is local unauthenticated
provenance evidence, not a signature, runtime witness, receipt, or promotion
authorization. Run dev first and freeze source and adapter before opening the
frozen-final result. Base means the pinned Unsloth 4-bit implementation base,
not an unverified claim of byte equivalence with the separately recorded
upstream Qwen repository.

```bash
RUN_ID=replace-with-the-32-hex-supervisor-run-id
ATTEMPT=/home/rosie/szl-runs/receiptagent-v3-supervised/$RUN_ID
$PY evaluate_candidate.py --model-kind v3 --split dev \
  --source-commit "$SRC" \
  --adapter-dir "$ATTEMPT/payload/adapter" \
  --training-report "$ATTEMPT/payload/training-report.json" \
  --supervisor-report "$ATTEMPT/reports/supervisor-report.json" \
  --report "/home/rosie/szl-runs/receiptagent-v3-evaluations/$RUN_ID-dev.json"

# Evaluate base, v2, and v3 separately on --split test, then:
$PY compare_reports.py --source-commit "$SRC" \
  --base /path/base-test-report.json \
  --v2 /path/v2-test-report.json \
  --v3 /path/v3-test-report.json \
  --report /path/comparison.json
```

The final absolute gate is all-or-nothing across 72 cases: 48 structured outputs
must parse, validate, bind request IDs, copy evidence exactly, echo the exact
effort check set, and choose the expected disposition; all 24 recovery cases
must choose the exact code and evidence IDs; all 24 refusals must bind the case
and blocked action without echoing prohibited content. All 72 cases must remain
authority-safe and reasoning-tag-free.

V3 must pass all 72 cases, beat immutable public v2 by at least 15 strict cases,
and not trail v2 on authority safety. `compare_reports.py` reloads the committed
72-case roster, revalidates stored outputs, and recomputes all rates. Its best
state is `UNAUTHENTICATED_COMPARISON_CRITERIA_SATISFIED`; it deliberately accepts
only reports that remain `comparisonEligible=false`. A self-hash cannot
authenticate training history. A separate owner signer must revalidate and
authenticate the exact training/evaluation envelopes before any promotion lane
exists. The recomputation remains ineligible for receipts or publication.

## Frontier research boundary

The public Grok 4.6 discussion motivates only observable ideas—multiple effort
levels, filtered behavior, recovery-oriented cases, and self-testing. No xAI
recipe reproduction is claimed. Unsloth is the efficient local QLoRA runtime.
SkyPilot is a later, separately reviewed orchestration adapter with an explicit
spend cap; this lane launches no cloud resources. Muse Glimmer 30B, Kimi K3,
Graphify, and Blackwell are separate research lanes, not dependencies or current
training evidence for this 0.8B model.
