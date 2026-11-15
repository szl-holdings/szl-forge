# Evidence to decision: eight-case acceptance example

**Proposal-only.** Nothing here has action authority. Nothing is signed, sent, written or executed. All data is synthetic: fictional software-ops facts for two fictional tenants. There is no clinical, personal, financial or tactical data. This example makes no network, model or provider calls. Every generation goes to a deterministic local spy model.

## What it shows

A governed request path either answers from admissible evidence, with a citation bound to a verbatim excerpt and a pinned digest, or refuses at a named point with a stable reason code. It makes zero model calls when it refuses before generation. The path is checked against a frozen answer key and compared with a plain-prompting path over the same corpus and the same spy.

## Run (Python >= 3.11, stdlib only)

```
cd examples/evidence_to_decision
python acceptance.py                       # frozen 8 cases + 12 scenarios, 7 guard cases, 2 freeze controls
python -m unittest discover -s . -p "test_*.py" -t .
python mutation_sweep.py                   # removes each tagged harness guard; refuses if the clean suite is red
```

`acceptance.py` exits 0 only if every part passes. Exit 3 means a frozen input changed and the run was refused.

The mutation sweep distinguishes completed behavioral failures from invalid execution.
Timeouts, bare crashes, import/setup failures, incomplete acceptance reports and changed
test counts never count as kills. A caught guard exception remains a behavioral failure
under the frozen acceptance rules. Unit errors qualify only when their innermost traceback
frame is the copied mutated harness, with the same test and skip counts as the clean run.
The result discloses baseline coverage and every layer's outcome; invalid evidence makes
the sweep exit 1. Existing disabled tests remain disclosed rather than silently counted.

To include the production seam tests in the temporary sweep copies, set
`E2D_FORGE_ROOT` to the absolute repository root containing `inference/production.py`.
Without it those optional tests are skipped in copied directories. The deliberately
disabled planted-failure test is normally skipped in either mode.

## Files

| File | Role |
|---|---|
| `eight_case_harness.py` | Harness: spy model, governed controller, plain comparator, scoring |
| `request_lifecycle.py` | Request-lifecycle reference model (pinned by `FREEZE.json`) |
| `cases.json`, `corpus_manifest.json`, `corpus/` | Frozen case inputs and synthetic corpus |
| `frozen_answer_key.json`, `FREEZE.json` | Frozen labels and hashes, written before the harness existed |
| `guard_cases.json`, `guard_answer_key.json`, `GUARD_FREEZE.json` | Guard-coverage extension, pinned before its first run |
| `acceptance.py` | One-command end-to-end acceptance |
| `a11oy_forge_envelope_adapter.py` | Reference adapter: rights re-checked at retrieval, hydration and egress |
| `forge_seam_bridge.py` | Reference glue from the adapter to `inference/production.py` seams |
| `mutation_sweep.py` | Guard-removal sweep with red-baseline refusal |
| `test_*.py` | unittest suites; `test_forge_production_seams.py` skips if `inference/production.py` is absent |
| `SPEC.md` | Case table, refusal points, frozen rules, guard extension |
| `MANIFEST.json` | Path, bytes and sha256 of every file in this directory |

## Labels

MEASURED means it ran here against the spy. NOT_RUN means it was not executed. UNKNOWN means there is no basis to state it. Governance properties are MEASURED against the spy only.

## NOT_RUN / UNKNOWN

- Answer correctness against a real model: NOT_RUN.
- Total and first-token latency: NOT_RUN. Spy wall time is not model latency and is not reported.
- Token counts and cost: UNKNOWN (no tokenizer pinned, no provider, no dated pricing).
- Energy: NOT_RUN (no meter).
- The real envelope witness and the real retrieval/hydration backends: NOT_RUN. Local spies stand in for them.
- Real-model trial counts and quality thresholds: PROPOSED, not set.

## Continuous integration

`tests/test_evidence_to_decision_example.py` collects the example suites and acceptance
command under the repository's base gate. `tests/test_evidence_mutation_outcomes.py`
checks the mutation scorer's failure classification, baseline refusal and cleanup.
The full mutation sweep remains a separate command. Repository-wide lint applies here.

Licence: the repository licence (Apache-2.0).
