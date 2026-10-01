# SZL Frontier Operator

This standalone Python operator inspects protected GitHub pull requests, captures failed logs with redaction, and independently checks the Forge evaluation publication on Hugging Face. It runs once and exits. It creates no scheduler, background controller, training job, provider deployment, or automatic merge queue.

Requirements: Python 3.11 or later and an installed, authenticated `gh` CLI with read access to repository policy, Actions logs, and artifacts. The code uses only the Python standard library. Public Hugging Face evidence reads require no token. Never paste a token into a command or report.

Run these commands from the Forge repository root in PowerShell. The offline adversarial tests also run in the repository's existing Python CI:

```powershell
python -m pytest -q tests/test_frontier_operator.py tests/test_frontier_operator_artifact.py
python -I -B tools/szl_frontier_operator.py report --output-dir reports/frontier-operator
python -I -B tools/szl_frontier_operator.py snapshot --repo szl-holdings/platform --pr 764
python -I -B tools/szl_frontier_operator.py failed-logs --repo szl-holdings/platform --pr 764 --output-dir reports/failed-logs
python -I -B tools/szl_frontier_operator.py verify-forge-174
python -I -B tools/szl_frontier_operator.py hf-dataset
```

`report` saves `frontier-operator-status.json`. `failed-logs` saves an index and only failed workflow logs for the currently observed PR head. All commands print machine-readable JSON. Exit code 0 means that the requested observation completed without a reported blocker; 2 means failed verification or a conservative refusal; 3 means an external read was unavailable or an attempted merge could not be conclusively verified. `MEASURED` means observed, not qualified. The report intentionally reports an already merged PR as ineligible for another merge.

The operator checks all pages of checks, statuses, workflow runs, review threads, commits, and repository rules. It requires source-bound signing evidence, current protected checks and resolved required review state; zero checks cannot pass. Skipped or neutral conclusions require an explicit policy entry naming the kind, check name, exact head SHA, conclusion, app ID for checks, reason, and source reference. An exception file does not grant review authority or bypass GitHub protection.

Linear-history requirements exclude the merge-commit method. Team/file-pattern reviewer rules are not implemented and therefore block qualification when configured, even if an aggregate review decision says approved. Hugging Face reads enforce HTTPS and the exact approved hostname before each redirect and before consuming the final response.

`guard-merge` is read-only by default. Its explicit `--execute --expected-head <40-character SHA>` option performs a single protected merge only after two fresh qualifications agree on head, base and policy. It does not acquire a cross-task writer lease. Confirm ownership of the specific mutation before using execution mode; the operator cannot replace another task's source or provider ownership. A missing response after an issued merge is `UNKNOWN_AFTER_ATTEMPT` and requires authoritative readback before any retry.

The Forge verifier pins the merged PR's exact push-to-main source revision, workflow and attempt. It verifies the artifact's GitHub SHA-256 digest, reads a bounded ZIP entirely in memory, and rejects duplicate, unsafe, encrypted, symlink, excessive or malformed entries. It compares receipt, bundle and summary bytes with both the artifact's recorded immutable Hugging Face publication revision and the currently observed dataset revision. It then validates source module hashes, fixture coverage and requests, independently recomputes response scores and aggregate comparisons, checks provider routes and attempts, and rereads the immutable model metadata files. Remote Python is parsed as data and never executed. The independent scorer supports the reviewed Forge core hash and refuses a changed core pending a deliberate implementation review.

The receipt remains unsigned, and GitHub artifact binding does not establish an independent author's signature or a new inference replay. A source model revision does not attest the provider's served weights. Provider execution revision, hardware and runtime version must remain unavailable. Four public synthetic fixtures are not held-out evaluation, broad capability measurement, production qualification, or AGI evidence.

Historical Forge PR #174 has verifiable source, workflow, artifact and publication integrity, but its fallback records a successful HTTP transport with malformed model JSON. The command therefore returns `MEASURED_FAIL` while preserving `pipeline_integrity: VERIFIED`, `semantic_safety: UNQUALIFIED`, and `model_qualification: UNQUALIFIED`. The newer deterministic fallback contract is understood separately: an exact static escalation envelope can pass the guard while the baseline model's semantics remain `FAILED`. Every verified result retains production `HOLD`, promotion `NONE`, and no AGI claim. The default #174 command checks that historical PR; it does not silently substitute a newer run.

HTTP objects, subprocess stdout/stderr, artifact expansion, members, pagination and command duration are bounded. Logs and reported errors are redacted before local persistence. Network errors are retained as unavailable evidence. Domain application contracts, current deployments, model training, independent replay, and production authorization are outside this operator's verification scope.
