# Native estate repair operator

This standalone Python 3.11+ program uses the owner's already authenticated
GitHub CLI. It invokes **only** A11oy's existing `Estate release train` workflow,
with its approved exact-source vertical plan. It is not another publisher,
scheduler, credential broker, or production-readiness certificate.

Forge owns this local operator; A11oy owns the release train, source pinning,
canonical HF writer, readiness probes and release evidence. Tracks Forge #293,
A11oy #2010 and Frontier #96. The separate Omni evaluation remains HOLD.

## Run

From the reviewed Forge checkout, inspect first (read-only):

```powershell
python -I -B tools/szl_estate_operator.py --expected-source a7bf14a576bc79b4db1945384c55a3c56b109670
```

The exact SHA above was observed on September 13; it is not a floating alias.
After reviewing the source and approving the native repair, run:

```powershell
python -I -B tools/szl_estate_operator.py --expected-source a7bf14a576bc79b4db1945384c55a3c56b109670 --execute --wait
```

No administrator shell, pasted token, Python package install, direct Space write,
new allocation, local training, benchmark, or protection change is required.
The CLI's native GitHub permissions and the workflow's existing controls remain
mandatory. If source or any pinned controller file changes, stop and review;
do not automatically replace the approved SHA or the controller pins.

## What execution means

The default command only performs reads. `--execute` explicitly approves the
existing repair inputs for the given source. Preflight requires protected main,
exact native workflow/config/planner Git blob bytes, an active workflow and a
complete bounded active-writer enumeration. It checks source again immediately
before the request. That check is not a distributed lock: native writer
serialization and downstream source/plan checks still own the final race.

The operator creates a source-scoped exclusive journal under
`~/.szl-estate-operator/`, flushes the mutation intent to disk before POST, and
attempts dispatch once. A missing/invalid response is `UNKNOWN_AFTER_ATTEMPT`,
never an automatic retry. GitHub API version 2026-03-10 returns the exact run ID;
an empty legacy response remains uncertain. The operator never guesses the
newest matching run. Repeated execution for the same source refuses to replace
its journal. Do not delete it to bypass an uncertain outcome.

A known run can be reobserved without any mutation:

```powershell
python -I -B tools/szl_estate_operator.py --expected-source <approved-sha> --inspect-run <returned-id> --wait
```

`--wait` blocks locally for at most 65 minutes; it does not create a background
service. After timeout the native workflow may continue; keep its ID and use
read-only recovery. An explicitly necessary retry after an inspected native
failure belongs to the normal Actions UI, not a hidden operator loop.

## Evidence bounds

Dispatch acceptance is not deployment or alignment. The observer validates the
exact repository/workflow/event/source/attempt, independently hashes the native
artifact, parses bounded regular ZIP members without extraction, checks the
linked estate/inventory bytes and public-only scope, preserves native blockers,
and rejects empty/missing/contradictory required-component observations. It binds
observation time to the native run and refreshes the required GitHub authority
vector before accepting an aligned native result. It does not independently
rerun network probes or authenticate an external author's signature.

`NATIVE_ALIGNMENT_RECEIPT_VERIFIED` means that bounded native evidence passed
these checks. `production_authorization` remains false. Optional components,
model quality, GPU behavior, runtime qualification and authenticated HF
inventory-v2 are not silently included. The original native terminal gate is
not weakened: a successful-looking receipt cannot override its failed run.

Exit 0: read-only preflight completed or native alignment receipt verified.
Exit 2: HOLD, pending, uncertain or refused. Read the state and retained run ID;
a nonzero exit never authorizes a blind repeated dispatch.

## Verification

```powershell
python -m unittest discover -s tests -p test_szl_estate_operator.py -v
python -O -m unittest discover -s tests -p test_szl_estate_operator.py -v
```

The dedicated workflow runs these offline tests on Linux/Windows and Python
3.11/3.12 without production credentials. Synthetic fixtures exercise refusal,
mutation uncertainty and receipt integrity; their successful tests do not prove
that a live repair was dispatched. Existing repository security and base tests
remain independent and mandatory.
