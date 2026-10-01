# Native estate repair operator

This standalone Python 3.11+ program uses the owner's already authenticated
GitHub CLI. It invokes **only** A11oy's existing `Estate release train` workflow,
with its approved exact-source vertical plan. It is not another publisher,
scheduler, credential broker, or production-readiness certificate.

Forge owns this local operator; A11oy owns the release train, source pinning,
canonical HF writer, readiness probes and release evidence. Tracks Forge #293,
A11oy #2010 and Frontier #96. The separate Omni evaluation remains HOLD.

## Run

The reviewed A11oy repository is `szl-holdings/a11oy`, at exact source revision
`0f389189edf9930068f1dfbe9cdda851ca197dda` (supplied review evidence E1,
`szl.reviewed-native-control/v1`). This review binds six Git blob identities:

- `.github/workflows/estate-release-train.yml`:
  `fd4611973b1b271d9b8e2bec5d4e0e102b9d1792`, the existing native release workflow.
- `.github/workflows/hf-sync.yml`:
  `e3913a40e9c4884eb60707a0058504ada9e142cc`, the canonical HF writer workflow,
  including the current Finance rebind job covered by E1's review.
- `scripts/estate_repair_dispatch.py`:
  `81f0e83729390ee66b633a8d45e5f1139374ba41`, the unchanged source-bound planner.
- `config/estate-release-train.v1.json`:
  `84e998eca4f2f4b6cabc992f12a1c466875be848`, the unchanged release configuration.
- `scripts/estate_child_completion.py`:
  `077524c306189dd0315e07c492a49081dba17399`, the complete-set child completion barrier.
- `.github/workflows/repair-cloudflare-product-edge.yml`:
  `ad703a8be9601913601d1e53ff070bac87d861a6`, the canonical edge workflow.

E1 reports that native preflight refused the changed release workflow identity;
the previous operator bound four controls and omitted the child completion helper
and edge workflow. Updating these pins does not approve an alternate publisher.

From the reviewed Forge checkout, inspect first (read-only):

```powershell
python -I -B tools/szl_estate_operator.py --expected-source 0f389189edf9930068f1dfbe9cdda851ca197dda
```

This is an exact reviewed source, not a floating alias or a claim that protected
main is still at that revision. Preflight must establish that independently.
E1 keeps publication HOLD: fresh writer-state enumeration is rate-limited and
current source receipt/inventory failures remain under investigation. No dispatch,
provider write, publication or training is authorized or certified by this review.

Only after resolving HOLD and separately approving the exact native repair does
the execution syntax apply:

```powershell
python -I -B tools/szl_estate_operator.py --expected-source 0f389189edf9930068f1dfbe9cdda851ca197dda --execute --wait
```

No administrator shell, pasted token, Python package install, direct Space write,
new allocation, local training, benchmark, or protection change is required.
The CLI's native GitHub permissions and the workflow's existing controls remain
mandatory. If source or any pinned controller file changes, stop and review;
do not automatically replace the approved SHA or the controller pins.

Focused offline regression coverage is in
`tests/test_szl_estate_controller_pins.py`. It compares the full mapping with E1
and uses explicitly synthetic Git blobs to exercise all six source-bound reads,
per-control identity/byte refusals, active writers, rate limits and source advance.
Run it alongside the existing suite:

```powershell
python -m unittest discover -s tests -p test_szl_estate_controller_pins.py -v
```

Synthetic fixture success cannot verify the actual reviewed blob contents or live
writer/receipt state. This edit proposal includes no executed checks or live actions.

## What execution means

The default command only performs reads. `--execute` explicitly approves the
existing repair inputs for the given source. Preflight requires protected main,
exact Git blob identities and bytes for all six reviewed controls, an active
workflow and a complete bounded active-writer enumeration. It checks source again
immediately before the request. That check is not a distributed lock: native writer
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
