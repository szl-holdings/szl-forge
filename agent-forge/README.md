# Owned Agent Control v2

Owned Agent Control (OAC) is a fail-closed controller for **explicitly registered
local process trees that OAC launches on Windows**. It combines Windows Job
Object lifecycle enforcement, quorum-signed one-shot isolation requests,
append-only context evidence, and a locally hash-chained audit ledger.

The package is beta software. Its Windows enforcement must pass the real
Windows self-test before an installation is treated as qualified.

## Truth boundary

| Capability | What is implemented | What is not claimed |
|---|---|---|
| Process lifecycle | Starts a registered executable suspended, attaches it to an OAC-owned Windows Job Object, then resumes it | Control over services, containers, VMs, remote agents, or processes OAC did not launch |
| Isolation | Terminates the exact supervised process tree and durably blocks later OAC starts for that target | A hostile-security sandbox or containment from other processes running as the same Windows identity |
| Authorization | Short-lived Ed25519 quorum envelopes, exact audience/target binding, and one-shot replay state | Cloud IAM, provider credential revocation, network policy, or remote account control |
| Evidence | SQLite constraints, append-only triggers, signed-request binding, and a local hash-chained audit ledger | Immutable, independently witnessed, or externally notarized evidence |
| Context | Deterministic enriched-context traces, entropy allocation, cross-step consistency, legal state-transition and policy gates | Model safety certification, permission expansion, or a substitute for process enforcement |
| A11oy | Schema-bound read-only projection of context and audit evidence | Process-control authority; `a11oy_process_control` is always `false` |

Context generation and Windows isolation are intentionally separate. Context
must stabilize before a target may start, but an authorized stop remains
available even when context generation fails.

Context inputs are trusted operator assertions. OAC validates their schema,
state transitions, repeated-value consistency, and declared policy flags; it
does not independently determine whether those assertions are semantically safe.

## Requirements

- Windows 10/11 or Windows Server 2022 for real process enforcement
- 64-bit CPython 3.11, 3.12, or 3.13
- PowerShell 7 or newer for the PowerShell runbook examples
- NTFS state directory on a fixed local drive
- Two or more independent Ed25519 operator keys for isolation quorum

Portable schema, parser, context, audit, and packaging tests run on Linux. A
Linux test pass does **not** qualify the Windows enforcement boundary.

## Install from source

From the repository root:

```powershell
py -3.12 -m venv .venv-oac
.\.venv-oac\Scripts\python.exe -m pip install --upgrade pip
.\.venv-oac\Scripts\python.exe -m pip install --requirement .\agent-forge\requirements.lock
.\.venv-oac\Scripts\python.exe -m pip install --no-deps .\agent-forge
.\.venv-oac\Scripts\owned-agent-control.exe --version
.\.venv-oac\Scripts\owned-agent-control.exe doctor
.\.venv-oac\Scripts\owned-agent-control.exe self-test
```

`doctor` reports readiness without changing controller state. `self-test`
creates disposable state and launches a harmless parent/child target. Only the
`VERIFIED_LOCAL_WINDOWS_ENFORCEMENT` result qualifies that machine/runtime.
`doctor` is not a complete ACL, sync-folder, endpoint-security, or hostile
same-identity review.

## Controlled workflow

The full operator procedure is in [RUNBOOK.md](RUNBOOK.md). At a high level:

1. Initialize the protected state directory.
2. Generate and add at least two operator public keys; seal the trust store.
3. Register an exact executable path, argv list, working directory, and hash.
4. Generate and record a stabilized context trace.
5. Start the registered target under the supervisor.
6. Create a short-lived isolation request, collect quorum signatures, verify,
   then apply it once.
7. Verify the audit chain and export A11oy evidence if needed.

Example context input: [examples/context-input.json](examples/context-input.json).

```powershell
$State = 'C:\ProgramData\SZL\OwnedAgentControl'
$Oac = '.\.venv-oac\Scripts\owned-agent-control.exe'

& $Oac init --state-dir $State
& $Oac register-demo --state-dir $State --target owned-agent:demo
& $Oac context-generate --state-dir $State --target owned-agent:demo `
  --input .\agent-forge\examples\context-input.json
& $Oac context-show --state-dir $State --target owned-agent:demo
```

## A11oy projection

`context-export` writes a projection validated against
[`schemas/a11oy-owned-agent-control-projection.schema.json`](schemas/a11oy-owned-agent-control-projection.schema.json).
The same schema ships inside the Python package. The projection is evidence for
ingestion or display and contains these fixed truth markers:

```json
{
  "capabilities": {
    "a11oy_process_control": false,
    "a11oy_read_only_projection": true,
    "local_windows_supervisor_control": true
  }
}
```

A11oy remains a verifier/projector. It cannot start, stop, sign for, or mutate
the locally controlled process.

## Development and qualification

```bash
python -m unittest discover -s agent-forge/tests -v
python -m compileall -q agent-forge/src agent-forge/tests
python -m build --wheel agent-forge
```

CI runs the portable suite on Python 3.11–3.13, builds and installs the wheel,
and runs the behavioral self-test on a GitHub-hosted Windows Server 2022 runner.
See [THREAT-MODEL.md](THREAT-MODEL.md) before production use.

## Status language

- `VERIFIED_LOCAL_WINDOWS_ENFORCEMENT`: the exact Windows runtime completed the
  behavioral process-tree isolation self-test.
- `PORTABLE_CONTRACTS_VERIFIED`: non-Windows contracts passed; Windows process
  enforcement remains unverified.
- `NOT_READY`: a required dependency, platform property, or state protection is
  absent. Do not relabel this as operational.
