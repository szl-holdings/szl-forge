# Windows host-memory guard for supervised v3 attempts

This source implements the containment repair tracked in
[Forge #546](https://github.com/szl-holdings/szl-forge/issues/546). The reported
owner smoke reached model loading with about 1.4 GiB of available Windows
physical memory and zero optimizer steps. WSL guest memory and GPU telemetry
do not measure the Windows host's remaining physical memory or system commit.

The guard must admit the attempt before the isolated model worker is launched.
It then remains independent of the GPU supervisor's sampling loop. A healthy
guard is a necessary local evidence input; it is not proof that the complete
training run fits, that training finished, or that a model qualifies for release.

## Native observations and provisional thresholds

`windows_host_memory_sampler.ps1` uses Windows PowerShell 5.1 through the existing
WSL interop socket. Its source bytes, the guard, supervisor, launcher, and other
required lane components must match the exact committed source before use.
The observer calls native APIs in a persistent process; it does not import the
model, open held-out data, or receive a credential-bearing environment.

| Observation or rule | Source contract |
|---|---|
| Available physical memory | `GlobalMemoryStatusEx().ullAvailPhys` |
| Available system commit | `(GetPerformanceInfo().CommitLimit - CommitTotal) * PageSize` |
| Available physical floor | 8 GiB, provisional |
| Available system commit floor | 8 GiB, provisional |
| Native sampling interval | 2 seconds |
| Maximum native/WSL lease gap | 8 seconds |
| Native UTC versus receipt-time skew | At most 5 seconds |
| Identity | Exact run ID, source revision, and native sampler source SHA-256 |
| Ordering | Contiguous sequence, advancing native monotonic clock, advancing WSL receipt clock |

Available **process** commit, `MEMORYSTATUSEX.ullAvailPageFile`, is not used for
the system-wide commit floor. The API definitions are documented by Microsoft:
[MEMORYSTATUSEX](https://learn.microsoft.com/windows/win32/api/sysinfoapi/ns-sysinfoapi-memorystatusex),
[GlobalMemoryStatusEx](https://learn.microsoft.com/windows/win32/api/sysinfoapi/nf-sysinfoapi-globalmemorystatusex),
and [PERFORMANCE_INFORMATION](https://learn.microsoft.com/windows/win32/api/psapi/ns-psapi-performance_information).

Both floors are conservative admission and stop conditions, not measured
capacity estimates. They cannot be lowered by a CLI argument. The existing
80 C thermal ceiling, GPU cadence, source identity, worker isolation, and
training/evaluation split rules remain required.

## Independent containment and termination

The supervisor starts `szl-ra3-host-memory-<runId>.service` before GPU readiness.
The worker binds to both this guard and its exact supervisor with systemd
`BindsTo` and `After`. A crashed or failed guard therefore stops the worker even
if the GPU supervisor is blocked in another observation. The guard itself is
bound to the supervisor. The successful guard unit retains its exit result
until the supervisor exits; a failed helper does not become a successful
retained service.

Each usable sample is written and fsynced before its lease is published in the
supervisor-owned report directory, outside the worker filesystem namespace.
Missing, malformed, stale, out-of-order, wrong-source, low-memory, or terminated
native observation causes a sticky veto. The native observer also exits when
its controlling pipe closes, so helper death does not intentionally leave an
orphaned Windows sampler.

On failure the guard first attempts to flush the native witness and sticky
abort, then stops the exact worker. Unit name, invocation ID, and control group
are checked before signaling; a replacement invocation is never escalated.
The ordinary worker stop has a 20-second bound, followed by an exact-invocation
kill escalation if needed and a bounded cgroup-empty check. Failure to confirm
containment permits only an exact-invocation stop of the outer supervisor.
An evidence write failure cannot postpone the worker stop.

The supervisor records a bounded `HOST_MEMORY_*` terminal cause and exit code
81. Failure does not turn into successful training, an evaluation input, or
release eligibility. This is a cooperative boundary within the same owner
account. A SHA-256 digest binds retained bytes; it is not authentication against
a malicious process with that account's authority.

## Completion and downstream evidence

The supervisor can request completion only after its worker has finished and
the training payload has passed the existing checks. The guard requires a new
native observation after that request, coverage of the complete worker
monotonic interval, the exact successful worker result, and an empty worker
cgroup. The supervisor then requires the exact helper's successful process
exit before accepting the terminal file. A crash after writing a healthy seal,
a mismatched request, or an abort file is still a veto.

The private attempt report directory contains:

| File | Purpose |
|---|---|
| `host-memory-sample-NNNN.json` | Write-once native sample and WSL receipt times |
| `host-memory-lease.json` | Atomic latest lease with full sample-list digest |
| `host-memory-abort.json` | Write-once veto, when a failure occurs |
| `host-memory-finalize.json` | Exact supervisor completion request |
| `host-memory-terminal.json` | Write-once terminal report with all samples and unit bindings |

The evaluator replays the native evidence and rejects missing, stale, failed,
aborted, incomplete, or source-mismatched reports. Authentication also requires
the full report: `mint_training_receipt(..., host_memory_report=report)` binds
its digest as `hostMemoryGuardReportSha256`. Development and final evaluation
receipts must reference that same authenticated digest.

Release preparation requires `--host-memory-report` pointing to
`host-memory-terminal.json` beside the exact supervisor report. It rejects an
abort in that directory and replays the guard report before assembling the
authenticated packet. The publisher repeats the packet's evidence validation.
The guard report remains a `NOT_FOR_PUBLICATION` evidence file, with the role
`MEASURED_HOST_MEMORY_GUARD_REPORT`; it is not uploaded as a public model asset.
Older training evidence without this guard is not accepted by this source
contract. There is no compatibility switch to waive the new requirement.

## Verification and remaining owner proof

The offline suite covers threshold edges, missing/stale/error samples, replay
and identity mismatches, sticky aborts, write failure, exact-worker stop order,
stop timeout and replacement-invocation controls, incomplete runtime coverage,
crash after a healthy seal, mismatched completion requests, evaluator vetoes,
and authenticated training/evaluation binding. A harmless local subprocess
control verifies a real worker is stopped on synthetic pressure while an
unrelated process remains alive. Its unit-control adapter is simulated; it does
not claim to prove systemd or WSL containment.

The Windows CI control executes the real native APIs using both Windows
PowerShell 5.1 and PowerShell 7. It also checks consecutive native samples and
bounded exit when the controlling pipe closes. The emitted records explicitly
state that training, WSL interop, owner worker containment, full-run capacity,
and model qualification have not been proved by those controls.

Run the native control with PowerShell 7. It uses that running host's
`pwsh.exe`, so duplicate installations on PATH cannot combine into an invalid
executable name. Timestamp checks use the original JSON string and require
explicit UTC within five seconds; PowerShell's automatic conversion to the
machine's local timezone does not change the native evidence.

Before an owner training retry is considered cleared, retain actual evidence
from the intended Windows/WSL host for all of the following:

1. Native physical and system-commit sampling through the admitted WSL interop
   socket and exact source components.
2. A harmless isolated worker stopped under injected low-memory, missing/stale
   sample, native watcher death, and helper death conditions; unrelated
   processes remain alive and the exact worker cgroup is empty.
3. Completion/abort races and a bounded stop-timeout exercise with retained
   native witness, helper/worker identities, and final supervisor receipt.
4. Only after these fault controls pass, a guarded smoke that retains the
   existing 80 C, host-memory, source, isolation, and data gates.

Keep #546's owner-hardware proof open until those receipts exist. Source tests
and native API CI do not provide those owner receipts or authorize a claim of
successful full training.
