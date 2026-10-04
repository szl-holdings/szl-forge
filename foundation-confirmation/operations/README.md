# Local workbench startup

These Windows operations manage the original, unchanged Foundation Confirmation
v0.4 workbench. They install one **current-user logon** task, `SZL Foundation
Confirmation`, with an Interactive token and Limited privileges. No password or
administrator elevation is requested. Installation does not start the workbench.
This is local synthetic research execution, and the registered benchmark remains
**FAILED**. Public runtime deployment and readiness are qualified separately.

## Install and verify

Use Windows PowerShell 5.1 or newer as the intended desktop user. Supply the
absolute extracted lab directory and original sealed archive. Both must be below
that user's profile on the Windows system volume; network/device paths and
reparse points are rejected. These initial commands select legacy mode. To use
an isolated CPU environment, follow **Admit a CPU environment** below. For legacy
mode, install the original pinned Python dependencies
first. The Python selection is the original launcher's Python 3.11 preference,
then an installed `python` command under the current user's profile.

```powershell
.\install-workbench.ps1 -LabRoot 'C:\Users\USER\Documents\Foundation\lab' -ArchivePath 'C:\Users\USER\Documents\Foundation\release.zip' -Port 18767 -ValidateOnly
.\install-workbench.ps1 -LabRoot 'C:\Users\USER\Documents\Foundation\lab' -ArchivePath 'C:\Users\USER\Documents\Foundation\release.zip' -Port 18767
Start-ScheduledTask -TaskName 'SZL Foundation Confirmation' -TaskPath '\'
```

The admission boundary pins the original ZIP, release manifest, verifier, server,
launcher, and three executable research modules. The verifier is checked against
its known sealed SHA-256 **before** execution. It then verifies all 70 release
files, the three trained and three initial checkpoints, and 5,184 evaluation
records. A caller-supplied manifest cannot authorize changed executable source.
The frozen directory and its trials are preserved.

Managed copies and installation evidence live in
`%USERPROFILE%\Documents\SZL\FoundationConfirmation`, outside the release. The
explicit profile path is shared with native Task Scheduler; a packaged desktop
application can see a virtualized AppData directory that native tasks cannot
read. The path must pass the same profile, system-volume and reparse-point
checks as the frozen lab. The task invokes
that installed supervisor. Its action, arguments, current-user SID, trigger,
privileges, retry settings, and script hashes must match the owned receipt.
Existing conflicting tasks, receipts, or managed files are refused.

New installations use Scheduler priority `5` (Normal), recorded as
`task_priority: 5` in a `windows-installation/v2` receipt and checked again after
registration and before each launch. Priority mismatches fail admission. Historical
`v1` receipts without that field remain valid only with their original Scheduler
default priority `7` (BelowNormal) and the hash-bound historical bounded-recovery
installer, so an owned installation can be verified and
removed before upgrading. An existing installation is never silently reprioritized.
The process-priority mapping follows [Microsoft's Scheduler priority contract](https://learn.microsoft.com/en-us/windows/win32/taskschd/tasksettings-priority).

If an earlier installation used `%LOCALAPPDATA%\SZL\FoundationConfirmation`,
retain that installation evidence and verify its exact owned task with the
earlier scripts before removing that task. The new installer never takes over
an existing task or treats the earlier receipt as proof for the new directory.
The frozen lab and its trial receipts remain in place during this migration.

Before installation, run the validation-only helper with the same paths:

```powershell
.\test-startup.ps1 -LabRoot 'C:\Users\USER\Documents\Foundation\lab' -ArchivePath 'C:\Users\USER\Documents\Foundation\release.zip' -Port 18767
```

It parses the scripts, admits the complete sealed release, checks native task
definitions without registering them, and rejects changed archive/verifier
bytes, unsafe paths, extra task actions, elevated or foreign owners, and a changed
retry policy. If this lab is already running, it also witnesses readiness and
the process handle, and confirms that an occupied port cannot be admitted without
an ownership witness. Its temporary verifier fixture is removed within the
operations source directory; production tasks, services and trial receipts are
not changed.

The supervisor invokes the unchanged `start.ps1` with a hidden window and waits
for that launcher alone to exit. It then separately verifies and tracks the
running service; waiting for the launcher's entire descendant tree would block
those checks for the lifetime of the service. It checks
the service receipt against the process's exact executable and command arguments,
creation timestamp, PID, and exclusive loopback listener. The process handle is
compared at CIM's declared microsecond timestamp precision, with exact equality
and no time tolerance. The health response and all three checkpoint bindings
must pass before it waits for that owned
process. Task Scheduler therefore tracks the running service instead of a
launcher that has already exited. After that exact owned process exits, the
supervisor allows **at most two retries, 60 seconds apart**, within the same task
run. Each retry re-verifies the sealed source, installation, task, predecessor
and loopback port before invoking the unchanged launcher. Admission, ownership,
readiness and launcher errors stop the supervisor immediately; they do not
authorize retries. Scheduler-level retries are disabled so they cannot multiply
the supervisor's bounded budget. There is no endless restart loop. The task
permits manual demand starts and has no time limit while the owned service is
healthy. Its receipt declares `BOUNDED_SUPERVISOR_RETRY`, the retry limit and
interval, and the current recovery attempt.

Before every launch, an occupied port without the lab's verified service receipt
is rejected. The original launcher is not invoked in that case, so a port
conflict cannot overwrite its service receipt. Both the scheduled supervisor and
its original launcher request hidden windows.

Check `installation.json`, `supervisor-status.json`, the task's Running state,
`http://127.0.0.1:18767/api/status`, and the original `state/service.json`. A
registration receipt proves the registered task definition; separately retain
on-demand execution, owned-process restart, and unchanged trial receipt readbacks
before claiming startup has been verified. These scripts do not claim that a
machine reboot was tested or that execution begins before desktop logon.

CPU startup diagnostics retain `ENTERED`, `RETURNED` and `FAILED` phase events
with UTC and monotonic elapsed time. Supervisor and launcher processes each own
an exclusive JSONL file under
`%USERPROFILE%\Documents\SZL\FoundationConfirmationDiagnostics`, outside managed
Operations and the frozen lab. The CPU child writes its events to the existing
per-attempt stderr file identified by `launch.<attempt>.json`. Logs bind a
process/session and contain error types, without exception messages or argv.
They are unsigned diagnostic observations, not readiness, authorization, user
trial receipts or automatic recovery evidence. A missing `RETURNED` event shows
the last observed entered phase; it does not establish why that phase stopped.

Telemetry is advisory: an I/O failure reports `UNAVAILABLE` where its warning
stream remains writable and leaves the original operation and exception intact.
Disk exhaustion or process termination can leave missing or partial events.
Diagnostic I/O can add latency; it cannot extend the existing 180-second child
clock, authorize a retry, skip an admission or change the two 60-second retries.
Retain diagnostic files alongside the other evidence before an explicit owned
upgrade; the uninstaller never deletes this separate diagnostic directory.

To deliberately stop the service without recovery restarting it, first verify
the installed receipt and exact task definition, disable and stop that owned task,
then use the original lab's `stop.ps1` if its exact owned service remains alive.
Disabling the logon trigger alone does not stop a supervisor that is already
running. Do not terminate an unrelated process or use port ownership alone as
authorization to stop it.

## Upgrade an existing installation

When an existing same-directory installation already uses this recovery policy,
running the new installer returns `ALREADY_REGISTERED`; it verifies the receipt
and does **not** replace the supervisor or upgrade a historical `v1` task to Normal
priority. Earlier receipts without this recovery
policy are refused. In either case, deploy the current supervisor and policy by
explicitly uninstalling the owned installation with its currently installed
helper, then reinstalling from the new protected source.

Before removal, preserve `installation.json` and any `supervisor-status.json`
outside the managed directory. Use the **currently installed** helper and
uninstaller to verify that installation's script hashes, current-user identity,
and exact task definition. Only after those checks pass, disable its retry
trigger and stop its running task:

```powershell
$installedRoot = "$env:USERPROFILE\Documents\SZL\FoundationConfirmation"
& "$installedRoot\uninstall-workbench.ps1" -ValidateOnly
. "$installedRoot\supervise-workbench.ps1"
$installed = Read-FoundationInstallation
$ownedTask = Get-ScheduledTask -TaskName $installed.task_name -TaskPath $installed.task_path
Assert-FoundationTask $ownedTask $installed
Disable-ScheduledTask -TaskName $installed.task_name -TaskPath $installed.task_path
if ([string]$ownedTask.State -eq 'Running') {
    Stop-ScheduledTask -TaskName $installed.task_name -TaskPath $installed.task_path
}
& "$installedRoot\uninstall-workbench.ps1"
```

Stopping the task can also stop its owned service descendants, so this is a
deliberate local workbench interruption. The uninstaller preserves the sealed
lab, archive, and `state/trials`. Then run the new checkout's validation and
installation commands above, start the newly registered task on demand, and
retain fresh readiness, exact process, supervisor-wait, and controlled-recovery
evidence. Report supervisor recovery separately from any actual Scheduler restart;
manual demand recovery cannot prove automatic recovery. A conflicting task,
changed script, or unexpected managed file remains
a refusal; it is never removed as an upgrade shortcut.

## Admit a CPU environment

The optional CPU mode runs the frozen workbench with the eleven exact official
wheels in `cpu-runtime-lock.json`, including Torch `2.10.0+cpu`. It retains the
original Torch 2.10 model semantics and immutable checkpoints. It requires an
explicit native Python 3.11 AMD64 executable, a dedicated environment created
with `include-system-site-packages = false`, and the retained official wheelhouse.
Provision only those pinned wheels into the owned environment with hash checking,
no shared cache, and no dependency substitution. The admission helper verifies
the complete installed importable payload against the locked wheel bytes, rejects
additional source, extensions, bytecode, site hooks and links, and binds the
native executable hash, environment paths, payload digest and package versions.
Installed payload hashing uses four bounded I/O workers after every pathname is
admitted. Each file is read in full; missing, added or changed bytes still fail.
An NTFS-compressed file is acceptable when its bytes and hashes are unchanged.

Use all three explicit CPU arguments when validating or installing:

```powershell
& .\foundation-confirmation\operations\install-workbench.ps1 `
  -LabRoot $lab -ArchivePath $archive `
  -PythonExecutable $nativePython311 `
  -CpuEnvironmentRoot $ownedCpuEnvironment -CpuWheelhouse $officialWheelhouse `
  -CpuStateDirectory $freshCpuState -ValidateOnly
```

After the protected source is published, the same command without `-ValidateOnly`
registers the current-user task. Existing installations still require the
receipt-verified uninstall/reinstall procedure above; supplying CPU arguments
does not replace an existing managed installation. Exact repeated CPU arguments
verify the existing environment and task before returning `ALREADY_REGISTERED`.
CPU state defaults to `lab/state/cpu-runtime`; it must be fresh on installation
and cannot alias or contain the original `lab/state/trials` directory. CPU user
trials, logs and service receipt are retained in that separate state directory.
Uninstall preserves both sets of trial evidence, the environment and wheelhouse.

The managed launcher starts the hash-bound native image with `-I -S -B`, activates
only the admitted CPU site directory, and loads each frozen Python module by
compiling the exact bytes whose hash was checked. It executes learned selectors
17, 23 and 41 with checkpoint/source bindings before constructing the listener.
These startup probes do not mint user trial receipts or qualify the registered
scientific gate, which remains `FAILED`. Health and status keep their respective
5 second and 10 second bounds. A monotonic 180 second startup limit is enforced
through the retained handle of the direct newly launched child, including guard
setup and endpoint verification; a late readiness response fails admission.
Startup failure preserves its logs and cannot authorize supervisor recovery.
Every launch attempt uses fresh GUID-named stdout, stderr and attempt evidence;
a later successor cannot overwrite an earlier failed startup's diagnostics.
The original two retries, 60 second delay, exact ownership checks and zero
Scheduler retries remain in force.

Recovery admits a previous service receipt only after its complete launch
contract matches and native PID/creation-time identity establishes exit. Windows
can keep an exited process object available while the supervisor retains its
handle; the runner therefore verifies that exact object's creation time and
zero-duration native exit signal on one query handle. A live process, reused PID,
malformed identity, access denial, or unknown native result is refused and leaves
the predecessor receipt bytes in place. An absent PID is admitted only for native
error `87` at the query instant. The runner closes only the query handle it opened.
This follows [Microsoft's process lifetime contract](https://learn.microsoft.com/en-us/windows/win32/procthread/terminating-a-process).

The supervisor imports the built-in Utility module from the running shell's
`PSHOME`. A mixed PowerShell 5.1/7 module search path cannot substitute another
shell's hashing or JSON commands. This changes no user or machine settings.

Run the network-free controls with:

```powershell
python -I -S -B -m unittest discover -s tests -p test_foundation_cpu_environment.py -v
powershell -NoProfile -NonInteractive -File .\foundation-confirmation\operations\test-cpu-startup.ps1
pwsh -NoProfile -NonInteractive -File .\foundation-confirmation\operations\test-cpu-startup.ps1
```

The controls use small owned fixtures, never register a task, never import Torch
and never change a production service. The existing `test-startup.ps1` also
checks the immutable release and any already-running exact-owned legacy service.
The Foundation workflow runs the Python controls and both native shells before
its existing publisher may run. Local fixture results do not prove deployment,
actual managed CPU startup, reboot behavior or scientific generality. Each of
those claims requires its own fresh evidence.
Both native-shell controls execute `test-cpu-predecessor.py`, which keeps its own
bounded child handle open after exit, refuses the live child and a changed
creation identity, and verifies exact receipt replacement only after native exit.
The Python suite also runs this actual Windows fixture and platform-independent
negative native-API controls. These fixtures are separate from an actual managed
service recovery witness.

## Remove startup

```powershell
& "$env:USERPROFILE\Documents\SZL\FoundationConfirmation\uninstall-workbench.ps1" -ValidateOnly
& "$env:USERPROFILE\Documents\SZL\FoundationConfirmation\uninstall-workbench.ps1"
```

Uninstall verifies the current-user receipt and exact task definition before
unregistering the task. It removes only its explicitly named managed script and
receipt files. It does not stop a running workbench, delete the extracted source,
or delete `state/trials`. A missing task can be cleaned up only when a valid owned
installation receipt remains. Unexpected files prevent cleanup.
