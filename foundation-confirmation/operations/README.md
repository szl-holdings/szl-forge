# Local workbench startup

These Windows operations manage the original, unchanged Foundation Confirmation
v0.4 workbench. They install one **current-user logon** task, `SZL Foundation
Confirmation`, with an Interactive token and Limited privileges. No password or
administrator elevation is requested. Installation does not start the workbench.
This is local synthetic research execution; the public evidence explorer remains
a recorded replay, and the registered benchmark remains **FAILED**.

## Install and verify

Use Windows PowerShell 5.1 or newer as the intended desktop user. Supply the
absolute extracted lab directory and original sealed archive. Both must be below
that user's profile on the Windows system volume; network/device paths and
reparse points are rejected. Install the original pinned Python dependencies
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
`%LOCALAPPDATA%\SZL\FoundationConfirmation`, outside the release. The task invokes
that installed supervisor. Its action, arguments, current-user SID, trigger,
privileges, retry settings, and script hashes must match the owned receipt.
Existing conflicting tasks, receipts, or managed files are refused.

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

The supervisor invokes the unchanged `start.ps1` with a hidden window. It checks
the service receipt against the process's exact executable and command arguments,
creation timestamp, PID, and exclusive loopback listener. It also checks the
health response and all three checkpoint bindings before waiting for that owned
process. Task Scheduler therefore tracks the running service instead of a
launcher that has already exited. A service exit produces a failed supervisor
result, allowing **two retries, one minute apart**. There is no endless restart
loop. The task permits manual demand starts and has no time limit while the owned
service is healthy.

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

To deliberately stop the service without the retry policy restarting it, first
disable the owned task and then use the original lab's `stop.ps1`. Do not terminate
an unrelated process or use port ownership alone as authorization to stop it.

## Remove startup

```powershell
& "$env:LOCALAPPDATA\SZL\FoundationConfirmation\uninstall-workbench.ps1" -ValidateOnly
& "$env:LOCALAPPDATA\SZL\FoundationConfirmation\uninstall-workbench.ps1"
```

Uninstall verifies the current-user receipt and exact task definition before
unregistering the task. It removes only its explicitly named managed script and
receipt files. It does not stop a running workbench, delete the extracted source,
or delete `state/trials`. A missing task can be cleaned up only when a valid owned
installation receipt remains. Unexpected files prevent cleanup.
