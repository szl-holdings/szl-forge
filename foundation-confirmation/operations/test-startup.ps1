# SPDX-License-Identifier: Apache-2.0
param(
    [Parameter(Mandatory = $true)][string]$LabRoot,
    [Parameter(Mandatory = $true)][string]$ArchivePath,
    [ValidateRange(1, 65535)][int]$Port = 18767
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'supervise-workbench.ps1') -LabRoot $LabRoot -ArchivePath $ArchivePath -Port $Port -ValidateOnly
$checks = [Collections.Generic.List[string]]::new()

function Confirm-Rejection {
    param([string]$Name, [scriptblock]$Action, [string]$ExpectedMessage)
    $rejected = $false
    try { & $Action | Out-Null }
    catch {
        if ($_.Exception.Message -notmatch $ExpectedMessage) { throw }
        $rejected = $true
    }
    if (-not $rejected) { throw "Admission incorrectly accepted: $Name" }
    $checks.Add($Name)
}

function New-TaskFixture {
    param($Receipt)
    $paths = Get-FoundationPaths
    # Real native CIM definitions, never registered with Task Scheduler.
    return [pscustomobject]@{
        TaskName = $paths.Task; TaskPath = '\'
        Actions = @(New-ScheduledTaskAction -Execute $paths.Shell -Argument $Receipt.action.arguments -WorkingDirectory $Receipt.lab_root)
        Principal = New-ScheduledTaskPrincipal -UserId $paths.Sid -LogonType Interactive -RunLevel Limited
        Triggers = @(New-ScheduledTaskTrigger -AtLogOn -User $paths.Sid)
        Settings = New-FoundationTaskSettings
    }
}

$fixtureRoot = $null
try {
    foreach ($script in Get-ChildItem -LiteralPath $PSScriptRoot -Filter '*.ps1') {
        $tokens = $null; $parseErrors = $null
        [Management.Automation.Language.Parser]::ParseFile($script.FullName, [ref]$tokens, [ref]$parseErrors) | Out-Null
        if ($parseErrors.Count -gt 0) { throw ($parseErrors | Out-String) }
    }
    $checks.Add('ALL_POWERSHELL_SOURCES_PARSE')
    $sharedPaths = Get-FoundationPaths
    $expectedOperations = Join-Path $sharedPaths.Profile 'Documents\SZL\FoundationConfirmation'
    if ($sharedPaths.Operations -cne $expectedOperations -or
        $sharedPaths.Receipt -cne (Join-Path $expectedOperations 'installation.json') -or
        $sharedPaths.Supervisor -cne (Join-Path $expectedOperations 'supervise-workbench.ps1')) {
        throw 'Managed startup files must use the shared current-profile Documents directory.'
    }
    Assert-FoundationPath $sharedPaths.Operations -Directory -MayNotExist | Out-Null
    $checks.Add('MANAGED_STARTUP_USES_SHARED_PROFILE_DOCUMENTS_OUTSIDE_APPDATA')
    $admitted = Test-FoundationRelease $LabRoot $ArchivePath
    $checks.Add('PINNED_ARCHIVE_AND_COMPLETE_70_FILE_RELEASE_VERIFIED')
    Confirm-Rejection 'NETWORK_PATH_REJECTED' { Assert-FoundationPath '\\localhost\C$\Users' -Directory } 'absolute local filesystem'
    Confirm-Rejection 'PROFILE_ESCAPE_REJECTED' { Assert-FoundationPath ($admitted.LabRoot + '\..\..\..\..\..\..\..\Windows') -Directory } 'below the current user profile'
    Confirm-Rejection 'ALTERNATE_STREAM_REJECTED' { Assert-FoundationPath ($ArchivePath + ':payload') } 'absolute local filesystem'
    Confirm-Rejection 'UNPINNED_ARCHIVE_REJECTED' { Test-FoundationRelease $LabRoot (Join-Path $PSScriptRoot 'README.md') } 'Pinned release hash mismatch'

    # A modified verifier must be rejected before Python is allowed to execute it.
    $operationRoot = Assert-FoundationPath $PSScriptRoot -Directory
    $fixtureRoot = Join-Path $operationRoot ('.startup-validation-' + [guid]::NewGuid().ToString('N'))
    [IO.Directory]::CreateDirectory($fixtureRoot) | Out-Null
    foreach ($name in @('release-manifest.json', 'verify_release.py', 'server.py', 'start.ps1', 'core.py', 'policy.py', 'experiment.py')) {
        [IO.File]::Copy((Join-Path $admitted.LabRoot $name), (Join-Path $fixtureRoot $name), $false)
    }
    [IO.File]::WriteAllText((Join-Path $fixtureRoot 'verify_release.py'), "raise RuntimeError('UNTRUSTED_VERIFIER_EXECUTED')", [Text.UTF8Encoding]::new($false))
    Confirm-Rejection 'MODIFIED_VERIFIER_REJECTED_BEFORE_EXECUTION' { Test-FoundationRelease $fixtureRoot $admitted.ArchivePath } 'Pinned release hash mismatch: .*verify_release.py'
    $transitionPath = Join-Path $fixtureRoot 'atomic-receipt.json'
    Write-FoundationJson $transitionPath @{ state = 'PREPARED'; installation_id = 'fixture-owned-installation' }
    if ((Get-Content -LiteralPath $transitionPath -Raw | ConvertFrom-Json).state -cne 'PREPARED') {
        throw 'The atomic receipt did not preserve the prepared state.'
    }
    Write-FoundationJson $transitionPath @{ state = 'REGISTERED'; installation_id = 'fixture-owned-installation' }
    $transition = Get-Content -LiteralPath $transitionPath -Raw | ConvertFrom-Json
    if ($transition.state -cne 'REGISTERED' -or $transition.installation_id -cne 'fixture-owned-installation') {
        throw 'The atomic receipt update did not preserve the second transition.'
    }
    $checks.Add('ATOMIC_RECEIPT_SECOND_TRANSITION_VERIFIED')
    $cimInstant = [datetime]::Parse('2026-10-02T01:59:46.2526220Z').ToUniversalTime()
    foreach ($extraTick in 0..9) {
        if ((ConvertTo-FoundationCimTimestamp $cimInstant.AddTicks($extraTick)) -ne $cimInstant) {
            throw 'The process handle timestamp did not match its CIM microsecond identity.'
        }
    }
    $checks.Add('PROCESS_HANDLE_TIMESTAMP_MATCHES_CIM_MICROSECOND_PRECISION')
    if ((ConvertTo-FoundationCimTimestamp $cimInstant.AddTicks(10)) -eq $cimInstant) {
        throw 'A distinct next-microsecond process identity was admitted.'
    }
    $checks.Add('NEXT_MICROSECOND_PROCESS_IDENTITY_REJECTED')

    # A real launcher exits with a known failure code while its bounded child
    # remains running. Waiting for a process tree would consume the entire child
    # lifetime and prevent the separate service-ownership checks from running.
    $parentScript = Join-Path $fixtureRoot 'launcher-parent.py'
    $childScript = Join-Path $fixtureRoot 'launcher-child.py'
    $childPidPath = Join-Path $fixtureRoot 'launcher-child.pid'
    $finishPath = Join-Path $fixtureRoot 'launcher-child.finish'
    $completedPath = Join-Path $fixtureRoot 'launcher-child.completed'
    $parentCode = @'
import pathlib, subprocess, sys
child = subprocess.Popen([sys.executable, '-I', '-B', sys.argv[1], sys.argv[3], sys.argv[4]], creationflags=subprocess.CREATE_NO_WINDOW)
pathlib.Path(sys.argv[2]).write_text(str(child.pid), encoding='ascii')
sys.exit(7)
'@
    $childCode = @'
import pathlib, sys, time
finish = pathlib.Path(sys.argv[1])
deadline = time.monotonic() + 20
while not finish.exists() and time.monotonic() < deadline:
    time.sleep(0.1)
pathlib.Path(sys.argv[2]).write_text('complete', encoding='ascii')
'@
    [IO.File]::WriteAllText($parentScript, $parentCode, [Text.UTF8Encoding]::new($false))
    [IO.File]::WriteAllText($childScript, $childCode, [Text.UTF8Encoding]::new($false))
    $childHandle = $null
    try {
        # Warm the native ownership helpers before creating the bounded child.
        # Shell/module compilation latency must not consume its test lifetime.
        @(Get-FoundationArgv ('"' + $admitted.Python + '"')) | Out-Null
        Get-CimInstance Win32_Process -Filter "ProcessId=$PID" -ErrorAction Stop | Out-Null
        $fixtureArguments = '-I -B "' + $parentScript + '" "' + $childScript + '" "' + $childPidPath + '" "' + $finishPath + '" "' + $completedPath + '"'
        $fixtureExit = Invoke-FoundationLauncher $admitted.Python $fixtureArguments $fixtureRoot
        if ($fixtureExit -ne 7) { throw 'Launcher exit code was not retained.' }
        if (Test-Path -LiteralPath $completedPath) { throw 'Launcher wait incorrectly included the bounded descendant process.' }
        $fixturePid = [int](Get-Content -LiteralPath $childPidPath -Raw)
        $childHandle = Get-Process -Id $fixturePid -ErrorAction Stop
        $childProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$fixturePid" -ErrorAction Stop
        $childArguments = @(Get-FoundationArgv $childProcess.CommandLine)
        $expectedChild = @($admitted.Python, '-I', '-B', $childScript, $finishPath, $completedPath)
        if ($childHandle.HasExited -or $childHandle.Path -cne $admitted.Python -or
            (ConvertTo-FoundationCimTimestamp $childHandle.StartTime) -ne $childProcess.CreationDate.ToUniversalTime() -or
            $childArguments.Count -ne $expectedChild.Count) {
            throw ('The bounded fixture child ownership differs: exited={0}; executable={1}; creation_ticks_delta={2}; argument_count={3}.' -f
                $childHandle.HasExited, ($childHandle.Path -ceq $admitted.Python),
                ($childHandle.StartTime.ToUniversalTime().Ticks - $childProcess.CreationDate.ToUniversalTime().Ticks),
                ($childArguments.Count -eq $expectedChild.Count))
        }
        for ($index = 0; $index -lt $expectedChild.Count; $index++) {
            if ($childArguments[$index] -cne $expectedChild[$index]) { throw 'The bounded fixture child command differs.' }
        }
        $checks.Add('LAUNCHER_EXIT_OBSERVED_WITH_OWNED_DESCENDANT_STILL_RUNNING')
        $checks.Add('LAUNCHER_NONZERO_EXIT_CODE_RETAINED')
    } finally {
        # Ask only this exact fixture child to exit naturally; no process is killed.
        [IO.File]::WriteAllText($finishPath, 'finish', [Text.UTF8Encoding]::new($false))
        if ($null -ne $childHandle) { Wait-Process -InputObject $childHandle -Timeout 25 -ErrorAction Stop }
    }

    # Three real bounded launcher exits exercise the retry budget. The delay
    # callback records the production interval without sleeping during validation.
    $exitScript = Join-Path $fixtureRoot 'recovery-exit.py'
    [IO.File]::WriteAllText($exitScript, 'import sys; sys.exit(7)', [Text.UTF8Encoding]::new($false))
    $script:recoveryAttempts = [Collections.Generic.List[int]]::new()
    $script:recoveryDelays = [Collections.Generic.List[int]]::new()
    Confirm-Rejection 'OWNED_EXIT_RECOVERY_BUDGET_EXHAUSTED_AFTER_TWO_RETRIES' {
        Invoke-FoundationRecoveryLoop -Cycle {
            param($Attempt)
            $script:recoveryAttempts.Add($Attempt)
            $code = Invoke-FoundationLauncher $admitted.Python ('-I -B "' + $exitScript + '"') $fixtureRoot
            if ($code -ne 7) { throw 'The bounded recovery fixture exit differs.' }
            return @{ status = 'OWNED_SERVICE_EXITED'; owned_service_exit_verified = $true }
        } -Delay {
            param($Seconds)
            $script:recoveryDelays.Add($Seconds)
        }
    } 'recovery budget exhausted'
    if (($script:recoveryAttempts -join ',') -cne '0,1,2' -or ($script:recoveryDelays -join ',') -cne '60,60') {
        throw 'Recovery did not enforce exactly two retries at the declared 60-second interval.'
    }
    $checks.Add('THREE_REAL_LAUNCHER_EXITS_ENFORCE_TWO_60_SECOND_RETRIES')
    $script:recoveryAttempts.Clear(); $script:recoveryDelays.Clear()
    Confirm-Rejection 'ADMISSION_ERROR_PREVENTS_ANY_RECOVERY_RETRY' {
        Invoke-FoundationRecoveryLoop -Cycle {
            param($Attempt)
            $script:recoveryAttempts.Add($Attempt)
            throw 'Fixture ownership admission refused.'
        } -Delay { param($Seconds) $script:recoveryDelays.Add($Seconds) }
    } 'Fixture ownership admission refused'
    if ($script:recoveryAttempts.Count -ne 1 -or $script:recoveryDelays.Count -ne 0) { throw 'Admission failure was retried.' }
    Confirm-Rejection 'UNVERIFIED_EXIT_PREVENTS_ANY_RECOVERY_RETRY' {
        Invoke-FoundationRecoveryLoop -Cycle { return @{ status = 'OWNED_SERVICE_EXITED'; owned_service_exit_verified = $false } } -Delay {
            param($Seconds) $script:recoveryDelays.Add($Seconds)
        }
    } 'did not verify the owned service exit'
    if ($script:recoveryDelays.Count -ne 0) { throw 'An unverified exit was retried.' }

    $paths = Get-FoundationPaths
    $receipt = [pscustomobject]@{ schema = 'szl.foundation-confirmation.windows-installation/v2'; task_priority = 5;
        lab_root = $admitted.LabRoot; action = [pscustomobject]@{
        executable = $paths.Shell; arguments = Get-FoundationTaskArguments $admitted.LabRoot $admitted.ArchivePath $Port
    } }
    $task = New-TaskFixture $receipt
    Assert-FoundationTask $task $receipt
    $checks.Add('NATIVE_CURRENT_USER_TASK_DEFINITION_ADMITTED_WITHOUT_REGISTRATION')
    $task.Settings.Priority = 7
    Confirm-Rejection 'BELOW_NORMAL_TASK_PRIORITY_REJECTED' { Assert-FoundationTask $task $receipt } 'conflicting task definition'
    $task.Settings.Priority = 5
    $accountName = [Security.Principal.SecurityIdentifier]::new($paths.Sid).Translate([Security.Principal.NTAccount]).Value
    $task = New-TaskFixture $receipt
    $task.Triggers[0].UserId = $accountName
    Assert-FoundationTask $task $receipt
    $checks.Add('REGISTERED_LOGON_TRIGGER_ACCOUNT_NORMALIZATION_ADMITTED')
    $task.Principal.UserId = $accountName
    Assert-FoundationTask $task $receipt
    $checks.Add('REGISTERED_PRINCIPAL_AND_TRIGGER_ACCOUNT_NORMALIZATION_ADMITTED')
    $task = New-TaskFixture $receipt
    $task.Triggers[0].UserId = 'S-1-5-18'
    Confirm-Rejection 'FOREIGN_LOGON_TRIGGER_SID_REJECTED' { Assert-FoundationTask $task $receipt } 'conflicting task definition'
    $task = New-TaskFixture $receipt
    $task.Triggers[0].UserId = [Security.Principal.SecurityIdentifier]::new('S-1-5-18').Translate([Security.Principal.NTAccount]).Value
    Confirm-Rejection 'FOREIGN_LOGON_TRIGGER_ACCOUNT_REJECTED' { Assert-FoundationTask $task $receipt } 'conflicting task definition'
    $task = New-TaskFixture $receipt
    $task.Triggers[0].UserId = ''
    Confirm-Rejection 'EMPTY_LOGON_TRIGGER_IDENTITY_REJECTED' { Assert-FoundationTask $task $receipt } 'conflicting task definition'
    $task = New-TaskFixture $receipt
    $task.Triggers[0].UserId = $env:COMPUTERNAME + '\SZL-Missing-' + [guid]::NewGuid().ToString('N')
    Confirm-Rejection 'UNKNOWN_LOGON_TRIGGER_ACCOUNT_REJECTED' { Assert-FoundationTask $task $receipt } 'conflicting task definition'
    $task = New-TaskFixture $receipt
    $task.Triggers[0].UserId = 'S-1-invalid'
    Confirm-Rejection 'MALFORMED_LOGON_TRIGGER_SID_REJECTED' { Assert-FoundationTask $task $receipt } 'conflicting task definition'
    $task = New-TaskFixture $receipt
    $task.Actions = @($task.Actions[0], $task.Actions[0])
    Confirm-Rejection 'ADDITIONAL_TASK_ACTION_REJECTED' { Assert-FoundationTask $task $receipt } 'conflicting task definition'
    $task = New-TaskFixture $receipt
    $task.Principal = New-ScheduledTaskPrincipal -UserId 'SYSTEM' -LogonType ServiceAccount -RunLevel Highest
    Confirm-Rejection 'FOREIGN_ELEVATED_TASK_OWNER_REJECTED' { Assert-FoundationTask $task $receipt } 'conflicting task definition'
    $task = New-TaskFixture $receipt
    $task.Actions = @(New-ScheduledTaskAction -Execute $paths.Shell -Argument '-NoProfile -Command exit' -WorkingDirectory $admitted.LabRoot)
    Confirm-Rejection 'CONFLICTING_TASK_ARGUMENTS_REJECTED' { Assert-FoundationTask $task $receipt } 'conflicting task definition'
    $task = New-TaskFixture $receipt
    $task.Settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -RestartCount 2 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit ([TimeSpan]::Zero)
    Confirm-Rejection 'ADDITIONAL_SCHEDULER_RETRIES_REJECTED' { Assert-FoundationTask $task $receipt } 'conflicting task definition'

    $service = Get-FoundationService $admitted.LabRoot $admitted.Python $Port -MayBeAbsent
    $serviceWitness = 'ABSENT'
    if ($null -ne $service) {
        Assert-FoundationLaunchPort $Port $service
        $checks.Add('VERIFIED_PREDECESSOR_LISTENER_ADMITTED')
        Confirm-Rejection 'LISTENER_WITHOUT_OWNERSHIP_WITNESS_REJECTED' { Assert-FoundationLaunchPort $Port $null } 'without this lab receipt'
        $tracked = Get-Process -Id $service.Receipt.pid -ErrorAction Stop
        if ((ConvertTo-FoundationCimTimestamp $tracked.StartTime) -ne ([datetime]$service.Receipt.process_created).ToUniversalTime() -or
            $tracked.Path -ne $admitted.Python) { throw 'Live owned process handle did not match its receipt.' }
        $checks.Add('EXISTING_SERVICE_OWNERSHIP_READINESS_AND_PROCESS_HANDLE_VERIFIED')
        $serviceWitness = 'READY'
    }
    @{ status = 'VERIFIED'; checks = @($checks); count = $checks.Count; existing_service = $serviceWitness;
        tasks_registered = $false; services_started_or_stopped = $false; production_receipts_changed = $false } | ConvertTo-Json -Depth 4
} finally {
    if ($null -ne $fixtureRoot -and (Test-Path -LiteralPath $fixtureRoot)) {
        $safeFixture = Assert-FoundationPath $fixtureRoot -Directory
        $safeOperations = Assert-FoundationPath $PSScriptRoot -Directory
        if (-not $safeFixture.StartsWith($safeOperations + '\', [StringComparison]::OrdinalIgnoreCase) -or
            [IO.Path]::GetFileName($safeFixture) -notmatch '^\.startup-validation-[0-9a-f]{32}$') {
            throw 'Refused validation fixture cleanup outside its exact owned operations directory.'
        }
        Remove-Item -LiteralPath $safeFixture -Recurse -Force
    }
}
