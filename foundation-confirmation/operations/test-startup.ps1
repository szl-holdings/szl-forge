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
        Settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -RestartCount 2 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
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

    $paths = Get-FoundationPaths
    $receipt = [pscustomobject]@{ lab_root = $admitted.LabRoot; action = [pscustomobject]@{
        executable = $paths.Shell; arguments = Get-FoundationTaskArguments $admitted.LabRoot $admitted.ArchivePath $Port
    } }
    $task = New-TaskFixture $receipt
    Assert-FoundationTask $task $receipt
    $checks.Add('NATIVE_CURRENT_USER_TASK_DEFINITION_ADMITTED_WITHOUT_REGISTRATION')
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
    $task.Settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit ([TimeSpan]::Zero)
    Confirm-Rejection 'CHANGED_RESTART_POLICY_REJECTED' { Assert-FoundationTask $task $receipt } 'conflicting task definition'

    $service = Get-FoundationService $admitted.LabRoot $admitted.Python $Port -MayBeAbsent
    $serviceWitness = 'ABSENT'
    if ($null -ne $service) {
        Assert-FoundationLaunchPort $Port $service
        $checks.Add('VERIFIED_PREDECESSOR_LISTENER_ADMITTED')
        Confirm-Rejection 'LISTENER_WITHOUT_OWNERSHIP_WITNESS_REJECTED' { Assert-FoundationLaunchPort $Port $null } 'without this lab receipt'
        $tracked = Get-Process -Id $service.Receipt.pid -ErrorAction Stop
        if ($tracked.StartTime.ToUniversalTime() -ne ([datetime]$service.Receipt.process_created).ToUniversalTime() -or
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
