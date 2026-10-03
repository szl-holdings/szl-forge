# SPDX-License-Identifier: Apache-2.0
param(
    [Parameter(Mandatory = $true)][string]$LabRoot,
    [Parameter(Mandatory = $true)][string]$ArchivePath,
    [ValidateRange(1, 65535)][int]$Port = 18767,
    [switch]$ValidateOnly,
    [string]$PythonExecutable, [string]$CpuEnvironmentRoot, [string]$CpuWheelhouse, [string]$CpuStateDirectory
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'supervise-workbench.ps1') -LabRoot $LabRoot -ArchivePath $ArchivePath -Port $Port -ValidateOnly:$ValidateOnly -PythonExecutable $PythonExecutable -CpuEnvironmentRoot $CpuEnvironmentRoot -CpuWheelhouse $CpuWheelhouse -CpuStateDirectory $CpuStateDirectory

try {
    $paths = Get-FoundationPaths
    $admitted = Test-FoundationRelease $LabRoot $ArchivePath
    if ($PythonExecutable) {
        $script:foundationCpuRuntime = Get-FoundationCpuRuntime -Executable $PythonExecutable -EnvironmentRoot $CpuEnvironmentRoot -Wheelhouse $CpuWheelhouse -StateDirectory $CpuStateDirectory -Root $admitted.LabRoot
    }
    $service = Get-FoundationService $admitted.LabRoot $admitted.Python $Port -MayBeAbsent
    Assert-FoundationLaunchPort $Port $service
    $task = Get-ScheduledTask -TaskName $paths.Task -TaskPath '\' -ErrorAction SilentlyContinue
    $operationsPath = Assert-FoundationPath $paths.Operations -Directory -MayNotExist
    if (Test-Path -LiteralPath $paths.Receipt) {
        $existing = Read-FoundationInstallation
        if ($existing.lab_root -ne $admitted.LabRoot -or $existing.archive_path -ne $admitted.ArchivePath -or $existing.port -ne $Port -or
            [bool]$existing.cpu_runtime -ne [bool]$PythonExecutable -or ($existing.cpu_runtime -and
            ($existing.cpu_runtime.environment_root -cne $CpuEnvironmentRoot -or $existing.cpu_runtime.wheelhouse -cne $CpuWheelhouse -or $existing.cpu_runtime.python_executable -cne $PythonExecutable))) {
            throw 'A different workbench installation already owns this name and receipt.'
        }
        if ($null -eq $task) { throw 'An owned receipt exists without its task. Uninstall the receipt explicitly before reinstalling.' }
        Assert-FoundationTask $task $existing
        @{ status = 'ALREADY_REGISTERED'; task_name = $paths.Task; installation = $existing; service_started = $false } | ConvertTo-Json -Depth 8
        exit 0
    }
    if ($null -ne $task) { throw 'The task name is already in use without our installation receipt.' }
    if (Test-Path -LiteralPath $operationsPath) {
        if (@(Get-ChildItem -LiteralPath $operationsPath -Force).Count -gt 0) {
            throw 'The operations directory contains unowned files; nothing was overwritten.'
        }
    }
    if ($ValidateOnly) {
        @{ status = 'VERIFIED'; admission = $admitted; task_name = $paths.Task; owner_sid = $paths.Sid;
            task_registered = $false; service_started = $false; operations_root = $operationsPath } | ConvertTo-Json -Depth 8
        exit 0
    }
    if ($script:foundationCpuRuntime) {
        $cpuState = $script:foundationCpuRuntime.state_directory
        if ((Test-Path -LiteralPath $cpuState) -and @(Get-ChildItem -LiteralPath $cpuState -Force).Count -gt 0) { throw 'Choose a fresh CPU state directory; existing trial evidence is preserved.' }
        [IO.Directory]::CreateDirectory($cpuState) | Out-Null
    }
    [IO.Directory]::CreateDirectory($operationsPath) | Out-Null
    Assert-FoundationPath $operationsPath -Directory | Out-Null
    $scriptHashes = @{}
    $managedNames = @('install-workbench.ps1', 'supervise-workbench.ps1', 'uninstall-workbench.ps1')
    if ($script:foundationCpuRuntime) { $managedNames += @('launch-cpu-workbench.ps1', 'cpu_environment.py', 'cpu_workbench.py', 'cpu-runtime-lock.json') }
    foreach ($name in $managedNames) {
        $source = Assert-FoundationPath (Join-Path $PSScriptRoot $name)
        $target = Join-Path $operationsPath $name
        [IO.File]::Copy($source, $target, $false)
        $hash = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant()
        Assert-FoundationHash $target $hash
        $scriptHashes[$name] = $hash
    }
    $arguments = Get-FoundationTaskArguments $admitted.LabRoot $admitted.ArchivePath $Port $script:foundationCpuRuntime
    $receipt = @{ schema = 'szl.foundation-confirmation.windows-installation/v1'; installation_id = [guid]::NewGuid().ToString('N');
        installed_utc = [datetime]::UtcNow.ToString('o'); owner_sid = $paths.Sid; task_name = $paths.Task; task_path = '\';
        lab_root = $admitted.LabRoot; archive_path = $admitted.ArchivePath; port = $Port; operations_root = $operationsPath;
        state = 'PREPARED'; scripts = $scriptHashes; source_admission = $admitted.Verification;
        archive_sha256 = '869e318dd5f328205dd181ee836ef267bd2ae278f6430a9e8ddc661fbc689d03';
        action = @{ executable = $paths.Shell; arguments = $arguments; working_directory = $admitted.LabRoot };
        startup_scope = 'CURRENT_USER_LOGON'; machine_reboot_start_claimed = $false; on_demand_execution_verified = $false;
        recovery_mechanism = 'BOUNDED_SUPERVISOR_RETRY'; retry_limit = 2; retry_interval_seconds = 60; scheduler_restart_count = 0 }
    if ($script:foundationCpuRuntime) { $receipt.cpu_runtime = $script:foundationCpuRuntime }
    Write-FoundationJson $paths.Receipt $receipt
    $action = New-ScheduledTaskAction -Execute $paths.Shell -Argument $arguments -WorkingDirectory $admitted.LabRoot
    $trigger = New-ScheduledTaskTrigger -AtLogOn -User $paths.Sid
    $principal = New-ScheduledTaskPrincipal -UserId $paths.Sid -LogonType Interactive -RunLevel Limited
    # The supervisor owns the bounded recovery budget. Scheduler retries must
    # remain disabled so the two independent mechanisms cannot multiply retries.
    $settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
    # No -Force: an intervening task-name conflict is a failure, never a takeover.
    Register-ScheduledTask -TaskName $paths.Task -TaskPath '\' -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Description 'Verified local SZL Foundation Confirmation research workbench; current-user logon only.' | Out-Null
    $registered = Get-ScheduledTask -TaskName $paths.Task -TaskPath '\' -ErrorAction Stop
    Assert-FoundationTask $registered $receipt
    $receipt.state = 'REGISTERED'
    $receipt.registration_readback_utc = [datetime]::UtcNow.ToString('o')
    $receipt.registration_readback_verified = $true
    Write-FoundationJson $paths.Receipt $receipt
    @{ status = 'REGISTERED'; task_name = $paths.Task; receipt = $paths.Receipt; service_started = $false;
        next_verification = 'On-demand execution, owned-process restart and retained-receipt checks remain required.' } | ConvertTo-Json
} catch {
    Write-Error $_ -ErrorAction Continue
    exit 1
}
