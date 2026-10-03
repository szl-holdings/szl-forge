# SPDX-License-Identifier: Apache-2.0
# Native, network-free controls. Only short-lived fixture children are launched.
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'supervise-workbench.ps1')
$checks = [Collections.Generic.List[string]]::new()
function Reject {
    param([string]$Name, [scriptblock]$Action, [string]$Message)
    $failed = $false
    try { & $Action | Out-Null } catch { if ($_.Exception.Message -notmatch $Message) { throw }; $failed = $true }
    if (-not $failed) { throw "Control was admitted: $Name" }
    $checks.Add($Name)
}
$fixture = Join-Path $PSScriptRoot ('.cpu-controls-' + [guid]::NewGuid().ToString('N'))
$children = @()
try {
    foreach ($source in Get-ChildItem -LiteralPath $PSScriptRoot -Filter '*.ps1') {
        $tokens = $null; $errors = $null
        [Management.Automation.Language.Parser]::ParseFile($source.FullName, [ref]$tokens, [ref]$errors) | Out-Null
        if ($errors.Count -gt 0) { throw ($errors | Out-String) }
    }
    $checks.Add('ALL_SOURCES_PARSE')
    $paths = Get-FoundationPaths
    $root = Join-Path $paths.Profile 'Documents\cpu-controls-lab'
    foreach ($state in @($root, (Join-Path $root 'state'), (Join-Path $root 'state\trials'), (Join-Path $root 'state\trials\child'))) {
        Reject ('FROZEN_STATE_ALIAS_' + [IO.Path]::GetFileName($state)) { Assert-FoundationCpuState $state $root } 'separate from the original'
    }
    Assert-FoundationCpuState (Join-Path $root 'state\cpu-runtime') $root
    $checks.Add('SEPARATE_CPU_STATE_ADMITTED')
    $cpu = @{ python_executable=(Join-Path $paths.Profile 'Python\python.exe'); python_image_sha256=('a' * 64);
        environment_root=(Join-Path $paths.Profile 'cpu-env'); wheelhouse=(Join-Path $paths.Profile 'wheels');
        state_directory=(Join-Path $root 'state\cpu-runtime'); environment_binding_sha256=('b' * 64) }
    $archive = Join-Path $paths.Profile 'release.zip'
    $argv = @(Get-FoundationCpuArgv $cpu $root $archive 18767)
    if (($argv[1..3] -join ',') -cne '-I,-S,-B' -or $argv[-1] -cne '180') { throw 'CPU isolation/deadline arguments differ.' }
    $checks.Add('CPU_ARGV_ISOLATION_AND_DEADLINE')
    $receipt = @{ schema='szl.foundation-confirmation.cpu-service/v1'; pid=123; executable=$cpu.python_executable;
        server_script=$argv[4]; lab_root=$root; port=18767; state_directory=$cpu.state_directory;
        environment_root=$cpu.environment_root; wheelhouse=$cpu.wheelhouse; python_image_sha256=$cpu.python_image_sha256;
        environment_binding_sha256=$cpu.environment_binding_sha256; argv=$argv[4..($argv.Count-1)]; startup_timeout_seconds=180 }
    Assert-FoundationCpuReceipt $receipt $cpu $root $archive 18767
    $checks.Add('COMPLETE_CPU_RECEIPT_ADMITTED')
    foreach ($key in @('state_directory','environment_root','wheelhouse','python_image_sha256','executable')) {
        $bad = $receipt.Clone(); $bad[$key] = 'wrong'
        Reject ('MISMATCHED_PREDECESSOR_' + $key) { Assert-FoundationCpuReceipt $bad $cpu $root $archive 18767 } 'launch contract differs'
    }
    $bad = $receipt.Clone(); $bad.argv = @($receipt.argv.Clone()); $bad.argv[-1] = '181'
    Reject 'MISMATCHED_PREDECESSOR_ARGV' { Assert-FoundationCpuReceipt $bad $cpu $root $archive 18767 } 'argv differs'
    [IO.Directory]::CreateDirectory($fixture) | Out-Null
    $checkpoints = @(foreach ($seed in @(17,23,41)) { @{ seed=$seed; sha256=('c' * 64); trained_fingerprint=('d' * 64) } })
    Write-FoundationJson (Join-Path $fixture 'training-summary.json') @{ checkpoints=$checkpoints }
    $probes = @(foreach ($checkpoint in $checkpoints) { @{ model_seed=$checkpoint.seed;
        request=@{ model_seed=$checkpoint.seed; policy='learned'; seed=20261002; index=10007; family='shared_bias' };
        binding=@{ checkpoint_sha256=$checkpoint.sha256; model_fingerprint=$checkpoint.trained_fingerprint };
        result_sha256=('e' * 64); receipt_minted=$false } })
    $receipt.startup_execution_probes = $probes
    $status = @{ ready=$true; state='READY'; checkpoints_verified=3; runtime_kind='ADMITTED_CPU_ONLY';
        environment_binding_sha256=$cpu.environment_binding_sha256; models_loaded=@(17,23,41);
        startup_execution_probes=$probes; startup_probes_are_user_receipts=$false; receipt_minted=$false }
    Assert-FoundationCpuReadiness $status $receipt $cpu $fixture
    $checks.Add('ALL_THREE_ACTUAL_STARTUP_BINDINGS_ADMITTED')
    $badStatus = $status.Clone(); $badStatus.models_loaded = @()
    Reject 'CHECKPOINT_ONLY_CPU_READINESS' { Assert-FoundationCpuReadiness $badStatus $receipt $cpu $fixture } 'startup execution/readiness'
    $badStatus = $status.Clone(); $badStatus.startup_execution_probes = @($probes[0],$probes[0],$probes[2])
    $badReceipt = $receipt.Clone(); $badReceipt.startup_execution_probes = $badStatus.startup_execution_probes
    Reject 'DUPLICATE_ACTUAL_STARTUP_SELECTOR' { Assert-FoundationCpuReadiness $badStatus $badReceipt $cpu $fixture } 'probe differs'
    $badReceipt = $receipt.Clone(); $badReceipt.startup_execution_probes = @()
    Reject 'STARTUP_RECEIPT_RUNTIME_DIVERGENCE' { Assert-FoundationCpuReadiness $status $badReceipt $cpu $fixture } 'probe receipt differs'
    $script:foundationCpuRuntime = $cpu
    $script:foundationArchivePath = $archive
    $cpu.state_directory = $fixture
    $bad = $receipt.Clone(); $bad.state_directory = 'wrong'
    $serviceFile = Join-Path $fixture 'service.json'
    Write-FoundationJson $serviceFile $bad
    $before = (Get-FileHash -LiteralPath $serviceFile).Hash
    Reject 'ABSENT_PID_CANNOT_HIDE_MISMATCHED_RECEIPT' { Get-FoundationService $root $cpu.python_executable 18767 -MayBeAbsent } 'launch contract differs'
    if ((Get-FileHash -LiteralPath $serviceFile).Hash -cne $before) { throw 'Predecessor evidence was modified.' }
    $checks.Add('MISMATCHED_RECEIPT_BYTES_PRESERVED')
    $script:foundationCpuRuntime = $null
    $cpu.state_directory = Join-Path $root 'state\cpu-runtime'
    $action = @{ executable=$paths.Shell; arguments=(Get-FoundationTaskArguments $root $archive 18767 $cpu); working_directory=$root }
    $taskReceipt = @{ action=$action; lab_root=$root }
    $task = [pscustomobject]@{ TaskName=$paths.Task; TaskPath='\';
        Actions=@(New-ScheduledTaskAction -Execute $paths.Shell -Argument $action.arguments -WorkingDirectory $root);
        Principal=(New-ScheduledTaskPrincipal -UserId $paths.Sid -LogonType Interactive -RunLevel Limited);
        Triggers=@(New-ScheduledTaskTrigger -AtLogOn -User $paths.Sid);
        Settings=(New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries) }
    Assert-FoundationTask $task $taskReceipt
    $checks.Add('NATIVE_CPU_TASK_ACTION_ADMITTED_WITHOUT_REGISTRATION')
    $task.Actions = @(New-ScheduledTaskAction -Execute $paths.Shell -Argument ($action.arguments + ' -CpuStateDirectory wrong') -WorkingDirectory $root)
    Reject 'CONFLICTING_CPU_TASK_ARGUMENTS' { Assert-FoundationTask $task $taskReceipt } 'conflicting task definition'
    $child = Start-Process -FilePath $paths.Shell -ArgumentList '-NoProfile -NonInteractive -WindowStyle Hidden -Command "Start-Sleep -Seconds 15"' -WindowStyle Hidden -PassThru
    $children += $child
    $clock = [Diagnostics.Stopwatch]::StartNew()
    $guard = New-FoundationCpuDeadline $child 800
    try {
        if (-not $child.WaitForExit(4000)) { throw 'Direct child deadline did not terminate its retained handle.' }
        if ($clock.ElapsedMilliseconds -gt 4000 -or $child.ExitCode -ne 124) { throw 'Direct child deadline result differs.' }
        Reject 'LATE_READINESS_CANNOT_PASS' { $guard.Complete() } 'startup deadline exceeded'
        $checks.Add('NATIVE_DEADLINE_TERMINATES_ONLY_DIRECT_NEW_CHILD')
    } finally { $guard.Dispose() }
    $child = Start-Process -FilePath $paths.Shell -ArgumentList '-NoProfile -NonInteractive -WindowStyle Hidden -Command "Start-Sleep -Seconds 2"' -WindowStyle Hidden -PassThru
    $children += $child
    $guard = New-FoundationCpuDeadline $child 1000
    $guard.Complete(); $guard.Dispose()
    if (-not $child.WaitForExit(6000) -or $child.ExitCode -ne 0) { throw 'Successful completion did not disarm the direct child deadline.' }
    $checks.Add('COMPLETED_READINESS_DISARMS_DEADLINE')
    $child = Start-Process -FilePath $paths.Shell -ArgumentList '-NoProfile -NonInteractive -WindowStyle Hidden -Command "Start-Sleep -Seconds 15"' -WindowStyle Hidden -PassThru
    $children += $child
    $clock = [Diagnostics.Stopwatch]::StartNew()
    Start-Sleep -Milliseconds 600
    $guard = New-FoundationCpuDeadline $child 500 $clock
    try {
        Reject 'GUARD_SETUP_CANNOT_EXTEND_ORIGINAL_DEADLINE' { $guard.Complete() } 'startup deadline exceeded'
        if (-not $child.WaitForExit(4000) -or $child.ExitCode -ne 124) { throw 'Late guard setup did not preserve the original monotonic deadline.' }
    } finally { $guard.Dispose() }
    @{ schema='szl.foundation-confirmation.cpu-source-controls/v1'; status='VERIFIED'; observed_utc=[datetime]::UtcNow.ToString('o');
        shell=$PSVersionTable.PSVersion.ToString(); count=$checks.Count; checks=@($checks);
        tasks_registered=$false; production_services_changed=$false; model_imported=$false; provider_writes=$false } | ConvertTo-Json -Depth 5
} finally {
    foreach ($child in $children) { $child.Dispose() }
    if (Test-Path -LiteralPath $fixture) {
        $safe = Assert-FoundationPath $fixture -Directory
        if (-not $safe.StartsWith($PSScriptRoot.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase) -or
            [IO.Path]::GetFileName($safe) -notmatch '^\.cpu-controls-[0-9a-f]{32}$') { throw 'Refused fixture cleanup outside this owned source directory.' }
        Remove-Item -LiteralPath $safe -Recurse -Force
    }
}
