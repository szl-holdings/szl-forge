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
$fixtureParent = (Get-FoundationPaths).Profile
$fixture = Join-Path $fixtureParent ('.cpu-controls-' + [guid]::NewGuid().ToString('N'))
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
    $taskReceipt = @{ schema='szl.foundation-confirmation.windows-installation/v2'; task_priority=5; action=$action; lab_root=$root }
    $task = [pscustomobject]@{ TaskName=$paths.Task; TaskPath='\';
        Actions=@(New-ScheduledTaskAction -Execute $paths.Shell -Argument $action.arguments -WorkingDirectory $root);
        Principal=(New-ScheduledTaskPrincipal -UserId $paths.Sid -LogonType Interactive -RunLevel Limited);
        Triggers=@(New-ScheduledTaskTrigger -AtLogOn -User $paths.Sid);
        Settings=(New-FoundationTaskSettings) }
    if ($task.Settings.Priority -ne 5 -or $task.Settings.RestartCount -ne 0 -or
        [string]$task.Settings.MultipleInstances -cne 'IgnoreNew' -or $task.Settings.ExecutionTimeLimit -cne 'PT0S') {
        throw 'New native Scheduler settings differ from the Normal priority/recovery contract.'
    }
    Assert-FoundationTask $task $taskReceipt
    $checks.Add('NATIVE_CPU_TASK_ACTION_ADMITTED_WITHOUT_REGISTRATION')
    $checks.Add('NEW_NATIVE_TASK_SETTINGS_BIND_NORMAL_PRIORITY_WITH_ZERO_SCHEDULER_RETRIES')
    foreach ($priority in @(0,4,6,7,10)) {
        $task.Settings.Priority = $priority
        Reject ('CONFLICTING_CPU_TASK_PRIORITY_' + $priority) { Assert-FoundationTask $task $taskReceipt } 'conflicting task definition'
    }
    $task.Settings.Priority = 5
    foreach ($priority in @($null, '5', 5.0, 4, 7)) {
        $badReceipt = $taskReceipt.Clone(); $badReceipt.task_priority = $priority
        $priorityType = if ($null -eq $priority) { 'NULL' } else { $priority.GetType().Name }
        Reject ('INVALID_RECEIPT_TASK_PRIORITY_' + $priorityType + '_' + [string]$priority) { Assert-FoundationTask $task $badReceipt } 'installation task priority contract'
    }
    $badReceipt = $taskReceipt.Clone(); $badReceipt.Remove('task_priority')
    Reject 'V2_RECEIPT_MISSING_TASK_PRIORITY' { Assert-FoundationTask $task $badReceipt } 'installation task priority contract'
    $badReceipt = $taskReceipt.Clone(); $badReceipt.schema = 'szl.foundation-confirmation.windows-installation/v3'
    Reject 'UNKNOWN_INSTALLATION_SCHEMA' { Assert-FoundationTask $task $badReceipt } 'installation task priority contract'
    $badReceipt = $taskReceipt.Clone(); $badReceipt.schema = $null
    Reject 'NULL_INSTALLATION_SCHEMA' { Assert-FoundationTask $task $badReceipt } 'installation task priority contract'
    $badReceipt = $taskReceipt.Clone(); $badReceipt.schema = @('szl.foundation-confirmation.windows-installation/v2')
    Reject 'ARRAY_INSTALLATION_SCHEMA' { Assert-FoundationTask $task $badReceipt } 'installation task priority contract'
    $badReceipt = $taskReceipt.Clone(); $badReceipt.schema = 'szl.foundation-confirmation.windows-installation/v1'
    Reject 'V1_SCHEMA_CANNOT_HIDE_V2_PRIORITY_FIELD' { Assert-FoundationTask $task $badReceipt } 'installation task priority contract'
    $legacyReceipt = $badReceipt.Clone(); $legacyReceipt.Remove('task_priority')
    $downgradedReceipt = $legacyReceipt.Clone()
    $downgradedReceipt.scripts = @{ 'install-workbench.ps1'=(Get-FileHash -LiteralPath (Join-Path $PSScriptRoot 'install-workbench.ps1') -Algorithm SHA256).Hash.ToLowerInvariant() }
    $task.Settings.Priority = 7
    Reject 'V1_SCHEMA_AND_PRIORITY_FIELD_DOWNGRADE_CANNOT_AUTHORIZE_NEW_INSTALLER' { Assert-FoundationTask $task $downgradedReceipt } 'installation task priority contract'
    $legacyReceipt.scripts = @{ 'install-workbench.ps1'='2dded72edba06cd048c0ce17c3c36a66e80feeba2b31d990e5c2d3da45349b24' }
    $task.Settings.Priority = 7
    Assert-FoundationTask $task $legacyReceipt
    Assert-FoundationTask $task ($legacyReceipt | ConvertTo-Json -Depth 4 | ConvertFrom-Json)
    $checks.Add('LEGACY_V1_DEFAULT_PRIORITY_ADMITTED_FOR_OWNED_VALIDATION_AND_REMOVAL')
    $legacyCpuHash = $legacyReceipt.scripts.'install-workbench.ps1'
    $legacyReceipt.scripts.'install-workbench.ps1' = '0cb4523b74cbd197c86fae827294a648d4aea496f7a53428af176aacf3b49cfb'
    Assert-FoundationTask $task $legacyReceipt
    $checks.Add('LEGACY_V1_BOUNDED_RECOVERY_INSTALLER_DEFAULT_PRIORITY_ADMITTED')
    $legacyReceipt.scripts.'install-workbench.ps1' = $legacyCpuHash
    $badReceipt = $legacyReceipt | ConvertTo-Json -Depth 4 | ConvertFrom-Json
    $badReceipt.scripts.'install-workbench.ps1' = 'unknown'
    Reject 'UNKNOWN_V1_INSTALLER_SOURCE' { Assert-FoundationTask $task $badReceipt } 'installation task priority contract'
    $task.Settings.Priority = 5
    Reject 'LEGACY_V1_PRIORITY_MUTATION_REJECTED' { Assert-FoundationTask $task $legacyReceipt } 'conflicting task definition'
    $downgradedReceipt = $taskReceipt | ConvertTo-Json -Depth 4 | ConvertFrom-Json
    $downgradedReceipt.schema = 'szl.foundation-confirmation.windows-installation/v1'
    Reject 'JSON_V1_SCHEMA_CANNOT_HIDE_V2_PRIORITY_FIELD' { Assert-FoundationTask $task $downgradedReceipt } 'installation task priority contract'
    $jsonReceipt = $taskReceipt | ConvertTo-Json -Depth 4 | ConvertFrom-Json
    Assert-FoundationTask $task $jsonReceipt
    $checks.Add('JSON_V2_RECEIPT_NORMAL_PRIORITY_ADMITTED')
    $task.Actions = @(New-ScheduledTaskAction -Execute $paths.Shell -Argument ($action.arguments + ' -CpuStateDirectory wrong') -WorkingDirectory $root)
    Reject 'CONFLICTING_CPU_TASK_ARGUMENTS' { Assert-FoundationTask $task $taskReceipt } 'conflicting task definition'
    # This is a stdlib-only test image, not production environment admission.
    $fixturePython = Join-Path $paths.Profile 'AppData\Local\Programs\Python\Python311\python.exe'
    if (-not (Test-Path -LiteralPath $fixturePython)) { $fixturePython = (Get-Command python -CommandType Application -ErrorAction Stop).Source }
    $nativeRaw = & $fixturePython -I -S -B (Join-Path $PSScriptRoot 'test-cpu-predecessor.py')
    if ($LASTEXITCODE -ne 0) { throw 'The actual retained-handle predecessor fixture failed.' }
    $native = ($nativeRaw -join "`n") | ConvertFrom-Json
    if ($native.schema -cne 'szl.foundation-confirmation.native-predecessor-controls/v1' -or $native.status -cne 'VERIFIED' -or
        $native.class -cne 'MEASURED' -or $native.count -ne 5 -or $native.only_owned_fixture_child -ne $true -or
        $native.production_services_changed -ne $false -or $native.model_imported -ne $false) { throw 'Native predecessor fixture evidence differs.' }
    foreach ($check in $native.checks) { $checks.Add([string]$check) }
    Initialize-FoundationCpuDeadline # Match production: compile before the child exists.
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
    # Keep the successful child alive until after the deadline. A slow shell
    # startup must not be mistaken for a guard that failed to disarm.
    $signal = Join-Path $fixture 'completed-child.signal'
    $quotedSignal = $signal.Replace("'", "''")
    $childCommand = "while (-not (Test-Path -LiteralPath '$quotedSignal')) { Start-Sleep -Milliseconds 100 }"
    $encodedCommand = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($childCommand))
    $child = Start-Process -FilePath $paths.Shell -ArgumentList ('-NoProfile -NonInteractive -WindowStyle Hidden -EncodedCommand ' + $encodedCommand) -WindowStyle Hidden -PassThru
    $children += $child
    $guard = New-FoundationCpuDeadline $child 5000
    try {
        $guard.Complete()
        if ($child.WaitForExit(5500)) { throw 'Completed guard terminated the direct child before release.' }
        Set-Content -LiteralPath $signal -Value 'release' -NoNewline -Encoding Ascii
        if (-not $child.WaitForExit(20000) -or $child.ExitCode -ne 0) { throw 'Successful completion did not disarm the direct child deadline.' }
    } finally {
        $guard.Dispose()
        if (-not $child.HasExited) { $child.Kill(); $child.WaitForExit(5000) | Out-Null }
    }
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
        if (-not $safe.StartsWith($fixtureParent.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase) -or
            [IO.Path]::GetDirectoryName($safe) -cne $fixtureParent -or
            [IO.Path]::GetFileName($safe) -notmatch '^\.cpu-controls-[0-9a-f]{32}$') { throw 'Refused fixture cleanup outside this owned profile directory.' }
        Remove-Item -LiteralPath $safe -Recurse -Force
    }
}
