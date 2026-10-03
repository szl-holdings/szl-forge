# SPDX-License-Identifier: Apache-2.0
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'supervise-workbench.ps1')
Initialize-FoundationPhaseLog -Role 'cpu-launcher'
try {
Write-FoundationPhaseEvent 'installation_admission' 'ENTERED'
$installation = Read-FoundationInstallation
if (-not $installation.cpu_runtime) { throw 'This launcher requires an admitted CPU installation.' }
Write-FoundationPhaseEvent 'installation_admission' 'RETURNED'
$paths = Get-FoundationPaths
Assert-FoundationTask (Get-ScheduledTask -TaskName $paths.Task -TaskPath '\' -ErrorAction Stop) $installation
Write-FoundationPhaseEvent 'release_verification' 'ENTERED'
$admitted = Test-FoundationRelease $installation.lab_root $installation.archive_path
Write-FoundationPhaseEvent 'release_verification' 'RETURNED'
Write-FoundationPhaseEvent 'predecessor_admission' 'ENTERED'
$prior = Get-FoundationService $admitted.LabRoot $admitted.Python $installation.port -MayBeAbsent
Assert-FoundationLaunchPort $installation.port $prior
if ($prior) { throw 'An admitted service already runs; the CPU launcher did not replace it.' }
Write-FoundationPhaseEvent 'predecessor_admission' 'RETURNED'
$cpu = $script:foundationCpuRuntime
$state = Assert-FoundationPath $cpu.state_directory -Directory
$argv = @(Get-FoundationCpuArgv $cpu $admitted.LabRoot $admitted.ArchivePath $installation.port)
$arguments = (@($argv | Select-Object -Skip 1 | ForEach-Object {
    if ($_ -match '["\x00-\x1f]') { throw 'Unsafe CPU launcher argument.' }
    '"' + [string]$_ + '"'
}) -join ' ')
$priorSettings = @{}
$attempt = [guid]::NewGuid().ToString('N')
$stdout = Join-Path $state ('service.' + $attempt + '.stdout.log')
$stderr = Join-Path $state ('service.' + $attempt + '.stderr.log')
$attemptReceipt = Join-Path $state ('launch.' + $attempt + '.json')
foreach ($path in @($stdout, $stderr, $attemptReceipt)) {
    if (Test-Path -LiteralPath $path) { throw 'The exclusive CPU launch evidence path already exists.' }
}
Write-FoundationPhaseEvent 'child_creation' 'ENTERED'
Initialize-FoundationCpuDeadline # Compile before any child exists.
$clock = [Diagnostics.Stopwatch]::StartNew()
try {
    foreach ($name in @('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')) {
        $priorSettings[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
        [Environment]::SetEnvironmentVariable($name, '1', 'Process')
    }
    $child = Start-Process -FilePath $cpu.python_executable -ArgumentList $arguments -WorkingDirectory $admitted.LabRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
} finally {
    foreach ($name in $priorSettings.Keys) { [Environment]::SetEnvironmentVariable($name, $priorSettings[$name], 'Process') }
}
$guard = $null
try {
$guard = New-FoundationCpuDeadline $child 180000 $clock
Write-FoundationPhaseEvent 'child_creation' 'RETURNED'
Write-FoundationPhaseEvent 'child_readiness_admission' 'ENTERED'
Write-FoundationJson $attemptReceipt @{ schema='szl.foundation-confirmation.cpu-launch-attempt/v1';
    attempt_id=$attempt; observed_utc=[datetime]::UtcNow.ToString('o'); pid=$child.Id;
    process_created=(ConvertTo-FoundationCimTimestamp $child.StartTime).ToString('o');
    native_executable=$cpu.python_executable; argv=$argv; stdout=$stdout; stderr=$stderr;
    environment_binding_sha256=$cpu.environment_binding_sha256; startup_timeout_seconds=180 }
while ($clock.ElapsedMilliseconds -lt 180000) {
    $child.Refresh()
    if ($child.HasExited) { throw 'The new CPU child exited; its startup error is retained in CPU state logs.' }
    $servicePath = Join-Path $state 'service.json'
    if (Test-Path -LiteralPath $servicePath) {
        $receipt = Get-Content -LiteralPath $servicePath -Raw | ConvertFrom-Json
        if ($receipt.pid -eq $child.Id) {
            # Exactly one readiness admission after all-three preload publishes
            # the receipt. Endpoint limits remain 5 seconds and 10 seconds.
            $owned = Get-FoundationService $admitted.LabRoot $admitted.Python $installation.port
            if ($owned.Receipt.pid -ne $child.Id -or
                (ConvertTo-FoundationCimTimestamp $child.StartTime) -ne ([datetime]$owned.Receipt.process_created).ToUniversalTime()) { throw 'The CPU child identity changed before launch admission.' }
            # Endpoint admission only. Complete still charges all diagnostic I/O.
            Write-FoundationPhaseEvent 'child_readiness_admission' 'RETURNED'
            $guard.Complete() # Monotonic recheck after the bounded endpoints.
            @{ status='ADMITTED_CPU_READY'; pid=$child.Id; models_loaded=$owned.Status.models_loaded; receipt_minted=$false } | ConvertTo-Json
            exit 0
        }
    }
    Start-Sleep -Milliseconds 200
}
throw 'The bounded CPU startup deadline elapsed; no readiness retry was attempted.'
} finally {
    if ($null -ne $guard) { $guard.Dispose() }
    else {
        # Setup failed before the guard took ownership. Only this direct child
        # returned by Start-Process may be terminated, never a receipt/port PID.
        $child.Refresh()
        if (-not $child.HasExited) { $child.Kill(); $child.WaitForExit(5000) | Out-Null }
    }
}
} catch {
    Write-FoundationPhaseFailure $_
    throw
} finally {
    Close-FoundationPhaseLog
}
