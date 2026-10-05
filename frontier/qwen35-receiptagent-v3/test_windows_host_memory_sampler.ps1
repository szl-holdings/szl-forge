# Native API and controlling-pipe lifetime controls on CI Windows hosts.
[CmdletBinding()]
param([Parameter(Mandatory = $true)][ValidatePattern('^[0-9a-f]{40}$')][string]$SourceRevision)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Get-ValidatedNativeUtcStamp {
    param(
        [Parameter(Mandatory = $true)][string]$RawJson,
        [DateTimeOffset]$Now = [DateTimeOffset]::UtcNow
    )
    $document = [System.Text.Json.JsonDocument]::Parse($RawJson)
    try {
        # ConvertFrom-Json may convert ISO strings to local DateTime values.
        # Validate the native representation before that timezone conversion.
        $stamp = $document.RootElement.GetProperty('observedAt').GetString()
        if (-not $stamp -or (-not $stamp.EndsWith('Z') -and -not $stamp.EndsWith('+00:00'))) {
            throw 'native timestamp has no explicit UTC offset'
        }
        $observed = [DateTimeOffset]::Parse($stamp, [Globalization.CultureInfo]::InvariantCulture)
        if ([Math]::Abs(($Now - $observed).TotalSeconds) -gt 5 -or
            $observed.Offset -ne [TimeSpan]::Zero) {
            throw 'native timestamp is not fresh UTC'
        }
        return $stamp
    }
    finally { $document.Dispose() }
}

$referenceTime = [DateTimeOffset]::Parse('2026-01-01T12:00:00+00:00')
foreach ($stamp in @('2026-01-01T12:00:00+00:00', '2026-01-01T12:00:00Z',
                      '2026-01-01T12:00:05Z', '2026-01-01T11:59:55Z')) {
    $rawCase = @{ observedAt = $stamp } | ConvertTo-Json -Compress
    $retained = Get-ValidatedNativeUtcStamp -RawJson $rawCase -Now $referenceTime
    if ($retained -cne $stamp) { throw 'native timestamp representation changed' }
}
foreach ($stamp in @('2026-01-01T07:00:00-05:00', '2026-01-01T12:00:06Z',
                      '2026-01-01T11:59:54Z', '2026-01-01T12:00:00', 'invalid', $null)) {
    $rejected = $false
    try {
        $rawCase = @{ observedAt = $stamp } | ConvertTo-Json -Compress
        $null = Get-ValidatedNativeUtcStamp -RawJson $rawCase -Now $referenceTime
    }
    catch { $rejected = $true }
    if (-not $rejected) { throw 'invalid native timestamp was accepted' }
}

$sampler = Join-Path $PSScriptRoot 'windows_host_memory_sampler.ps1'
$samplerSha = (Get-FileHash -LiteralPath $sampler -Algorithm SHA256).Hash.ToLowerInvariant()
$executables = @(
    (Join-Path $env:SystemRoot 'System32/WindowsPowerShell/v1.0/powershell.exe'),
    (Join-Path $PSHOME 'pwsh.exe')
)
foreach ($executable in $executables) {
    $runId = [Guid]::NewGuid().ToString('N')
    $start = [System.Diagnostics.ProcessStartInfo]::new()
    $start.FileName = $executable
    $start.UseShellExecute = $false
    $start.RedirectStandardOutput = $true
    $start.RedirectStandardError = $true
    foreach ($argument in @('-NoLogo', '-NoProfile', '-NonInteractive', '-File', $sampler,
                            '-RunId', $runId, '-SourceRevision', $SourceRevision,
                            '-SamplerSourceSha256', $samplerSha, '-Once')) {
        $start.ArgumentList.Add($argument)
    }
    $process = [System.Diagnostics.Process]::new()
    $process.StartInfo = $start
    try {
        if (-not $process.Start()) { throw 'native sampler did not start' }
        $stdout = $process.StandardOutput.ReadToEndAsync()
        $stderr = $process.StandardError.ReadToEndAsync()
        if (-not $process.WaitForExit(15000)) {
            $process.Kill($true)
            throw 'native sampler exceeded the admission deadline'
        }
        if ($process.ExitCode -ne 0) { throw 'native sampler exited unsuccessfully' }
        $raw = $stdout.GetAwaiter().GetResult()
        if ($raw.Length -gt 4096) { throw 'native sample exceeded its size bound' }
        $row = $raw | ConvertFrom-Json
        if ($row.schema -cne 'szl.windows-host-memory-sample/v1' -or
            $row.runId -cne $runId -or $row.sourceRevision -cne $SourceRevision -or
            $row.samplerSourceSha256 -cne $samplerSha -or $row.sequence -ne 0) {
            throw 'native sample identity differs'
        }
        foreach ($name in @('availablePhysicalBytes', 'totalPhysicalBytes', 'committedBytes',
                            'commitLimitBytes', 'availableCommitBytes',
                            'nativeMonotonicTicks', 'nativeMonotonicFrequency')) {
            $value = $row.$name
            if (($value -isnot [int] -and $value -isnot [long] -and $value -isnot [uint64]) -or $value -lt 0) {
                throw 'native counter is not a nonnegative integer'
            }
        }
        if ($row.totalPhysicalBytes -le 0 -or $row.commitLimitBytes -le 0 -or
            $row.availablePhysicalBytes -gt $row.totalPhysicalBytes -or
            $row.committedBytes -gt $row.commitLimitBytes -or
            $row.availableCommitBytes -ne ($row.commitLimitBytes - $row.committedBytes) -or
            $row.nativeMonotonicFrequency -le 0) {
            throw 'native memory counters are inconsistent'
        }
        $row.observedAt = Get-ValidatedNativeUtcStamp -RawJson $raw
        [ordered]@{
            schema = 'szl.windows-native-memory-control/v1'
            state = 'NATIVE_API_OBSERVER_VERIFIED'
            executable = [System.IO.Path]::GetFileName($executable)
            sample = $row
            trainingAttempted = $false
            wslInteropVerified = $false
            workerContainmentVerified = $false
            fullRunCapacityProven = $false
            qualificationEligible = $false
            publicationEligible = $false
        } | ConvertTo-Json -Depth 5 -Compress
    }
    finally { $process.Dispose() }

    # A helper death closes this pipe. Verify the real native observer exits
    # promptly on EOF, without retaining a Windows observer after WSL exits.
    $start = [System.Diagnostics.ProcessStartInfo]::new()
    $start.FileName = $executable
    $start.UseShellExecute = $false
    $start.RedirectStandardInput = $true
    $start.RedirectStandardOutput = $true
    $start.RedirectStandardError = $true
    foreach ($argument in @('-NoLogo', '-NoProfile', '-NonInteractive', '-File', $sampler,
                            '-RunId', $runId, '-SourceRevision', $SourceRevision,
                            '-SamplerSourceSha256', $samplerSha)) {
        $start.ArgumentList.Add($argument)
    }
    $process = [System.Diagnostics.Process]::new()
    $process.StartInfo = $start
    try {
        if (-not $process.Start()) { throw 'persistent native observer did not start' }
        $stderr = $process.StandardError.ReadToEndAsync()
        $firstLine = $process.StandardOutput.ReadLineAsync()
        if (-not $firstLine.Wait(15000)) { throw 'persistent native observer was silent' }
        $first = $firstLine.GetAwaiter().GetResult() | ConvertFrom-Json
        if ($first.sequence -ne 0 -or $first.runId -cne $runId) {
            throw 'persistent native observer first identity differs'
        }
        $secondLine = $process.StandardOutput.ReadLineAsync()
        if (-not $secondLine.Wait(8000)) { throw 'persistent native observer lost its cadence' }
        $second = $secondLine.GetAwaiter().GetResult() | ConvertFrom-Json
        if ($second.sequence -ne 1 -or $second.runId -cne $runId -or
            $second.nativeMonotonicTicks -le $first.nativeMonotonicTicks -or
            $second.nativeMonotonicFrequency -ne $first.nativeMonotonicFrequency) {
            throw 'persistent native observer sequence differs'
        }
        $process.StandardInput.Close()
        if (-not $process.WaitForExit(5000)) { throw 'native observer did not exit after helper pipe EOF' }
        if ($process.ExitCode -ne 0) { throw 'native observer EOF exit was unsuccessful' }
        [ordered]@{
            schema = 'szl.windows-native-memory-lifetime-control/v1'
            state = 'NATIVE_OBSERVER_EOF_EXIT_VERIFIED'
            executable = [System.IO.Path]::GetFileName($executable)
            firstSequence = $first.sequence
            secondSequence = $second.sequence
            sourceRevision = $SourceRevision
            samplerSourceSha256 = $samplerSha
            trainingAttempted = $false
            wslInteropVerified = $false
            workerContainmentVerified = $false
            fullRunCapacityProven = $false
            qualificationEligible = $false
            publicationEligible = $false
        } | ConvertTo-Json -Compress
    }
    finally {
        if (-not $process.HasExited) {
            $process.Kill($true)
            $process.WaitForExit()
        }
        $process.Dispose()
    }
}
