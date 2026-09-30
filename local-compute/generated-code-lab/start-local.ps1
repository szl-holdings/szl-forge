# SPDX-License-Identifier: Apache-2.0
[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$runtimeDir = Join-Path $PSScriptRoot 'runtime'
$null = New-Item -ItemType Directory -Path $runtimeDir -Force
$tcpProbe = [Net.Sockets.TcpClient]::new()
try {
    $connect = $tcpProbe.ConnectAsync('127.0.0.1', 11439)
    if ($connect.Wait(1000) -and $tcpProbe.Connected) {
        throw 'Port 11439 is already in use; inspect its owner instead of starting a duplicate.'
    }
} catch [System.AggregateException] {
    # Connection refused is expected when the task-specific port is free.
} finally { $tcpProbe.Dispose() }
$ollamaExe = (Get-Command ollama.exe -ErrorAction Stop).Source
$pythonExe = (Get-Command python.exe -ErrorAction Stop).Source
# Process-local environment only: no user/machine settings or services changed.
$settings = @{
    OLLAMA_HOST = '127.0.0.1:11439'; OLLAMA_NO_CLOUD = '1'; OLLAMA_NOPRUNE = '1'
    OLLAMA_ORIGINS = 'http://127.0.0.1:11439'; OLLAMA_NUM_PARALLEL = '1'
    OLLAMA_MAX_LOADED_MODELS = '1'; OLLAMA_CONTEXT_LENGTH = '4096'; OLLAMA_KEEP_ALIVE = '2m'
    OLLAMA_MODELS = (Join-Path $env:USERPROFILE '.ollama\models')
}
$prior = @{}
$logId = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ') + '-' + [Guid]::NewGuid().ToString('N').Substring(0, 8)
try {
    foreach ($key in $settings.Keys) {
        $prior[$key] = [Environment]::GetEnvironmentVariable($key, 'Process')
        [Environment]::SetEnvironmentVariable($key, $settings[$key], 'Process')
    }
    $localProcess = Start-Process -FilePath $ollamaExe -ArgumentList 'serve' -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $runtimeDir ($logId + '.stdout.log')) `
        -RedirectStandardError (Join-Path $runtimeDir ($logId + '.stderr.log'))
} finally {
    foreach ($key in $prior.Keys) { [Environment]::SetEnvironmentVariable($key, $prior[$key], 'Process') }
}
$localProcess | Select-Object Id,ProcessName
& $pythonExe -B (Join-Path $PSScriptRoot 'lab.py') ready --timeout 60
if ($LASTEXITCODE -ne 0) { throw ('Readiness failed; inspect task-local PID ' + $localProcess.Id + ' and logs ' + $logId) }
Write-Output 'Verified task-local endpoint: http://127.0.0.1:11439. Cloud features disabled. No auto-start service installed.'
