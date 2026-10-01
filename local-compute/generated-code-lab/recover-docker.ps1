# SPDX-License-Identifier: Apache-2.0
# Explicitly authorized local Docker restart after normal restart failed.
# No filesystem cleanup, factory reset, service policy changes, or WSL termination.
[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$installationRoot = 'C:\Program Files\Docker\Docker'
$pluginRoot = 'C:\Program Files\Docker\cli-plugins'
$launcher = Join-Path $installationRoot 'Docker Desktop.exe'
if (-not (Test-Path -LiteralPath $launcher -PathType Leaf)) { throw 'Docker launcher is missing' }
$ownedNames = @('Docker Desktop.exe', 'com.docker.backend.exe', 'com.docker.build.exe', 'docker-mcp.exe', 'docker-agent.exe', 'docker-offload.exe')
$snapshot = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -in $ownedNames })
$stopped = @()
# Validate the full snapshot before stopping any process. Docker also installs
# docker-agent under its sibling cli-plugins directory on this host.
foreach ($item in $snapshot) {
    $current = Get-CimInstance Win32_Process -Filter ('ProcessId = ' + $item.ProcessId)
    if (-not $current) { continue }
    $owned = $current.ExecutablePath -and (
        $current.ExecutablePath.StartsWith($installationRoot + '\', [StringComparison]::OrdinalIgnoreCase) -or
        $current.ExecutablePath.Equals((Join-Path $pluginRoot $current.Name), [StringComparison]::OrdinalIgnoreCase))
    if (-not $owned -or $current.CreationDate -ne $item.CreationDate) {
        throw ('Cannot confirm Docker ownership for PID ' + $item.ProcessId)
    }
}
foreach ($item in $snapshot) {
    $current = Get-CimInstance Win32_Process -Filter ('ProcessId = ' + $item.ProcessId)
    if (-not $current) { continue }
    if ($current.CreationDate -ne $item.CreationDate) { throw 'Process identity changed; refusing stale PID' }
    if (-not $current.ExecutablePath -or -not (
        $current.ExecutablePath.StartsWith($installationRoot + '\', [StringComparison]::OrdinalIgnoreCase) -or
        $current.ExecutablePath.Equals((Join-Path $pluginRoot $current.Name), [StringComparison]::OrdinalIgnoreCase))) {
        throw ('Cannot confirm Docker ownership for PID ' + $item.ProcessId)
    }
    if ($current) {
        Stop-Process -Id $item.ProcessId -Force -ErrorAction Stop
        $stopped += [pscustomobject]@{ pid = $current.ProcessId; name = $current.Name; path = $current.ExecutablePath }
    }
}
$remaining = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -in $ownedNames })
if ($remaining.Count -gt 0) { throw 'Docker processes reappeared; refusing to race another launcher' }
$started = Start-Process -FilePath $launcher -WindowStyle Hidden -PassThru
[pscustomobject]@{
    stopped = $stopped
    launchedPid = $started.Id
    launchedPath = $launcher
    deletedFiles = 0
    volumesModified = $false
    wslTerminated = $false
} | ConvertTo-Json -Depth 4
