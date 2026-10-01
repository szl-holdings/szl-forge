# SPDX-License-Identifier: Apache-2.0
param([switch]$ValidateOnly)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'supervise-workbench.ps1') -ValidateOnly:$ValidateOnly

try {
    $paths = Get-FoundationPaths
    $receipt = Read-FoundationInstallation
    $task = Get-ScheduledTask -TaskName $paths.Task -TaskPath '\' -ErrorAction SilentlyContinue
    if ($null -ne $task) { Assert-FoundationTask $task $receipt }
    $known = @('install-workbench.ps1', 'supervise-workbench.ps1', 'uninstall-workbench.ps1', 'installation.json', 'supervisor-status.json')
    $unknown = @(Get-ChildItem -LiteralPath $paths.Operations -Force | Where-Object { $_.PSIsContainer -or $_.Name -notin $known })
    if ($unknown.Count -gt 0) { throw 'The managed directory contains unowned files; uninstall refused.' }
    if (Test-Path -LiteralPath (Join-Path $paths.Operations 'supervisor-status.json')) {
        $status = Get-Content -LiteralPath (Join-Path $paths.Operations 'supervisor-status.json') -Raw | ConvertFrom-Json
        if ($status.schema -ne 'szl.foundation-confirmation.windows-supervisor/v1' -or $status.owner_sid -ne $paths.Sid -or
            $status.installation_id -ne $receipt.installation_id) { throw 'Unowned supervisor evidence was found.' }
    }
    if ($ValidateOnly) {
        @{ status = 'OWNERSHIP_VERIFIED'; task_present = ($null -ne $task); task_removed = $false; service_stopped = $false } | ConvertTo-Json
        exit 0
    }
    if ($null -ne $task) {
        Unregister-ScheduledTask -TaskName $paths.Task -TaskPath '\' -Confirm:$false -ErrorAction Stop
        if ($null -ne (Get-ScheduledTask -TaskName $paths.Task -TaskPath '\' -ErrorAction SilentlyContinue)) {
            throw 'Task removal readback failed; managed files were preserved.'
        }
    }
    # Explicit files only. Neither the lab nor its state/trials are deletion targets.
    foreach ($name in $known) {
        $target = Join-Path $paths.Operations $name
        if (Test-Path -LiteralPath $target) {
            Assert-FoundationPath $target | Out-Null
            Remove-Item -LiteralPath $target -ErrorAction Stop
        }
    }
    if (@(Get-ChildItem -LiteralPath $paths.Operations -Force).Count -eq 0) {
        Remove-Item -LiteralPath $paths.Operations -ErrorAction Stop
    }
    @{ status = 'UNINSTALLED'; task_removed = $true; service_stopped = $false; lab_and_trial_receipts_preserved = $true } | ConvertTo-Json
} catch {
    Write-Error $_ -ErrorAction Continue
    exit 1
}
