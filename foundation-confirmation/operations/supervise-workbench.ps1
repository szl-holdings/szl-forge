# SPDX-License-Identifier: Apache-2.0
param(
    [string]$LabRoot,
    [string]$ArchivePath,
    [ValidateRange(1, 65535)][int]$Port = 18767,
    [switch]$ValidateOnly
)
$ErrorActionPreference = 'Stop'

function Assert-FoundationWindows {
    if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT -or $PSVersionTable.PSVersion.Major -lt 5) {
        throw 'Windows PowerShell 5.1 or newer on Windows is required.'
    }
}

function Get-FoundationPaths {
    Assert-FoundationWindows
    $profileRoot = [IO.Path]::GetFullPath([Environment]::GetFolderPath('UserProfile')).TrimEnd('\')
    # Packaged desktop applications may see a virtualized LocalApplicationData
    # directory that the native Task Scheduler process cannot see. The explicit
    # profile Documents path has one shared filesystem view for both processes.
    $operations = Join-Path $profileRoot 'Documents\SZL\FoundationConfirmation'
    return @{
        Profile = $profileRoot
        Operations = [IO.Path]::GetFullPath($operations)
        Receipt = Join-Path $operations 'installation.json'
        Supervisor = Join-Path $operations 'supervise-workbench.ps1'
        Shell = Join-Path $env:WINDIR 'System32\WindowsPowerShell\v1.0\powershell.exe'
        Sid = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
        Task = 'SZL Foundation Confirmation'
    }
}

function Assert-FoundationPath {
    param([string]$Path, [switch]$Directory, [switch]$MayNotExist)
    $paths = Get-FoundationPaths
    if ([string]::IsNullOrWhiteSpace($Path) -or $Path -notmatch '^[A-Za-z]:\\' -or
        $Path -match '["\x00-\x1f]' -or $Path.Substring(2).Contains(':')) {
        throw 'Use an absolute local filesystem path without device, network, or stream syntax.'
    }
    $full = [IO.Path]::GetFullPath($Path).TrimEnd('\')
    $prefix = $paths.Profile + '\'
    if (-not $full.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase) -or
        -not [IO.Path]::GetPathRoot($full).Equals([IO.Path]::GetPathRoot($paths.Profile), [StringComparison]::OrdinalIgnoreCase) -or
        -not [IO.Path]::GetPathRoot($full).Equals([IO.Path]::GetPathRoot($env:WINDIR), [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Paths must remain below the current user profile on the Windows system volume.'
    }
    $cursor = $full
    while ($cursor) {
        if (Test-Path -LiteralPath $cursor) {
            $item = Get-Item -LiteralPath $cursor -Force
            if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw "Reparse points are not admitted: $cursor"
            }
        }
        $parent = [IO.Path]::GetDirectoryName($cursor)
        if ($parent -eq $cursor) { break }
        $cursor = $parent
    }
    if (-not $MayNotExist) {
        $item = Get-Item -LiteralPath $full -Force
        if ([bool]$item.PSIsContainer -ne [bool]$Directory) { throw "Unexpected path type: $full" }
    }
    return $full
}

function Get-FoundationPython {
    $preferred = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python311\python.exe'
    if (Test-Path -LiteralPath $preferred) { return Assert-FoundationPath $preferred }
    return Assert-FoundationPath (Get-Command python -CommandType Application -ErrorAction Stop).Source
}

function Assert-FoundationHash {
    param([string]$Path, [string]$Expected)
    if ((Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $Expected) {
        throw "Pinned release hash mismatch: $Path"
    }
}

function Test-FoundationRelease {
    param([string]$Root, [string]$Archive)
    $rootPath = Assert-FoundationPath $Root -Directory
    $archiveFile = Assert-FoundationPath $Archive
    $pinned = @{
        'release-manifest.json' = '03a13779b09f2e8ad3dd53d928ac460a395d329742f9f43f7e5892fba672c877'
        'verify_release.py' = 'b7b5d7775fadc6933144c16e17db2ee64e63c18b43d412a593eba927f53672f9'
        'server.py' = 'ffa50276f1956f78dc9738a36f1855e95a71c1621c52e6b1c8a6a54ddcbc55fb'
        'start.ps1' = 'cb49ca92d5a207a1a25ac64a9a5a7bf313b62fd0f293a8be603ddee73b07ff78'
        'core.py' = '573740e2f3e138b82508b49e29ad55c11e2a6e29c07fd44fe49f973628632c69'
        'policy.py' = '02705491ed34ca23ef2e900768ad33dc96903c16729a9981ad72475bf4379919'
        'experiment.py' = 'a763286c6e29dd041ff8e09a3c49f6376e3c08383d5be38788d4b81f65854a11'
    }
    Assert-FoundationHash $archiveFile '869e318dd5f328205dd181ee836ef267bd2ae278f6430a9e8ddc661fbc689d03'
    foreach ($name in $pinned.Keys) {
        $file = Assert-FoundationPath (Join-Path $rootPath $name)
        Assert-FoundationHash $file $pinned[$name]
    }
    # Reject links throughout the package, including excluded writable state.
    $pending = [Collections.Generic.Queue[string]]::new()
    $pending.Enqueue($rootPath)
    while ($pending.Count -gt 0) {
        foreach ($item in Get-ChildItem -LiteralPath $pending.Dequeue() -Force) {
            if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw "Package contains a reparse point: $($item.FullName)"
            }
            if ($item.PSIsContainer) { $pending.Enqueue($item.FullName) }
        }
    }
    $python = Get-FoundationPython
    # Only the known-hash, stdlib verifier is executable at this admission stage.
    $raw = & $python -I -B (Join-Path $rootPath 'verify_release.py') --root $rootPath
    if ($LASTEXITCODE -ne 0) { throw 'Original directory verification failed.' }
    $verified = ($raw -join "`n") | ConvertFrom-Json
    if ($verified.status -ne 'VERIFIED' -or $verified.files -ne 70 -or
        $verified.manifest_sha256 -ne $pinned['release-manifest.json'] -or
        $verified.checkpoints -ne 3 -or $verified.evaluation_rows -ne 5184) {
        throw 'Original verifier did not establish the complete frozen release.'
    }
    # Recheck the execution boundary after verification and immediately before launch.
    foreach ($name in $pinned.Keys) { Assert-FoundationHash (Join-Path $rootPath $name) $pinned[$name] }
    return @{ LabRoot = $rootPath; ArchivePath = $archiveFile; Python = $python; Verification = $verified }
}

function Get-FoundationTaskArguments {
    param([string]$Root, [string]$Archive, [int]$TaskPort)
    $paths = Get-FoundationPaths
    return '-NoProfile -NonInteractive -WindowStyle Hidden -File "' + $paths.Supervisor + '" -LabRoot "' + $Root + '" -ArchivePath "' + $Archive + '" -Port ' + $TaskPort
}

function Write-FoundationJson {
    param([string]$Path, $Value)
    $safe = Assert-FoundationPath $Path -MayNotExist
    $temporary = $safe + '.' + [guid]::NewGuid().ToString('N') + '.tmp'
    try {
        $bytes = [Text.UTF8Encoding]::new($false).GetBytes(($Value | ConvertTo-Json -Depth 12))
        $stream = [IO.File]::Open($temporary, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
        try { $stream.Write($bytes, 0, $bytes.Length); $stream.Flush($true) } finally { $stream.Dispose() }
        if (Test-Path -LiteralPath $safe) { [IO.File]::Replace($temporary, $safe, [System.Management.Automation.Language.NullString]::Value) }
        else { [IO.File]::Move($temporary, $safe) }
    } finally {
        if (Test-Path -LiteralPath $temporary) { Remove-Item -LiteralPath $temporary }
    }
}

function Read-FoundationInstallation {
    $paths = Get-FoundationPaths
    Assert-FoundationPath $paths.Operations -Directory | Out-Null
    Assert-FoundationPath $paths.Receipt | Out-Null
    $receipt = Get-Content -LiteralPath $paths.Receipt -Raw | ConvertFrom-Json
    if ($receipt.schema -ne 'szl.foundation-confirmation.windows-installation/v1' -or
        $receipt.owner_sid -ne $paths.Sid -or $receipt.task_name -ne $paths.Task -or
        $receipt.task_path -ne '\' -or $receipt.operations_root -ne $paths.Operations -or
        $receipt.action.executable -ne $paths.Shell -or $receipt.state -notin @('PREPARED', 'REGISTERED')) {
        throw 'The installation receipt does not identify this current-user installation.'
    }
    $root = Assert-FoundationPath $receipt.lab_root -Directory
    $archive = Assert-FoundationPath $receipt.archive_path
    if ($receipt.port -isnot [long] -and $receipt.port -isnot [int]) { throw 'Invalid recorded port.' }
    if ($receipt.port -lt 1 -or $receipt.port -gt 65535 -or
        $receipt.action.arguments -ne (Get-FoundationTaskArguments $root $archive $receipt.port) -or
        $receipt.action.working_directory -ne $root) { throw 'Installation action mismatch.' }
    foreach ($name in @('install-workbench.ps1', 'supervise-workbench.ps1', 'uninstall-workbench.ps1')) {
        $expected = $receipt.scripts.$name
        if ($expected -notmatch '^[0-9a-f]{64}$') { throw 'Incomplete managed script receipt.' }
        Assert-FoundationHash (Assert-FoundationPath (Join-Path $paths.Operations $name)) $expected
    }
    return $receipt
}

function Resolve-FoundationTaskSid {
    param([string]$Identity)
    try {
        if ([string]::IsNullOrWhiteSpace($Identity)) { throw 'Missing task identity.' }
        if ($Identity -match '^S-1-') {
            return [Security.Principal.SecurityIdentifier]::new($Identity).Value
        }
        # Task Scheduler may normalize a registered SID to its account name.
        return [Security.Principal.NTAccount]::new($Identity).Translate([Security.Principal.SecurityIdentifier]).Value
    } catch {
        throw 'A conflicting task definition was found; no task was changed.'
    }
}

function Assert-FoundationTask {
    param($Task, $Receipt)
    $paths = Get-FoundationPaths
    $principalSid = $null
    $triggerSid = $null
    if ($null -ne $Task) {
        $principalSid = Resolve-FoundationTaskSid ([string]$Task.Principal.UserId)
        if (@($Task.Triggers).Count -eq 1) {
            $triggerSid = Resolve-FoundationTaskSid ([string]$Task.Triggers[0].UserId)
        }
    }
    if ($null -eq $Task -or $Task.TaskName -ne $paths.Task -or $Task.TaskPath -ne '\' -or
        @($Task.Actions).Count -ne 1 -or $Task.Actions[0].Execute -ne $Receipt.action.executable -or
        $Task.Actions[0].Arguments -ne $Receipt.action.arguments -or
        $Task.Actions[0].WorkingDirectory -ne $Receipt.lab_root -or
        $principalSid -ne $paths.Sid -or [string]$Task.Principal.LogonType -ne 'Interactive' -or
        [string]$Task.Principal.RunLevel -ne 'Limited' -or @($Task.Triggers).Count -ne 1 -or
        $Task.Triggers[0].CimClass.CimClassName -ne 'MSFT_TaskLogonTrigger' -or $triggerSid -ne $paths.Sid -or
        $Task.Settings.RestartCount -ne 2 -or $Task.Settings.RestartInterval -ne 'PT1M' -or
        [string]$Task.Settings.MultipleInstances -ne 'IgnoreNew' -or $Task.Settings.ExecutionTimeLimit -ne 'PT0S') {
        throw 'A conflicting task definition was found; no task was changed.'
    }
}

function Get-FoundationArgv {
    param([string]$CommandLine)
    if (-not ('SZL.FoundationCommandLine' -as [type])) {
        Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
namespace SZL {
    public static class FoundationCommandLine {
        [DllImport("shell32.dll", SetLastError=true)] static extern IntPtr CommandLineToArgvW([MarshalAs(UnmanagedType.LPWStr)] string line, out int count);
        [DllImport("kernel32.dll")] static extern IntPtr LocalFree(IntPtr value);
        public static string[] Parse(string line) {
            int count; IntPtr ptr = CommandLineToArgvW(line, out count);
            if (ptr == IntPtr.Zero) throw new InvalidOperationException("Command line parsing failed");
            try { var result = new string[count]; for (int i=0;i<count;i++) result[i]=Marshal.PtrToStringUni(Marshal.ReadIntPtr(ptr,i*IntPtr.Size)); return result; }
            finally { LocalFree(ptr); }
        }
    }
}
'@
    }
    return [SZL.FoundationCommandLine]::Parse($CommandLine)
}

function Get-FoundationService {
    param([string]$Root, [string]$Python, [int]$ServicePort, [switch]$MayBeAbsent)
    $path = Join-Path $Root 'state\service.json'
    if (-not (Test-Path -LiteralPath $path)) {
        if ($MayBeAbsent) { return $null }
        throw 'No service process receipt exists.'
    }
    Assert-FoundationPath $path | Out-Null
    $receipt = Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
    if (($receipt.pid -isnot [int] -and $receipt.pid -isnot [long]) -or $receipt.pid -le 0 -or
        $receipt.server_script -ne (Join-Path $Root 'server.py') -or
        $receipt.executable -ne $Python -or $receipt.port -ne $ServicePort) {
        throw 'The service receipt belongs to a different launch; no process was changed.'
    }
    $process = Get-CimInstance Win32_Process -Filter "ProcessId=$([int]$receipt.pid)" -ErrorAction Stop
    if ($null -eq $process) {
        if ($MayBeAbsent) { return $null }
        throw 'The recorded service process is absent.'
    }
    $created = ([datetime]$receipt.process_created).ToUniversalTime()
    $arguments = @(Get-FoundationArgv $process.CommandLine)
    $expected = @($Python, '-X', 'utf8', '-B', (Join-Path $Root 'server.py'), '--port', [string]$ServicePort)
    if ($process.ExecutablePath -ne $Python -or $process.CreationDate.ToUniversalTime() -ne $created -or
        $arguments.Count -ne $expected.Count) { throw 'Service process ownership could not be verified.' }
    for ($index = 0; $index -lt $expected.Count; $index++) {
        if ($arguments[$index] -cne $expected[$index]) { throw 'Service process command differs from the owned launch.' }
    }
    $listeners = @(Get-NetTCPConnection -LocalPort $ServicePort -State Listen -ErrorAction SilentlyContinue)
    if ($listeners.Count -ne 1 -or $listeners[0].LocalAddress -ne '127.0.0.1' -or $listeners[0].OwningProcess -ne $receipt.pid) {
        throw 'The loopback port is not exclusively owned by the recorded service.'
    }
    $health = Invoke-RestMethod -Uri "http://127.0.0.1:$ServicePort/healthz" -TimeoutSec 5
    $status = Invoke-RestMethod -Uri "http://127.0.0.1:$ServicePort/api/status" -TimeoutSec 10
    if ($health.service -ne 'szl-foundation-confirmation' -or $health.process_id -ne $receipt.pid -or
        $status.ready -ne $true -or $status.state -ne 'READY' -or $status.checkpoints_verified -ne 3) {
        throw 'The owned service has not demonstrated release readiness.'
    }
    return @{ Receipt = $receipt; Process = $process; Status = $status }
}

function Assert-FoundationLaunchPort {
    param([int]$ServicePort, $OwnedService)
    if ($null -eq $OwnedService -and @(Get-NetTCPConnection -LocalPort $ServicePort -State Listen -ErrorAction SilentlyContinue).Count -gt 0) {
        throw 'The requested port belongs to a process without this lab receipt. The launcher was not invoked.'
    }
}

# The installer and uninstaller dot-source only these definitions.
if ($MyInvocation.InvocationName -eq '.') { return }

try {
    $admitted = Test-FoundationRelease $LabRoot $ArchivePath
    if ($ValidateOnly) {
        @{ status = 'VERIFIED'; admission = $admitted; task_registered = $false; service_started = $false } | ConvertTo-Json -Depth 8
        exit 0
    }
    $paths = Get-FoundationPaths
    $installation = Read-FoundationInstallation
    if ($installation.lab_root -ne $admitted.LabRoot -or $installation.archive_path -ne $admitted.ArchivePath -or $installation.port -ne $Port) {
        throw 'Launch arguments do not match the owned installation receipt.'
    }
    Assert-FoundationTask (Get-ScheduledTask -TaskName $paths.Task -TaskPath '\' -ErrorAction Stop) $installation
    # Verify a live predecessor before letting the unchanged launcher reuse its receipt.
    $predecessor = Get-FoundationService $admitted.LabRoot $admitted.Python $Port -MayBeAbsent
    Assert-FoundationLaunchPort $Port $predecessor
    $launchArguments = '-NoProfile -NonInteractive -File "' + (Join-Path $admitted.LabRoot 'start.ps1') + '" -Port ' + $Port
    $launcher = Start-Process -FilePath $paths.Shell -ArgumentList $launchArguments -WorkingDirectory $admitted.LabRoot -WindowStyle Hidden -Wait -PassThru
    if ($launcher.ExitCode -ne 0) { throw 'The unchanged workbench launcher failed. See retained state/service.stderr.log.' }
    $owned = Get-FoundationService $admitted.LabRoot $admitted.Python $Port
    $run = @{ schema = 'szl.foundation-confirmation.windows-supervisor/v1'; owner_sid = $paths.Sid;
        installation_id = $installation.installation_id; started_utc = [datetime]::UtcNow.ToString('o');
        status = 'WAITING_FOR_OWNED_SERVICE'; process = $owned.Receipt; source_admission = $admitted.Verification;
        checkpoints_verified = $owned.Status.checkpoints_verified; task_tracks_service = $true }
    Write-FoundationJson (Join-Path $paths.Operations 'supervisor-status.json') $run
    $tracked = Get-Process -Id $owned.Receipt.pid -ErrorAction Stop
    # Pin the process handle to the creation timestamp, even if the PID exits and is reused.
    if ($tracked.StartTime.ToUniversalTime() -ne ([datetime]$owned.Receipt.process_created).ToUniversalTime() -or
        $tracked.Path -ne $admitted.Python) { throw 'The service changed before supervision began.' }
    Wait-Process -InputObject $tracked -ErrorAction Stop
    $run.status = 'OWNED_SERVICE_EXITED'
    $run.completed_utc = [datetime]::UtcNow.ToString('o')
    Write-FoundationJson (Join-Path $paths.Operations 'supervisor-status.json') $run
    # Ending the service is a failure for the supervisor; Task Scheduler owns the bounded retries.
    exit 1
} catch {
    Write-Error $_ -ErrorAction Continue
    exit 1
}
