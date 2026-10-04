# Native Windows observations only. No training, files, networking, or process control.
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][ValidatePattern('^[0-9a-f]{32}$')][string]$RunId,
    [Parameter(Mandatory = $true)][ValidatePattern('^[0-9a-f]{40}$')][string]$SourceRevision,
    [Parameter(Mandatory = $true)][ValidatePattern('^[0-9a-f]{64}$')][string]$SamplerSourceSha256,
    [switch]$Once
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)

# Available process commit (MEMORYSTATUSEX.ullAvailPageFile) is deliberately
# unused. System-wide available commit is (CommitLimit-CommitTotal)*PageSize.
# https://learn.microsoft.com/windows/win32/api/sysinfoapi/ns-sysinfoapi-memorystatusex
Add-Type -TypeDefinition @'
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
using System.Threading;
public static class SzlHostMemoryNative {
    public static volatile bool StopRequested;
    public static void StartControlReader() {
        Thread reader = new Thread(() => {
            try { Console.In.ReadLine(); }
            finally { StopRequested = true; }
        });
        reader.IsBackground = true;
        reader.Start();
    }
    [StructLayout(LayoutKind.Sequential)]
    public struct MemoryStatus {
        public UInt32 Length, MemoryLoad;
        public UInt64 TotalPhys, AvailPhys, TotalPageFile, AvailPageFile;
        public UInt64 TotalVirtual, AvailVirtual, AvailExtendedVirtual;
    }
    [StructLayout(LayoutKind.Sequential)]
    public struct PerformanceInfo {
        public UInt32 Size;
        public UIntPtr CommitTotal, CommitLimit, CommitPeak, PhysicalTotal;
        public UIntPtr PhysicalAvailable, SystemCache, KernelTotal, KernelPaged;
        public UIntPtr KernelNonpaged, PageSize;
        public UInt32 HandleCount, ProcessCount, ThreadCount;
    }
    [DllImport("kernel32.dll", SetLastError=true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    static extern bool GlobalMemoryStatusEx(ref MemoryStatus status);
    [DllImport("psapi.dll", SetLastError=true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    static extern bool GetPerformanceInfo(out PerformanceInfo info, UInt32 size);
    public static UInt64[] Read() {
        MemoryStatus status = new MemoryStatus();
        status.Length = (UInt32)Marshal.SizeOf(typeof(MemoryStatus));
        PerformanceInfo performance;
        if (!GlobalMemoryStatusEx(ref status) ||
            !GetPerformanceInfo(out performance, (UInt32)Marshal.SizeOf(typeof(PerformanceInfo))))
            throw new Win32Exception(Marshal.GetLastWin32Error());
        UInt64 pages = performance.PageSize.ToUInt64();
        UInt64 total = performance.CommitTotal.ToUInt64();
        UInt64 limit = performance.CommitLimit.ToUInt64();
        if (pages == 0 || total > limit || status.AvailPhys > status.TotalPhys)
            throw new InvalidOperationException("invalid native memory counters");
        checked {
            return new UInt64[] { status.AvailPhys, status.TotalPhys,
                total*pages, limit*pages, (limit-total)*pages };
        }
    }
}
'@

$clock = [System.Diagnostics.Stopwatch]::StartNew()
[long]$sequence = 0
[long]$nextSampleMs = 0
if (-not $Once) { [SzlHostMemoryNative]::StartControlReader() }
try {
    while ($clock.Elapsed.TotalSeconds -le 10860) {
        if ([SzlHostMemoryNative]::StopRequested) { exit 0 }
        if ($sequence -ge 5500) { throw 'sample bound exceeded' }
        $values = [SzlHostMemoryNative]::Read()
        $row = [ordered]@{
            schema = 'szl.windows-host-memory-sample/v1'
            runId = $RunId
            sourceRevision = $SourceRevision
            samplerSourceSha256 = $SamplerSourceSha256
            observedAt = [DateTimeOffset]::UtcNow.ToString('o')
            sequence = $sequence
            nativeMonotonicTicks = [System.Diagnostics.Stopwatch]::GetTimestamp()
            nativeMonotonicFrequency = [System.Diagnostics.Stopwatch]::Frequency
            availablePhysicalBytes = $values[0]
            totalPhysicalBytes = $values[1]
            committedBytes = $values[2]
            commitLimitBytes = $values[3]
            availableCommitBytes = $values[4]
        }
        [Console]::Out.WriteLine(($row | ConvertTo-Json -Compress))
        [Console]::Out.Flush()
        if ($Once) { exit 0 }
        $sequence++
        $nextSampleMs += 2000
        [long]$delayMs = $nextSampleMs - $clock.ElapsedMilliseconds
        if ($delayMs -lt -8000) { throw 'native cadence lost' }
        if ($delayMs -gt 0) { [System.Threading.Thread]::Sleep([int]$delayMs) }
    }
    throw 'native observation deadline expired'
}
catch {
    # Never echo the exception or arbitrary caller/provider data.
    [Console]::Error.WriteLine('HOST_MEMORY_NATIVE_OBSERVATION_FAILED')
    exit 74
}
