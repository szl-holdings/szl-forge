& {
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

# ============================ SETTINGS ============================
$MergeGreen      = $true     # merge ONLY fully green PRs (live re-check + head-SHA lock); BEHIND PRs get branch updates
$RepairPRs       = $true     # failing/conflicting PRs: flaky reruns, @dependabot rebase/recreate, formatter pushes,
                             # base-merge pushes; everything else gets a redacted repair brief for the coding agent
$CommentPRs      = $true     # one short classification comment per PR head (never log lines)
$OpenTrainingPR  = $true     # open ONE pull request to szl-forge with the generated fail-closed candidate kits
$TrainMode       = 'smoke'   # off | smoke (1 optimizer step per bound candidate, proves the path) | full (needs confirmed bindings)
$ConfirmBindings = $false    # $true = pass --confirm-binding for full runs (you vouch for the heuristically bound local corpora)
$ExecuteHF       = $false    # $true = open Hugging Face quarantine PRs (PRs only, never direct commits, never deletes)
$CloneAll        = $false    # $true = shallow-clone every active repo for a full-file audit (slower, more complete)
$TrainingPython  = ''        # optional: python.exe of a qualified venv (torch+CUDA, transformers, peft); auto-detected under ~\szl-*\.venv
$LlamaCpp        = ''        # optional: llama.cpp checkout for GGUF export + parity probe
$CorpusRoots     = ''        # optional: extra ';'-separated folders holding JSONL corpora (default scans ~\szl-*)
$OperatorRepo    = 'szl-holdings/szl-forge'
$OperatorRef     = 'main'   # pin a 40-hex commit of tools/estate_payload for an exact-source run; never a floating alias in evidence
$OperatorPath    = 'tools/estate_payload'
# ==================================================================

$principal = New-Object Security.Principal.WindowsPrincipal ([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Open Windows PowerShell with "Run as administrator", then paste this block again.'
}
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Update-SessionPath {
    $env:Path = [Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [Environment]::GetEnvironmentVariable('Path','User')
}
function Invoke-Native {
    param([string]$File, [string[]]$ArgList = @())
    $old = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $out = & $File @ArgList 2>&1 | ForEach-Object { "$_" }
        $code = $LASTEXITCODE
    } finally { $ErrorActionPreference = $old }
    [pscustomobject]@{ Code = $code; Out = (@($out) -join "`n") }
}
function Get-LastLine([string]$Text) {
    (($Text -split "`n") | Where-Object { $_.Trim() } | Select-Object -Last 1)
}
function Install-IfMissing([string]$Command, [string]$WingetId) {
    if (Get-Command $Command -ErrorAction SilentlyContinue) { return }
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) { throw "$Command is missing and winget is unavailable. Install $WingetId manually, then rerun." }
    Write-Host "Installing $WingetId ..." -ForegroundColor Yellow
    $r = Invoke-Native 'winget' @('install','--id',$WingetId,'-e','--silent','--accept-package-agreements','--accept-source-agreements')
    Update-SessionPath
    if (-not (Get-Command $Command -ErrorAction SilentlyContinue)) { throw "$Command still unavailable after install. Open a new Administrator PowerShell and rerun.`n$($r.Out)" }
}
function Find-Python {
    foreach ($name in @('py','python')) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if (-not $cmd) { continue }
        $pre = @(); if ($name -eq 'py') { $pre = @('-3') }
        $r = Invoke-Native $cmd.Source ($pre + @('-c','import sys; print(sys.version_info[0]*100+sys.version_info[1])'))
        $last = Get-LastLine $r.Out
        if ($r.Code -eq 0 -and $last -and $last.Trim() -match '^\d+$' -and [int]$last.Trim() -ge 311) {
            return [pscustomobject]@{ Exe = $cmd.Source; Pre = $pre }
        }
    }
    $null
}
function Get-RepoFile([string]$RepoPath, [string]$Dest) {
    # Exact-source fetch through the authenticated gh CLI; base64 keeps bytes intact regardless of console code page.
    $r = Invoke-Native 'gh' @('api', ("repos/{0}/contents/{1}?ref={2}" -f $OperatorRepo, $RepoPath, $OperatorRef), '--jq', '.content')
    if ($r.Code -ne 0 -or -not $r.Out.Trim()) { throw "Could not fetch $RepoPath at $OperatorRef from $OperatorRepo.`n$($r.Out)" }
    $bytes = [Convert]::FromBase64String(($r.Out -replace '\s',''))
    New-Item -ItemType Directory -Force -Path (Split-Path $Dest -Parent) | Out-Null
    [System.IO.File]::WriteAllBytes($Dest, $bytes)
    if ((Get-Item $Dest).Length -lt 100) { throw "Fetched file is suspiciously small: $Dest" }
}

Write-Host "`n=== PRECHECK ===" -ForegroundColor Cyan
Install-IfMissing 'git' 'Git.Git'
Install-IfMissing 'gh'  'GitHub.cli'
$Py = Find-Python
if (-not $Py) {
    Write-Host 'Installing Python 3.12 ...' -ForegroundColor Yellow
    $null = Invoke-Native 'winget' @('install','--id','Python.Python.3.12','-e','--silent','--accept-package-agreements','--accept-source-agreements')
    Update-SessionPath
    $Py = Find-Python
    if (-not $Py) { throw 'Python 3.11+ still unavailable. Open a new Administrator PowerShell and rerun.' }
}
Write-Host "OK: Python $($Py.Exe) $($Py.Pre -join ' ')"

Remove-Item Env:GH_TOKEN -ErrorAction SilentlyContinue
$st = Invoke-Native 'gh' @('auth','status','--hostname','github.com')
if ($st.Code -ne 0) {
    & gh auth login --hostname github.com --git-protocol https --web --scopes 'repo,workflow,read:org'
    if ($LASTEXITCODE -ne 0) { throw 'GitHub login did not complete.' }
    $st = Invoke-Native 'gh' @('auth','status','--hostname','github.com')
}
if (($MergeGreen -or $RepairPRs -or $OpenTrainingPR) -and $st.Out -notmatch 'workflow') {
    Write-Host 'Adding workflow scope (needed to merge or push changes that touch .github/workflows) ...' -ForegroundColor Yellow
    & gh auth refresh --hostname github.com --scopes 'repo,workflow,read:org'
}
$null = Invoke-Native 'gh' @('auth','setup-git')
Write-Host "OK: GitHub $(Get-LastLine (Invoke-Native 'gh' @('api','user','--jq','.login')).Out)"

if (-not $env:HF_TOKEN) {
    $sec = Read-Host 'Hugging Face token (Enter to skip; write access only needed when ExecuteHF is $true)' -AsSecureString
    $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($sec)
    try { $plain = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr) } finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr) }
    if ($plain) { $env:HF_TOKEN = $plain.Trim() }
}
if ($ExecuteHF -and -not $env:HF_TOKEN) { Write-Warning 'ExecuteHF needs HF_TOKEN; quarantine stays dry-run.'; $ExecuteHF = $false }
Write-Host ("OK: Hugging Face token " + $(if ($env:HF_TOKEN) { 'present' } else { 'absent (public audit only)' }))

$Base    = Join-Path $env:USERPROFILE 'SZL-PAYLOAD-V4'
$Venv    = Join-Path $Base '.venv'
$RefTag  = ($OperatorRef -replace '[^A-Za-z0-9._-]','_'); if ($RefTag.Length -gt 12) { $RefTag = $RefTag.Substring(0, 12) }
$OpDir   = Join-Path $Base ("operator-" + $RefTag)
$PyFile  = Join-Path $OpDir 'szl_payload_v4.py'
New-Item -ItemType Directory -Force -Path $Base, $OpDir | Out-Null

Write-Host "`n=== FETCH OPERATOR $OperatorRepo@$OperatorRef ===" -ForegroundColor Cyan
$Files = @('szl_payload_v4.py') + @('candidate_lib.py','qualify_runtime.py','curriculum.py','train_candidate.py','evaluate_candidate.py',
          'export_gguf.py','render_card.py','test_candidate_contract.py','README.md','MODEL_CARD.tmpl.md' | ForEach-Object { "kit/$_" })
foreach ($f in $Files) {
    Get-RepoFile "$OperatorPath/$f" (Join-Path $OpDir ($f -replace '/','\'))
}
Write-Host "OK: $($Files.Count) files -> $OpDir"

$VenvPy = Join-Path $Venv 'Scripts\python.exe'
if (-not (Test-Path $VenvPy)) {
    $r = Invoke-Native $Py.Exe ($Py.Pre + @('-m','venv',$Venv))
    if ($r.Code -ne 0) { throw "Virtual environment creation failed:`n$($r.Out)" }
}
$r = Invoke-Native $VenvPy @('-m','pip','install','--upgrade','--disable-pip-version-check','pip','requests>=2.31','huggingface_hub>=0.24','ruff','black')
if ($r.Code -ne 0) { throw "Dependency install failed:`n$($r.Out)" }
$r = Invoke-Native $VenvPy (@('-m','py_compile',$PyFile) + @(Get-ChildItem (Join-Path $OpDir 'kit') -Filter '*.py' | ForEach-Object { $_.FullName }))
if ($r.Code -ne 0) { throw "Operator failed to compile:`n$($r.Out)" }
Write-Host 'OK: Dependencies ready, operator and kit compile'

$env:SZL_ROOT             = Join-Path $Base 'out'
$env:SZL_MERGE            = $(if ($MergeGreen)      { '1' } else { '0' })
$env:SZL_REPAIR           = $(if ($RepairPRs)       { '1' } else { '0' })
$env:SZL_COMMENT_PRS      = $(if ($CommentPRs)      { '1' } else { '0' })
$env:SZL_OPEN_TRAINING_PR = $(if ($OpenTrainingPR)  { '1' } else { '0' })
$env:SZL_TRAIN_MODE       = $TrainMode
$env:SZL_CONFIRM_BINDINGS = $(if ($ConfirmBindings) { '1' } else { '0' })
$env:SZL_EXECUTE          = $(if ($ExecuteHF)       { '1' } else { '0' })
$env:SZL_CLONE            = $(if ($CloneAll)        { '1' } else { '0' })
if ($TrainingPython) { $env:SZL_TRAIN_PY = $TrainingPython }
if ($LlamaCpp)       { $env:SZL_LLAMA_CPP = $LlamaCpp }
if ($CorpusRoots)    { $env:SZL_CORPUS_ROOTS = $CorpusRoots }
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8       = '1'

Write-Host "`n=== RUNNING PAYLOAD v4 ===" -ForegroundColor Cyan
Write-Host "Merge green: $MergeGreen | Repair PRs: $RepairPRs | Training PR: $OpenTrainingPR | Train: $TrainMode | HF quarantine PRs: $ExecuteHF" -ForegroundColor DarkGray
& $VenvPy -u $PyFile
$Exit = $LASTEXITCODE

$LatestFile = Join-Path $env:SZL_ROOT 'LATEST.txt'
$RunDir = $null
if (Test-Path $LatestFile) { $RunDir = (Get-Content $LatestFile -Raw).Trim() }

if ($Exit -eq 2) { throw "Discovery failed, so nothing else ran. See the trace files in $RunDir" }
if ($Exit -eq 1) { Write-Warning "One or more lanes were BLOCKED; the others finished. See lanes.json and *.trace in $RunDir" }

Write-Host "`n=== COMPLETE ===" -ForegroundColor Green
Write-Host "Run folder: $RunDir"
foreach ($f in @('SCORECARD.md','CODEX-v4.md','CODEX-PR-REPAIRS.md','OWNER-GPU-RUNBOOK.md')) {
    $p = Join-Path $RunDir $f
    if ($RunDir -and (Test-Path $p)) { Write-Host "Opening $p"; Start-Process notepad.exe $p }
}
}
