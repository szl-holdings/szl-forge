#Requires -Version 5.1
<#
.SYNOPSIS
  GEH v8 owner-metal runbook - Lean 4 governed verification harness + Chaski r4 evidence reconciliation.

.DESCRIPTION
  Restartable, evidence-bound. Order: PREFLIGHT -> MATERIALIZE -> PYTHON -> LEAN BUILD -> GEH RUNS -> CHASKI RECONCILE -> [GATE] -> RECEIPT.
  Every stage prints observable evidence and stops nonzero on failure. Nothing is committed, pushed, uploaded, published,
  merged, trained, or promoted by this script. The only writes are under:
     <Root>\tools\geh_v8\            (harness files + Lean build cache)
     <Root>\chaski_r4\evidence\geh\  (GEH receipts, chain, Article-12 log, keys)
     <Root>\chaski_r4\evidence\      (runbook receipt; optional canonical rerun receipt with -RunCanonicalGate)

.PARAMETER Root      szl-forge checkout (default C:\Users\steph\szl-forge)
.PARAMETER Bundle    path to geh_v8_bundle.zip (downloaded from the Computer thread). If omitted, files must already be in <Root>\tools\geh_v8
.PARAMETER SkipLeanBuild   reuse an existing Lean build (after a successful first run)
.PARAMETER RunCanonicalGate  ALSO run chaski_r4\bakeoff_canonical_four_way_r4.py --run into a NEW receipt (never overwrites the committed one)

.EXAMPLE
  powershell -NoProfile -ExecutionPolicy Bypass -File "$env:USERPROFILE\Downloads\geh_v8_runbook.ps1" -Bundle "$env:USERPROFILE\Downloads\geh_v8_bundle.zip"
#>
[CmdletBinding()]
param(
    [string]$Root = "C:\Users\steph\szl-forge",
    [string]$Bundle = "",
    [switch]$SkipLeanBuild,
    [switch]$RunCanonicalGate
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$ProgressPreference = "SilentlyContinue"

function Banner([string]$Text) {
    Write-Host ""
    Write-Host ("=" * 88) -ForegroundColor DarkCyan
    Write-Host ("  " + $Text) -ForegroundColor Cyan
    Write-Host ("=" * 88) -ForegroundColor DarkCyan
}
function Sha256([string]$Path) { return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() }
function Require-File([string]$Path) { if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "Required file is missing: $Path" } }
function Require-Dir([string]$Path)  { if (-not (Test-Path -LiteralPath $Path -PathType Container)) { throw "Required directory is missing: $Path" } }
function Run-Step([string]$Label, [string]$Exe, [string[]]$ArgList, [string]$Cwd) {
    Write-Host ("> " + $Label) -ForegroundColor Yellow
    Write-Host ("  " + $Exe + " " + ($ArgList -join " ")) -ForegroundColor DarkGray
    Push-Location -LiteralPath $Cwd
    try { & $Exe @ArgList; $code = $LASTEXITCODE } finally { Pop-Location }
    if ($code -ne 0) { throw "$Label failed with exit code $code. Runbook INCOMPLETE; nothing was promoted." }
}

# ------------------------------------------------------------------ PREFLIGHT
Banner "PREFLIGHT"
$Root = (Resolve-Path -LiteralPath $Root).Path
Set-Location -LiteralPath $Root
$Stamp    = Get-Date -Format "yyyyMMdd_HHmmss"
$Tools    = Join-Path $Root "tools\geh_v8"
$LeanProj = Join-Path $Tools "geh-lean"
$Evidence = Join-Path $Root "chaski_r4\evidence"
$GehEvid  = Join-Path $Evidence "geh"
New-Item -ItemType Directory -Force -Path $Tools, $Evidence | Out-Null

foreach ($tool in @("lake", "elan", "git")) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) { throw "$tool not on PATH. Install elan (https://github.com/leanprover/elan); lutar-lean already needs it." }
}
$VenvPy = Join-Path $Root ".venv-ra-v2\Scripts\python.exe"
$HaveVenv = Test-Path -LiteralPath $VenvPy -PathType Leaf
if ($HaveVenv) { $Py = $VenvPy } else {
    $PyCmd = Get-Command py -ErrorAction SilentlyContinue
    if ($null -eq $PyCmd) { throw "No Python: expected $VenvPy or the py launcher." }
    $Py = $PyCmd.Source
    Write-Host "  .venv-ra-v2 missing; falling back to the py launcher (GEH needs no CUDA; the chaski gate does)" -ForegroundColor Yellow
}
Write-Host "Root:        $Root"
Write-Host "Python:      $Py"
Write-Host "Lean tools:  $((Get-Command lake).Source)"
Write-Host "Stamp:       $Stamp"

# ------------------------------------------------------------------ MATERIALIZE
Banner "MATERIALIZE GEH v8 FILES"
if ($Bundle -ne "") {
    Require-File $Bundle
    Write-Host "Bundle SHA-256: $(Sha256 $Bundle)"
    Expand-Archive -LiteralPath $Bundle -DestinationPath $Tools -Force
}
$Manifest = @{
    "szl_geh_v8.py" = "25008681bed2563c2caa683ae342933fd1ae72e7b2c89ef0c16bb021f7a736b7"
    "szl_geh_verify.py" = "6647ef7fb207cbf01e4c69275735fc632009a0558625208899bcdef2e2787040"
    "chaski_margin_probe.py" = "6c9a22b089150897d54d1a9a566eeefdc26179e99bc9f196853b3fb898b0d34e"
    "tests\test_geh_v8.py" = "c5a601f482da7e605ee02171c633ab931b082a90a24ec5f5f2f270b5fec47b6b"
    "README.md" = "c45f15347da292fcd6970fb996b4360ee83d101eadea9f1a77bf8f40718c4602"
    "geh-lean\lakefile.toml" = "60b374e3ac39ecf8e9f8bf79c7ffee008c9fcbb0bfafab5530cd7d8bec373cb8"
    "geh-lean\lean-toolchain" = "692487121da24843c4cc39e8225d0e016615914dc602328195691014cc98c526"
    "geh-lean\GEH.lean" = "f74c2fc9e4eac1557049738948f0138ceb49b664ddd5c6b5a8cb08941691f64f"
    "geh-lean\GEH\Compliance\GehGuard.lean" = "6ed9fb9146c62fded4714f7f0359e14d4509a8b51b1d370a88dd8065c2aa2709"
    "geh-lean\GEH\Compliance\SelfTest.lean" = "fa10b82adcfb15f3f74dfadf0864b79487d1a7ecf79e3edae51c7455c287990c"
    "geh-lean\GEH\Evidence\README.md" = "a0436a50dfb08b7daee5e1a1895c2b38cdec992e6c758c02eca5eceb2c6500e9"
    "geh-lean\GEH\RedTests\Red_Axiom.lean" = "de661cc73bae1ad3a3a07ab236c520d3c3ba450fa4777b10847ada936a2efa47"
    "geh-lean\GEH\RedTests\Red_NativeDecide.lean" = "fdae35f1092bbf5fe3b50aebe3cb1a27429527d8eefdd5fabf92fe483f57be46"
    "geh-lean\GEH\RedTests\Red_Sorry.lean" = "8005c605eb2ecbe5a69acf5962cc15c236c53ab068d6a48b4e855d5f0d5e62ee"
}
$Drift = @()
foreach ($rel in $Manifest.Keys) {
    $p = Join-Path $Tools $rel
    if (-not (Test-Path -LiteralPath $p -PathType Leaf)) { $Drift += "MISSING $rel"; continue }
    $h = Sha256 $p
    if ($h -ne $Manifest[$rel]) { $Drift += "DRIFT   $rel  expected=$($Manifest[$rel].Substring(0,16)) actual=$($h.Substring(0,16))" }
    else { Write-Host ("  ok  {0}  {1}" -f $h.Substring(0,16), $rel) -ForegroundColor DarkGreen }
}
if ($Drift.Count -gt 0) { $Drift | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }; throw "Harness files do not match the shipped manifest. Do not run unverified harness code." }

# ------------------------------------------------------------------ PYTHON
Banner "PYTHON DEPENDENCIES (venv-local only)"
$EAP = $ErrorActionPreference; $ErrorActionPreference = "Continue"
try { & $Py -c "import cryptography, pytest; print('cryptography', cryptography.__version__, '| pytest', pytest.__version__)" 2>$null; $probe = $LASTEXITCODE } finally { $ErrorActionPreference = $EAP }
if ($probe -ne 0) {
    Write-Host "  installing cryptography + pytest into the selected interpreter" -ForegroundColor Yellow
    & $Py -m pip install --quiet cryptography pytest
    if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
    & $Py -c "import cryptography, pytest; print('cryptography', cryptography.__version__, '| pytest', pytest.__version__)"
}

# ------------------------------------------------------------------ LEAN BUILD
Banner "LEAN BUILD (leanprover/lean4:v4.18.0, the lutar-lean pin)"
Require-Dir $LeanProj
$Toolchain = (Get-Content -LiteralPath (Join-Path $LeanProj "lean-toolchain") -Raw).Trim()
if ($Toolchain -ne "leanprover/lean4:v4.18.0") { throw "lean-toolchain drifted: $Toolchain" }
if (-not $SkipLeanBuild) {
    Run-Step "lake update (clones leanprover-community/repl @ v4.18.0)" "lake" @("update") $LeanProj
    Run-Step "lake build (GEH.Compliance.*: guard + kernel self-tests)" "lake" @("build") $LeanProj
    Run-Step "lake build repl" "lake" @("build", "repl") $LeanProj
    # RED TESTS: each must FAIL the build (foreign axiom / native_decide / sorry)
    foreach ($red in @("Red_Axiom.lean", "Red_NativeDecide.lean", "Red_Sorry.lean")) {
        Push-Location -LiteralPath $LeanProj
        $EAP = $ErrorActionPreference; $ErrorActionPreference = "Continue"
        try { & lake env lean (Join-Path "GEH\RedTests" $red) 2>&1 | Out-Null; $code = $LASTEXITCODE } finally { Pop-Location; $ErrorActionPreference = $EAP }
        if ($code -eq 0) { throw "RED TEST $red COMPILED: the axiom guard is not enforcing. Stop." }
        Write-Host "  red test $red correctly FAILED the build (exit $code)" -ForegroundColor DarkGreen
    }
} else { Write-Host "  -SkipLeanBuild: reusing existing build" -ForegroundColor Yellow }

# ------------------------------------------------------------------ GEH RUNS
Banner "GEH v8 RUNS (fail-closed; evidence -> $GehEvid)"
$Harness = Join-Path $Tools "szl_geh_v8.py"
$Verifier = Join-Path $Tools "szl_geh_verify.py"
$Common = @("--root", $Root, "--lean-project", $LeanProj)
$RunId = ("owner" + $Stamp.Substring(0,8) + $Stamp.Substring(9,4)).ToLower()
Run-Step "self-check (structural + kernel-grounded)" $Py (@($Harness) + $Common + @("--selfcheck")) $Tools
Run-Step "emit tool-use spec" $Py (@($Harness) + $Common + @("--emit-toolspec")) $Tools
Run-Step "governed proof -> DSSE receipt" $Py (@($Harness) + $Common + @("--transport", "repl", "--run-id", ($RunId + "p"), "--prove", "forall (p q : Prop), p -> (p -> q) -> q", "--script", "intro p q hp hpq, exact hpq hp")) $Tools
Run-Step "randomized crypto benchmark K=10 on the kernel" $Py (@($Harness) + $Common + @("--transport", "repl", "--run-id", ($RunId + "b"), "--salt", $Stamp.Substring(0,8), "--bench-crypto", "10")) $Tools
Run-Step "pytest" $Py @("-m", "pytest", "-q", (Join-Path $Tools "tests")) $Tools
Run-Step "verify chain (harness)" $Py (@($Harness) + $Common + @("--verify-chain")) $Tools
Run-Step "verify chain (zero-dependency offline verifier)" $Py @($Verifier, "--evidence", $GehEvid) $Tools

# ------------------------------------------------------------------ CHASKI RECONCILE (read-only)
Banner "CHASKI R4 EVIDENCE RECONCILIATION (read-only)"
# Thread-anchored constants (from the committed szl-forge receipts, commit bcdd1d85, 2026-09-17):
$RxA_r4   = "b116832a34628407655b50b7729babfcf91f797e2e3fe768810373a4c74eb298"  # canonical receipt A (0/5, 3/6) dir-hash of r4 adapter
$RxB_r4   = "e1abc37a5c41a82b0fc2cd98ccd6edbb2a08fceb8ca9e883bcfb76b861c221cf"  # four-way receipt B (5/5, 6/6) dir-hash of r4 adapter
$TrainW   = "f1a2cdc313795775966280bc8648367005700327dd28010da8d2f88d2a5e2a02"  # training_receipt.json adapter_weights_sha256 (retrained r4, 2026-09-17T02:36Z)
$DsPlan   = "8965f3910100d818810b4bf95832c5f1f49c95900f4bd943c5fcafa9fe645a40"  # training_plan.json dataset_sha256 (original curriculum)
$DsRcpt   = "0fea0d85f2ca4cd55d7ce51b8399a409d7046d2cd77ffc131945ee20c706535a"  # training_receipt.json dataset_sha256 (fixed curriculum)
$RunnerOk = "f4ca282a29e2fa2f0246ebcb6e3298b7b3f023d686ba883d948459bcaf56dadb"  # pristine canonical runner
$HubR2W   = "6f12981ea5df5e22d3493eefb20d75db75c1c88961ba6621a35af09edd0cdde6"  # SZLHOLDINGS/chaski-r2 adapter_model.safetensors LFS sha256 (Hub, 2026-09-30)
$R4Dir    = Join-Path $Root "chaski_r4\chaski-r4-adapter"
$R2Dir    = Join-Path $Root "chaski_r2\chaski-r2-adapter"
$Runner   = Join-Path $Root "chaski_r4\bakeoff_canonical_four_way_r4.py"
$Train    = Join-Path $Root "chaski_r4\train.jsonl"
$Recon = [ordered]@{ runner_pristine = $false; r4_on_disk_is = "NOT FOUND"; r4_weights_match_training_receipt = $false }
if (Test-Path -LiteralPath $Runner) { $Recon.runner_sha256 = Sha256 $Runner; $Recon.runner_pristine = ($Recon.runner_sha256 -eq $RunnerOk) }
if (Test-Path -LiteralPath $Train)  { $Recon.train_sha256 = Sha256 $Train; $Recon.train_matches = @{plan = ($Recon.train_sha256 -eq $DsPlan); receipt = ($Recon.train_sha256 -eq $DsRcpt)} }
foreach ($pair in @(@("r4", $R4Dir), @("r2", $R2Dir))) {
    $name = $pair[0]; $dir = $pair[1]
    $w = Join-Path $dir "adapter_model.safetensors"
    if (-not (Test-Path -LiteralPath $w)) { $Recon["${name}_adapter"] = "MISSING $w"; continue }
    $fileHash = Sha256 $w
    # the runner's dir-hash: sha256(filename + NUL + bytes) over sorted *.safetensors (same formula as the canonical runner)
    $dirHash = (& $Py $Harness --dirhash $dir).Trim()
    $Recon["${name}_weights_sha256"] = $fileHash
    $Recon["${name}_runner_dirhash"] = $dirHash
}
if ($Recon.Contains("r4_runner_dirhash")) {
    $which = "UNKNOWN adapter (matches neither committed receipt)"
    if ($Recon.r4_runner_dirhash -eq $RxA_r4) { $which = "ORIGINAL r4 (receipt A: 0/5 drafts, 3/6 refusals, 2026-09-16)" }
    if ($Recon.r4_runner_dirhash -eq $RxB_r4) { $which = "RETRAINED r4 (receipt B: 5/5 drafts, 6/6 refusals, 2026-09-17T02:41Z)" }
    $Recon.r4_on_disk_is = $which
    $Recon.r4_weights_match_training_receipt = ($Recon.r4_weights_sha256 -eq $TrainW)
}
if ($Recon.Contains("r2_weights_sha256")) { $Recon.r2_local_matches_hub_published = ($Recon.r2_weights_sha256 -eq $HubR2W) }
$Recon | ConvertTo-Json -Depth 4 | Write-Host

Write-Host ""
Write-Host "STATE BLOCK (model-qualification-gates)" -ForegroundColor Cyan
@"
Artifact: chaski-r4 local adapter ($($Recon.r4_on_disk_is))
Canonical source: szl-holdings/szl-forge chaski_r4/ @ bcdd1d85 (runner f4ca282a...dadb)
Training: TRAINED_CHALLENGER (training_receipt.json 2026-09-17T02:36Z, weights f1a2cdc3..., dataset 0fea0d85...)
Evaluation: MEASURED but SPLIT: receipt A (original adapter, torch 2.11.0+cu128): 0/5, 3/6 | receipt B (retrained adapter, torch 2.10.0+cu130): 5/5, 6/6
Publication: UNPUBLISHED (chaski-r4 absent from the Hub; correct)
Promotion: NOT_PROMOTABLE
Blocking gate: canonical receipt path still holds receipt A; no canonical rerun of the RETRAINED adapter exists in a pinned env with r2 control in the same run
Next bounded action: -RunCanonicalGate (writes a NEW receipt; promote to canonical only by explicit copy after review)
Protected state: committed receipts A and B, adapters, gate files, pristine runner; none modified by this runbook
"@ | Write-Host

# ------------------------------------------------------------------ OPTIONAL GATE
$GateReceipt = $null
if ($RunCanonicalGate) {
    Banner "CANONICAL FOUR-WAY GATE (explicit; NEW receipt path)"
    Require-File $Runner
    if (-not $HaveVenv) { throw "The canonical gate requires the CUDA runtime at $VenvPy (torch 2.11.0+cu128 family). Refusing to evaluate on a different interpreter." }
    if ($Recon.runner_pristine -ne $true) { throw "Runner is not the pristine f4ca282a build; refusing to run a non-canonical evaluator." }
    $GateReceipt = Join-Path $Evidence ("canonical_rerun_" + $Stamp + ".receipt.json")
    $A5050 = Join-Path $Root "chaski\chaski-5050-adapter"
    if (-not (Test-Path -LiteralPath $A5050)) { $A5050 = Join-Path $Root "chaski-5050\chaski-5050-adapter" }
    $GateArgs = @($Runner, "--run", "--chaski-r2-adapter", $R2Dir, "--chaski-r4-adapter", $R4Dir, "--receipt", $GateReceipt)
    if (Test-Path -LiteralPath $A5050) { $GateArgs += @("--chaski-5050-adapter", $A5050) }
    & $Py -c "import torch,transformers,peft;print('torch',torch.__version__,'transformers',transformers.__version__,'peft',peft.__version__,'cuda',torch.cuda.is_available())"
    Run-Step "canonical runner --run" $Py $GateArgs (Join-Path $Root "chaski_r4")
    $R = Get-Content -LiteralPath $GateReceipt -Raw | ConvertFrom-Json
    foreach ($c in $R.candidates) { Write-Host ("  {0,-18} drafts {1}/{2}  refusals {3}/{4}  adapter_sha {5}" -f $c.id, $c.json_draft_valid, $c.json_draft_total, $c.adversarial_refused, $c.adversarial_total, $c.adapter_sha256) }
    Write-Host "  torch in receipt: $($R.gpu.torch)   receipt SHA-256: $(Sha256 $GateReceipt)"
    Write-Host "  The committed canonical receipt was NOT overwritten. Promote by explicit copy only after review." -ForegroundColor Yellow
}

# ------------------------------------------------------------------ RECEIPT
Banner "RUNBOOK RECEIPT"
$ChainHead = (Get-Content -LiteralPath (Join-Path $GehEvid "geh_chain.jsonl") | Select-Object -Last 1 | ConvertFrom-Json).sha256
$Out = [ordered]@{
    kind = "szl-geh-v8-runbook-receipt"; generated = (Get-Date).ToString("o"); root = $Root; python = $Py
    lean_toolchain = $Toolchain; harness_sha256 = (Sha256 $Harness); verifier_sha256 = (Sha256 $Verifier)
    geh_chain_head = $ChainHead; geh_evidence_dir = $GehEvid; chaski_reconciliation = $Recon
    canonical_rerun_receipt = $GateReceipt
    safety = "No training, adapter modification, merge, upload, publication, deployment, Git commit, Git push, or promotion occurred."
}
$ReceiptPath = Join-Path $Evidence ("geh_v8_runbook_" + $Stamp + ".json")
$Out | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $ReceiptPath -Encoding UTF8
Write-Host "RUNBOOK_RECEIPT=$ReceiptPath"
Write-Host "RUNBOOK_RECEIPT_SHA256=$(Sha256 $ReceiptPath)"
Write-Host "GEH_CHAIN_HEAD=$ChainHead"
Write-Host "COMPLETE: all gates observable above. Nothing was promoted." -ForegroundColor Green
