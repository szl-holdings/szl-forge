# SZL Forge bootstrap — ONE command runs the whole SZL-1 pipeline on this laptop.
#
#   iwr https://raw.githubusercontent.com/szl-holdings/szl-forge/main/forge.ps1 -OutFile "$env:TEMP\forge.ps1"; powershell -ExecutionPolicy Bypass -File "$env:TEMP\forge.ps1"
#
# Honest: prints exactly what it is doing at each step; stops on real failure
# with the real error on screen. It never claims success it didn't see.

$ErrorActionPreference = "Continue"
function Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Fail($msg) {
  Write-Host "`n[szl-forge] FAILED: $msg" -ForegroundColor Red
  Write-Host "Screenshot the last ~20 lines above." -ForegroundColor Yellow
  exit 1
}

Step "0/6 Python check"
$pv = ""
try { $pv = (& python --version 2>&1) | Out-String } catch { }
$pv = $pv.Trim()
if (-not $pv -or $pv -notmatch "^Python ") {
  Fail "python not found. Install it with: winget install -e --id Python.Python.3.12  (then CLOSE and REOPEN PowerShell and re-run this one command)"
}
Write-Host $pv
if ($pv -notmatch "Python 3\.1[1-3]\.") { Fail "need Python 3.11-3.13, got: $pv" }

Step "1/6 Forge folder"
$dir = Join-Path $env:USERPROFILE "szl-forge"
New-Item -ItemType Directory -Force -Path $dir | Out-Null
Set-Location $dir
Write-Host "working in $dir"

Step "2/6 Kit files"
$base = "https://raw.githubusercontent.com/szl-holdings/szl-forge/main"
foreach ($f in @("train_szl.py", "szl_dataset.jsonl", "Modelfile")) {
  curl.exe -sL -o $f "$base/$f"
  if (-not (Test-Path $f) -or (Get-Item $f).Length -lt 100) { Fail "could not download $f" }
  Write-Host "  $f OK"
}

Step "3/6 Unsloth"
& python -c "import unsloth" 2>$null
if ($LASTEXITCODE -ne 0) {
  Write-Host "installing unsloth (one time, ~5-10 min)..."
  & python -m pip install unsloth
  if ($LASTEXITCODE -ne 0) { Fail "pip install unsloth failed" }
} else {
  Write-Host "already installed"
}

Step "4/6 CUDA torch (RTX 5050 = Blackwell needs the cu128 build)"
& python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>$null
if ($LASTEXITCODE -ne 0) {
  Write-Host "torch has NO CUDA - reinstalling torch+torchvision+torchaudio together as the CUDA 12.8 build (one time, ~3 GB download)..."
  & python -m pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
  if ($LASTEXITCODE -ne 0) { Fail "CUDA torch install failed" }
  & python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"
  if ($LASTEXITCODE -ne 0) { Fail "torch still cannot see the GPU after the cu128 reinstall" }
  Write-Host "note: pip may warn 'unsloth requires torch<2.11' - that warning is conservative; training has run fine on newer torch. Only downgrade if a REAL runtime error appears."
}
& python -c "import torch; print('GPU visible to torch:', torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) { Fail "could not read the GPU name from torch" }

Step "5/6 Train SZL-1 (first run downloads the ~2 GB base model; training itself is minutes-scale)"
& python train_szl.py
if ($LASTEXITCODE -ne 0) { Fail "training stopped - the real error is right above this line" }

Step "6/6 Birth into Ollama (GGUF path - direct safetensors import corrupted the voice, MEASURED 2026-07-12)"
$ollama = Get-Command ollama -ErrorAction SilentlyContinue
if ($ollama) {
  curl.exe -sL -o rebirth.ps1 "$base/rebirth.ps1"
  if (-not (Test-Path "rebirth.ps1") -or (Get-Item "rebirth.ps1").Length -lt 100) { Fail "could not download rebirth.ps1" }
  & powershell -ExecutionPolicy Bypass -File .\rebirth.ps1
  if ($LASTEXITCODE -ne 0) { Fail "birth failed - the real error is above" }
} else {
  Write-Host "ollama is not on PATH in this shell. When ready, run:" -ForegroundColor Yellow
  Write-Host "  iwr $base/rebirth.ps1 -OutFile rebirth.ps1; powershell -ExecutionPolicy Bypass -File .\rebirth.ps1"
}
