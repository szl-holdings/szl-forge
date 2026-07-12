# SZL Forge rebirth — fix SZL-1's voice WITHOUT retraining.
#
# Diagnosis (MEASURED 2026-07-12): after forge.ps1 finished, szl1 answered
# '@@@@...' at temperature 0 — even in raw mode. The Ollama-imported weights
# are corrupted. The trained merge at .\szl-model is the input this script
# re-imports properly: convert to F16 GGUF with llama.cpp's pure-Python
# converter, then quantize to q4_K_M (fits the 8 GB GPU, fast).
#
#   iwr https://raw.githubusercontent.com/szl-holdings/szl-forge/main/rebirth.ps1 -OutFile "$env:TEMP\rebirth.ps1"; powershell -ExecutionPolicy Bypass -File "$env:TEMP\rebirth.ps1"
#
# Honest: prints exactly what it is doing; stops on real failure with the
# real error on screen. It never claims success it didn't see.

$ErrorActionPreference = "Continue"
function Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Fail($msg) {
  Write-Host "`n[szl-forge] FAILED: $msg" -ForegroundColor Red
  Write-Host "Screenshot the last ~20 lines above." -ForegroundColor Yellow
  exit 1
}

Step "1/5 Trained model check"
$dir = Join-Path $env:USERPROFILE "szl-forge"
if (-not (Test-Path (Join-Path $dir "szl-model"))) { Fail "no trained model at $dir\szl-model - run forge.ps1 first" }
Set-Location $dir
Write-Host "found $dir\szl-model"

Step "2/5 llama.cpp converter (pure Python - nothing to compile)"
if (-not (Test-Path "llama.cpp\convert_hf_to_gguf.py")) {
  $git = Get-Command git -ErrorAction SilentlyContinue
  if ($git) { & git clone --depth 1 https://github.com/ggml-org/llama.cpp }
  if (-not (Test-Path "llama.cpp\convert_hf_to_gguf.py")) {
    Write-Host "git unavailable or clone failed - downloading tarball instead..."
    curl.exe -sL -o llama-cpp.tar.gz https://github.com/ggml-org/llama.cpp/archive/refs/heads/master.tar.gz
    tar -xzf llama-cpp.tar.gz
    $ex = Get-ChildItem -Directory -Filter "llama.cpp-*" | Select-Object -First 1
    if ($ex) { Rename-Item $ex.FullName "llama.cpp" }
  }
}
if (-not (Test-Path "llama.cpp\convert_hf_to_gguf.py")) { Fail "could not get the llama.cpp converter" }
& python -m pip install -q gguf mistral-common sentencepiece protobuf
if ($LASTEXITCODE -ne 0) { Fail "pip install for the converter failed" }

Step "3/5 Convert merge -> F16 GGUF (the step the old birth was getting wrong)"
& python llama.cpp\convert_hf_to_gguf.py szl-model --outfile szl1-f16.gguf --outtype f16
if ($LASTEXITCODE -ne 0 -or -not (Test-Path "szl1-f16.gguf")) { Fail "GGUF conversion failed - the real error is above" }

Step "4/5 Rebirth into Ollama (quantized q4_K_M)"
curl.exe -sL -o Modelfile.gguf https://raw.githubusercontent.com/szl-holdings/szl-forge/main/Modelfile.gguf
if (-not (Test-Path "Modelfile.gguf") -or (Get-Item "Modelfile.gguf").Length -lt 100) { Fail "could not download Modelfile.gguf" }
& ollama rm szl1 2>$null
& ollama create szl1 --quantize q4_K_M -f Modelfile.gguf
if ($LASTEXITCODE -ne 0) { Fail "ollama create failed - the real error is above" }

Step "5/5 First words (the REAL test - must NOT be @ spam)"
& ollama run szl1 "Who are you and who do you belong to? One sentence."
Write-Host ""
Write-Host "[szl-forge] If that reads like SZL-1, rebirth worked - tell Alloy 'szl1 reborn'." -ForegroundColor Green
Write-Host "If it is STILL @ spam, the merge itself is suspect - tell Alloy, do NOT just retrain (same seed would likely reproduce the same merge)." -ForegroundColor Yellow
