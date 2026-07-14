# SZL-Khipu-1.5B-BrainNavigator rebirth — import the merged model into Ollama.
# SPDX-License-Identifier: Apache-2.0
# (c) 2026 Lutar, Stephen P. - SZL Holdings
#
# train_khipu.py writes a merged 16-bit safetensors folder to .\khipu-model. As
# with SZL-1, importing that folder directly into Ollama can corrupt the voice
# (@-spam at temp 0). The reliable path is to convert to F16 GGUF with
# llama.cpp's pure-Python converter, then quantize to q4_K_M and import that.
#
# Honest: prints exactly what it is doing; stops on real failure with the real
# error on screen. It NEVER claims success it did not see, and it NEVER touches
# szl1 (the 3B sovereign model) or receiptagent -- this births a SEPARATE
# 'khipu' model.
$ErrorActionPreference = "Continue"
function Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Fail($msg) {
  Write-Host "`n[khipu] FAILED: $msg" -ForegroundColor Red
  Write-Host "Screenshot the last ~20 lines above." -ForegroundColor Yellow
  exit 1
}

Step "1/5 Merged model check"
$dir = $PSScriptRoot
if (-not $dir) { $dir = Get-Location }
Set-Location $dir
if (-not (Test-Path (Join-Path $dir "khipu-model"))) {
  Fail "no merged model at $dir\khipu-model - run train_khipu.py first"
}
Write-Host "found $dir\khipu-model"

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

Step "3/5 Convert merge -> F16 GGUF"
& python llama.cpp\convert_hf_to_gguf.py khipu-model --outfile khipu-f16.gguf --outtype f16
if ($LASTEXITCODE -ne 0 -or -not (Test-Path "khipu-f16.gguf")) { Fail "GGUF conversion failed - the real error is above" }

Step "4/5 Rebirth into Ollama (quantized q4_K_M, name 'khipu')"
if (-not (Test-Path "Modelfile.khipu.gguf")) { Fail "Modelfile.khipu.gguf not found next to this script" }
& ollama rm khipu 2>$null
& ollama create khipu --quantize q4_K_M -f Modelfile.khipu.gguf
if ($LASTEXITCODE -ne 0) { Fail "ollama create failed - the real error is above" }

Step "5/5 First words (the REAL test - must be a JSON PLAN, not @ spam)"
$smoke = '{"query":"Which handle records the rolling 24h spend-cap policy?","candidates":[{"nodeId":"node://khipu-synthetic/0000000000000000","nodeKind":"CLAIM","label":"DECLARED","note":"synthetic handle - topic tag policy-spend-cap; no node content is embedded."}]}'
& ollama run khipu $smoke
Write-Host ""
Write-Host "[khipu] If that reads like a JSON retrieval PLAN (NAVIGATE citing the handle, or ABSTAIN), rebirth worked." -ForegroundColor Green
Write-Host "Next: python sanity_gate.py, then python eval_khipu.py (produces the signed eval receipt)." -ForegroundColor Green
Write-Host "If it is STILL @ spam, the merge is suspect - tell Alloy, do NOT just retrain." -ForegroundColor Yellow
