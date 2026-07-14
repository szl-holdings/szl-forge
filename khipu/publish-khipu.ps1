<#
  publish-khipu.ps1 — one-paste publisher for the forged Khipu artifacts.

  Run from the khipu folder AFTER a PASS eval (Step 6 of RUNBOOK-KHIPU.md):

      powershell -ExecutionPolicy Bypass -File publish-khipu.ps1

  What it does (and nothing else):
    1. Verifies the three trust-root files exist and the owner keyId matches
       the pinned value below (refuses to publish on mismatch).
    2. Commits the three receipts to szl-holdings/szl-forge/khipu/
       (gh CLI -> git -> browser upload page, whichever you have), then
       VERIFIES they are actually on GitHub main before claiming so.
    3. Uploads receipts + merged weights (khipu-model/) + LoRA adapter
       (khipu-adapter/ -> adapter/) to the Hub repo, matching the
       SZL-Forge-1.5B-ReceiptAgent layout, then VERIFIES the Hub file list.
       Weights and adapter are REQUIRED unless you pass -NoWeights/-NoAdapter,
       so a partial publish can never masquerade as a complete one.

  Honesty invariants (same as the runbook): publishing is a repo-existence
  fact ONLY. trainingStatus/evalStatus flip solely when the signed receipts
  verify server-side (Alloy /api/forge/family). Private key material
  (*.pem, khipu_owner_ed25519*) is never read, and folder uploads exclude
  those patterns explicitly — the key stays on your metal.
#>
param(
  [string]$KhipuDir = $PSScriptRoot,
  [string]$HfRepo = "SZLHOLDINGS/SZL-Khipu-1.5B-BrainNavigator",
  # Owner keyId observed at the 2026-07 khipu forge run. Distinct from the
  # ReceiptAgent owner key by design (each forge births its own key).
  # Override ONLY if you deliberately rotated the khipu owner key.
  [string]$ExpectedKeyId = "89540347a69b789e",
  [switch]$SkipGitHub,
  [switch]$SkipHub,
  [switch]$NoWeights,
  [switch]$NoAdapter,
  [switch]$IncludeGguf,
  # Also sha256-compare uploaded LFS files (weights) against local bytes. Slower.
  [switch]$DeepVerify
)
$ErrorActionPreference = "Stop"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
function Say([string]$m)  { Write-Host "[publish-khipu] $m" }
function Fail([string]$m) { Write-Host "[publish-khipu] FAIL: $m" -ForegroundColor Red; exit 1 }
function Test-Exit([string]$what) { if ($LASTEXITCODE -ne 0) { Fail "$what failed (exit $LASTEXITCODE) - nothing past this point ran." } }

if (-not $KhipuDir) { $KhipuDir = (Get-Location).Path }
Set-Location $KhipuDir
$receipts = @("owner_pubkey.json", "training_receipt.signed.json", "eval_receipt.signed.json")
Say "courier v4.3 - auth-aware fallback (gh -> git -> browser); byte-verify gates every path. If you do not see this version line, you are running stale bytes."

# ---- 1. Trust-root sanity + keyId pin -------------------------------------
# Wrong-folder rescue: if the receipts are not beside us (e.g. launched from an
# admin shell in system32), hop to the kit's canonical location - but only if
# ALL THREE receipts are there. Never guess between partial candidates.
$kitDefault = Join-Path $env:LOCALAPPDATA "Temp\szl-forge-main\khipu"
$missingHere = @($receipts | Where-Object { -not (Test-Path (Join-Path $KhipuDir $_)) })
if ($missingHere.Count -gt 0 -and (Test-Path $kitDefault)) {
  $foundThere = @($receipts | Where-Object { Test-Path (Join-Path $kitDefault $_) })
  if ($foundThere.Count -eq $receipts.Count) {
    Say "receipts not in $KhipuDir - found all 3 in $kitDefault, switching there."
    $KhipuDir = $kitDefault
    Set-Location $KhipuDir
  }
}
foreach ($f in $receipts) {
  if (-not (Test-Path $f)) { Fail "$f not found in $KhipuDir - run the forge steps first (RUNBOOK-KHIPU.md)." }
}
$pub = Get-Content owner_pubkey.json -Raw | ConvertFrom-Json
if (-not $pub.keyId) { Fail "owner_pubkey.json has no keyId field - file looks wrong or truncated." }
if ($pub.keyId -ne $ExpectedKeyId) {
  Fail ("owner keyId is '$($pub.keyId)' but this script pins '$ExpectedKeyId'. " +
        "If you rotated the key on purpose, re-run with -ExpectedKeyId $($pub.keyId). Refusing to publish a surprise key.")
}
foreach ($f in @("training_receipt.signed.json", "eval_receipt.signed.json")) {
  $j = Get-Content $f -Raw | ConvertFrom-Json
  if (-not $j.signatureBase64 -and -not $j.signature) { Fail "$f parses but has no signature field - not a signed receipt." }
}
Say "trust root OK: keyId $($pub.keyId) (pinned), 3 files present and signed."
Say "NOTE: signatures are re-verified server-side by Alloy; this script only refuses obvious mistakes."

# ---- 2. Receipts -> szl-forge (the committed trust root) -------------------
if ($SkipGitHub) { Say "SkipGitHub set - not committing receipts to szl-forge." }
else {
  $msg = "khipu: commit owner trust root (pubkey + signed training/eval receipts), keyId $($pub.keyId)"
  # Route by what actually WORKS, not what merely exists: a gh that is not
  # logged in falls through to git; a git that cannot push falls through to
  # the browser page. The byte-identity verification below gates all paths.
  $committed = $false
  $ghOk = $false
  if (Get-Command gh -ErrorAction SilentlyContinue) {
    # PS 5.1 + ErrorActionPreference=Stop turns REDIRECTED native stderr into a
    # terminating error, so probe auth via cmd.exe redirection instead.
    cmd /c "gh auth status 1>nul 2>nul"
    if ($LASTEXITCODE -eq 0) { $ghOk = $true }
    else { Say "gh CLI found but not logged in (optional: 'gh auth login') - trying the next path ..." }
  }
  if ($ghOk) {
    Say "using gh CLI to commit receipts to szl-holdings/szl-forge/khipu/ ..."
    foreach ($f in $receipts) {
      $b64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes((Join-Path $KhipuDir $f)))
      $sha = ""
      try { $sha = (gh api "repos/szl-holdings/szl-forge/contents/khipu/$f" --jq .sha 2>$null) } catch {}
      $ghArgs = @("api", "--method", "PUT", "repos/szl-holdings/szl-forge/contents/khipu/$f",
                  "-f", "message=$msg", "-f", "content=$b64")
      if ($sha) { $ghArgs += @("-f", "sha=$sha") }
      gh @ghArgs | Out-Null
      Test-Exit "gh api PUT khipu/$f (gh IS logged in, so this is a real error - permissions?)"
      Say "  committed khipu/$f"
    }
    $committed = $true
  }
  elseif (Get-Command git -ErrorAction SilentlyContinue) {
    Say "using git clone/commit/push (any stumble falls back to the browser page) ..."
    $tmp = Join-Path $env:TEMP "szl-forge-publish"
    if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp }
    git clone --depth 1 https://github.com/szl-holdings/szl-forge $tmp
    if ($LASTEXITCODE -eq 0) {
      foreach ($f in $receipts) { Copy-Item (Join-Path $KhipuDir $f) (Join-Path $tmp "khipu\$f") -Force }
      Push-Location $tmp
      git add khipu/owner_pubkey.json khipu/training_receipt.signed.json khipu/eval_receipt.signed.json
      $staged = $null
      if ($LASTEXITCODE -eq 0) { $staged = git status --porcelain }
      if ($LASTEXITCODE -eq 0 -and -not $staged) {
        Say "  receipts already identical on main - nothing to commit."
        $committed = $true
      }
      elseif ($LASTEXITCODE -eq 0 -and $staged) {
        git -c user.name="Lutar, Stephen P." -c user.email="stephenlutar2@gmail.com" commit -s -m $msg
        if ($LASTEXITCODE -eq 0) {
          git push origin HEAD:main
          if ($LASTEXITCODE -eq 0) { $committed = $true; Say "  receipts pushed to main." }
        }
      }
      Pop-Location
      if (-not $committed) { Say "git path did not complete (credentials or identity) - falling back to the browser upload page." }
    }
    else { Say "git clone failed - falling back to the browser upload page." }
  }
  if (-not $committed) {
    Say "opening the GitHub upload page - this needs no command-line login, just your browser session."
    Say "ACTION: drag these 3 files from this folder onto the page, commit to main:"
    $receipts | ForEach-Object { Say "    $KhipuDir\$_" }
    Start-Process "https://github.com/szl-holdings/szl-forge/upload/main/khipu"
    Read-Host "press Enter here AFTER the upload page shows the commit succeeded"
  }
  # Verify on GitHub regardless of which path ran (public repo, anonymous API):
  # not just that the paths exist, but that main's bytes EQUAL this folder's bytes.
  foreach ($f in $receipts) {
    $remote = $null
    try { $remote = Invoke-RestMethod -UseBasicParsing "https://api.github.com/repos/szl-holdings/szl-forge/contents/khipu/$($f)?ref=main" }
    catch { Fail "verification: khipu/$f is NOT on szl-forge main. The commit did not land - fix that before trusting any output above." }
    $localB64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes((Join-Path $KhipuDir $f)))
    $remoteB64 = ($remote.content -replace "\s", "")
    if ($remoteB64 -ne $localB64) {
      Fail "verification: khipu/$f on main differs from the local file - an OLD or WRONG copy is committed. Re-run the commit step (browser path: make sure the upload actually finished)."
    }
  }
  Say "VERIFIED: all 3 receipt files on szl-forge main are byte-identical to this folder."
}

# ---- 3. Weights + receipts -> the Hub (ReceiptAgent layout) ----------------
if ($SkipHub) { Say "SkipHub set - not uploading to $HfRepo. (GitHub trust root only.)"; exit 0 }
if (-not $NoWeights -and -not (Test-Path "khipu-model")) {
  Fail "khipu-model/ (merged weights) not found in $KhipuDir. Re-run from the folder that has it, or pass -NoWeights to publish receipts only - refusing to half-publish silently."
}
if (-not $NoAdapter -and -not (Test-Path "khipu-adapter")) {
  Fail "khipu-adapter/ (LoRA) not found in $KhipuDir. Re-run from the folder that has it, or pass -NoAdapter - refusing to half-publish silently."
}
$py = $null
foreach ($cand in @("python", "py")) {
  if (Get-Command $cand -ErrorAction SilentlyContinue) { $py = $cand; break }
}
if (-not $py) { Fail "no python found - the forge needed it, so run this from the same shell that trained." }
cmd /c "$py -m pip show huggingface_hub 1>nul 2>nul"
if ($LASTEXITCODE -ne 0) {
  Say "installing huggingface_hub ..."
  & $py -m pip install -U "huggingface_hub[cli]"
  Test-Exit "pip install huggingface_hub"
}
& $py -c "from huggingface_hub import whoami; print('[publish-khipu] Hub login OK:', whoami()['name'])"
if ($LASTEXITCODE -ne 0) {
  Say "not logged in to the Hub - a login prompt follows (paste a WRITE token for SZLHOLDINGS):"
  & $py -c "from huggingface_hub import login; login()"
  Test-Exit "Hub login"
}
# Folder uploads exclude private-key material and local GGUF builds by pattern.
$ignore = "['*.pem', 'khipu_owner_ed25519*', '*.gguf']"
Say "uploading receipts to $HfRepo ..."
& $py -c "from huggingface_hub import HfApi; a = HfApi(); [a.upload_file(path_or_fileobj=f, path_in_repo=f, repo_id='$HfRepo', commit_message='khipu: owner trust root (pubkey + signed receipts)') for f in ['owner_pubkey.json', 'training_receipt.signed.json', 'eval_receipt.signed.json']]"
Test-Exit "receipt upload"
if (-not $NoWeights) {
  Say "uploading merged weights (khipu-model/ -> repo root, like the ReceiptAgent; *.pem/owner-key/gguf excluded) ..."
  & $py -c "from huggingface_hub import HfApi; HfApi().upload_folder(folder_path='khipu-model', repo_id='$HfRepo', commit_message='khipu: merged weights (owner-metal build)', ignore_patterns=$ignore)"
  Test-Exit "weights upload"
}
if (-not $NoAdapter) {
  Say "uploading LoRA adapter (khipu-adapter/ -> adapter/; same exclusions) ..."
  & $py -c "from huggingface_hub import HfApi; HfApi().upload_folder(folder_path='khipu-adapter', path_in_repo='adapter', repo_id='$HfRepo', commit_message='khipu: LoRA adapter', ignore_patterns=$ignore)"
  Test-Exit "adapter upload"
}
if ($IncludeGguf) {
  Get-ChildItem -Filter *.gguf | ForEach-Object {
    Say "uploading $($_.Name) -> gguf/ ..."
    & $py -c "from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj='$($_.Name)', path_in_repo='gguf/$($_.Name)', repo_id='$HfRepo', commit_message='khipu: GGUF build')"
    Test-Exit "GGUF upload ($($_.Name))"
  }
}
# Verify the Hub actually has what we claim it has - receipts byte-identical
# (downloaded + sha256-compared), folders name+size (-DeepVerify: sha256 too).
if (-not (Test-Path "verify_publish.py")) {
  Say "fetching verify_publish.py (your kit copy predates it) ..."
  Invoke-WebRequest -UseBasicParsing "https://raw.githubusercontent.com/szl-holdings/szl-forge/main/khipu/verify_publish.py" -OutFile "verify_publish.py"
}
$vArgs = @("verify_publish.py", "--repo", $HfRepo)
if (-not $NoWeights) { $vArgs += "--need-weights" }
if (-not $NoAdapter) { $vArgs += "--need-adapter" }
if ($DeepVerify)     { $vArgs += "--deep" }
& $py @vArgs
Test-Exit "Hub verification (verify_publish.py)"

Say "DONE - and verified, not just attempted:"
if (-not $SkipGitHub) { Say "  - szl-forge main: 3 receipt files byte-identical to this folder (GitHub API compare)" }
Say "  - ${HfRepo}: receipts byte-identical$(if (-not $NoWeights) { '; weights name+size' })$(if (-not $NoAdapter) { '; adapter name+size' })$(if ($DeepVerify) { ' (deep sha256)' }) - no key material remotely."
Say "  - Final judge stays Alloy /api/forge/family: receipts verifying there - that, and only that, changes trainingStatus/evalStatus."
