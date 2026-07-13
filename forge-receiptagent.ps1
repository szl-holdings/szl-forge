# SZL-Forge-1.5B-ReceiptAgent -- one-shot owner forge (paste-once bootstrap).
#
# Run it with ONE line, from ANY PowerShell window (system32 is fine):
#   irm https://raw.githubusercontent.com/szl-holdings/szl-forge/main/forge-receiptagent.ps1 | iex
#
# It pulls the PUBLIC kit (no git, no login), fixes GPU torch, makes/reuses your
# owner key, trains QLoRA Qwen2.5-1.5B, rebirths into Ollama, runs the held-out
# eval, then prints the THREE files to send back to Alloy.
#
# BINDING honesty doctrine: this runs entirely on YOUR metal with YOUR key. The
# PRIVATE key never leaves ~/.a11oy. Nothing is signed unless training + eval
# actually ran here. Nothing upgrades Lambda.
$ErrorActionPreference = 'Stop'
function Say($m, $c = 'Cyan') { Write-Host $m -ForegroundColor $c }
Say "== SZL-Forge-1.5B-ReceiptAgent : one-shot forge =="

# [1/6] fetch the kit from the PUBLIC szl-forge repo (works from any directory)
$root = Join-Path $env:TEMP 'szl-forge-main'
$zip = Join-Path $env:TEMP 'szl-forge-kit.zip'
Say "[1/6] downloading kit..."
if (Test-Path $root) { Remove-Item -Recurse -Force $root }
Invoke-WebRequest 'https://codeload.github.com/szl-holdings/szl-forge/zip/refs/heads/main' -OutFile $zip -UseBasicParsing
Expand-Archive -Path $zip -DestinationPath $env:TEMP -Force
$kit = Join-Path $root 'receiptagent'
if (-not (Test-Path (Join-Path $kit 'train_receiptagent.py'))) { throw "kit not found at $kit" }
Set-Location $kit
Say "    kit: $kit"

# [2/6] python deps (idempotent). NB: 'pip install unsloth' can pull CPU-only
#       torch on Windows -- we fix the cu128 build AFTER so the CUDA build wins.
Say "[2/6] python deps..."
python -m pip install -q --disable-pip-version-check unsloth trl datasets cryptography jsonschema

# [3/6] make sure torch actually sees the GPU (Blackwell / RTX 50xx needs cu128;
#       unsloth caps torch<2.11 so pin the matching 2.10 triple).
Say "[3/6] GPU torch check..."
$cuda = (python -c "import torch;print(1 if torch.cuda.is_available() else 0)" 2>$null)
if ("$cuda".Trim() -ne '1') {
  Say "    CPU-only torch -> installing cu128 (torch+vision+audio, large download)..." 'Yellow'
  python -m pip install "torch==2.10.*" "torchvision==0.25.*" "torchaudio==2.10.*" --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
  $cuda = (python -c "import torch;print(1 if torch.cuda.is_available() else 0)" 2>$null)
}
python -c "import torch;print('    torch',torch.__version__,'| CUDA',torch.cuda.is_available())"
if ("$cuda".Trim() -ne '1') { throw "GPU still not visible to torch. Fix your NVIDIA driver / cu128, then re-run the one-liner." }

# [4/6] owner key. Private key -> ~/.a11oy (never leaves this machine). If a key
#       already exists we REUSE it (no rotation) and just re-export the pubkey.
Say "[4/6] owner key..."
$pem = Join-Path $env:USERPROFILE '.a11oy\receiptagent_owner_ed25519.pem'
if (Test-Path $pem) {
  Say "    existing key found -> reusing (no rotation), re-exporting owner_pubkey.json"
  python -c "import base64,hashlib,json,os;from cryptography.hazmat.primitives import serialization;p=os.path.join(os.path.expanduser('~'),'.a11oy','receiptagent_owner_ed25519.pem');k=serialization.load_pem_private_key(open(p,'rb').read(),password=None);s=k.public_key().public_bytes(serialization.Encoding.DER,serialization.PublicFormat.SubjectPublicKeyInfo);i=hashlib.sha256(s).hexdigest()[:16];open('owner_pubkey.json','w',encoding='utf-8').write(json.dumps({'algo':'ed25519','publicKeySpkiBase64':base64.b64encode(s).decode(),'keyId':i},indent=2)+chr(10));print('    keyId',i)"
} else {
  python sign_receipt.py keygen
}

# [5/6] the long part: train -> rebirth into Ollama -> held-out eval.
Say "[5/6] train -> rebirth -> eval (long; first run downloads the base model)..."
$env:HF_HUB_ENABLE_HF_TRANSFER = '0'  # robust resumable downloads on flaky wifi
python train_receiptagent.py
powershell -ExecutionPolicy Bypass -File .\rebirth-receiptagent.ps1
python eval_receiptagent.py

# [6/6] done -- report the three files to send back to Alloy.
Say "[6/6] DONE" 'Green'
$need = 'owner_pubkey.json', 'training_receipt.signed.json', 'eval_receipt.signed.json'
Say "Send these THREE files to Alloy (drag them into the chat):" 'Green'
foreach ($f in $need) {
  if (Test-Path $f) { Say ("   [OK]      " + (Resolve-Path $f).Path) 'Green' }
  else { Say ("   [MISSING] " + $f + "  (a step above failed -- copy the red error to Alloy)") 'Red' }
}
try { explorer.exe $kit } catch {}
Say "Folder: $kit"
