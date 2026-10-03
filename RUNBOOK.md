# SZL Forge Runbook — train SZL-1 on the laptop (RTX 5050)

> **Shortcut:** all steps below run as ONE command now — see "One command"
> in [README.md](./README.md) (`forge.ps1`). This runbook remains the
> step-by-step version of the same pipeline.

Historical goal: fine-tune an open base model into **SZL-1** on SZL hardware.
This recipe is not a current recovery or serving authorization. Its former
direct import can replace an installed model without establishing clean source
lineage or qualification. The installed repeated-`@` exports are held under
[recovery issue #264](https://github.com/szl-holdings/szl-forge/issues/264).
Use `local-compute/README.md` for bounded new experiments; preserve existing
model names, merges, reports, claims, and locks.

Sources (REPORTED): Unsloth docs state RTX 50-series (Blackwell) is supported,
Windows works without WSL, Python 3.11–3.13, and a 3B QLoRA run fits in
5–8 GB VRAM. This runbook has NOT yet been executed end-to-end on this
laptop — treat each step as measured only when it succeeds on screen.

Every step below is ONE single-line PowerShell command. Screenshot anything
surprising.

## Step -1 — preflight and dataset proof (new, runs anywhere)

Before touching the GPU, prove the environment and the dataset. Both tools
are stdlib-only and fail closed with honest MEASURED/UNAVAILABLE labels:

```powershell
python tools/validate_sft_dataset.py szl_dataset.jsonl --min-examples 8
python tools/forge_preflight.py --dataset szl_dataset.jsonl --out receipts/preflight.json
```

The validator refuses malformed JSONL, missing `messages`, unknown roles, and
empty content — the exact failure class that once sent a package file to the
trainer. The preflight writes a hashed `szl.forge-preflight/v1` receipt and
exits non-zero (`NOT_READY`) if Python, disk, the training stack, CUDA, or
the dataset cannot support a run. A READY preflight is **not** a training
receipt: the run still needs its own artifact + evaluation + receipt evidence.

Well-formed is not the same as doctrine-conformant. Before training, also
gate the dataset's *content* through the szl-nemo doctrine kernel
(`pip install "szl-nemo @ git+https://github.com/szl-holdings/szl-nemo.git@<pinned-sha>"`,
then one command per dataset):

```powershell
python tools/nemo_doctrine_gate.py szl_dataset.jsonl --persona finetuned --write-receipt receipts/doctrine
```

The gate fails closed on fabrication-label, honest-unknown, Lambda-theorem,
trust-ceiling, and fine-tune-provenance violations in the training text
itself, and binds the run into a deterministic hash-chained receipt ledger
(`szl.nemo.receipt.v1`). `--persona finetuned` is the SZL-1/ReceiptAgent
posture: R3 is mirrored — a fine-tune question must be answered by
*affirming* the SZL fine-tune, not by Nemo's "not fine-tuned" disclosure.
Choosing the persona is a doctrine decision, so the flag never defaults.

## Step 0 — check Python

```powershell
python --version
```

Want: `Python 3.11.x`–`3.13.x`. If missing or too old:

```powershell
winget install -e --id Python.Python.3.12
```

(then CLOSE and REOPEN PowerShell so `python` is on PATH)

## Step 1 — install Unsloth (one-time, ~5–10 min)

```powershell
pip install unsloth
```

## Step 2 — get the forge kit

```powershell
mkdir "$env:USERPROFILE\szl-forge" -Force; cd "$env:USERPROFILE\szl-forge"; curl.exe -L -o train_szl.py https://raw.githubusercontent.com/szl-holdings/szl-forge/main/train_szl.py; curl.exe -L -o szl_dataset.jsonl https://raw.githubusercontent.com/szl-holdings/szl-forge/main/szl_dataset.jsonl; curl.exe -L -o Modelfile https://raw.githubusercontent.com/szl-holdings/szl-forge/main/Modelfile
```

## Step 3 — train (first run downloads ~2 GB base model)

```powershell
cd "$env:USERPROFILE\szl-forge"; python train_szl.py
```

Loss numbers and a saved merge establish only that this historical training
recipe ran. They do not qualify the merge for canonical import.

## Step 4 — hold canonical import

Do not run the historical `ollama create szl1` command. A separately named
candidate requires exact base, adapter, tokenizer, merge, converter, and input
lineage; preserved rollback bytes; adequate storage; and fresh held-out
before/after evidence. Issue #264 remains open until those gates are met.

## Step 5 — hold routing

A self-identification answer is a smoke check, not model qualification. Do not
set `SOVEREIGN_MODEL=szl1` or treat the installed repeated-`@` export as a
working sovereign model.

## Honest notes

- This is FINE-TUNING an open base model (Qwen2.5-3B-Instruct) into SZL's
  own — identity, doctrine, and weights on SZL's disk. Pre-training a
  frontier model from scratch genuinely does need datacenter metal; nobody
  should claim otherwise.
- Disk: ~30 GB free recommended (model cache + merged export).
- If `pip install unsloth` or training fails, screenshot the last ~20 lines —
  Windows wheels for triton/bitsandbytes are the usual suspects and each has
  a known fix.
