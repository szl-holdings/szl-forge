# SZL Forge Runbook — train SZL-1 on the laptop (RTX 5050)

> **Shortcut:** all steps below run as ONE command now — see "One command"
> in [README.md](./README.md) (`forge.ps1`). This runbook remains the
> step-by-step version of the same pipeline.

Goal: fine-tune an open base model into **SZL-1** — SZL Holdings' own model —
entirely on SZL's own hardware, then serve it through Ollama like any other
estate model.

Sources (REPORTED): Unsloth docs state RTX 50-series (Blackwell) is supported,
Windows works without WSL, Python 3.11–3.13, and a 3B QLoRA run fits in
5–8 GB VRAM. This runbook has NOT yet been executed end-to-end on this
laptop — treat each step as measured only when it succeeds on screen.

Every step below is ONE single-line PowerShell command. Screenshot anything
surprising.

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

Success looks like: loss numbers ticking down, then
`[szl-forge] DONE. Next: ollama create szl1 -f Modelfile`

## Step 4 — birth the model into Ollama

```powershell
cd "$env:USERPROFILE\szl-forge"; ollama create szl1 -f Modelfile
```

## Step 5 — first words

```powershell
ollama run szl1 "Who are you and who do you belong to?"
```

If it answers as SZL-1, sovereign model of SZL Holdings — the estate has its
own model. Alloy then switches `SOVEREIGN_MODEL=szl1` and every default run
in the cockpit is served by a model SZL trained itself.

## Honest notes

- This is FINE-TUNING an open base model (Qwen2.5-3B-Instruct) into SZL's
  own — identity, doctrine, and weights on SZL's disk. Pre-training a
  frontier model from scratch genuinely does need datacenter metal; nobody
  should claim otherwise.
- Disk: ~30 GB free recommended (model cache + merged export).
- If `pip install unsloth` or training fails, screenshot the last ~20 lines —
  Windows wheels for triton/bitsandbytes are the usual suspects and each has
  a known fix.
