# SZL-Nemo Runbook — put Nemotron on the tower (RTX 4060 Ti)

Goal: serve **SZL-Nemo** — NVIDIA's open Nemotron 3 Nano 4B wrapped in SZL's
honesty doctrine — from the tower's Ollama, as the third target in Alloy's
sovereign failover chain (tower·szl → laptop → tower·nemo).

This follows the pattern of the LangChain × NVIDIA **NemoClaw Deep Agents
blueprint** (July 2026): an open model you control + your own harness + a
governed runtime, tuned together. SZL's version: open Nemotron weights, the
Alloy backbone as harness, and SZL's receipt/guardrail stack as governance.

Honest facts (REPORTED from ollama.com, 2026-07-11): `nemotron-3-nano:4b` is a
**2.8 GB** download with a 256K context window. The 30B/latest tags are 24 GB —
too big for estate GPUs; do not pull them.

Every step is ONE command, run **on the tower (omen)** in PowerShell.

## Step 0 — power on the tower

Just turn it on. The boot autostart brings up Ollama + the Cloudflare tunnel.
Confirm from any machine: https://gpu.a-11-oy.com/api/version answers.

## Step 1 — pull the open model (~2.8 GB download)

```powershell
ollama pull nemotron-3-nano:4b
```

## Step 2 — get the SZL-Nemo recipe

```powershell
cd "$env:USERPROFILE"; curl.exe -L -o Modelfile.nemo https://raw.githubusercontent.com/szl-holdings/szl-forge/main/Modelfile.nemo
```

## Step 3 — birth SZL-Nemo into Ollama

```powershell
ollama create szl-nemo -f "$env:USERPROFILE\Modelfile.nemo"
```

## Step 4 — first words

```powershell
ollama run szl-nemo "Who are you, and did SZL train your weights?"
```

An honest SZL-Nemo says it is Nemotron served on SZL metal and that SZL did
**not** fine-tune its weights.

## Step 5 — nothing

Alloy is already wired: the sovereign fleet's third slot points at the tower
with model `szl-nemo`. Once Steps 1–3 are done, cockpit runs can be served by
SZL-Nemo whenever the primary SZL-1 target is unavailable.

## Honest notes

- SZL-Nemo is a **doctrine wrapper**, not an SZL fine-tune. The weights are
  NVIDIA's open Nemotron 3 Nano weights, unchanged (see the model's license on
  ollama.com/library/nemotron-3-nano).
- No benchmarks have been measured on SZL hardware yet — quality is UNKNOWN
  until measured.
- A future szl-forge run MAY fine-tune a Nemotron base the way `train_szl.py`
  fine-tunes Qwen; that has not been done and is not claimed.
