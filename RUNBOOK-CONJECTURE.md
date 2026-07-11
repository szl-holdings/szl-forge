# Conjecture Machine Runbook — point the sovereign stack at the formula corpus

Goal: run **SZL's own model, on SZL's own metal, against SZL's own formula
corpus** — asking it for proof sketches, lemma decompositions, and
counterexample searches for each formula, and saving every attempt.

**What this is (honest tier):** an *advisory* research aid. It asks a language
model for the *shape* of an argument. It does **NOT** prove anything. A formula
stays a **CONJECTURE** until a real Lean proof checks — which is done separately
in `szl-holdings/lutar-lean`, not here. The one formula about Λ uniqueness is
locked to **Conjecture-1** and can never be upgraded by this tool, by design.

Every step below is ONE command. It needs only Python 3.8+ (standard library —
no `pip install`).

## Step 0 — check Python

```powershell
python --version
```

Want `Python 3.8`–`3.13`. If missing: `winget install -e --id Python.Python.3.12`
(then close and reopen PowerShell).

## Step 1 — get the kit

```powershell
mkdir "$env:USERPROFILE\conjecture" -Force; cd "$env:USERPROFILE\conjecture"; curl.exe -L -o conjecture_machine.py https://raw.githubusercontent.com/szl-holdings/szl-forge/main/conjecture_machine.py; curl.exe -L -o thesis_formula_index.json https://raw.githubusercontent.com/szl-holdings/szl-forge/main/thesis_formula_index.json
```

## Step 2 — prove the wiring WITHOUT touching the network (dry run)

```powershell
cd "$env:USERPROFILE\conjecture"; python conjecture_machine.py --dry-run
```

Success looks like: it prints the **actual** formula count (80 in the shipped
snapshot), writes one `DRY_RUN` record per formula under
`conjecture_runs\<timestamp>\`, and ends with `proven: 0`. No network was used.

## Step 3 — turn on the tower

The sovereign endpoint is the tower's Ollama at `https://gpu.a-11-oy.com/v1`.
Turn the tower on; the boot autostart brings up Ollama + the Cloudflare tunnel.
Confirm from any machine: `https://gpu.a-11-oy.com/api/version` answers.
(If it does not answer, the tower is off — the machine will simply record
`UNAVAILABLE` for every formula and exit cleanly. That is honest, not a failure.)

## Step 4 — run the machine for real (a few formulas first)

```powershell
cd "$env:USERPROFILE\conjecture"; python conjecture_machine.py --limit 5
```

This asks the sovereign model (`llama3-szl-finetuned-q4:latest` by default) for
an advisory sketch of the first 5 formulas and writes them to
`conjecture_runs\<timestamp>\`. Each file's `status` is `CONJECTURE` (advisory
sketch received) or `UNAVAILABLE` (endpoint down). `proven` is always `false`.

## Step 5 — run the full corpus

```powershell
cd "$env:USERPROFILE\conjecture"; python conjecture_machine.py
```

## Optional — use the SZL-Nemo slot instead

```powershell
cd "$env:USERPROFILE\conjecture"; $env:MODEL="szl-nemo"; python conjecture_machine.py --limit 5
```

## Where the output goes

- `conjecture_runs\<timestamp>\<FORMULA_ID>.json` — one file per formula:
  the statement, the model's advisory sketch (`model_output`), the honest
  `lean_check` stub, and the doctrine flags.
- `conjecture_runs\<timestamp>\_manifest.json` — the run summary: endpoint,
  model, endpoint status, tallies, and `proven_count: 0`.

## Honest notes

- **Nothing here is a proof.** `model_output` is an advisory sketch. `proven`
  is hard-coded `false` and `lean_check.passed` is always `false` until a real
  `lake build` against `szl-holdings/lutar-lean` is wired in. The corpus
  entries carry their own `status` labels (e.g. `GREEN`) — those are the
  dataset's **staged-advisory** tags, NOT live proofs, and this machine does
  not launder them into "proven".
- **Λ uniqueness stays Conjecture-1.** `TH10 — Uniqueness of Lutar Invariant`
  is doctrine-locked: its record shows `lambda_uniqueness_locked: true` and its
  status can never be upgraded, even if a Lean check were later supplied.
- **The shipped snapshot has 80 formulas.** The upstream
  `thesis-formula-index` metadata field `entry_count` reads **68**, which is
  stale — the machine counts the actual array (80) and prints a NOTE about the
  disagreement rather than trusting the metadata.
- **Endpoint status as of 2026-07-11:** `https://gpu.a-11-oy.com/v1` returned
  **HTTP 530** (tower tunnel had no live connector — tower off). Turn the tower
  on (Step 3) before expecting live sketches.
- To point at a different sovereign node:
  `python conjecture_machine.py --endpoint https://gpu2.a-11-oy.com/v1`
  (or set `A11OY_MODEL_BASE_URL`). CF-Access-gated nodes read
  `A11OY_GPU_CF_ACCESS_ID` / `A11OY_GPU_CF_ACCESS_SECRET` from the environment.
