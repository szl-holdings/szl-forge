# RUNBOOK-EVAL — measuring SZL-1 honestly

This is the eval half of the SZL-Forge kit. After you have trained and served
SZL-1 (see `RUNBOOK.md`), `eval_szl.py` turns SZL's own public benchmark into
**MEASURED** numbers you can trust — or an honest **UNKNOWN** if the run cannot
be scored. It never invents a figure.

## One command (after training + `ollama create szl1`)

```bash
python eval_szl.py
```

That's it. Pure Python standard library — no `pip install`, no HF login. It
defaults to the local Ollama OpenAI-compatible endpoint and model:

```bash
python eval_szl.py --base-url http://localhost:11434/v1 --model szl1
```

Point it at the tower instead, or smoke-test the first few items first:

```bash
python eval_szl.py --base-url https://gpu.a-11-oy.com/v1 --model szl1
python eval_szl.py --limit 5          # quick 5-item sanity pass
```

Other flags: `--api-key` (if the endpoint is gated), `--timeout`, `--output`,
`--system-prompt`.

## What it does

1. Downloads two public SZL datasets from Hugging Face (anonymously):
   - `SZLHOLDINGS/k-verify-benchmark-v1` — 100 scored items + integrity manifest.
   - `SZLHOLDINGS/alloy-sovereign-eval-runs` — prior measured runs + the
     append-only row schema this harness mirrors.
2. Recomputes the SHA256 of the benchmark JSONL and checks it against the
   manifest's declared hash (MEASURED integrity: `MATCH`/`MISMATCH`).
3. Sends every question to your endpoint and scores the reply against the
   dataset's own rubric fields.
4. Writes `eval_results.json`.

## What the output means

`eval_results.json` carries three metrics, each labeled honestly:

- **accuracy** (MEASURED) — of the 85 verifiable items, how many the model
  answered correctly (numeric-tolerance / exact-containment matcher).
- **huklla_refusal** (MEASURED) — of the 15 unverifiable "trap" items (future,
  private, unknowable), how many the model correctly **refused** instead of
  confabulating.
- **khipu_verifiability** (UNSCORED by default, with reason) — this needs the
  model/agent to emit a signed receipt whose SHA256 recomputes. A bare
  completion endpoint does not, so the harness leaves it UNSCORED and tells you
  why. Route the same benchmark through Alloy (a11oy.code) to score this metric.

Provenance stamped on every result: run timestamp (UTC), model id, endpoint,
and the exact HF dataset revision SHAs scored against. Metric values are
MEASURED; the dataset revision is REPORTED (by the HF API); the manifest hash is
DECLARED; the integrity verdict is MEASURED.

Honest edge cases:
- Endpoint unreachable → `endpoint_status: NO_API_ACCESS` + the observed error;
  metrics become UNSCORED. **No number is fabricated.**
- A metric that the dataset shape cannot support → UNSCORED + reason.

The `runs_rows` array in the output mirrors the
`SZLHOLDINGS/alloy-sovereign-eval-runs` schema so you can append your results to
that dataset. Note that `energy` and `receipt` fields are honestly UNKNOWN/NONE
here — this stdlib harness has no NVML meter or receipt signer; route through
Alloy for MEASURED energy and SIGNED receipts.

## Doctrine (binding)

Until you run `eval_szl.py` on real metal and read genuine numbers out of
`eval_results.json`, every model card stays **"Benchmarks: None yet"** and
performance stays **UNKNOWN**. No benchmark figure may be written onto a card,
README, or post before a real run produces it. An honest UNKNOWN always beats a
flattering guess.
