# eval/ — held-out generation + refusal probes (model-publish-gate, Gate 2)

- `heldout_generate.yaml` — the held-out generation eval set. These prompts
  must be withheld from every training split. `eval/run_heldout.py` fails
  closed (exit 1) if this file is missing.
- `run_heldout.py` --config <yaml> --model <dir> [--results <json>] \
    [--assert-min-pass-rate 0.80] --out release/heldout-eval.json
  Modes: score a pre-computed results file, or run a local transformers model
  CPU-only. If neither is possible the gate fails closed.
- `run_refusal.py` --model <dir> --assert-no-regression — compares against
  a pre-existing reviewed `baselines/refusal.json`. Assertion mode never creates
  or updates a baseline. Missing, incomplete, inconsistent, oversized or invalid
  evidence fails before generation. Tolerance must be finite and in [0, 1].
- `baselines/` — committed refusal baselines.

## Refusal comparison evidence

The baseline must contain the complete current probe-ID set, strict boolean
outcomes, an integer `probe_count`, and a finite `refusal_rate` agreeing with
those outcomes at the report's six-decimal precision. At most 64 KiB of baseline
JSON is accepted. Duplicate keys and nonfinite JSON are rejected. The supplied
baseline byte digest is printed for content identification, not as a signature
or an independent source/review attestation.

A first evaluation without `--assert-no-regression` remains report-only and
writes no baseline. A new comparator must be established and reviewed separately;
never bootstrap it from the candidate during a publication gate. Do not replace
committed historical baselines merely to obtain a pass.

These refusal results are lexical-marker proxies, not verified semantic safety
judgments. This input-contract repair changes no probe prompt, marker, generation
method, model staging, held-out split, weight, or existing baseline byte. Raw
prompt versus model-native chat-template evaluation and semantic refusal review
remain independent methodology requirements. No fresh model evaluation or
production authorization follows from passing the regression tests below.

```bash
python -m unittest discover -s tests -p 'test_refusal_assertion_baseline.py' -v
python -O -m unittest discover -s tests -p 'test_refusal_assertion_baseline.py' -v
```
