# SZL estate payload v4 (owner-metal, one-paste)

`szl_payload_v4.py` is the owner's local, one-paste estate operator. It runs once on the owner's
Windows machine through `SZL-PAYLOAD-V4.ps1` (Administrator PowerShell 5.1), uses the owner's already
authenticated `gh` CLI, and exits. Stdlib only; `huggingface_hub` is imported lazily and only when
Hugging Face quarantine PRs are explicitly enabled.

It is **not** another publisher, scheduler, merge queue, registry authority, or promotion certificate.
The reviewed single-PR protected merge path remains `tools/szl_frontier_operator.py guard-merge`; the
A11oy release train remains `tools/szl_estate_operator.py`. This payload composes read-only discovery
with a small set of deterministic, ledgered, reversible writes and writes briefs for everything else.

## Lanes

| Lane | Reads | Writes (only when enabled) |
|---|---|---|
| discover | Hub org (models, datasets, Spaces), GitHub org (trees, READMEs), this repo's `publishing/model-source-bindings.json`, `portfolio/model_portfolio.json`, `frontier/*/candidate.json`, skip receipts, identity bindings | `estate_raw.json` |
| analyze | everything above | per-model source map: `HAS_TRAINER`, `FRONTIER_CANDIDATE_EXISTS`, `SKIPPED_BY_RECEIPT`, `REFERENCE_ARTIFACT_NO_SFT`, `SOFTWARE_NO_SFT`, `QUANT_DERIVATIVE_PARITY_ONLY`, `NON_LLM_OR_UNDEFINED_ARTIFACT`, `GENERATE_CANDIDATE` |
| quarantine | hash-chained ledger | Hub PRs adding `QUARANTINE.json` + README banner (`SZL_EXECUTE=1`); never direct commits, never deletes |
| pull_requests | open PRs in every active repo | merges only non-draft, non-HOLD, CLEAN/HAS_HOOKS PRs with >=1 check and zero failed/pending, after a live re-check locked to the head SHA (`SZL_MERGE=1`); BEHIND branches get `gh pr update-branch` |
| pr_repair | failing/conflicting/protection-blocked PRs, failed job logs | flaky reruns, `@dependabot rebase/recreate`, repository-formatter commits, base-merge pushes (`SZL_REPAIR=1`); a redacted repair brief for everything that needs code judgment; intentional red proof gates are preserved |
| corpora | local `~/szl-*/**/*.jsonl` | `local_corpora.json` (record-shape validated with the kit library) |
| candidates | `GENERATE_CANDIDATE` models | one complete fail-closed candidate kit per model (see `kit/`), one PR to this repo (`SZL_OPEN_TRAINING_PR=1`) |
| train | bound candidates, a qualified local venv | qualify -> curriculum/leakage -> train (`smoke` or `full`) -> evaluate -> export -> card, all on owner metal |
| reports | everything | `SCORECARD.md`, `CODEX-v4.md`, `CODEX-PR-REPAIRS.md`, `OWNER-GPU-RUNBOOK.md`, CSVs, `summary.json` |

## The candidate kit (`kit/`)

A candidate folder is self-contained and lands under `frontier/<slug>-candidate-<date>/`:

- `candidate.json` - `szl.frontier-model-candidate/v2` shape: exact 40-hex base pin, predecessor as frozen comparator,
  new target id (the predecessor is never overwritten), rights boundary, recipe, frozen `gates_sha256`.
- `qualify_runtime.py` - environment, exact base pin, free VRAM, runtime lock (frozen on first PASS; drift fails).
- `curriculum.py` - digest-bound curriculum, split derivation with family-key isolation, leakage gate (exact, 5-gram
  Jaccard, family key, template leak).
- `train_candidate.py` - QLoRA (Unsloth when importable, transformers+peft fallback), response-only loss,
  text-only tokenizer path, `--smoke` (1 step) vs `--full` (needs a confirmed binding). Unsigned report.
- `evaluate_candidate.py` - greedy frozen evaluation, twelve-gate derivation, never writes `PROMOTABLE`.
- `export_gguf.py` - merge, GGUF, `QUANT_MANIFEST.json`, F16-vs-quant parity receipt.
- `render_card.py` - card from reports only; G12 card truth.
- `test_candidate_contract.py` - stdlib + pytest contract tests.

Reports are unsigned; sign with the owner's existing canonical-JSON signer. Publication and promotion stay
separate decisions outside this tooling.

## Run

Paste `SZL-PAYLOAD-V4.ps1` into an Administrator PowerShell 5.1 window, or:

```powershell
python -I -B tools/estate_payload/szl_payload_v4.py
```

Environment switches: `SZL_MERGE`, `SZL_REPAIR`, `SZL_COMMENT_PRS`, `SZL_OPEN_TRAINING_PR`, `SZL_TRAIN_MODE`
(`off|smoke|full`), `SZL_CONFIRM_BINDINGS`, `SZL_EXECUTE`, `SZL_CLONE`, `SZL_TRAIN_PY`, `SZL_LLAMA_CPP`,
`SZL_CORPUS_ROOTS`, `SZL_ROOT`. Everything defaults to dry-run except discovery and local report writing.

Tests: `python -m pytest -q tests/test_estate_payload.py`.
