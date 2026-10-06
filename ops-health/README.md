# OAC Ops Health v2 (`ops-health/`)

**Everything in this directory is SYNTHETIC.** OAC Ops Health v2 is a small logistic-regression
advisory for operational transport counters. Its kernel, `v2/ops-health/ops_health.py`, runs on
the Python standard library alone. It reads eight synthetic counters and returns `ALERT`,
`NO_ALERT` or `ABSTAIN` as a signal for operator attention. It is **advisory only**: its output
authority map is all-false, and it fails closed on missing, extra, non-finite, out-of-range or
prohibited input fields. It was trained, selected and tested only on generated data, so no
number here says anything about a real transport, device, site, population or workflow.

v2 was built under a protocol registered before any v2 code or data existed
(`v2/PREREGISTRATION.md`, frozen; amendments in `v2/AMENDMENTS.md`), and its sealed test splits
were opened exactly once (`runs/TEST_OPENED.json`). The verdict and every reported number are in
`v2/results/FINAL_RESULTS.md` and `v2/results/final_evaluation.json`.

This directory publishes nothing. No workflow writes any of it to Hugging Face.

## Where it comes from

The files under `v2/`, `runs/`, `math/` and `logs/` come from SZL's research repository, which
is not published. They are byte copies with one exception: 40 logs under `logs/v2/` that no hash
binds had their CR bytes turned into LF (`"transform": "eol_lf"` in `PROVENANCE.json`). In three
of them, `logs/v2/review/kernel_attack_*.txt`, lone CRs also became line breaks, so they have more
lines than the research copies. Every hash-bound file under these four directories is
byte-identical to its research copy. `PROVENANCE.md` and `PROVENANCE.json` give the research
commit and git blob of every file, the research copy's SHA-256 where the landed bytes differ, and
the rule for every SHA-256 value that a landed file binds.

**The origin layer is UNAVAILABLE here.** The frozen receipt
(`v2/ops-health/artifact_receipt.json`) records `source.commit` `92872f88…`, a commit of that
research repository. `freeze.verify_origin` resolves it with `git`, which only that repository can
do. So seven tests that need it skip here with the reason "receipt source commit 92872f88 is in
the unpublished research repository; origin layer UNAVAILABLE here". The kernel's own checks (its
self-hash and the model hash against the receipt) still run.

## Layout

```
ops-health/
  README.md, PROVENANCE.md, PROVENANCE.json   this file; where every file and bound hash comes from
  .gitattributes                              "* text eol=lf": the hash bindings are over LF bytes
  .gitignore                                  generated files that are never committed (below)
  v2/                PREREGISTRATION.md (frozen), AMENDMENTS.md, reports and developer notes
    ops-health/      the frozen kernel, model.json, artifact_receipt.json, example_input.json
    research/        the stdlib-only research code (imported as v2.research.*)
    tests/           the unit suite (DEV data only)
    data/            MANIFEST.json and the byte-exact copies of v1's public splits (legacy_v1/)
    results/         reports, search and mutation records, contenders, known-good retrains,
                     and the per-row score files of the single opening (results/final/)
  runs/              TEST_OPENED.json and the governed search receipts
  math/              the aggregator and the receipt and search adapters the research code loads
  logs/v2/, logs/drafts/   the raw outputs the reports and the draft card cite
  provenance/final_analyze.cbac497.py   the analysis-time script, as an LF copy (not importable)
  tools/             regenerate_data.py, verify_provenance.py, materialize_v1_hub.py, run_v2_tests.py
```

**Not committed, and regenerated instead:** the nine registered data splits
(`v2/data/*.jsonl` and `v2/data/sealed/*.jsonl`, 48 MB) and the five known-good train splits
(`v2/results/known_good/k*/train.jsonl`, 31 MB). `tools/regenerate_data.py` rebuilds each one with
the frozen generator and compares its SHA-256 and size with `v2/data/MANIFEST.json` and
`v2/results/known_good/INDEX.json`. It hashes bytes only and scores nothing.

**Materialized, not committed:** `hub/oac-v1/`, v1's six published files. They are already staged
in this repository (the one directory matching `*/huggingface/model/oac-system-health-v1/`), and
`tools/materialize_v1_hub.py` copies them after checking the hashes v2's records bind.

## How to check it

From the repository root, with Python 3.11 or later and no third-party packages:

```bash
python -B ops-health/tools/verify_provenance.py      # every landed file and every bound hash
python -B ops-health/tools/regenerate_data.py --verify-against ops-health/v2/data/MANIFEST.json
(cd ops-health && python -B -m v2.research.legacy_v1_source --verify)
python -B ops-health/tools/materialize_v1_hub.py     # hub/oac-v1/ from v1's staged files
python -I -B ops-health/v2/ops-health/ops_health.py --model ops-health/v2/ops-health/model.json \
  --receipt ops-health/v2/ops-health/artifact_receipt.json --input ops-health/v2/ops-health/example_input.json
python -B ops-health/tools/run_v2_tests.py           # unit suite; only the 7 origin skips allowed
```

`.github/workflows/oac-ops-health.yml` runs these steps on every change to this directory, on
Ubuntu with Python 3.11, 3.12 and 3.13 and on Windows with Python 3.12. It also checks that git
stores every file here with LF line endings. A copy with CRLF line endings is refused: the kernel
exits with code 2 because its self-hash no longer matches the receipt
(`logs/drafts/kernel_line_ending_check.txt`).

**Recompute the test-set numbers (optional, slow).** After `materialize_v1_hub.py`, run
`python -B -m v2.research.final_analyze` inside `ops-health/`. It reads only the saved score
files under `v2/results/final/`, never a data split, takes tens of minutes, and **rewrites**
`v2/results/final_evaluation.json`, `FINAL_RESULTS.md` and `reliability.svg` in place. Compare
the numbers with `git diff`: the run time, the git head and the script hash in the output differ
by design. The sealed splits cannot be opened again; the protocol allows one opening, and it has
happened.
