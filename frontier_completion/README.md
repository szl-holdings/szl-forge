# frontier_completion 0.1.0 — offline helper package

Standard-library helpers from the 2026-09-11 frontier completion handoff.
They accelerate local plan, scan, receipt, and scoring work inside this
repository. They are **not** a second SZL service, an installed production
model, a scheduler, or an execution authorizer.

Compare useful functions with the existing owner modules
(`inference/hf_frontier.py`, `inference/hf_tooling.py`,
`tools/evaluate_hf_tooling.py`) before adopting any helper as a call site.

## Bounds

- Production disposition remains **HOLD**.
- SHA-256 fields are UTF-8 byte integrity aids, **not signatures** and
  **not permission**.
- `plan` is offline. `scan` / `releases` require explicit `--live` and the
  exact executor `--source` SHA of the committed code being run. They never
  download weights or execute remote code.
- `plan_preflight` validates structure only. `schedulerMayLaunch` stays false.
- The quantization canary is a strict equal-byte screen, not a Pareto solver.
- A first scan is baseline capture, not evidence of a new release.

## Run

From the repository root (Python 3.11+, stdlib only). `tests/conftest.py`
already puts the repo root on `sys.path`:

```bash
python -m unittest tests.test_frontier_completion -v
python -m frontier_completion plan
```

Explicit network use, only after review, with the real committed executor:

```bash
python -m frontier_completion scan --live --source "$(git rev-parse HEAD)" --output /tmp/szl-scan-first.json
python -m frontier_completion scan --live --source "$(git rev-parse HEAD)" --previous /tmp/szl-scan-first.json --output /tmp/szl-scan-next.json
python -m frontier_completion releases --live --source "$(git rev-parse HEAD)" --output /tmp/szl-release-source-audit.json
```

Do not stamp an unrelated staging directory's Git identity as Forge source.

## Layout

This package lives at repo root as `frontier_completion/` so it does not
collide with the existing flat `tools/*.py` modules (`tools` is not a
Python package).

`work_queue.json` is the 27-item remaining-work queue from the handoff.
Owner assignments must be re-read from current repositories. An empty
evidence array is not an approved run.

## What this change does not do

- No model download, paid Jobs, training, or production deployment.
- No Hub / TRL / Tau product-runtime upgrade.
- No Lyte canonical republish.
- No merge to `main` without review.

File hashes of the original handoff inventory are recorded on the PR. The
committed tree drops unused imports required by Forge ruff F401 (E9/F gate).
Those remaining hashes establish byte integrity of the committed files, not
author trust.
