# Early-stop integration handoff — 2026-10-27

Evidence class: **MEASURED** for the source test result; **ROADMAP** for the integration sequence. This note is a bounded handoff for the existing VM2 training owner. It does not launch training, signal a process, modify the VM, grant budget, or confer publication authority.

## Verified source

The standalone source proposal is under `early-stop-proposal/`:

- `early_stop.py` SHA-256 `4A72A22B5B6539A8CBDD438A79C3420AB6EB68BBDAA148C359ECA30D8C4F1B12`
- `test_early_stop.py` SHA-256 `467C3AE549FFA39736921F4509BBC484203DC2E965EB5117333013C55BA5BE63`
- `README.md` SHA-256 `87709C58DEE845D7EDE93ED7F5232B6E6895BFEFE1D8BAB98F1BC23F1C47A4BE`

Root independently ran, from that directory:

```text
python -I -S -B -c "import sys,unittest; sys.path.insert(0,'.'); import test_early_stop; r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_early_stop)); raise SystemExit(not r.wasSuccessful())"
```

Result: **15/15 passed in 0.009 seconds**. The fixtures are synthetic. This does not qualify a model or establish training admission.

## Required owner integration sequence

1. **Freeze a new candidate/run/output identity.** Do not reuse `/home/Ubuntu/nemotron-50h-20261025` or its checkpoint directory. Bind the candidate ID, run ID, exact development-data SHA-256, evaluation interval, `direction=min`, `min_delta`, and patience into the run receipt before work begins.
2. **Keep final held-out refusal/abstention data outside the development set.** The helper accepts only the development loss stream. A final evaluation must be separately frozen and run only after training selection.
3. **At every completed optimizer boundary**, call `observe(...)` with the exact scheduled step and a complete checkpoint identity. The checkpoint hash should identify the complete recovery manifest, not only adapter bytes.
4. **Persist the early-stop state with the full recovery state.** The existing `recovery_state.save_checkpoint(...)` writes model state, optimizer, scheduler/scaler, module modes, Python/Torch/CUDA RNG and cursor. Add the canonical bytes from `early_stop.canonical_bytes(state)` and its hash to the same completed checkpoint manifest, or bind them as an additional immutable file before the manifest completion marker. The early-stop state must be restored together with optimizer/RNG/cursor; adapter-only resume is not exact.
5. **Restore fail-closed.** Supply the independently retained latest state hash to `early_stop.restore(...)`, and use the existing trusted recovery-manifest pin for `load_checkpoint(...)`. Reject any candidate/config/data drift, hash mismatch, stale/old state pin, cadence mismatch, noncanonical JSON, or stop-latched state that receives another metric.
6. **Honor the result in the external supervisor.** When `state.stopped` is true with reason `PATIENCE_EXHAUSTED`, stop the exact worker tree and capture a process identity receipt. The helper cannot interrupt a blocked worker.
7. **Close the budget guard only after stop is independently confirmed.** The existing `CampaignBudget.guard(...).__exit__` retains an active record unless `confirm_stopped()` returns true; do not report a clean budget close from an unconfirmed or changed process identity.
8. **Only after source, recovery, budget, held-out eval, and authority gates pass** may a separate owner-controlled deployment/promotion lane consider the candidate. Keep `training_admitted` and `publication_eligible` false until those gates have real receipts.

## Current boundary

The latest local VM readback (`vm-a6000-readback-20261025/snapshot-04.json`, SHA-256 `652ef41f00303aae838266fefbb220c99b1a8026d7994b5793c362b3431c7927`) measured a running GPU PID 40894 using the unchanged original trainer and reused checkpoint root. This chat did not launch or stop it. The existing owner remains the sole coordinator. The owner must establish the exact process/start identity, preservation, and no-restart window before applying this handoff.

The handoff is **NOT INTEGRATED**, **NOT DEPLOYED**, and **NOT A TRAINING APPROVAL** until the owner applies it in the controlled runner and returns exact receipts.

