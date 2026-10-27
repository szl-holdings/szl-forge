# Development-loss early-stop source proposal

Evidence class: SIMULATED for fixture trajectories. Training admission and
publication eligibility remain false. This helper has no model, process, network,
filesystem persistence, authorization, signature, budget, or training API.

## Frozen contract

`Config` binds candidate ID, run ID, development-dataset SHA-256, evaluation
interval, `direction=min`, a nonnegative decimal `min_delta`, and positive integer
patience. The first evaluation occurs at one interval; every following step is
exactly one interval later. The caller must freeze this schedule before the run.
This proposal does not silently accommodate historical step-1 checkpoint schemes.

Metrics are bounded plain decimal strings. Floats, exponents, NaN, infinity,
booleans and malformed numbers are rejected. Serialization removes trailing
fractional zeros and turns negative zero into `0`. Comparisons use explicit Decimal
precision, independent of ambient precision. An improvement must exceed min_delta
strictly; a tie or improvement exactly equal to min_delta consumes patience.

State retains last evaluation, best qualifying metric, exact checkpoint ID/hash,
best step, failed-improvement count, and stop reason. "Best" means best qualifying
improvement under the frozen delta, not every numerically lower value. When
patience is exhausted, the stop latch prevents further observations. Resume
requires canonical JSON bytes, the identical frozen configuration, and a latest
state hash retained independently by the caller.

## Integration requirements and limits

- The source is separate from the original trainer and the existing recovery and
  campaign-budget helpers. It has not been integrated, deployed, or run on the VM.
- Persist these bytes atomically with the completed optimizer checkpoint and
  bind their hash into the owner's recovery manifest. Patience/best identity are
  mutable training state and must resume with optimizer/RNG/cursor state.
- The checkpoint hash is a caller-supplied content identity, preferably the
  complete recovery-manifest digest. This helper neither reads those bytes nor
  proves they exist, are durable, or match a model. The caller verifies them.
- Checkpoint identity and metrics are trusted inputs. The module does not prove
  loss correctness, checkpoint freshness, training provenance, or model quality.
- The external latest-state pin must come from the authorized owner and an
  anti-rollback store. Deriving it from the candidate saved state defeats the
  integrity check. A caller accepting an old pin can replay an old state; no pure
  local-state helper can establish freshness independently.
- Initial state is for a fresh candidate only. It cannot reconstruct historical
  patience or create an exact resume from old adapter-only checkpoints.
- An external trainer/supervisor must actually honor the stop result. This helper
  cannot interrupt a blocked process, stop VM charges, replace campaign budgets,
  confer sole-writer authority, or resolve the unexplained running process.
- Development losses are selection data. No held-out refusal/abstention examples
  or final evaluation content are included in this package.

Only new synthetic tests should be run for this proposal. From this directory:

```text
python -I -S -B -c "import sys,unittest; sys.path.insert(0,'.'); import test_early_stop; r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_early_stop)); raise SystemExit(not r.wasSuccessful())"
```

No state files are written by that suite. Its synthetic losses and checkpoint
hashes do not describe a trained model or a real evaluation.

