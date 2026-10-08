# Preregistration amendments (continuation of PREREGISTRATION.md section 12)

The registered file `v2/PREREGISTRATION.md` stays byte-identical to its registration commit
`bca9225` (sha256 `145497789c7ad78a80a94cfd7f2b031ca9a20df49c06e0b3412af9070fa1ef4b`). The v2
kernel and `v2/research/registry.py` pin that hash. Amendments therefore live in this
append-only file rather than inside section 12. This is a deviation in form only, and REPORT.md
discloses it.

Every entry below was committed **before** any hyperparameter search, before any model was fit
on the registered train split, and before any sealed split was opened. The git history of this
file is the evidence. Decisions were made by the orchestrator under Stephen's standing
instruction to make judgment calls and hand back a green light
(`decide-then-green-light`).

Decided 2026-09-29, before stage 2. None of these changes the question, the primary metric, the
win condition, the splits, the seeds, the search space, or the shift regimes.

## AM-1 (answers AR-1): the upstream citation stays as-is

`v2/PREREGISTRATION.receipt.json` cites v1's generator path in `szl-forge@9a6fdd2c`, and that
path contains the word banned for v2 names. It is a provenance pointer to v1's source, not a v2
path, file, module, artifact, or card. Rewriting a registration record would weaken the evidence
more than the word harms it. The whitelist in `ForbiddenWordTest` stays limited to that one line.

## AM-2 (answers AR-2): non-converged fits cannot be selected

An IRLS fit that stops with `LINE_SEARCH_FAILED` is treated exactly like `CAP_REACHED`: it has
not converged. Such a trial is still run, receipted, and reported, but it is **ineligible for
selection**. If all 12 trials are ineligible, v2 does not exist and REPORT.md says so. The same
rule applies to the Platt calibrator fit.

## AM-3 (answers AR-3): how the "tampered metrics" mutant is realized

The receipt carries no metrics by design, because receipts attest integrity and origin, not
quality. The section 10 mutant "tampered metrics" is realized as **a metrics block added to the
receipt**. The integrity layer must refuse it. It is reported with the other receipt mutants.

## AM-4 (clarifies section 11): what a known-good retrain is

A known-good retrain re-runs the **selected recipe** end to end: the selected l2, IRLS, the
selected calibrator fit on the registered calibration split, the threshold re-selected on the
registered validation split by the section 6 rule, and Mondrian conformal on the registered
conformal split. Only the train split changes. For k = 1..5 it is drawn from the nominal regime
with 16,384 rows, using seed material `oac-ops-health-v2:{20260926+k}:train` and the section 3
derivation. These train splits are not sealed, and their hashes are recorded in the stage-2
results before any gate is evaluated.

## AM-5 (clarifies section 7): what "v1 as-is" means for Brier and ECE

For calibration comparisons, v1's score is the exact `operator_attention_score` emitted by the
Hub kernel at revision `dd7d1098…`. No recalibration is applied. v1's own card calls this score
"not production calibrated"; the W3 comparison applies to it unchanged.

## AM-6 (records a stage-1 fact): v2 refuses duplicate JSON keys, v1 accepts them

The v1 audit MEASURED that the v1 kernel accepts duplicate JSON keys (last value wins). The v2
kernel refuses them. The fail-closed probe suites of both contenders therefore differ by
construction. Section 10's `g_failclosed` is computed on v2's own suite, and v1's result on the
same probes is reported descriptively. This does not affect the win condition.
