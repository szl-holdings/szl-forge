# Owner candidate reconciliation: verify the loss boundary before training

Observed source: `b18f2604c309710847158443a907f9a20be9e95a` on 2026-10-04.
This is a dated source review, not current machine or model qualification.

## Concrete source repair

The shared kit and the four 20261001 candidate trainers used the length of a
separately rendered generation prompt as the loss-mask boundary without comparing
its token IDs to the beginning of the complete conversation. A different generation
prefix could therefore assign assistant loss to the wrong suffix. Truncation could
also remove all target tokens and silently drop the example.

`response_only_labels` now verifies a nonempty exact token prefix before masking,
requires a retained assistant suffix, rejects invalid IDs and lengths, preserves
valid token bytes and existing suffix truncation accounting, and emits fixed
content-free errors. Tensor construction explicitly uses integer dtype. A template
that does not satisfy this prefix contract needs its own qualified alignment path;
this repair does not guess one or rewrite the template.

The fix covers the kit plus WILLAY, Brain Navigator r2, triage, and study5 generated
trainers. Their manifests, frozen gates, data, base revisions, target identities,
optimizer settings, runtime and publication requirements remain unchanged.
The rest of each trainer is preserved, including differences between the kit and
historically generated copies. This is not a broad kit migration.

## Reproduction and tests

The exact original four-candidate trainer blob was
`f9e7f7e549719aa2f022209af127d19e63ff9f40`; the kit blob was
`09a01dd50a8af753579856ba63fcffaa2af3a826`. Reconstructed bytes matched those Git
identities before testing. Synthetic full IDs `[10,11,12,13]` and unrelated prompt
IDs `[90,91]` were incorrectly accepted as labels `[-100,-100,12,13]` before repair.
The repaired implementation rejects with `MASK_PROMPT_PREFIX_MISMATCH`.

Sixteen unique standard-library unittest methods exercise all five source owners,
including positive/negative prefix cases, incomplete targets, truncation, invalid
types, unchanged inputs, content-free diagnostics, mocked tensor construction, and
100 seeded positive cases per owner. Local Python 3.13.5 passed normally and under
optimized Python. These repeated cases are not additional distinct test methods.
An additional actual PyTorch 2.10.0+cpu execution reproduced the old defect and
verified the corrected positive mask and int64 tensors. No model was loaded and
zero optimizer steps ran. Python 3.11 grammar checks passed; hosted execution and
owner GPU qualification are separate evidence.

Run the focused suite without importing a candidate or model:

```text
python -I -B -m unittest discover -s tests -p test_candidate_prompt_mask.py -v
python -I -B -O -m unittest discover -s tests -p test_candidate_prompt_mask.py -v
```

The existing base Python workflow collects this new test file; no workflow gate or
publisher is replaced. Full current-head hosted checks remain required.

## Why the October 1 runbook is not a completion procedure

- WILLAY remains UNBOUND with no selected rows.
- The current Brain Navigator manifest explicitly rejects its earlier heuristic
  Brain corpus selection. It is UNBOUND, not the runbook's BOUND_LOCAL/1233.
  Do not resurrect that rejected corpus or train on live memory content.
- Both triage candidates name the same five-row untagged file and remain owner
  unconfirmed. Their data identity is not two independent corpora or adequate
  release evaluation. The established TypeSafe Triage source separately contains
  a 515-train/113-held family split; inspect its frozen protocol and rights rather
  than substituting it automatically or absorbing its holdout into gradients.
- The generated evaluation program still declares the policy-compliance gate
  unavailable. Other gates require comparator, adversarial, quantization and card
  evidence. Completing the listed commands cannot establish all twelve gates.
- Smoke and full modes write the same default adapter/report paths. Use distinct
  approved attempt directories through the existing SZL_CANDIDATE_OUT seam and
  freshly bound prerequisite reports; never overwrite smoke or failed evidence.
- The template's optimizer fallback and the generated trainers differ. An
  exception after partially updating a model cannot be treated as a clean restart.
  Review that separate attempt-identity/recovery contract before full execution.
- In Windows PowerShell 5.1 each native command's exit code needs checking; a bare
  newline command chain must not proceed from failed qualification into training,
  evaluation or export. A zero evaluator exit also does not mean its gates passed.

## Parallel work without contending for the GPU

Run source/manifest comparison, local path/hash checks and software tests in
bounded independent CPU workers. Do not create four simultaneous CUDA training
processes on one GPU. An active inference service is not authorization to stop it
or consume its memory. The October 4 independent host-memory guard source is
admitted separately; issue #546 still requires owner Windows/WSL fault controls.
Its dummy-worker tests and hardware acceptance cannot be replaced by a one-time
nvidia-smi sample. Preserve the GPU cadence, sticky abort and exact-worker stop
contracts; never shut down WSL wholesale to make a model fit.

Next admissible steps: source review and normal merge; reconcile local generated
copies without overwriting them; establish approved task-specific train/dev/final
bindings; qualify the exact tokenizer and worker under the coupled host/GPU guard;
run one bounded smoke per admitted device; independently review smoke before a
separate full attempt; then evaluate, export/reload, sign through the existing
signer, canonically publish and read back. Training, publication and promotion
remain distinct. No threshold or status is edited to manufacture eligibility.
