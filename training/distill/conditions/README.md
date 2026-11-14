# Distillation condition scripts (rounds 3 and 4), as run

These two files are **byte-identical copies** of the scripts that produced the
round-3 and round-4 teacher-gate receipts on Nov 7, 2026. Until this commit they
existed only as untracked files in `~/szl-forge/training/distill/` on the
training node (.145). They are committed here unchanged, so their hashes point
at an immutable revision. Do not edit them; a changed file is a different
script.

| File | Condition | sha256 |
|---|---|---|
| `distill_wave_ga.py` | Round 3, rule stated in a system prompt ("gate-assisted"), unconstrained decoding | `e749c75a89f6d97e8aba4a71ae2fb3917f622355d3915f59714e0a1dbd74fbb3` |
| `distill_wave_cd.py` | Round 4, copy-constrained decoding (`CopyConstrainedLogitsProcessor`) | `3f58335ea03f9d1e870fd6fd1b2dee990d822d9ef85d9ee1a060e9767f6bde79` |

Provenance: copied read-only from .145 on 2026-11-13; `sha256sum` was run on the
node and on the copy, and both match the values above.

Methods facts that apply to both scripts:
- **Gate.** Both embed the *original* grounding gate (`szl.distill.gate/v1`),
  whose code block is identical to `training/distill/distill_wave.py` at
  commit `ab0f1dd09adaf2196aedb550f8c2822cb42d39f6`. They do **not** include
  the later gate revisions on `main` (termination check, fence handling,
  dotted identifiers, generation prompt).
- **Context.** The source assistant turn is not stripped and no generation
  prompt is added, so the teacher continues a conversation that already holds
  the reference answer. Under copy-constraint the allowed tokens are those of
  the prompt plus the reference answer (plus EOS).
- **Defaults.** seed 3407, temperature 0.7, max_new_tokens 512. Invocation
  arguments were not logged, so these are the scripts' defaults, not proven
  run values.

Results these scripts produced (corpus acceptance, not a probe score; one run
each): round 3 accepted 2/150; round 4 pilot 31/150; round 4 extended to 210
prompts 46/210 (the first 150 outputs are the pilot's, regenerated under the
same seed; prompts 151-210 gave 15/60). The baseline (round 2, no instruction) was 4/150.
See the superseding trace on
`SZLHOLDINGS/frontier-54-runs` for receipt hashes.
