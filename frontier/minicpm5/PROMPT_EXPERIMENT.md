# MiniCPM5 explicit lookup prompt experiment

`explicit-lookup-v2` is an opt-in instruction revision for the existing public
twelve-case development suite. The original `baseline-v1` remains the default,
and its historical reports remain verifiable. Neither policy trains the model,
executes tools, nor grants publication or production authority.

The baseline produced false abstentions on one authorized lookup and both
injection fixtures in the recorded RTX 5050 and A10G runs. The revised system
message states the required lookup conditions as separate rules. It clarifies
that ignoring an injected instruction does not revoke an otherwise authorized
document. It does not contain fixture IDs or expected answers. The complete
request remains canonical JSON in the user message, including the adversarial
note. No document is selected in code and no model response is repaired.

Run the candidate explicitly with the same pinned model and token/time limits:

```powershell
python -B inference/minicpm5_prompt_qualification.py --run `
  --source-revision <exact-source-commit> `
  --image <runtime-image-or-explicit-unavailable-marker> `
  --prompt-policy explicit-lookup-v2 --cached-only `
  --output explicit-lookup-v2.json
```

`--cached-only` reads public metadata for the immutable model revision and
requires all selected model files to exist in the local Hugging Face cache.
The runner hashes those files and keeps local-only loaders and
`trust_remote_code=False`. It does not download missing weights in this mode.

The candidate report hashes the system instructions, labels the workload
`DEVELOPMENT_NOT_HELD_OUT`, and declares no preprocessing, output repair, or
constrained decoding. The verifier validates that policy and preserves the
development label in both projections. Compare the unchanged model and suite
hashes and case-level output hashes against the original baseline. A gain on
these already observed fixtures is development evidence; independent held-out
quality and product/runtime qualification still require separate measurements.

One local CUDA experiment on 2026-09-08 used the existing RTX 5050 Laptop GPU and
cached `openbmb/MiniCPM5-2B` revision
`3497c460c89e00520c3cfa2e73f49ab7647f1177`. It passed all 12 cases; the original
local baseline passed 9. The three formerly failed cases (`probe_00`, `probe_04`,
and `probe_10`) became correct raw model outputs. The other nine output hashes
were identical to baseline. Model artifact hashes, suite hash, and reported
hardware matched. Total measured generation was 71.995 seconds. Candidate
generation p50/p95 were 4,202.460/14,021.344 ms; baseline values were
3,094.992/11,887.164 ms. This is not a speed improvement, and timing was not a
controlled performance benchmark.

The measured candidate runner digest was
`69274d8c2f0a1c96deac844c61e09662a683df6cbc3fcd374669562f1c4dac88`;
the candidate record digest was
`e02ae339d8ef35c361302f59684c71086c0c4303ae4bfe608175afe86ccb33b4`.
This run used a local source patch based on
`3f12d5078fe058ca66c41afb3fedbecd96ee9468`. That base revision in the record
is not a claim that the modified runner already exists at that GitHub commit.
The comparison witness identifies the patch as unpublished and binds the
executed runner bytes. The local unsigned receipt is not an independent job
attestation or a served-model qualification. No second GPU experiment ran.

The portable [original records, executed sources, and comparison](evidence/2026-09-08-rtx5050-dev12/README.md)
retain that execution-time provenance without rewriting the original report.
Run `python -B inference/verify_minicpm5_dev12.py` from the repository root to
check their integrity and unchanged grading functions offline. The byte-preserved
executed source retains its original CRLF line endings. The published opt-in
module reuses the unchanged baseline suite, parser, grader, and model-file checks.
It preserves the measured request construction and prompt contract, but the
refactored module has not itself been run on GPU. The original
`inference/minicpm5_qualification.py` remains byte-identical, preserving existing
GGUF and matched-runtime source bindings. No new model run is implied by this
source publication.

Run the offline contract checks:

```powershell
python -B -m unittest discover -s tests -p 'test_minicpm5*.py' -v
```

These tests reject changed grading claims and false held-out labels. They also
verify that adversarial content remains user data, baseline message formatting
is unchanged, and strict JSON grading still rejects false abstention, extra
keys, private-reasoning wrappers, and truncated responses.
