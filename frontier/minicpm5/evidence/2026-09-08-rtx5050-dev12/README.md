# Local RTX 5050 development comparison

The original baseline and candidate receipts are retained byte for byte. On the
same public twelve-case development suite, the baseline passed 9/12 and the
explicit lookup candidate passed 12/12. No held-out result, training, serving
deployment, independent witness, or production promotion is established here.

The candidate ran once on September 8, 2026. Its recorded `sourceRevision`
`3f12d5078fe058ca66c41afb3fedbecd96ee9468` identifies the base of an unpublished
working-tree patch. It does **not** identify a commit containing that candidate.
`source/candidate/inference/minicpm5_qualification.py` preserves the actual
executed bytes and their SHA-256
`69274d8c2f0a1c96deac844c61e09662a683df6cbc3fcd374669562f1c4dac88`.
The historical comparison's `sourcePatchUnpublished: true` remains true for the
time of execution even when these artifacts are later committed for review.

`source/baseline/` preserves the baseline runner and verifier. The original
four-file patch records the prompt change. The source publication adds a separate
`inference/minicpm5_prompt_qualification.py` module and preserves the original
baseline runner byte for byte. The new module reuses the original grader, suite,
JSON parser, and model-file verification functions. Its prompt and request
construction match the archived experiment, but this refactored module has not
itself been measured on GPU. The offline verifier checks those contracts and
hashes the original snapshots. It never executes the archived sources or reruns
a model. Existing GGUF and matched-runtime evidence keeps its original baseline.

The complete requests, including injection text, remained model inputs. There
was no answer injection, document preselection, constrained decoding, output
repair, tool execution, or weight change. Both runs used the same recorded model
artifact hashes and hardware. The candidate resolved the three observed false
abstentions; the other nine output hashes were unchanged. Raw model responses
were not exported, so the stored grades cannot independently be recomputed from
these receipts. The verifier checks their integrity and internal consistency;
it does not convert unsigned local records into an independent attestation.

Timing was not a controlled benchmark. Candidate p50/p95 generation latency was
4202.460/14021.344 ms, versus baseline 3094.992/11887.164 ms. These observations
do not establish a speed improvement. Energy, billed cost, TTFT, and immutable
runtime-image qualification remain unmeasured or unverified.

From the repository root, verify the portable evidence without ML dependencies,
credentials, network calls, or model execution:

```shell
python -B inference/verify_minicpm5_dev12.py
python -B -m unittest discover -s tests -p 'test_minicpm5*.py' -v
```

The manifest hashes the original records and executed source snapshots. Its
integrity is repository evidence, not a signature. Both records and projections
retain `productionDisposition: HOLD`; the prompt policy remains opt-in.
