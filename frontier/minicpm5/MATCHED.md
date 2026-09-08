# MiniCPM5 same-worker F16/Q4 comparison

This follows merged Forge #190 and Frontier #39. The older handoff ZIP must not
replace either implementation. Forge owns this experiment. The existing native
GGUF evaluator, original twelve public synthetic cases, grader and historical
negative records are unchanged.

## The question

The historical float16/A10G and Q4/CPU runs each passed 9/12, but failed different
cases. Equal totals are not parity. This runner executes F16 then Q4_K_M on one
CPU worker, with the same pinned native archive/binary, evaluator, request bodies,
context, threads, sampling and token limits. It records per-case output/verdict
changes, not just aggregate scores.

## Bounded execution

Default `python -m inference.minicpm5_matched` prints a plan. No PR CI job starts
compute or downloads model weights. An explicitly authorized provider job runs:

```bash
python -m inference.minicpm5_matched run \
  --source-revision FULL_REVIEWED_GITHUB_SHA --output /tmp/matched.json
python -m inference.minicpm5_matched verify --record /tmp/matched.json
```

The enclosing job must have a 900-second hard timeout, no secrets, no exposed
ports, and no automatic retries. The controller uses a 780-second shared budget.
Two owned ephemeral native runs are sequential, each reusing the admitted
loopback-only authenticated evaluator and exact model/archive byte verification.
Raw model text/reasoning is not exported. Reports are emitted as base64 JSON with
nested content digests for lossless retrieval through Jobs logs. Model files are
not uploaded to any repository.

## Meaning of the result

EXECUTED means two complete, source-bound observations from the same worker,
not a passing model. The job exits 2 for any synthetic quality failure or output
hash difference; such evidence is retained rather than retried into green.
The verifier recomputes summaries, requires full source/variant/native identities,
validates nested metadata, rejects cross-job/worker substitutions and preserves
historical negative evidence. Fixtures are explicitly OFFLINE_FIXTURE, never live.

This is one ordered, cold-start-including synthetic comparison. Embedded template,
tokenizer and conversion lineage remain unproven. Same input-token counts would
not establish identical token sequences. No causal quantization loss, statistical
speedup, held-out quality, long-context quality, native tool parsing or production
readiness follows. All records remain unsigned, runtimeQualified=false,
modelOperational=false, productionDisposition=HOLD. Hashes prove record integrity,
not independent execution certification.

## Owners and follow-through

GitHub source -> Hugging Face execution evidence -> a-11-oy.com product state ->
a11oy.net proof/limits. Forge evaluates, Nemo witnesses and Serve validates recipes,
receipts and fallback around the existing studio. No new flagship or public model
listener. Codex must retain both original 9/12 failures and this new independently
identified experiment. Once admitted, Frontier may record the new evidence chain;
product/proof publishers must not infer LIVE from an EXECUTED experiment.
