# Verified, non-operational product/proof projections

The evaluator emits unsigned records. A digest is an integrity check, not a
signature, proof of job execution, or permission to deploy a model.

`inference/verify_minicpm5_report.py` checks the exact expected GitHub source and
executed-file digest supplied by the verifier, the pinned model and synthetic
suite, record digest, ordered unique cases, result/reason consistency, token
bounds, numeric measurements, recomputed summaries and the immutable HOLD flags.
It rejects both accidental corruption and a rehashed false summary. The expected
source and runner hash must come from GitHub/job evidence, not from the record
being verified. Actual job identity and logs require external verification.

```bash
python inference/verify_minicpm5_report.py \
  --report /tmp/minicpm5-qualification.json \
  --expected-source <full-github-commit> \
  --expected-runner-sha256 <sha256-of-executed-runner> \
  --output /tmp/minicpm5-projections.json
```

A successful verification creates local JSON projections for a-11-oy.com and
a11oy.net. Both explicitly keep modelOperational=false and
productionDisposition=HOLD. The proof projection carries measured case counts,
small-sample latency and unresolved gates. A SMOKE_FAIL remains visible. Invalid
verification replaces any stale output with an INVALID, non-operational record.
No domain, Space, credential, model default, or public runtime is modified by this
command. Existing GitHub -> Hugging Face -> product -> proof publication owners
must admit these bytes through their own tested publishing path.

Run all 43 offline tests with:

```bash
python -m unittest discover -s tests -p 'test_minicpm5*.py' -v
```

The synthetic fixtures in these tests are not model-execution evidence. This
projection contract complements the existing held-out gate harness; it does not
replace Named-N, Nemo envelopes, signed authority, or model-release gates.
