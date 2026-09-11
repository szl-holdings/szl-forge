# Khipu abstain retrain — separate candidate, preserved baseline

The owner-signed September 8 retrain is `SZLHOLDINGS/SZL-Khipu-1.5B-abstain`,
not the previously published `SZLHOLDINGS/SZL-Khipu-1.5B`. Public pinned receipt
reads establish distinct model-weight lineages. Original signed abstention is
**2/6**; candidate signed abstention is **3/6**. Both declare grounding **4/5**,
plan validity **11/11**, and zero hallucinated citations on these small fixtures.
These source receipts are not a fresh independent model evaluation.

The original `khipu/` receipts and the original portfolio binding remain unchanged.
This folder preserves the new signed wrappers byte-for-byte, including their CRLF
format, owner public key and all five unchanged receipt-bound synthetic dataset/
schema files. Those files reuse existing Git blobs; no curriculum was regenerated,
re-signed or silently altered. `identity.json` records exact observed model pins
and receipt identities. The separate Khipu-R2 Hub 3/6, grounding 5/5 result is NOT
this candidate and must not be substituted for either signed line.

```bash
python -m pip install -r khipu/requirements-verify.txt
python tools/verify_khipu_candidate_identity.py
python -m unittest discover -s tools -p 'test_khipu_candidate_identity.py' -v
```

The verifier reuses `verify_signed_receipts` for declared-key signatures,
training/evaluation linkage and dataset hashes. It additionally enforces the
exact model/revision/receipt mapping and original portfolio boundary. The new
read-only CI workflow preserves a source-checkout-linked integrity record and
refusal tests. It never runs model inference, modifies a Hub repository, downloads
weights, changes the public serving default, signs new receipts, or grants action
authority. Original tests retain their original 2/6 expectations; passing tests
cannot relabel a candidate as the original model.

The owner previously published adapter and F16 GGUF artifacts to the candidate
Hub repository. Their LFS metadata is recorded separately: a signed aggregate
weight hash is NOT the individual adapter/GGUF file hash. The presence of a GGUF
file does not establish a post-quantization benchmark or justify inheriting model
metrics. Actual local weight-byte checks, conversion lineage and runtime-specific
qualification remain pending. Existing L2 hidden-handle acceptance and production
authorization remain separate gates. This addition admits source evidence only:
**proposal-only, research candidate, HOLD**. The existing 16-entry admitted
portfolio, original model and quantized derivatives are not promoted or rebound.
