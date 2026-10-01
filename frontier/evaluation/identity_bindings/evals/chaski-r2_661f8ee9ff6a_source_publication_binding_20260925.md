# Chaski-R2 adapter source-byte binding correction

This additive [binding](chaski-r2_661f8ee9ff6a_source_publication_binding_20260925.json) connects the existing September 24 adapter
observation to immutable published Git source. The original evaluation receipt
and its original sidecar remain unchanged. The sidecar recorded hashes of
Windows CRLF working files; Git published LF bytes. Both exact identities and
the verified LF-to-CRLF transformation are recorded explicitly.

The new adapter observation used Hub revision
`661f8ee9ff6ab8b11dda6e7a9d42c14d3124c6dd`, `adapter_model.safetensors` SHA-256
`6f12981ea5df5e22d3493eefb20d75db75c1c88961ba6621a35af09edd0cdde6` and the listed adapter config. Independent
read-only review recomputed the evaluator's filename-plus-NUL-plus-bytes
aggregate `078ec09fb3e8e215d1b6a98bb7164a94c7bb332d847c8e1d66f565e767171107` from the existing pinned cache and
matched both the pinned and current Hub LFS metadata on September 25.

The retained run reports **5/5 JSON draft contracts and 6/6 refusal prefixes**
for the Chaski-R2 adapter on declared base
`huggingface:Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17`. The base comparator was 0/5 and 6/6; Chaski-5050
was unavailable. These reused, small named fixtures do not measure broad
quality, held-out generalization, or truth of the generated prose. This review
did not independently rehash the base model or tokenizer or rerun inference.

The separate September 15 gate failure remains preserved. The newer adapter
result neither erases that failure nor retroactively binds the older successful
reports. It does not evaluate the Hub repository's merged `model.safetensors`.
The original receipt also retains a stale sentence saying `gate_ran=false`
while its actual top-level field is `true`; this correction explicitly records
that distinction without rewriting the original evidence.

**Production disposition remains HOLD.** Publication/autonomy eligibility
remain false, promotion effect is NONE, output authority is PROPOSAL_ONLY,
and the evidence is UNSIGNED_HONEST.
