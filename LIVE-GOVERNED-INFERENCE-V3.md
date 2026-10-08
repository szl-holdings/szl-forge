# Public governed inference v3 — source contract

This is the proposed public wire contract, not a claim that a v3 runtime is
deployed. The protected-main publisher, exact-source readback, governed health,
and a retained live verifier report are separate release evidence.

The canonical publisher deploys this Space from protected `main` after merge;
an exact-source hosted v3 smoke test is therefore a **post-merge release gate**,
not a pre-merge check that can be demanded of an unpublished source revision.
Before merge, the compatibility decision, local tests, signed source commit, and
required exact-head CI must be accepted. A normal protected merge authorizes
the publish attempt, not a successful deployment. Keep the runtime disposition
`HOLD` until the publisher completes and the source-bound live verifier passes;
on any failed or unavailable stage, retain the failure receipt and `HOLD`.

## Research basis and protection boundary

Panfilov et al., [*Stealing Reasoning Traces from Proprietary LLM APIs*
(arXiv:2608.09867v1)](https://arxiv.org/abs/2608.09867v1), document risks from
portable client-held reasoning blocks and recommend removing opaque reasoning
fields from public transcripts. This v3 contract independently applies that
release-hygiene principle to the Forge Space: the public projection omits the
controller continuation, rejects unknown fields, and retires the former public
POST that could return it. No third-party reasoning traces or credentials are
needed for the regression tests.

This is an output-boundary control, **not** a claim to implement the paper's
provider-side authenticated-encryption context binding, key rotation, replay
revocation, or model-level defenses. It does not govern the Space's separately
documented legacy `/api/v1/infer` and `/v1/chat/completions` demo endpoints.
Those routes do not carry this controller continuation, but their existence
precludes a claim that every Space inference route is governed by v3.

`POST /api/v3/governed-infer` accepts the same bounded public request fields as
the former v2 route: `prompt`, `max_new_tokens`, and `k`. Its response schema is
`szl.forge.public-governed-inference/v3`. The response contains no `continuation`;
the internal controller still produces its private
`szl.forge.production-governed-inference/v2` result for in-process governance.
Unknown public fields or malformed nested receipt/observation data cause a
generic HTTP 503 `HOLD` without echoing the input.

The public response carries a `public_export_receipt` with policy
`strict-v1-private-continuation-omitted`, the internal v2 source schema, and a
SHA-256 digest of the exact public body excluding that export receipt. Both the
export receipt and the controller inference receipt are `UNSIGNED_LOCAL`; their
hashes establish internal consistency, not signer identity, authorization,
independent replay, or model qualification. The internal inference receipt's
payload and digest retain their v2 source semantics.

`POST /api/v2/governed-infer` returns a fixed HTTP 410 `BLOCKED` response with
the v3 successor path. It never calls the controller, redirects, echoes a
request, or publishes a continuation. Clients that depended on public
continuation cannot be made compatible without restoring the privacy defect;
they must use a separately authorized, signed A11oy process outside this Space.
The read-only v2 health, Formula Atlas, and Anatomy routes are unchanged.

The unversioned
`GET /.well-known/szl-governed-inference-contract.json` advertises the v3
request path and schema and records the retired v2 POST. The live verifier must
first bind the hosted release-manifest SHA-256 to the publisher's exact-source
checkout. It probes the retired v2 POST with an empty JSON object, which the old
v2 request model rejects before controller invocation; only HTTP 410 passes.
It then checks a v3 proposal with no continuation, matching source and model
revisions, digest consistency, and the existing no-tool/unsigned bounds
before a publication can be called verified. Until then, release disposition is
`HOLD` and hosted runtime state is `UNKNOWN`.
