# Public Kernel Hub fleet observation

`tools/observe_kernel_fleet.py` collects live, anonymous, read-only metadata from
`https://huggingface.co/api/kernels?author=SZLHOLDINGS&limit=100`. It complements
`hf_kernel_hub_ops.py`, whose historical report is deliberately not a live collector.

From the repository root:

```bash
python -m unittest -v tools.test_observe_kernel_fleet
python tools/observe_kernel_fleet.py \
  --source-revision "$(git rev-parse HEAD)" \
  --output kernel-fleet-observation.json
```

The output must be a new operator-selected file in an existing directory. The
collector never overwrites prior evidence. Source identity is supplied by the
operator; the workflow binds it to the actual checkout with `git rev-parse HEAD`.
The receipt additionally hashes the observer file itself. A hash is integrity
metadata, not an independent signature or approval.

## Scope and completeness

Two paginated list passes bracket per-item reads. Every listed repository is
restricted to the exact SZLHOLDINGS namespace. The collector obtains refs, resolves
main and any v1 branch to full commit IDs, reads metadata at those exact revisions,
and rechecks refs. Moving lists/refs, duplicate items, malformed responses, bounds,
or inaccessible endpoints keep aggregate counts null and the result HOLD.

The list API can omit optional SHA and tags. Missing list SHA is not a missing
repository: separate refs and pinned metadata establish the per-item revisions.
Omitted tags remain unknown, never automatically untagged. A missing v1 is only
reported after a complete refs response; a 403/404 is not a missing-v1 observation.
The kernel-type claim is grounded in the first-class API endpoint and rejects an
explicit contradictory repoType. It never infers type from a model's tags.

`observation_complete: true` and exit 0 mean all metadata reads in the declared
public scope completed with the specified stability checks. The status is
`OBSERVED_METADATA_ONLY`, while `fleet_qualification` remains `HOLD` and all runtime
and production-authority flags remain false. Exit 2 means incomplete observation.
Sequential reads are not an atomic snapshot and cannot prove absence of a transient
change between reads. Private assets and model mirrors are outside this scope.

## Evidence and security

Each response records its endpoint, start/end timestamps, HTTP status, raw bytes
encoded as base64 and SHA-256. Original raw JSON is retained, not reconstructed from
normalized rows. Response headers are not generally persisted; only the public
pagination Link is retained. The code does not load an HF token, ambient proxy,
cookie jar, kernel package, downloaded Python code, model weight, or publisher.

Only HTTPS GETs to the fixed HF origin and constrained metadata endpoints are
permitted. Redirects, changed pagination scope, duplicate cursors/items/JSON keys,
non-finite JSON, and oversized responses fail closed. Bounds: 100 items, 10 pages
per pass, 256 requests, 512 KiB per body, 16 MiB total body bytes, a 180-second
request-launch window and 12-second socket timeout. The hosted job has an additional
eight-minute hard timeout. There are no retries, paid HF jobs, installs, uploads to
HF, mirror deletions, schedules or new publication authorities.

The existing HF Kernel migration workflow runs the collector regressions and
public observation in a separate job, preserving the report as a 30-day Actions
artifact even on incomplete collection. That retention is not permanent immutable
storage. Preserve an admitted artifact digest/receipt through the existing evidence
publication path when needed; do not manufacture a second publisher.

## Still separate

Consumer inventory, source-to-provider build provenance, file-byte equivalence,
actual `kernels==0.16.1` CPU import/numerical tests, GPU tests, and downstream website
projections are not established by a metadata observation. No model-mirror deletion
or automatic production promotion follows from a complete result.

References: https://huggingface.co/docs/kernels/migration and
https://huggingface.co/docs/kernels/kernel-requirements.
