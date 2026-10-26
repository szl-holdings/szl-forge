# Estate wiring reconciliation — 2026-10-26

This is a dated source and read-only provider/HTTP observation, not a deployment,
model evaluation, security certification, compute grant or release attestation.
No DNS, Worker, Space, Snowflake, trainer or ledger mutation was performed for
this reconciliation. Training and publication admission remain false.

## Exact source observations

Authenticated GitHub reads on 2026-10-26 returned verified signed commits and
untruncated selected trees:

| Repository | Inspected head |
| --- | --- |
| szl-holdings/a11oy | c47b7030270a4024e5f7a34f2aa4c95b40f74ba0 |
| szl-holdings/platform | 29f4c0d1b983ab2e3dda135096942d9754ffdc17 |
| szl-holdings/szl-holdings.github.io | a4e35b424be41542d5b40a44ddb799de166d8cc4 |
| szl-holdings/szl-forge | bc69fe973933857562cf7b00084fff7786683f4a |

MEASURED here means the source/API/HTTP check was performed. Values declared by
a service remain DECLARED; they are not an independent hash of its running image.

## Source-to-surface map and measured gaps

| Surface | Source declaration | Readback / unresolved boundary |
| --- | --- | --- |
| A11oy Python/backend and console | [hf-sync.yml](https://github.com/szl-holdings/a11oy/blob/c47b7030270a4024e5f7a34f2aa4c95b40f74ba0/.github/workflows/hf-sync.yml) publishes Dockerfile COPY sources to SZLHOLDINGS/a11oy, binds SZL_GIT_SHA and reads /api/build-info; it does not change DNS. Docker packages serve.py, gdw_runtime.py and console static files. | Product and HF twin GET /api/build-info both returned 200 and identical decoded bodies, declaring revision 41ee66eb9ab13ec116448a8d3db555cd3c596178, Python 3.14.7 and working_tree UNKNOWN. GitHub compare shows inspected main seven commits ahead. Source convergence is not established. |
| Platform frontend preview | [source.json](https://github.com/szl-holdings/a11oy/blob/c47b7030270a4024e5f7a34f2aa4c95b40f74ba0/console/preview/platform-a11oy/source.json) pins Platform cc9fc2bd658241354c381e4fe78a11ba1d09dd3b. | Live /preview/platform-a11oy/source.json returned that pin with production_ready false, runtime_api_binding UNAVAILABLE and effectors 0. Its producer state is CANDIDATE_DEMO, not current-main frontend/backend convergence. |
| Product domain a-11-oy.com | Product/HF twin in the company origin-authority contract. | Authenticated Cloudflare GET measured a proxied apex CNAME to szlholdings-a11oy.hf.space. This routing and matching build-info do not establish model or API readiness. |
| Governed inference Worker route | [deploy-cloudflare-governed-inference.yml](https://github.com/szl-holdings/a11oy/blob/c47b7030270a4024e5f7a34f2aa4c95b40f74ba0/.github/workflows/deploy-cloudflare-governed-inference.yml) names a-11-oy.com/api/v2/*. | Authenticated Cloudflare workers/routes GET returned an empty list for this zone. The declared route was not present in that provider readback. No deployment was attempted. |
| Company frontend holdings.a-11-oy.com | [origin-authority.json](https://github.com/szl-holdings/szl-holdings.github.io/blob/a4e35b424be41542d5b40a44ddb799de166d8cc4/origin-authority.json) and CNAME declare GitHub Pages company entry point, separate from product and proof. | Root and origin-authority.json returned 200. The linked origin-status.json still describes observations dated 2026-09-04; it must not be treated as current health. |
| Proof a11oy.net | The company contract declares a separate public proof registry. | Authenticated Cloudflare GET measured four proxied apex A records; root returned 200 with title a11oy Proof Registry. Deployed source identity, live proof-chain verification and independent witness remain UNKNOWN. |
| Forge to Hugging Face | [publish-forge-lab.yml](https://github.com/szl-holdings/szl-forge/blob/bc69fe973933857562cf7b00084fff7786683f4a/.github/workflows/publish-forge-lab.yml), [publish-model-inference-lab.yml](https://github.com/szl-holdings/szl-forge/blob/bc69fe973933857562cf7b00084fff7786683f4a/.github/workflows/publish-model-inference-lab.yml) and [model-source-bindings.json](https://github.com/szl-holdings/szl-forge/blob/bc69fe973933857562cf7b00084fff7786683f4a/publishing/model-source-bindings.json) define curated publication paths. | The bindings file has four named entries, not complete bindings for the separately inventoried 77-model portfolio. Inventory, source tests and metadata publication are not all-model training/evaluation. |
| Platform Python documentation | [AGENTS.md](https://github.com/szl-holdings/platform/blob/29f4c0d1b983ab2e3dda135096942d9754ffdc17/AGENTS.md) names lib/a11oy-fabric-py and a11oy_fabric_py. | Neither path exists in the inspected untruncated tree; README contents GET returned 404. services/substrate-py-workers exists, but equivalence is not assumed. This requires the Platform owner's scoped documentation repair. |
| Snowflake | Existing Cloudflare/Snowflake coordinating lane retains ownership. | Current tenant/role/database/warehouse/connector and actual query/dataflow evidence were not established by this lane. UNAVAILABLE for this reconciliation. No SQL, warehouse resume, ingestion, spend or account change attempted. |

## Recording and next owned work

The inference lab's GET `/api/build-info` also returned 200 and declared Forge
source revision `bc69fe973933857562cf7b00084fff7786683f4a`, model revision
`67d60ec577730747055491640cfb91fc4a4b5d25`, and true source/hash checks. These are
producer-declared runtime fields observed over HTTP, not an independent weight
replay, full portfolio evaluation or an authorization grant. Separately, the
Forge catalog lab's `/api/build-info` returned 404; do not conflate that endpoint
with the inference lab or infer the entire catalog is unavailable from it.

This sanitized map and Forge source changes belong in normal signed GitHub review.
Local raw observations, command outputs, failure receipts and source/test hashes
are retained separately; this document does not claim every historical receipt
has been replicated to GitHub, Snowflake or the proof registry. No private data,
credential or operational account ID is copied into this public-facing map.

Next non-overlapping work is: finish explicit loader selection/returned-model
inspection with offline tests; reconcile exact model-family source/data/loader
bindings without retraining conversions; obtain the existing owners' current
deployment and Snowflake evidence; then make owner-coordinated fixes to stale
source maps and missing routes through their protected release paths. Do not
blindly deploy a declared route or rewrite a pinned preview to current main.

The existing app transport failed during owner messaging and monitor update;
neither delivery nor monitor repair is asserted. A previous heartbeat prompt is
not current runtime evidence. No competing worker or replacement monitor was
created, and no idle observation was converted into a compute allocation.

Preserve existing models/checkpoints/data, signed SKUs, receipts, other-writer
repositories and the cumulative campaign accounting. The USD 2000 / 180000-second
ceilings are not remaining allowance; historical consumption and remaining
authority are UNKNOWN. This source reconciliation does not change them.
