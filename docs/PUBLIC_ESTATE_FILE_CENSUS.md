# Public estate file census

This is an evidence collector beside the existing Forge frontier/estate operators,
not a new registry, publisher, queue, application backend, or production controller.
It fills the distinction between knowing a repository exists and actually enumerating
its tracked files. Related coordination: `szl-frontier#133`, `.github#740`, and the
existing native release acceptance in `szl-frontier#96` and `a11oy#2155`.

## Execute the actual observation

From a clean Forge checkout with Git and Python 3.11 or later:

```bash
python -m unittest -v tests.test_public_estate_files tests.test_public_estate_rate_control
python -O -m unittest -v tests.test_public_estate_files tests.test_public_estate_rate_control
mkdir -p reports
python -I -B tools/observe_public_estate_files.py --lane github \
  --source-revision "$(git rev-parse HEAD)" --output reports/public-files-github.json
python -I -B tools/observe_public_estate_files.py --lane huggingface \
  --source-revision "$(git rev-parse HEAD)" --output reports/public-files-huggingface.json
```

Choose new output names for subsequent observations. Each command reserves its output
before network reads; existing files are never overwritten. A killed process can leave
an incomplete output file, which is not a valid receipt. The dedicated read-only CI
runs both lanes separately and preserves their reports even when a lane fails.

The GitHub lane uses the existing `GITHUB_TOKEN` environment variable, when available,
only for requests to the fixed GitHub API. Contents-read is sufficient for public
repo trees; no organization-administration or repo-write credential is requested.
Without it, anonymous rate limits may make the observation incomplete. The HF lane
never reads a token or forwards the GitHub token. Neither lane inherits proxy or
cookie configuration. No downloaded source, build hook, model, or kernel executes.

## Scope and proof strength

GitHub scope is the public organization repository listing. Every repository is
rechecked public, resolved to its default-branch commit and tree, and enumerated at
that exact tree. Recursive responses marked truncated are discarded and complete
nonrecursive subtree requests are used instead. The collector reconstructs every
Git tree object (including directory sorting and empty trees) and checks its SHA-1
against the referenced object ID. Symlinks and submodules are opaque entries, never
followed. The public/default-branch metadata and commit are checked again afterward.
Unavailable final public visibility suppresses retained per-file entries for that repo.

This checks **tree structure and referenced blob IDs**, not downloaded blob contents,
LFS payloads, semantic correctness, source signatures, current branch protection,
or build provenance. Git SHA-1 object equality is not an independent attestation.
Per-repo before/after reads and repeated organization membership are sequential,
not an atomic global snapshot. Movement remains visible rather than silently changing
which revision was scanned. Submodule contents, other branches, and untracked files
are outside scope.

HF populations are separately listed anonymous public `models`, `datasets`, `spaces`,
and first-class `kernels`, each with exact `author=SZLHOLDINGS`. The collector resolves
each repository to a full SHA, paginates its immutable file tree, then rechecks that
revision and population membership. It records file/optional LFS object IDs and sizes
without downloading weights or executable payloads. These are **provider metadata**,
not independently verified artifact bytes. A model with a kernel tag is not counted
as a first-class kernel by inference. Runtime stage is included only when actually
returned in metadata; missing stage is unknown. A RUNNING stage is not an application
smoke test, numerical benchmark, frontend/backend contract, or deployment acceptance.

Private assets, collections and mutable storage buckets are explicitly excluded from
this schema. Their counts are **not zero** and whole-HF-ecosystem completion is not
claimed. The authenticated inventory and canonical portfolio's application-only view
must remain separate from these author-membership populations.

## Interpreting output

`FILE_METADATA_OBSERVED_NOT_QUALIFIED` and exit 0 mean the declared file-metadata
scope completed without observed membership/ref movement. `PARTIAL_OR_UNAVAILABLE`
and exit 2 preserve per-item failures; complete-scope counts stay null. Known file
subtotals are explicitly partial. Path-derived frontend, Python, test, workflow and
binding counts only route follow-up review. A missing path signal is not proof that
capability is absent, and a present file is not proof that it works.

Every result sets `semantic_review_complete=false`, `runtime_verified=false`,
`source_content_files_read=0`, and `production_authorization=false`. The observer's
actual Git checkout and file bytes are checked before reads. Reports include source
identity, observer SHA-256, observation windows, request status/byte counts and response
hashes. Raw HTTP response bodies are not retained: a response hash alone is not a
replayable HTTP proof. The GitHub normalized tree entries are sufficient to recompute
the recorded tree hashes. No private content or error response body is published.

Bounds per lane: 256 repositories per population, 250,000 tree entries, 100 HF pages
per collection, 512 distinct subtree requests per fallback, 1,500 HTTP requests,
8 MiB per response and 256 MiB total downloaded JSON. New request launches stop after
900 seconds; each socket has a 12-second timeout. This is not a strict total wall-time
limit against slow streaming; CI adds a 20-minute hard timeout. HF 429 handling is
bounded as described below; no schedules are installed. Thirty-day Actions retention
is not immutable external custody.

After ordinary source review and exact-head checks, use this evidence in the existing
estate audit and native release workflow. Fix qualified source first, publish through
the canonical HF controller, verify live product at a-11-oy.com, then expose admitted
proof at a11oy.net. Do not close runtime, provider-drift, GPU, checkpoint-custody, or
browser acceptance gaps merely because this census is complete.

Primary API contracts:
- https://docs.github.com/en/rest/git/trees
- https://docs.github.com/en/rest/repos/repos#list-organization-repositories
- https://huggingface.co/docs/huggingface_hub/guides/search
- https://huggingface.co/docs/kernels/migration

## Gated LFS metadata: retain observations without inventing hashes

A bounded anonymous diagnostic on 2026-09-17 at 01:51:28 UTC observed
`SZLHOLDINGS/SZL-Forge-1.5B-ReceiptAgent` at revision
`8aa18ba91259b043a886f78ae78f4ff7c08852dd`, public with `gated=auto` before and
after its tree request. Its 22 entries contain 20 files and two directories. Three
LFS objects returned a 64-asterisk mask instead of the content hash: the adapter,
model, and tokenizer. This explains the prior `HF_LFS_IDENTITY` exception for that
revision; it does not establish corrupted weights, missing files, or permission
to bypass model access terms. Native diagnostic:
https://huggingface.co/jobs/SZLHOLDINGS/6aab479df76d6a098a712293

The existing parser recognizes only that exact mask under explicit `auto` or
`manual` gating. It preserves the file's path, size and Git pointer object ID,
but emits `lfs_oid=null` and `lfs_identity_state=REDACTED`. A normal full lowercase
SHA-256 remains `OBSERVED`; absent LFS metadata is `NOT_REPORTED`, not a claim that
the file is definitely not LFS. Other malformed digests, malformed LFS objects,
size disagreements and invalid pointer sizes remain errors. A pointer object's
Git SHA must never replace an unavailable LFS content SHA.

Enumeration and identity completeness are separate. A fully traversed, rechecked
public tree may retain `tree_complete=true` and its known `file_count`, while
`file_metadata_identity_complete=false`, `complete=false`, and
`HF_LFS_IDENTITY_REDACTED` preserve the missing identity. The existing aggregate
keeps complete-scope totals null and the CLI exits 2. Thus a recognized redaction
does not turn a failed census green. A changed gate or source revision remains a
separate blocker. Both before/after metadata reads must explicitly report
`private=false`; an unavailable or changed final public identity withholds item
paths, path signals and file counts. No credentials are acquired by this repair.

Twenty added adversarial test methods cover these distinctions and preserve the
original 33 methods. The 53-method suite passed locally under normal and optimized
Python 3.13.5. This is repeated execution of the same suite, not 106 unique tests.
The exact provider diagnosis and unit fixtures do not replace a fresh native
whole-public-census observation or complete private/content/runtime qualification.

## Bounded HF request pacing and 429 handling

The existing sequential `Client` now schedules HF API launches at least one second
apart. Valid provider `RateLimit` API remaining/reset hints can lengthen that
interval. A 429 can use `Retry-After` seconds or an HTTP date, or a documented API
reset hint. When both are present, the longer wait wins. Missing hints reserve a
conservative 300-second wait; malformed, duplicate, oversized, or ambiguous hints
cause deferral instead of an invented short delay. Response dates make HTTP-date
waiting conservative under local/server clock skew.

Local policy limits are three attempts per GET, six scheduled retries per run,
and 600 seconds of reserved retry waits, all inside the unchanged 900-second
request-launch deadline and 1,500-request/byte limits. These are safety bounds,
not a claimed provider quota. A delay that does not fit is never shortened to
force another request. Once a circuit stop occurs, later HF calls in this client
record the same bounded failure without touching the network. A subsequent run
has no cached success, automatic resume, or persistent authorization.

429 responses and raised HTTP errors are closed before any wait, without reading
remote error bodies. Only exact fixed-origin GETs are retried. GitHub errors,
401/403, redirects, 5xx, transport errors, bad JSON and invalid artifact identities
are not silently retried or promoted. HF remains anonymous; no token, proxy,
resource family, request path, or inventory predicate is substituted.

Each real attempt retains status, time, body digest where available, attempt
number, actual prelaunch wait and sanitized numeric rate hints. `hf_retry` records
whether a retry was scheduled or deferred; `hf_rate_control` records the local
policy, reserved wait, scheduled retries and circuit stop. A scheduled retry is
not proof it was launched. A recovered HTTP request is not a complete census:
all original revision, membership, public-visibility, LFS, and completeness
checks still run. In particular, recognized gated LFS redaction stays incomplete.

The existing unit matrix invokes both the retained census tests and the new
`tests.test_public_estate_rate_control` suite. No action pin, workflow permission,
job timeout, publisher, schedule, or failure assertion is weakened. The new cases
use deterministic clocks and recording transports, not actual HF traffic. Both
original LFS functions and all 53 predecessor test methods are preserved. The
combined 110-method suite passed normally and optimized on local Python 3.13.5;
Python 3.11/3.12 hosted execution and a paced native census remain separate gates.

This is one sequential reader's backpressure, not a distributed quota coordinator.
Other jobs sharing an address can still consume quota. Socket timeout and the
request-launch deadline are not a hard per-body streaming deadline; the existing
CI job timeout remains the outer limit. Reconcile the current exact branch with
parallel work before source admission. Do not immediately loop a failed HF lane.

Primary rate-hint contract: https://huggingface.co/docs/hub/rate-limits .
The project retains its already bounded stdlib reader; no SDK or authentication
migration is bundled into this source repair.
