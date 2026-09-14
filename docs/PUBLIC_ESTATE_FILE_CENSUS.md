# Public estate file census

This is an evidence collector beside the existing Forge frontier/estate operators,
not a new registry, publisher, queue, application backend, or production controller.
It fills the distinction between knowing a repository exists and actually enumerating
its tracked files. Related coordination: `szl-frontier#133`, `.github#740`, and the
existing native release acceptance in `szl-frontier#96` and `a11oy#2155`.

## Execute the actual observation

From a clean Forge checkout with Git and Python 3.11 or later:

```bash
python -m unittest -v tests.test_public_estate_files
python -O -m unittest -v tests.test_public_estate_files
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
limit against slow streaming; CI adds a 20-minute hard timeout. No retries or schedules
are installed. Thirty-day Actions retention is not immutable external custody.

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
