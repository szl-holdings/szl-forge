# Forge evaluation payload — Ghlore 0.3.0 source

Tracking: `szl-holdings/szl-forge#207`.
Frontier successor: `szl-holdings/szl-frontier#79`.
Exact upstream: `huggingface/ghlore@87290a46c79e26ebb0d47575263b7ee34c1c1390` (Apache-2.0).

Implement and execute only a sandboxed, read-only evaluation against a fixed public non-secret repository corpus and versioned query set.

## Required execution evidence

Record the exact install/build receipt, dependency and license inventory, corpus/query digests, environment and all failures. Exercise the new source state rather than moving `main`.

- Start mismatched client/daemon wire versions and prove refusal; no compatibility downgrade.
- Keep `trust-network` off by default. If exercised, run only behind an independently enforced private test perimeter and prove loss/misconfiguration fails closed.
- Use read-only GitHub credentials. Do not create, label, close, edit or merge upstream issues/PRs.
- Verify retrieved prose is visibly marked as untrusted per line while Ghlore-generated provenance/assertions remain distinguishable.
- Verify `inflight` relationships are repository-scoped, freshness-bound and advisory. They may prevent duplicate work from being *suggested*, but cannot authorize closing, skipping, merging or modifying work.
- Verify body/file-list truncation is surfaced so missing history is not treated as proof of absence.
- Compare fixed queries against GitHub search and the existing estate-native retrieval path on relevance, citation correctness, freshness, deterministic reproducibility, latency/resource cost and empty-result behavior.
- Test stale rationale, prompt-injection history, bot/self-output loops, deleted/retained records, missing citations, backend outage, credential loss and rollback/disable.
- Re-open current source/tests before any historical rationale is used to justify a code change.

`UNAVAILABLE` is valid when an environment/backend cannot be qualified. It must not be converted into PASS. Even a complete PASS leaves this lane at EVALUATION until a separate promotion wave satisfies normal production controls.

Forbidden: autonomous merge/deploy, secret access, branch-protection mutation, private corpus ingestion, upstream writes, production route/default changes, A11oy exposure, implicit `trust-network` enablement, or historical text becoming instruction/policy authority.
