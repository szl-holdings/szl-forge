# Evidence lineage and hypothetical documentation impact

## Implemented boundary

The existing authenticated Model Lab now serves `/evidence` and
`/api/evidence-lineage` from one Python projection. The navigation entry is part
of the existing workbench, not another control plane. Both GET routes retain
Basic authentication, the loopback/host boundary, no-store, no-referrer, nosniff,
and the existing restrictive CSP. No JavaScript or remote request is required.

The input is the existing `szl.card-evidence-coverage/v1` public coverage format.
Configure an explicit owner-controlled local file and an independently obtained
expected SHA-256 using `SZL_LAB_EVIDENCE_SNAPSHOT` and
`SZL_LAB_EVIDENCE_SNAPSHOT_SHA256`, alongside the existing access-token setting.
Neither value is accepted from a web request. Both must be configured together.
The historical `docs/model-card-proof-20261113/closure-input.json` can be used as
an archived example. Its November 13 counts are not a current estate census.
A digest computed from the same untrusted input is not an independent trust root.
Do not publish private catalogs or expose the development HTTP server publicly.

The reader bounds bytes, model and record counts before constructing the view;
checks exact catalog membership, immutable repository identity, safe record paths,
fixed-host URLs, strict JSON and externally pinned snapshot bytes; and refuses
private scope or claimed release authority. It reuses the existing regular-file
reader. Missing/invalid input produces HTTP 503 with null counts and no local
path disclosure. Unknown selections produce a fixed HTTP 404. No weights,
record contents, pickle, model code, provider credentials or prompts are read.

## Identity and graph semantics

A documentation graph contains repository revisions, historical README reads,
structured-record references, declared source baselines and per-model documentation
projections. These are documentation inputs, not discovered deployment dependencies.
A source edge records a declaration, not proof that source built current weights.
The source route is not an allowlist for publication or execution.

Receipt identity includes repository, revision and relative path. Equal byte hashes
in two model repositories do not establish the same evaluation population, model
variant, signer, or result. Shared source references retain their exact revision
and path; they can support more than one documentation projection.

`invalidate=<node_id>` performs a deterministic, in-memory reachability calculation:
start with the selected input, then traverse its outgoing documentation edges.
Affected projections are marked `RECHECK_REQUIRED_SIMULATION`. The UI labels the
result WHAT-IF, offers a baseline link, and does not write the snapshot or change
real readiness. This is not causal inference, probabilistic confidence, a provenance
semiring implementation, a proof of a Lutar invariant, or a runtime incident report.
An unchanged baseline must be byte-for-byte identical before and after the query.

## Integration and evidence

The same projector supplies HTML and JSON. Selection, immutable links, missing
states and what-if results must agree across both routes. Jinja autoescaping and
restricted identity/path alphabets prevent metadata from becoming executable markup.
The existing wheel package-data rule includes the template. The code-only blueprint
source list includes both reader/router modules, template, and this document, while
retaining no committed Hub writer for the unpublished model identities.

The normal Model Lab CI discovers the new tests automatically. Tests cover auth,
HTTP method/host boundaries, source/API/UI agreement, missing evidence, input
substitution, digest tampering, exact membership, copied receipts, shared-source
impact, and baseline preservation. Any installed-Edge or loopback-server result
is local integration evidence, not hosted HF or production deployment qualification.

## Design research and original SZL adaptation

Palantir's Ontology SDK connects applications to shared object and action contracts;
its public React source guide also separates provider setup, loading and failures.
This implementation adopts the shared-contract idea without adding its SDK, copying
its code or visual design, or introducing a new writer. The Palantir/MIT paper
*Concept-Centric Software Development: An Experience Report* motivates independently
understandable concepts: evidence identity, documentation dependency, and scenario.

Primary references reviewed on 2026-11-16:

- https://www.palantir.com/docs/foundry/ontology-sdk/overview
- https://github.com/palantir/osdk-ts/blob/main/docs/react/getting-started.md
- https://www.palantir.com/docs/foundry/workflow-lineage/manage-security
- https://groups.csail.mit.edu/sdg/pubs/2023/concepts-onward-23.pdf
- https://www.sigmodrecord.org/publications/sigmodRecord/1209/03.principles.karvounarakis.pdf
- https://developers.openai.com/api/docs/guides/evaluation-best-practices
- https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents
- https://docs.newrelic.com/docs/ai-monitoring/opentelemetry-ai-monitoring/
- https://docs.nvidia.com/nemo/agent-toolkit/latest/resources/migration-guide.html
- https://www.trueanomaly.space/mosaic

The published Palantir Developers video `vSJ5H00V7Es` was located through its
description/chapter index and companion documentation; no full transcript or
frame-by-frame review is claimed. Vendor product descriptions are not independent
benchmark evidence. The provenance paper distinguishes alternative from joint
support; the current boolean dependency traversal deliberately does not claim that
richer algebra. OpenAI and Anthropic motivate contract-specific negative tests,
while New Relic motivates separating metadata from sensitive content.

NVIDIA's current migration guide states that NeMo Agent Toolkit was deprecated in
October 2026 and reaches end of life at year-end. It identifies Relay, Fabric and
Helix for observability/governance, execution adapters and evaluation respectively.
Do not add the deprecated toolkit as a new default dependency based on the earlier
broad stack recommendation. This work installs none of those products and changes
neither existing NeMo runtimes nor SZL's owned execution interfaces.

## Future work, not deployed claims

A separately reviewed extension could bind actual qualified checkpoints, dataset
revisions, graders, Lean obligations and runtime traces to this graph, preserving
their distinct trust domains. Evidence-directed re-evaluation could then propose
the minimum rechecks after a changed dependency. Existing Hatun admission, ORO job
ownership, Second Brain rights and canonical publishers would retain authority.
No comparative superiority, novel theorem, automated remediation, public deployment,
real model evaluation or production readiness is established by this UI.

Rollback is a reviewed revert of this feature and its blueprint closure additions.
No persistent data, provider configuration, weights or historical receipts migrate.
