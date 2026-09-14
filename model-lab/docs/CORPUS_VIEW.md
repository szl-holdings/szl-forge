# Corpus-review projection: existing producer, private workbench

This adds a read-only consumer of `szl.frontier.ultradata-pilot-review/v1` to the
existing Python Model Lab. The producer is `frontier_completion/data_review.py`
and its existing `review-data` CLI, developed in Forge #289. Do not install the
previous handoff's separate `corpus_review.py` parser alongside it.

## Contract and operator configuration

Only configure a legitimately held report on owner-controlled local storage.
Supply both `SZL_LAB_CORPUS_REPORT` (Path) and `SZL_LAB_CORPUS_REPORT_SHA256`
(externally obtained lowercase file SHA-256). Keep the existing Model Lab access
token local. Do not put reports, paths, tokens or source trajectories in GitHub,
HF cards or the public product/proof sites. No new credentials are introduced.
The existing owner-controlled filesystem boundary applies; this is not a defense
against a malicious process with the same operating-system identity.

`GET /corpus-review` renders the report summary in the existing Jinja template.
`GET /api/corpus-review` returns the allowlisted summary. Both require the same
operator authentication as the existing workbench. There is no report upload,
query-selected path, remote URL, training, model load, tool replay or publisher
endpoint. Missing or invalid reports yield HTTP 503 with null counts in the API;
the page explains the empty state. Unknown coverage is not zero.

The reader verifies the externally pinned file digest, producer receipt digest,
strict JSON, exact schema/field set, row numbering/types, count arithmetic,
producer state and denied-authority/review fields. It preserves `generatedAt` as
report-generation time. It is not model/data observation time. Reports over one
hour old are labeled stale; timestamps more than 30 seconds ahead are rejected.
The file limit is 256 KiB, and the producer's 256-row limit is retained.

No trajectory content, local paths, row hashes, exception contents or arbitrary
report explanations reach the UI/API. Dataset IDs are restricted to the two
producer-supported UltraData identities. Source/dataset revisions remain declared,
not authenticated by this reader. Matching a report hash is not a signature,
source-byte verification, reward verification, rights/privacy clearance,
decontamination or training authorization. Those permissions remain false.

## Source packaging and release boundary

The existing blueprint source allowlist includes this module and this document,
so the exported app does not contain a missing import. No report or corpus bytes
are included. Publisher functions, credentials, workflow triggers and admission
checks are unchanged. Manual protected-source publication and immutable provider
readback remain required. This source addition does not clear the shared HF
credential hold, Forge #289's required-check association blocker, or the separate
product/proof deployment acceptance.

## Tests and limits

`python -m pytest -q -c model-lab/pyproject.toml model-lab/tests/test_corpus_view.py`
exercises synthetic reports, tampering, typed counts, redaction, file checks,
settings, existing ASGI authentication and HTML/JSON routing. These are not real
UltraData records or a model benchmark. The producer's report contract is tested
with explicit fixtures, not a runtime import of unadmitted source. End-to-end
producer CLI plus consumer testing belongs to the integrated checkout after both
source changes land. Full native Model Lab tests and source-plan jobs remain the
existing workflow's responsibility; no competing workflow was added.
