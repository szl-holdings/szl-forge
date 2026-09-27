# Owner-declared document evidence adapter

Invariant: a local source quote may be called a verified **span** only when its
SHA-256 binding, Unicode codepoint offsets and exact text match a bounded source
packet. That never verifies the quote's truth, relevance, completeness, rights,
or the correctness of an answer using it.

This Python 3.11–3.13 standard-library adapter fills the document-intake gap around
the existing Evidence Lab. It does not create a new model or duplicate retrieval.
It prepares the exact `{question, context}` request body accepted by
`POST /api/answer-context` in the merged Evidence Lab PR 344, source merge
`a220bb8f33a9fb2825e3978821e9a96659ed470e`. The packet records required headers and
that immutable contract revision. **It sends nothing.** Future changes to that
API require a new compatibility review; no live integration is claimed here.

ReceiptAgent v3 remains a separate proposal-only model lineage. Its current
train/dev/test-only request schema is deliberately not reused for real document
requests. This adapter does not change model weights, curricula or signed receipts.

## Prepare a document

Use only material you are allowed to process locally that contains no sensitive
personal, customer, patient or credential data. Place a UTF-8 `.txt` or `.md` file
and a manifest in the same trusted local folder. Filenames must be simple
alphanumeric/hyphen/underscore basenames, not paths. No network shares, symlinks,
junctions, alternate data streams, reserved device names, drive-relative paths,
trailing dots/spaces or directory traversal are accepted. These checks apply to
output paths as well; trusted local directories must already exist.

Example manifest (replace the digest with the SHA-256 of the exact file bytes):

```json
{
  "schema": "szl.approved-document-manifest/v1",
  "documentId": "community-hours-v1",
  "title": "Community center public opening hours",
  "sourceFile": "hours.txt",
  "sourceSha256": "replace-with-64-lowercase-hex-characters",
  "question": "When is the community center open?",
  "ownerDeclarations": {
    "rightsBasis": "OWNER_AUTHORED",
    "rightsConfirmedByOwner": true,
    "sensitivity": "PUBLIC_NON_SENSITIVE",
    "permittedUse": "LOCAL_EVIDENCE_REVIEW"
  }
}
```

The other accepted rights bases are `LICENSE_PERMITS_LOCAL_PROCESSING` and
`EXPLICIT_PERMISSION`. These are owner assertions, **not independent legal,
privacy or consent verification**. This tool has no sensitive-data classifier.
Do not label private material as public to bypass that intake boundary.

```text
python -I -B operational/approved_documents/adapter.py prepare --manifest /local/manifest.json --output /local/new-packet.json
```

The output contains the complete source text and question in `request.body`.
It is sensitive if you supply sensitive input; protect or delete your own local
artifacts accordingly. There is no telemetry, upload, training, execution,
automatic submission or retention service. Input text is never interpreted as
instructions, HTML, Markdown, executable code or a URL to fetch.

Maximums: 64,000 source bytes, 16,000 source codepoints, 512 question codepoints,
65,536 encoded request-body bytes, 64,000 cumulative quoted UTF-8 bytes, 128 KiB
JSON input/output and 16 JSON value-nesting levels. Duplicate JSON keys, floating
point/nonfinite values, oversized integers, invalid UTF-8 and lone surrogates are
refused. Plain UTF-8 bytes and
line endings are preserved; offsets count Python/Unicode codepoints, not UTF-16
units or UTF-8 bytes. Display code in a browser must convert offsets correctly.

## Verify proposed citations

An external caller supplies a closed citation object, not a model response:

```json
{
  "sourceSha256": "replace-with-the-packet-document-digest",
  "spans": [{"start": 0, "end": 6, "quote": "Center"}]
}
```

```text
python -I -B operational/approved_documents/adapter.py verify --packet /local/new-packet.json --citations /local/citations.json --output /local/new-span-receipt.json
```

`VERIFIED_SPANS_ONLY` means every supplied quote matches, at most 16. Empty,
wrong-source, out-of-bounds, duplicate, over-budget, malformed or mismatched citations produce
`ABSTAIN`, never a partly accepted answer. Invalid packet bindings or invalid JSON
are refused before a receipt is written. Quotes can still be misleading, stale,
incomplete or irrelevant. No semantic entailment, answer correctness or clinical
authority is granted. Instructions inside source quotes remain untrusted data.
Distinct overlapping spans are permitted within the cumulative quote budget;
that does not constitute independent corroboration or additional sources.

Exit codes: `0` prepared/verified; `3` abstention receipt written; `2` invalid
input or output refusal. Outputs are exclusively created, never overwritten.
Both packets and receipts have self-digests for ordinary integrity checks, not
signatures or independent attestation. An actor able to rewrite data and hashes
can forge them. Use trusted stable local directories; path checks are not an OS
sandbox and do not defeat a hostile same-account process racing parent changes.
Output is validated by the same strict loader before creation, so emitted
packets and receipts stay within the loader's size and JSON contracts. A failed
filesystem write can leave a partial output; the program reports failure and
never silently replaces that file on retry. Inspect it and choose a new output
path after an I/O failure.

## Tests

```text
python -I -B -m unittest discover -s operational/approved_documents -p "test_*.py" -v
```

Tests use invented public-domain-style fixtures authored for this adapter. They
measure intake and span contracts, not real model inference, retrieval quality,
training, benchmark generalization, production deployment or accessibility.
