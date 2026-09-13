# Invariants release evidence inspection

This is an offline, read-only successor to the keyless preflight in PR #277.
It does not modify the publisher, exchange credentials, publish artifacts,
change trust bindings, execute Hub code, or make a production decision.

## Incident addressed

Forge runs 34727235269 and 34727278848 failed at Hub metadata reads after
GitHub authorization succeeded. The first publication-named ZIP held only
`invariants-release-authorization.json`. An uploaded archive, or a green
artifact-upload step, must not be treated as a completed publication receipt.

## Use

Download the exact publication archive using authenticated GitHub tooling.
Obtain its SHA-256, run/attempt, and publisher revision independently from
that run's GitHub artifact and workflow metadata. Obtain the source revision
from the authorized dispatch. Do not derive these expected values from the
untrusted report being inspected. A digest calculated only from the received
ZIP establishes no authenticity.

```bash
python tools/inspect_invariants_release_evidence.py \
  --archive /path/to/publication.zip \
  --archive-sha256 "$EXPECTED_GITHUB_ARTIFACT_SHA256" \
  --source-revision "$EXPECTED_SOURCE_SHA" \
  --publisher-revision "$EXPECTED_PUBLISHER_SHA" \
  --run-id "$EXPECTED_RUN_ID" \
  --run-attempt "$EXPECTED_ATTEMPT"
```

Supply the bare 64-character SHA-256 value, without a `sha256:` prefix.
The CLI emits only fixed diagnostic codes and invariant booleans; it never
prints untrusted receipt content, paths, repository names, or credentials.
It neither extracts the ZIP nor writes any report or repository file.

Exit 1 means rejected/unreadable. Exit 0 means only
`EVIDENCE_CONSISTENT_REVIEW_REQUIRED`. Even on exit 0 the result explicitly
retains `publication_verified=false`, `independent_hub_readback=false`,
`signature_independently_verified=false`, and `repository_mutation=NOT_ATTEMPTED`.
Never connect this exit code alone to release promotion or runtime loading.

## Checked contract

- The same bounded archive bytes are hashed and parsed; the caller must provide
  an independent expected digest. Only the two exact root report filenames are
  accepted. No path normalization, nested alternate location, or extraction.
- Maximum archive size is 2 MiB, each JSON member is bounded to 512 KiB. Reject
  duplicate ZIP members, encrypted or nonregular entries, unsupported compression,
  invalid JSON, duplicate JSON keys, nonfinite JSON numbers, and corrupt archives.
- Require a successful PUBLISH receipt, exact source/publisher/run/attempt and
  workflow identities, and type-sensitive agreement of both authorization copies.
- Require observed source/publisher protection, a valid source-signature claim,
  successful required checks with all reported app bindings, and a zoned timestamp.
- Require the closed four source-file records, matching source variants, and the
  recomputed artifact-tree digest using the existing publisher's algorithm.
- Require distinct model and kernel records, exact commit IDs, complete final
  readback claims, unchanged kernel main, consistent v1 parents and readback,
  the closed six staged kernel paths, and the existing pinned publication tools.

The inspector checks consistency of claimed evidence, not its truth. It does
not contact GitHub or Hugging Face, verify signatures cryptographically,
reconstruct source bytes, establish freshness, independently check grants, or
compare downloaded Hub bytes. Those remain mandatory external release tasks.
A forged but internally consistent report can pass this offline inspection;
independent artifact/run provenance and exact provider readback are essential.

## Qualification

```bash
python -m unittest discover -s tests -p test_invariants_release_evidence.py -v
python -O -m unittest discover -s tests -p test_invariants_release_evidence.py -v
```

Fixtures are explicitly synthetic. No successful fixture represents a real
publication, model-quality result, or live credential exchange. Existing
publisher credentials and integration are unchanged: do not retry publication
solely because this inspector is available.
