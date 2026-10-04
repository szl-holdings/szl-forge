# ReceiptAgent v4 held-out custody

This directory contains independently authored SIMULATED gold references for a
future protocol-conformance gate. It contains no trained weights, actual model
decodes, model scores, optimizer run, signature, authorization or promotion.
The author did not access the train/dev curriculum, its generator or predecessor
evaluation case contents while constructing this set.

Access class: PUBLIC_FROZEN_NOT_BLIND. Public case contents and generator do not
establish a blind benchmark. The test split must never be imported by any training
loader, used for target repair, or converted into a refusal/abstention curriculum.

The 180 cases are divided equally among three nonclinical software families:
package-license-ledger, schema-migration-compatibility and repository-access-matrix.
Each family has 20 DRAFT, 20 RECOVERY and 20 REFUSAL references. Within each family,
every recovery status and refused authority has four cases. All evidence labels
are SIMULATED. Artifact digests and canaries describe invented fixture content,
not real files, credentials, service observations or signed receipts.

The public v4 contract supplies typed routing and fixed claims. Therefore the
future gate measures JSON protocol transfer, request/evidence binding and copying;
it does not measure software reasoning, natural-language intent detection, legal
compliance, real evidence freshness, Khipu abstention or equivalence with v3 scores.

The declared future model gate is 180/180 full pair conformance, including 60/60
per branch and 12/12 per recovery status/refused authority. Invalid decodes count
as failures. No model has been evaluated on this set. A failure must retain BLOCKED;
do not alter the frozen test bytes or tune against revealed cases and reuse the
same set as independent held-out evidence.

`generate_heldout.py` imports only the exact-byte-pinned public contract. It does
not read train/dev files, load a model, install dependencies, use a GPU, perform
remote writes, sign anything, or grant execution/training/publication authority.
It validates all reference pairs before producing the two deterministic files.
Use an existing environment with jsonschema 4.26.0; from the repository root:

```sh
python -I -B frontier/qwen35-receiptagent-v4-json/heldout/generate_heldout.py --check
```

`--write` creates missing artifacts only and refuses to replace differing frozen
bytes. `--check` reproduces and compares without writing. Printed output contains
only hashes and aggregates. `manifest.json` binds seed 2601003, the original source
revision, generator/source/schema bytes, dataset bytes and future gate denominators.
Its MEASURED gold-reference validation is software conformance on SIMULATED data,
not a model evaluation or independent replay. It is unsigned. Cross-split semantic
leakage review remains UNAVAILABLE until a separate non-authoring reviewer supplies
an aggregate decision; random identifiers are not proof of split independence.

Before publication, the contract-source commitment was corrected to the canonical
LF checkout bytes (`373451dff44ba0fcd9f6cf445f19737ebf7566855a741377c7b0fa8388247b88`).
This was a source line-ending correction, not a protocol or case revision. The
frozen `test.jsonl` commitment remains
`973e5f5a272ad7c1304acec0db36fae77a6d0c2e0ec0c10c4401361b7889c575`.
The generator's no-overwrite rule remains unchanged.
