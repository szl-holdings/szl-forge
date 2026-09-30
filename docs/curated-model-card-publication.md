# Curated model-card input preparation

The closed WILLAY input route prepares a local diff and receipt from the existing
`willay/card/README.md` and a reviewed immutable metadata observation. It has no
Hub client, network, credential, upload, inference, training or promotion path.
It is not admitted to the existing model-card publishers.

Run with Python 3.11 or later, from the repository root:

```sh
python -B -m unittest discover -s tests -p test_prepare_curated_model_card.py -v
python -B tools/prepare_curated_model_card.py --profile willay
# Optional: create a new local review directory. Existing directories are refused.
python -B tools/prepare_curated_model_card.py --profile willay --output-dir willay-review
```

The default command writes nothing. The optional directory contains `README.md`,
`review.diff` and `receipt.json`; it never replaces the source card. Reviewers must
inspect the full diff before proposing a source-card edit.

## Input and evidence scope

`publishing/curated-model-card-profiles.json` binds the source card hash, reviewed
GitHub commit, model repository identity, observed Hub commit and observation
hash. The observation is a compact projection of the retained September 30
metadata pass and independently checked classification receipt. Their full
SHA-256 values, the classifier source hash and the evidence Library ID are in
the observation. The reviewed classification is an input, not a live classifier
or build attestation. No unmerged checker code is imported.

All 25 inventory entries, three recorded packages and both adapter bindings refer
to one immutable WILLAY commit. Complete package structure describes metadata
closure only. Weight bodies, tensor integrity, load compatibility, model quality,
training, evaluation, rights and release qualification were not evaluated.
Some supplemental JSON bodies were not retained; their raw digest fidelity is
inherited from the reviewed reader. The original first-pass unknown structure
result remains recorded alongside the supplemented observation.

Root and nested adapters keep their distinct base references. Missing immutable
base revisions remain UNKNOWN. No size threshold, file extension, or declared
merge receipt grants full-model, merged-model, compatibility or promotion status.

## Preservation and review

The preparer appends or replaces one exact `SZL-CURATED-METADATA:v1` block. It
requires the reviewed hash of all unmanaged bytes and preserves the existing
YAML and prose byte for byte, including the declared license and historical
failed evaluations. It does not parse or validate legal clearance. Missing YAML,
partial/duplicate markers, changed source bytes, invalid JSON, mixed revisions,
missing package files and stale observation hashes fail closed.

The receipt has full hashes for source, YAML, registry, observation, candidate and
diff. Its GitHub SHA denotes the reviewed input, not a release approval or a claim
that a checkout is fresh main. Preparing twice is idempotent. This input route
changes no card or publisher registry and grants no release/rights clearance.

## Ownership and release prerequisites

This first profile is WILLAY only. ReceiptAgent v3, Khipu r3 and Brain Navigator r2
source cards remain under Forge #417; their held corrections must be reconciled
before admitting additional inputs. Forge #424/#429, #426 and #434 are unchanged.
Khipu #79, model promotion and WO6 authentic owner-confirmation holds remain.

The read-only CI workflow runs offline fixtures and the default preparation; it
does not dispatch a publisher. These new paths are outside the reviewed Chaski,
Khipu and kernel-card publication path filters. Branch pushes cannot publish.

Any future rollout must use an explicitly owned, reviewed publisher route with
fresh source qualification, the expected target parent commit, provider
compare-and-swap, reviewed managed/unmanaged diffs, preserved real license,
idempotence, exact readback and partial-write evidence. Actual owner confirmation
and normal release checks remain separate prerequisites. No owner marker is
created by this preparation route.
