# Kernel provenance reconciliation

This is a metadata-only repair for six existing `SZLHOLDINGS` **model** repositories. It is not a kernel deployment or model-training release. The same-named Hub kernel repositories are outside the write set.

The September 19 audit originally identified eight targets. A fresh September 24 readback found that another writer had already corrected `szl-governed-norm` at `16776a536a875a609317537f3fa987c8604652bb` and `szl-lambda-gate` at `d1d021fa93b6072963aad9efff4e9f3b579a1e20`, with additive correction and CPU-test receipts. This release explicitly excludes those repositories and preserves their newer work. Its first eight-target rehearsal refused the stale plan before any write. This is intentional conflict detection, not a reason to weaken expected-parent checks.

The audited model trees lack the surrogate files claimed by their legacy root `MODEL_PROVENANCE.json`. Cards or kernel source already deny published weights, while earlier publishers update other files. This lane makes the root provenance a canonical reviewed artifact and binds its claims to the observed complete file trees.

## What is retained and what is asserted

`snapshots.json` contains public September 24, 2026 exact-revision metadata, tree records, request receipts, and the original provenance bytes. The old documents are preserved both as parsed historical evidence and verbatim base64 bytes in each replacement. Their historical claims, metrics, dependencies, and source assertions are **not** adopted as current verification.

Current assertions are limited to the observed artifact state: no recognized weight artifact was present, the named historical surrogate was absent, and no new training or runtime-quality claim is made. A training receipt's presence is not proof of its contents. Suffix scanning is not semantic analysis of every file; unsigned local audit records are not independent witnessing.

The manifest `../kernel-provenance-reconciliation.json` closes the registry to these targets:

- `SZLHOLDINGS/szl-blocked`
- `SZLHOLDINGS/szl-formulas`
- `SZLHOLDINGS/szl-govsign`
- `SZLHOLDINGS/szl-invariants`
- `SZLHOLDINGS/szl-ouroboros`
- `SZLHOLDINGS/szl-provctl`

Only `MODEL_PROVENANCE.json` can be replaced. No delete, executable artifact restoration, visibility change, Space restart, arbitrary target, or inferred human approval is supported.

## Operational commands

From the repository root, Python 3.11 or 3.12 can validate the reviewed source offline without third-party packages:

```console
python -B -m unittest discover -s tests -p test_kernel_provenance_evidence.py -v
python -B -m unittest discover -s tests -p test_reconcile_kernel_provenance.py -v
python -B tools/reconcile_kernel_provenance.py validate --repo-root .
```

For an immutable candidate checkout, install `huggingface-hub==1.29.0`, then run the read-only rehearsal with its actual 40-character Git commit:

```console
python -B tools/reconcile_kernel_provenance.py check --repo-root . --source-revision <exact-commit> --receipt <new-receipt.jsonl>
```

The workflow **Reconcile kernel provenance** runs offline tests on pull requests. It does **not** publish on push or merge. After protected review and release-owner coordination, a manual dispatch on current `main` defaults to `check`; an explicit `publish` selection is required for provider writes. The existing credential selector is reused without revealing token bytes, and the publisher checks write authorization for every target before its first commit.

Publication binds canonical manifest, evidence, and output bytes to immutable Git blobs. It refuses a checkout/source mismatch or a source commit that no longer owns canonical main. It preflights all six current provider trees and originals, and uses Hugging Face's expected-parent commit guard for each single-file commit. Every non-destination file must remain identical on exact readback. A new receipt path is mandatory; attempts are journaled before mutation and failures retain their evidence. Because the SDK can return an identical existing commit, verified artifact bytes do not imply this run authored the commit; receipts explicitly deny verified authorship.

There is no cross-repository atomic transaction. A later failure can leave an explicitly reported partial publication. Do not delete its receipt or retry blindly. First inspect recorded commit IDs and current provider state; already-applied exact output with an unchanged remainder of the tree is distinguishable from an unapproved drift. A changed original parent requires a new reviewed capture and manifest, not relaxing the guard.

Passing tests establishes the software contract. A successful check establishes current applicability. Only a successful retained publish/readback receipt establishes provider publication. None of these establish new SZL model weights, independent multi-person approval, training-data rights, measured general model quality, or live inference.

The API-level expected-parent and tree operations follow the official [Hugging Face HfApi reference](https://huggingface.co/docs/huggingface_hub/en/package_reference/hf_api). GitHub branch rules, actual credential grants, and human release ownership remain external prerequisites rather than invented local attestations.
