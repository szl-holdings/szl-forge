# Source-owned flagship collection reconciliation

The public [SZL Flagship Models collection](https://huggingface.co/collections/SZLHOLDINGS/szl-flagship-models-6a9315c1c853da528726dd8d) still contains 26 model-type repositories and says “Trained weights only.” The source-owned [`collection-rebuild.json`](collection-rebuild.json) instead specifies an **empty flagship** until a model qualifies. The source-owned [`collection-quarantine.json`](collection-quarantine.json) explicitly denies six of the current 26 from flagship placement. The reviewed collection API observation, exact item object IDs, exact model revisions, artifact classes, card promotion declarations, and all 26 decisions are in [`flagship-collection-reconciliation.v1.json`](flagship-collection-reconciliation.v1.json).

**Target:** keep every repository public and discoverable through the [public model artifact catalog](https://holdings.a-11-oy.com/frontier/), remove the 26 current collection memberships one at a time, and describe the resulting flagship shelf as intentionally empty. The six quarantine items remain under their source HOLD policy. This change does not qualify, train, delete, privatize, or serve any model. The `research` and `candidates` values in the manifest are source plans, not assertions of current Hub collection membership or release readiness.

## Source and provider gates

1. Merge this source change through the protected `szl-holdings/szl-forge` PR only after exact-head CI and signed-commit review. Confirm the current protected `main` SHA and that no competing collection writer is active. Do not dispatch the historical Space lifecycle controller for a model collection.
2. On a clean checkout of that exact protected `main`, read the public provider state:

   ```powershell
   py -3 -B -m publishing.flagship_collection_reconcile
   py -3 -B -m publishing.flagship_collection_reconcile --live
   ```

   The second command makes no write. It fails closed on unknown, duplicate, private, non-model, or replaced collection items. A changed model card or collection membership requires a new source review before applying.
3. Confirm `hf auth whoami` identifies `betterwithage` as a `SZLHOLDINGS` administrator with collection-write authority. The connected OAuth reader has only read scopes; it is not a collection publisher. The operator uses the local Hugging Face credential and never prints it. A repository write preflight alone does not establish collection management authority.
4. The operator verifies the local clean `main` against the exact remote protected `main` and checks its commit signature. GitHub web merge commits may require importing [GitHub’s published web-flow public key](https://docs.github.com/en/authentication/managing-commit-signature-verification/about-commit-signature-verification) locally before `git verify-commit HEAD` can verify them:

   ```powershell
   curl.exe -fsSL https://github.com/web-flow.gpg | gpg --import
   git verify-commit HEAD
   ```

   A missing public key is a setup gap, not permission to skip signature verification.

## One provider transition per invocation

From a fresh `--live` plan, inspect the chosen manifest row and exact card revision. Supply **one** listed Hub ID and the plan’s exact `provider_last_updated` value. For example, after reviewing the first item:

```powershell
$szlSourceSha = (git rev-parse HEAD).Trim()
py -3 -B -m publishing.flagship_collection_apply `
  --mode remove `
  --target-id SZLHOLDINGS/SZL-Khipu-1.5B `
  --expected-last-updated '2026-10-02T00:26:29.975Z' `
  --expected-source-sha $szlSourceSha `
  --execute
```

Refresh the plan before **every later removal**; never reuse the example timestamp. The operator verifies the exact collection object ID and source-owned 26-item baseline, reads provider state twice before the call, makes at most one `delete_collection_item` call, and reads back that exactly one membership disappeared. A removal changes only collection membership; the model repository stays public. The provider API has no atomic compare-and-set on `lastUpdated`, so sole-writer coordination remains necessary.

After all 26 removals read back as absent, a separate invocation updates only title and description from the manifest:

```powershell
$szlSourceSha = (git rev-parse HEAD).Trim()
py -3 -B -m publishing.flagship_collection_apply `
  --mode metadata `
  --expected-last-updated '<fresh provider_last_updated>' `
  --expected-source-sha $szlSourceSha `
  --execute
py -3 -B -m publishing.flagship_collection_reconcile --live
```

The terminal plan must report `ALIGNED`, zero flagship items, and aligned metadata. Independently read the public collection and the company catalog. The catalog covers artifact discovery, not model promotion. If the operator reports `UNKNOWN_AFTER_ATTEMPT`, inspect the provider before any retry; do not assume the write failed. Neither this plan nor a provider HTTP success qualifies a model. [Hugging Face’s collection API reference](https://huggingface.co/docs/huggingface_hub/package_reference/collections) documents the item-object-ID removal and metadata update methods used here.
