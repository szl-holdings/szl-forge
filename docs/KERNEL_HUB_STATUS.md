# Kernel Hub fleet: historical claims, not a live observation

Current fleet qualification: **HOLD / live observation unavailable from this tool**.
The offline report generator does not query Hugging Face. Running it cannot establish
current inventory, branch presence, artifact integrity, publisher trust, or runtime
compatibility. A model-type mirror's existence also does not prove that the matching
first-class kernel repository is missing or unmigrated.

## September 13 correction

`tools/hf_kernel_hub_ops.py` previously stamped hard-coded counts with the current
`observed_at` and accepted arbitrary nonempty revision strings other than `main`.
Its historical-data compatibility function is retained, but supplying a synthetic
observation timestamp now raises an error. New code uses `historical_snapshot()`.

The v2 report has separate `generated_at`, a null current `observed_at`, null current
`counts`, `collection_performed: false`, and source-linked `historical_claims`.
It always reports HOLD and never grants execution or production permission.
Existing dated JSON receipts are not overwritten or retroactively reinterpreted.

Consumer declarations require a valid repository identity, an explicit declared
`repo_type: kernel`, and a full lowercase 40- or 64-hex revision. These are syntax
checks only: a random digest can pass syntax without identifying any real commit.
Missing types are not defaulted to a provider observation. Branches such as `v1`,
`release`, and `main` must be resolved and independently verified before they can
satisfy SZL's immutable-pin policy. Major-version arguments remain supported by
upstream; this is a stricter SZL declaration policy, not a claim that upstream
removed version support. The declared repo type is metadata for this check, not
an extra argument to pass to `get_kernel`.

`trust_remote_code=True` remains blocked for review. An allowlist must contain
exactly the named repository, with no wildcard, additional repository, duplicate,
or case substitution. `False` retains the loader's default policy but does not
prove the publisher is trusted. Nulls, strings and integer lookalikes are rejected.
A passing declaration returns `state: PINNED`, not `ADMITTED`, with
`validation_scope: STATIC_DECLARATION_ONLY`, `provider_resolved: false`,
`publisher_trust_verified: false`, `executed: false`, and
`production_authorization: false`. `ok` means syntax passed, nothing more.

## Historical source claims retained

The earlier [source record](https://github.com/szl-holdings/szl-forge/blob/4ed693c2b7ea43ec1c23cdbb31bba08601802b30/docs/KERNEL_HUB_STATUS.md)
reported a September 11, 2026, 23:07 UTC unauthenticated list observation of
`GET https://huggingface.co/api/kernels?author=SZLHOLDINGS`: 14 objects, of which
9 carried the `kernel` tag and 5 did not. The five named objects were
`szl-governed-norm`, `governed-inference-meter`, `szl-maskmod`, `szl-block-kv`, and
`szl-receipt-attn`. This repair has not independently reproduced that observation.

A separate earlier same-day claim named missing `v1` branches for `szl-maskmod`,
`szl-block-kv`, `szl-receipt-attn`, and `YARQA-ATTN`. Its exact time is unknown and
must not inherit the 23:07 timestamp. Those branches were not rechecked by that
list/tag sweep. Historical list, tag and branch predicates are not interchangeable.
Do not add these counts to membership or inventory counts from other scopes.

## Run and interpret

From the repository root:

```bash
python -m unittest -v tools.test_hf_kernel_hub_ops
python tools/hf_kernel_hub_ops.py
```

Tests exit 0 on success. The report command prints JSON and exits **2** because
current live collection is unavailable/HOLD. This is intentional, not a successful
migration signal. To retain it, choose a new file in an existing directory:

```bash
python tools/hf_kernel_hub_ops.py --output kernel-hub-history.json
```

That command also exits 2. Existing files (including the former
`docs/KERNEL_HUB_FLEET_LIVE.json`) are never overwritten. No files are written by
default. The existing HF Kernel 0.16.1 migration workflow now runs these offline
regressions before its unchanged single-repository provider-verification steps.
A green metadata check still does not establish a kernel-client import witness.

## Remaining independent migration work

Use the canonical publishers; do not create another Hub writer or delete mirrors.
Collect a fresh, paginated, item-level first-class-kernel observation with explicit
scope, timestamps, raw response hashes, per-item errors and completeness. Verify
refs and their target commits separately. An inaccessible endpoint is unavailable,
not an empty collection, missing repository or evidence of migration failure.

Inventory actual `kernels.get_kernel` / layer consumers, including import aliases
and documentation snippets, without conflating local functions named `get_kernel`.
Bind proposed consumer pins to the canonical source and observed provider revision;
never manufacture a digest or silently widen trust. Verify artifact bytes and run
credentialless `kernels==0.16.1` CPU import/numerical/receipt checks. GPU parity
requires an actual supported device and remains separate from CPU evidence.

Upstream's [migration guide](https://huggingface.co/docs/kernels/migration) specifies
September 13, 2026 as the start date for legacy model-type kernel takedowns, not an
exact UTC-midnight deadline or proof that every mirror has been deleted. Its
[kernel requirements](https://huggingface.co/docs/kernels/kernel-requirements)
describe the first-class repo type and exact-repository trust allowlists.

The source-only repair does not publish to HF, load kernels, start compute jobs,
change branch protection, close runtime acceptance gaps, or promote production.
