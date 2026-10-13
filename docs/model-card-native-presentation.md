# Native model-card presentation

## Preserved contract

This presentation change updates eight existing model-type card sources. The
six Chaski and Khipu family cards replace the old moving 112-pixel image with
the exact approved 48-pixel Orbit, distinguish Command Lab, canonical source
and reviewed evidence links, and attribute the artwork. Their artifact/stage
summaries, visible use limits, dated provider observations and complete
technical disclosures retain their bytes.

The two OAC model cards gain the same compact identity and navigation after
their existing critical boundary or opening synthetic-only description. All
original text stays in its original order. Removing the two marked additions
reconstructs each previous OAC card byte for byte. Its model, data, training,
evaluation, clinical, device and authority boundaries remain unchanged.

Every YAML frontmatter byte is preserved. This is ordinary Markdown and an
image, with no custom CSS, scripts, new model metadata, executable loader,
artifact, training, evaluation or publication-authority change. A document
host controls its own typography and colors; these cards do not install the
KANCHAY CSS bundle.

The artwork is the approved
[native Orbit icon at `27d12e77`](https://github.com/szl-holdings/szl-brand/blob/27d12e77fc0f9f1a0e5fae8b34adf2380d51d070/kit/logos/orbit-v2/suite/szl-icon-32.svg):
2,969 bytes, SHA-256
`6e2b7d72a3af3d9602b79de41df0194ede6496a9661f057108ec224c0efc92e6`.
The pinned [kit license](https://github.com/szl-holdings/szl-brand/blob/27d12e77fc0f9f1a0e5fae8b34adf2380d51d070/kit/LICENSE)
is CC BY 4.0. The footer attributes the artwork separately from each
artifact's existing license.

## Actual authoring and publication paths

| Model-type artifact | Canonical README source | Existing writer |
| --- | --- | --- |
| `SZLHOLDINGS/chaski` | `chaski/card/README.md` | `publish-chaski-card.yml` / `tools/publish_chaski_card.py`, closed `chaski` profile |
| `SZLHOLDINGS/chaski-5050` | `chaski-5050/card/README.md` | Same writer, closed `chaski-5050` profile |
| `SZLHOLDINGS/chaski-r2` | `chaski-r2/card/README.md` | Same writer, closed `chaski-r2` profile |
| `SZLHOLDINGS/chaski-r4` | `chaski_r4/card/README.md` | Same writer, closed `chaski-r4` profile |
| `SZLHOLDINGS/SZL-Khipu-1.5B` | `khipu/card/README.md` | `publish-khipu-card.yml` / `tools/publish_khipu_card.py`, closed `khipu` profile |
| `SZLHOLDINGS/KHIPU-R2` | `khipu-r2/card/README.md` | Same writer, closed `khipu-r2` profile |
| `SZLHOLDINGS/oac-system-health-v1` | `clinical-gateway/huggingface/model/oac-system-health-v1/README.md` | `publish-oac-hub.yml` / `tools/publish_oac_hub.py`, closed `oac-v1-model` profile |
| `SZLHOLDINGS/oac-ops-health-v2` | `publishing/oac-ops-health-v2/README.md` | Same writer, closed `oac-v2-model` profile |

The family writers control only `README.md`, the existing `holo-banner.svg`
and generated `szl-source-binding.json`. All banner bytes and qualification
fields stay unchanged. They run through normal reviewed main publication,
fresh-source checks, credential validation, immutable parent concurrency
controls and exact readback. Card publication cannot promote a model or
clear a failed gate. The source-binding record will name the newly accepted
source; no precomputed source commit is fabricated inside these cards.

Each OAC profile has a fixed six-file package and permits replacement of
`README.md` only. The v1 model uses the existing main publisher. The v2 model
requires an explicit `workflow_dispatch` on current main with `publish=true`;
its normal push remains outside v2 publication authority. Existing asset
locks, current-source, reviewed-parent and package-parity controls remain.
No dispatch or Hub mutation is part of this source change.

The OAC dataset is a separate eight-file profile with **empty replacement
authority**. Its README and data are unchanged. The legacy
`oac-clinical-transport-health-v1` model is outside this closed registry and
is also unchanged. No source/profile is added to the existing writer.

### Overlapping nano writers are excluded

`publish-kernel-mirror-cards.yml` is a separate matrix. Its public targets
include `TinyKhipu-Nano` and `ReceiptAgent-Nano`, whose currently served
technical bodies match the accepted Khipu source and differ from the Forge
alternate cards. A watched card change can trigger that whole matrix. This
change touches none of its watched card paths, profile registry or workflow;
it does not use a different card edit to overwrite those nano bodies.
Both existing writer configurations remain intact pending explicit source
conflict resolution. (On 2026-10-13 the five converged kernel profiles were
removed from this matrix because their kernel repositories now own their
model twins; the nano profiles were not changed.)

The historical `publishing/model-card-presentation.v1.json` and separate
review-route records are preserved as historical evidence. Their old hashes
are not silently replaced with new presentation hashes.

## Observed published baseline

On 2026-10-11, public metadata followed by immutable README reads established
that all eight published cards were byte-identical to the corresponding
files at accepted
[`a41356937a6c52180e569b9066c6e291ae607ed1`](https://github.com/szl-holdings/szl-forge/tree/a41356937a6c52180e569b9066c6e291ae607ed1).

| Artifact | Immutable pre-change Hub revision |
| --- | --- |
| Chaski | `00a8bb51dd2040a2584ac690e212ce40aae2cce6` |
| Chaski-5050 | `a8126aedc1dc0d5faafde52426522b05ce86cd36` |
| Chaski-R2 | `d1a5558d267bfd74d7507917f429143d01f5259b` |
| Chaski-R4 | `c12950da09107744d3e696e5aaca98466917228b` |
| Khipu 1.5B | `8a5c666c4fc1ec75a51b8bd9784b4504e807757a` |
| KHIPU-R2 | `a66a35997aa2577ed17c661e331961fce609956a` |
| OAC System Health v1 | `28242f5b8fe46bcfc5ea63b34a904dc1dc2d60f1` |
| OAC Ops Health v2 | `a824a32d91a383d33a1e1e595f11b8362d1b4efa` |

The six family source bindings identify Forge source
`156b8e9dc63948d9f079f62c92837acb49983b3b`, their exact prior README/banner
digests, and the retained negative qualification. The README bytes at that
published state still match the accepted baseline above. The two OAC cards
use their own package and commit-history contract; a missing family-style
binding is not substituted with an invented one.

These are pre-change card-delivery observations. They establish no new
weight verification, training replay, runtime availability, evaluation,
provider support or delivery of this new presentation.

## Verification

The source invariant was captured before editing: only eight existing card
paths change. All **1,797 other tracked files** remain byte-identical to the
accepted baseline, including the root README, workflows, publishers, model
and data files, receipts, SDK payloads and historical presentation records.
All eight frontmatter hashes and all six marked technical-body hashes
match. Removing only the two new marked OAC additions recovers each entire
old card exactly.

| Preserved body | SHA-256 |
| --- | --- |
| Chaski technical region | `63df0b24435e6d83df1e1fef7a019f248faec76ff17d87522203a0b27cd8073f` |
| Chaski-5050 technical region | `1f7a0b1eccd8fe2c0e0c0e48fec030974035921aa4bbdb97ccd1c30b0af903a9` |
| Chaski-R2 technical region | `75100ada321dc1c8a810c8359d85ceef9d1c508bf882b11b4befd83d4f2b826d` |
| Chaski-R4 technical region | `714f07505dcf6ac0d624019ac3a1c45808c4610bb00b168f4e7d122b5dcb4414` |
| Khipu 1.5B technical region | `eee71933fd0d9e701fbf60a659aa1cbb2427ba6a32f9995012950dff524b3ac1` |
| KHIPU-R2 technical region | `c0b066006e5741a07de73cc7cb5fa20c32cde0618a6242878d274acca125bb53` |
| OAC v1 original complete card | `13bdc96f21d722100eb9f36c68c8db62ca395363dafc0e6501fb93d666394302` |
| OAC v2 original complete card | `8f63155003dff012cd319708ceee532e360bf9f716b9c75a19b8baf2d898de5a` |

**365 existing tests passed** using the publisher's exact
`huggingface-hub==1.23.0` and `pytest==9.1.1`. Eight parameter cases were
intentionally skipped because the Khipu-specific main-change guard does not
apply to a Chaski profile. The selected tests cover the three publishers,
their workflow restrictions, family claim honesty, source freshness,
concurrency, credential validation and OAC asset locks. Unmocked transport
was denied before imports and zero attempts occurred.

An independent local CommonMark 3.0 preview passed **33 checks**: each card
at 320 and 1280 pixels in light and dark document styles, loaded exact-size
native identity, no document overflow, preserved visible limits, OAC
navigation targets and family technical-disclosure keyboard behavior.
Representative desktop/mobile renders were visually inspected. Eleven
historical external image URLs inside the preserved technical material were
blocked, without fetching or substituting their artwork. No unexpected
external request, attempted write or page error occurred. This preview uses
approximated document-host CSS; it is not a qualification of the Hugging Face
renderer or those historical images.

Normal source review and hosted checks precede acceptance. The existing
writers must provide new source-bound immutable readbacks after acceptance;
until then the new presentation is source-ready, not verified as published.
