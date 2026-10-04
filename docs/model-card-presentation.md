# Model-card presentation

The canonical inputs for eighteen Forge-managed Hub cards now open with a small
SZL mark, a short purpose statement, the actual artifact type, its current stage,
navigation, and the limits a reader needs before use. The complete earlier
technical body is retained inside **Technical details and evidence**.

## Preservation contract

The source baseline is
`2b4d7a0f69e68d60cb0f35f09c9bac837d66acdc`.
Every existing frontmatter byte, including licenses, library names, base-model
relations, and eligibility flags, is preserved. Every byte after the original
frontmatter is preserved between the `SZL-CARD-TECHNICAL:v1` markers. The original
banners, code examples, dated measurements, signatures, and evidence links remain
in that body.

[`publishing/model-card-presentation.v1.json`](../publishing/model-card-presentation.v1.json)
records the eighteen exact source paths, publishers, original frontmatter/body
digests, output digests, visible limits, and dated provider observations. An empty
provider mapping is an observation at its recorded Hub revision and time, not a
promise about future availability. No new card header claims hosted inference.

## Publication routes

| Publisher | Profiles | Source disposition |
| --- | ---: | --- |
| `publish-kernel-mirror-cards.yml` | 12 | Eleven public profiles; ReceiptAgent 1.5B remains gated |
| `publish-khipu-card.yml` | 2 | Research cards with their abstention release blockers visible |
| `publish-chaski-card.yml` | 4 | Research cards with failed, quarantined, or not-promotable states visible |

Land the shared mark in `szl-holdings/.github` at
`profile/assets/szl/logos/szl_mark_holographic.svg` before the card fanout. A
protected-main merge invokes the existing card publishers. Their write sets,
source checks, conditional Hub writes, readback, and access gates are unchanged.
The gated profile remains blocked by its publisher.

The separate WILLAY and ReceiptAgent v3 preparation contracts retain their fixed
source hashes and one-shot review bounds. Other Forge card files outside these
three publisher registries are not part of this change.
[`publishing/model-card-separate-routes.v1.json`](../publishing/model-card-separate-routes.v1.json)
records those fourteen unchanged source files, their Hub identities, and the
known separate or unresolved publication routes. Together the two maps account
for thirty-two Forge source cards; they do not claim all-estate publication.

## Verification

Before the files changed, all eighteen candidate cards passed their existing
publisher asset validators and exact frontmatter/body preservation assertions.
The nine publisher, concurrency, freshness, evidence, and credential-selection
suites passed with **298 passed and 8 inapplicable Khipu-guard cases skipped**.
No inference, model download, Hub write, release promotion, or runtime deployment
was performed while preparing this source change.

When maintaining these cards, keep the decision-relevant limits above the
disclosure. Use an executable **Try** route only when its exact artifact and
runtime are established; otherwise keep **Explore in Command Lab**. Preserve
the distinction between an adapter's evaluation and a separate merged checkpoint.
