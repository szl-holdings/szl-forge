# Source-to-card evidence closure — 13 November 2026

This continuation implements source changes, not a blanket model promotion or a completed Hugging Face release. It builds on the audit admitted by Forge #730. Historical observations remain dated and are not overwritten with a newer, differently scoped count.

## What is implemented

The dated anonymous-public catalog contains 107 model-type repository identities. Pagination is exhausted, with 105 readable README bodies and two gated HTTP 401 results. Eighteen new or changed immutable records were refreshed; 89 same-revision observations were retained from the earlier audit. No catalog-versus-read revision race was observed in this snapshot. Private visibility is unqualified. Eight newly observed `v2-trained` cards lack license metadata; a filename is not evidence of training or license clearance.

`closure-input.json` binds every catalog identity to its immutable revision, observation time, README result and bounded JSON evidence reads. `catalog-identities.json` is the retained listing projection. The validator rejects omitted, duplicate or extra identities, partial pagination, unsafe paths, mixed repository/revision URLs, booleans used as counts, unread-card digests, non-finite JSON and output filename collisions. Its scope is the supplied dated public snapshot, not discovery of private inventory.

Six source cards have additive evidence panels: Khipu 1.5B, KHIPU-R2, Chaski, Chaski-5050, Chaski-R2 and Chaski-R4. Removing their exact managed panels recovers every original canonical Git byte. YAML, licenses, negative results, eligibility flags and technical disclosures remain unchanged. The model/source/publisher mapping is closed to those six existing profiles. Unknown mappings grant no write authority.

The original unfinished local renderer accepted four malformed cases: boolean file counts, a digest attached to an unread card, duplicate evidence records and a control character in an evidence path. The four regressions fail on that prior local implementation and pass on this implementation. This is not a claim that the unfinished renderer was previously deployed.

## Publication routes are not interchangeable

`publication-route-review.json` accounts for all 107 identities without reassigning owners:

| Classification | Repositories | Meaning |
| --- | ---: | --- |
| Closed Forge routes checked | 6 | The only source cards edited by this change |
| External kernel owner declared | 13 | Existing kernel ownership policy remains authoritative; Forge must not republish them |
| Historical source requires reconciliation | 14 | An old source path is not current publication authority |
| Source owner not resolved | 74 | Explicit open work; not an invitation to create a second writer |

The older presentation maps do not override `publishing/kernel-hub-ownership.json`. Kernel/model namespace overlap is not a count of separately trained neural models. A source route, signed receipt, HTTP 200 or provider RUNNING state is not runtime qualification.

## Training-history evidence

`training-curves.json` contains four immutable checkpoint histories and 139 actual recorded points. Their projections were matched exactly to the retained public trainer-state JSON bodies and the recorded SHA-256 digests. The four histories are triage study5, abstention-v2, physics-master-v1 and quant-master-v1. They are not evaluations, controlled comparisons or proof of equivalence to current released weights.

The optional renderer draws separate logarithmic-loss figures, without smoothing or invented observations. Plotting requires Matplotlib in the execution environment; validation does not. No dependency, model loader, training job or publisher is installed or changed by this module.

```bash
python -m tools.model_card_evidence_passport --manifest docs/model-card-proof-20261113/closure-input.json
python -m tools.model_card_evidence_passport --manifest docs/model-card-proof-20261113/closure-input.json --output-dir NEW_REVIEW_DIRECTORY
python -m tools.render_model_training_curves --input docs/model-card-proof-20261113/training-curves.json
python -m tools.render_model_training_curves --input docs/model-card-proof-20261113/training-curves.json --output-dir NEW_CHART_DIRECTORY
```

Outputs are review artifacts, not Hub uploads. Existing output directories are refused. Model-card renderer tests run in the repository's existing `tests/` collection; no CI workflow or gate is relaxed.

## Acceptance and open dependencies

Completion requires the actual source owner, exact-head checks and independent review; normal protected-main publication by the existing publisher; then immutable deployed-byte readback. This change alters no publisher workflow, target registry, hardware, credential, model weight or release eligibility.

The organization page remains under `.github#1008`. Its concurrently updated head was detected before applying the older local integration. A separate isolated diagnosis reproduces three inventory-contract failures after its generated inventory paragraph was replaced with a different public-page observation. Restoring the generated historical paragraph and keeping the newer observation outside that block repairs the binding without hiding either snapshot. That diagnostic is not permission to overwrite the active owner branch, refresh effect-policy pins, self-approve or bypass release controls.

The independently observed A11oy service still reports frontier `NOT_READY`. Card improvements do not close GPU-to-application reachability, signed execution evidence, operational persistence, release review or model-quality gaps.

## Regression and rollback

Run the new passport and training-curve suites together with the existing Khipu/Chaski publisher, source freshness, concurrent-write, credential and claim-honesty suites. Preserve negative controls and skipped-case explanations. Local source rendering in Edge is an approximation of document presentation, not certification of the Hugging Face renderer.

Before release, close or revise the proposal through normal review. After acceptance, revert the source change through the protected path; the existing publishers restore reviewed card bytes. Never rewrite historical training/evaluation records to make a presentation pass.

## Reference practices

The design follows primary documentation practices: explicit model metadata and limitations, reproducible evaluation settings, separate artifact variants, and clear source/measurement boundaries. It copies no third-party scores or branding.

- Hugging Face model cards: https://huggingface.co/docs/hub/model-cards
- Hugging Face organization cards: https://huggingface.co/docs/hub/organizations-cards
- Moonshot model documentation: https://huggingface.co/moonshotai/Kimi-K3.1-Pro
- NVIDIA model documentation: https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4
