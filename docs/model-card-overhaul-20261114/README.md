# Estate model-card overhaul — evidence manifest and card content schema (2026-11-14)

## Scope and observation

- **Scope:** the 138 public model-type repositories under `SZLHOLDINGS` on the Hub. Registry membership, not a trained-model count. The 45 private model repositories are out of scope for this pass.
- **Freshness observation:** all 138 public `README.md` files carry listing-level update timestamps in the window **2026-11-14T02:53Z–02:57Z**, consistent with the wave-4 train/publish run. The 2026-11-13 evidence audit's observed Hub revisions may therefore be superseded on the Hub side.
- **Problem this lane fixes:** sampled public cards re-stamped in that window present identical per-model claims regardless of the underlying artifact: a uniform `honesty-battery 3/6` model-index metric, identical QLoRA training tables, uniform receipts-count badges, and LoRA limitations on artifacts that are not LoRA adapters. Under the estate's fail-closed doctrine these are unverifiable per-model claims and must not stand.

## What this adds

1. `manifest-20261114.json` — a machine-readable per-model evidence manifest for all 138 public repositories: evidence-read counts from the 2026-11-13 coverage pass, publisher mapping status, atelier classification where a source card exists, the six receipt-verified bindings from the 2026-11-13 audit (exact Hub revision + training-record SHA-256), and explicit license gaps. Absent evidence is recorded as absent, never inferred.
2. This document — the card content schema that governs any future card rewrite.

## Card content schema (governs all future card rewrites)

1. **No metric without a binding.** A `model-index` entry, benchmark score, or battery result may appear only when the manifest carries a verified receipt binding for that repository, and the card links the exact receipt artifact. All other cards carry an explicit **Not measured** section instead.
2. **Training sections come from real contracts.** Training/provenance tables are populated only from a real per-model training contract (for example `training/w29/*.json`-style records with base model, datasets, method, hyperparameters, seed) or a verified training receipt. No template tables copied across models.
3. **Failures stay visible.** Dated, receipt-linked results are kept including failures (for example the retained chaski 0/5 drafts result). No favorable-only headline; no cross-model or cross-run inheritance of results.
4. **Artifact honesty for non-trained repositories.** Kernels, registries, quantizations, roadmap organs and stubs say what the artifact is (per the atelier classification vocabulary: MEASURED, SYNTHETIC, HUB, SIGNED, KERNEL, ROADMAP, STUB) and carry no training or evaluation sections.
5. **Licenses are never invented.** The five repositories flagged `license_gap` in the manifest show an explicit license gap until the owner supplies the real declaration. No license is inferred from siblings or defaults.
6. **Evidence binding before republish.** Each rewritten card records its source card hash, reviewed source revision, and observed Hub revision, following the existing curated-card profile pattern (`publishing/curated-model-card-profiles.json`). Publisher rollout requires an explicitly owned, reviewed publisher route with compare-and-swap, exact readback, and owner confirmation, per `docs/curated-model-card-publication.md`.

## Batch plan (proposed, each batch owner-gated)

- **Batch 0 (this PR):** documentation and manifest only. No card bytes, no Hub writes, no publisher changes.
- **Batch 1:** the six receipt-verified repositories plus `SZLHOLDINGS/WILLAY` (curated profile exists) — full-evidence cards.
- **Batch 2:** the 33 repositories with an atelier source classification — cards aligned to their atelier source card and class.
- **Batch 3:** the 38 covered repositories with ≥1 evidence JSON read and no receipt binding yet — per-model observation pass first, then honest not-measured-or-evidenced cards.
- **Batch 4:** the 63 covered repositories with zero evidence reads plus the 31 repositories new since the coverage pass — training contracts from wave logs where they exist; otherwise explicit not-measured cards.

## Non-changes

- No Hub repository, card, dataset, or Space bytes are changed by this PR.
- No workflow, publisher manifest, permission, credential, gate, or source pin is changed.
- The wave-4 card stamping path itself is unchanged in this PR; locating and constraining that writer is a separate follow-up recorded here as open work.
- Nothing in this PR claims runtime, weight, compatibility, or release qualification for any repository.

## Open work (explicit)

- Map the writer that stamped all 138 cards in the 2026-11-14 window and bring it under this schema, so a future stamp cannot overwrite curated cards.
- Publisher mapping for the 132 repositories still `UNMAPPED` in the coverage pass.
- License declarations for the five `license_gap` repositories (owner input required).
- Per-model observation pass for Batch 3/4 repositories.
