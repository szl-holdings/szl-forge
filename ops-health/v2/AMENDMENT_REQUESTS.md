# Amendment requests against the frozen preregistration

`v2/PREREGISTRATION.md` is frozen (sha256 `145497789c7ad78a80a94cfd7f2b031ca9a20df49c06e0b3412af9070fa1ef4b`,
re-measured unchanged on 2026-09-29) and was not edited. The problems below are requests for
Stephen to decide. If he accepts one, it becomes a dated entry in section 12 of the
preregistration, made before the sealed splits are opened. Until then, each is a disclosure item
for REPORT.md. Source: the stage-1 review, `logs/v2/review/STAGE1_REVIEW.md`.

## AR-1: the registration receipt records an upstream path that uses the banned word

- **What (MEASURED).** `v2/PREREGISTRATION.receipt.json`, key `v1_generator_source`, records the
  v1 generator path in `szl-forge` at `9a6fdd2c`. That path's top-level package directory name
  contains the word that CONTRACT §6 bans from v2 paths, names, and cards. It is the only hit of
  a case-insensitive search of `v2/` outside v1's authority-map key (evidence:
  `logs/v2/review/STAGE1_REVIEW.md`, finding A7).
- **Why it was not fixed in place.** The receipt is registration evidence. It was committed with
  the protocol in `bca9225`, so rewriting it would change the registration record.
- **Requested amendment.** Record the upstream path with the placeholder
  `{V1_GATEWAY_PACKAGE_DIR}/tools/train_operational_health_model.py`, as
  `v2/research/v1_gd.py` already does. Keep the commit `9a6fdd2c7bdb1660de009cef974d7df46866960a`.
  Alternatively, accept the upstream citation as-is, as a provenance pointer that is not a v2
  path, name, or card. `v2/tests/test_review_hardening.py::ForbiddenWordTest` whitelists exactly
  this one line, pending the decision.

## AR-2: the IRLS stop reason LINE_SEARCH_FAILED is not named in section 6

- **What (MEASURED from source).** `v2/research/lr.py` fits by Newton steps with step halving. If
  60 halvings fail to decrease the objective, it stops with `LINE_SEARCH_FAILED`. Section 6 names
  only convergence (max |Δθ| < 1e-10) and the 100-iteration cap. The kernel accepts all three
  stop reasons, and records `converged` as false for both non-convergent ones.
- **Requested clarification.** Treat `LINE_SEARCH_FAILED` exactly like `CAP_REACHED`: it is
  reported and never silently accepted. On DEV data, the four IRLS fits (l2 = 0, 1e-4, 1e-3,
  1e-2) each converged, in 6 or 7 iterations (`logs/v2/review/stats_reference_after.txt`).

## AR-3: the "tampered metrics" receipt mutant has no metrics field to tamper with

- **What (MEASURED).** Section 10 lists "tampered metrics" among the receipt-corruption mutants.
  The v2 receipt deliberately carries no metrics, because receipts attest integrity and origin
  only (section 6). An added or altered `metrics` key is refused by the exact-key check
  (`logs/v2/review/kernel_attack_after.txt`, `receipt_metrics_key_added`).
- **Requested clarification.** In stage 2, realize the mutant as "a metrics block added to the
  receipt", which the integrity layer must catch.
