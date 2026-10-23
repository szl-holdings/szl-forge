# Amendment requests against the registered v3 protocol

Everything here is SYNTHETIC. `v3/PREREGISTRATION.md` (sha256
`24c239e7e57049180b609b70e1b283b1d3f61253122e9e189232c69a1a3e1d0e`, registered at
`970ce6ce533f77c2c679f27d31ba86b7c43bb7a4`, tag `oac-v3-registered`) and the seal re-registration
`v3/REREGISTRATION_r1.md` (at `80eeaf12be7dc4d73e5f81e2bda9c05b6257ab23`, tag
`oac-v3-registered-r1`) are not edited. The requests below are undecided (section 15.2). Until a
decision is recorded in `v3/AMENDMENTS.md`, every lane follows the registered text, and where the
registered text is silent, it follows the reading stated in the request and discloses it.

Each request below is, in the requester's reading (MODELED), a clarification under section 15.4:
it concerns a detail that section 15.3 does not fix, and it changes no frozen item. Each must be
decided and, if accepted, committed to `v3/AMENDMENTS.md` before the step it affects.

Source: the build plan `v3/BUILD_PLAN.md` (ws v3a-setup, 2026-10-23).

## AR-1: which segment holds ABL-n's 40 descriptive retrains (affects steps 4 to 6)

- **What (REPORTED from the registered text).** Section 6.6 declares 227 fits in one bounded
  sequence "in three committed segments (section 13 steps 4, 5 and 6)": the main fit (1), the
  ablation fits (6), 100 band retrains, 80 known-good retrains (20 for each of the four H1 arms),
  and ABL-n's 40 descriptive retrains on the first 16,384 rows of `train_band_001..020` and of
  `train_kg_01..20`. Section 13 assigns the main and ablation fits to step 4, the band retrains to
  step 5 and the known-good retrains to step 6. It names no segment for ABL-n's 40 retrains.
- **Requested clarification.** Segment 1 (step 4): the 7 fits of section 6.6, in the order main,
  ABL-D, ABL-R, ABL-DR, ABL-n, ABL-jitter, ABL-calibration's Platt fit. Segment 2 (step 5): the
  100 band retrains `train_band_001..100` in index order, then ABL-n's 20 retrains on the first
  16,384 rows of `train_band_001..020` in index order (120). Segment 3 (step 6): for k = 01..20 in
  order, the four H1 arms v3 (`train_kg_k`), ABL-D (`train_kg_k_noD`), ABL-R (`train_kg_k`) and
  ABL-DR (`train_kg_k_noD`), then ABL-n's 20 retrains on the first 16,384 rows of
  `train_kg_01..20` (100). The total stays 227, and the cap stays 227.
- **Why it changes no frozen item.** The counts, the splits, the roles and the recipe are
  unchanged; only the segment that each declared fit belongs to, and the order, are written down.
  ABL-n's retrains are descriptive and enter neither the tolerance vector nor H1.

## AR-2: a segment interrupted by a host-side failure (affects steps 4 to 6)

- **What (REPORTED).** Section 6.6 allows only the stop reasons SEQUENCE_EXHAUSTED and
  CAP_REACHED, and section 15.6 covers code defects. The registered text does not say what
  happens when a segment's run stops for a reason outside the code, such as a host restart, a
  full disk or a killed session. Session stops from account usage limits happened repeatedly
  between 2026-10-14 and 2026-10-22 (REPORTED: `v3/REREGISTRATION_r1.md`).
- **Requested clarification.** An interrupted run is recorded with stop reason `INTERRUPTED` and
  its partial trace is kept and committed as evidence. It counts as no segment result. The whole
  segment is then re-run from its first fit at the same commit P, from a clean tree, with the
  runtime pins. Every attempt is reported. The cap of 227 applies to each attempt, not to their
  sum. Because the recipe is deterministic (section 6.7 item 9), each fit that completed in the
  interrupted attempt must give a byte-identical `model.json` in the re-run, and the re-run checks
  this; a difference is a defect under section 15.6.
- **Why it changes no frozen item.** No fit is added, dropped or chosen. The segment's result is
  the first complete run, and the earlier attempt cannot be selected against, because a complete
  run is fixed by P and the seeds.

## AR-3: the freeze manifest's list of earlier tags includes `oac-v3-registered-r1` (affects step 8)

- **What (REPORTED).** Section 3.5 says the freeze manifest lists "the target commit and the tag
  object id of every earlier `oac-v3-*` tag", followed by the list as it stood at registration:
  `oac-v3-registered`, `oac-v3-fit`, `oac-v3-band`, `oac-v3-tolerances` and `oac-v3-kg`. The seal
  re-registration added the tag `oac-v3-registered-r1` (tag object
  `7a036f228886b8d29b501aee3277dc1983c21175`, target `80eeaf12…`, MEASURED with `git ls-remote`
  on 2026-10-23).
- **Requested clarification.** The manifest lists every `oac-v3-*` tag that exists before
  `oac-v3-frozen`, `oac-v3-registered-r1` included, and any later `oac-v3-registered-rN`.
- **Why it changes no frozen item.** The rule "every earlier `oac-v3-*` tag" already covers it;
  the parenthesis predates the tag. The manifest format (canonical JSON, the fields, the digest
  rule) is unchanged.
