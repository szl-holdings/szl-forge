# v3 preregistration draft: review dispositions and disputes

Scope: the 22 issues that review A (13: statistics and design) and review B (9: evidence,
contract and provenance) raised against `v3/PREREGISTRATION.DRAFT.md` and
`v3/DESIGN_RATIONALE.md` at commit `63a5f0d`. Review evidence:
`logs/verify/v3-prereg-review-a/modeled_arithmetic_20261013.txt` and
`logs/verify/v3-prereg-review-b/` (`checks_20261013.txt`, `ledger_20261013.txt`). Everything is
SYNTHETIC. No v3 code, data or fit exists. This file claims no result.

## Disputes

**None.** Every issue was checked against its cited source, and every one holds. The cited v2
lines say what the reviewers quote (REPORTED or MEASURED in those files). Two numeric statements
are qualified below; neither changes a fix.

## Qualifications (the fix is applied; one detail differs from the reviewer's text)

| issue | reviewer statement | qualification and evidence |
| --- | --- | --- |
| B5 (t₁₉ false alarms) | "roughly 10–20%" chance of at least one known-good false alarm | That range treats the 220 comparisons as independent (ledger §C). Review A's quadrature, which shares the reference and the SD across fixtures, gives about 0.087 for the parameter bands plus about 0.009 for the threshold band, about 0.09–0.10 (MODELED: review A evidence §1). The draft cites A's figure and mentions B's. Both agree that the original 1–2% was wrong, and the fix is the same. |
| B1 (kernel contract) | "Those probes are dict payloads" | `v2/research/gates.py` lists the probe classes and says field names carry dummy numeric values; it does not say "dict payloads" (REPORTED: docstring lines 25–29). The substance holds: no listed class is a duplicate-key or authority-map check, and v2 tests both separately (REPORTED: `v2/tests/test_failclosed_probes.py`, `v2/tests/test_review_hardening.py`). The draft now says exactly that. |

## Variants (the reviewer offered options, or a choice between reviewers was needed)

| issue | options offered | adopted | why |
| --- | --- | --- | --- |
| A1 / B5 | ≥ 100 band retrains, or c from t₁₉ at a family-wise target | 100 band retrains, c = 4 kept, a stated family-wise target ≤ 0.05 | c from t₁₉ needs c ≈ 4.3–4.6 and raises the tls band miss to roughly 0.07–0.12; retrains cost no disk (rationale D6) |
| A2 | an H3-only set of about 100,000 pairs, or a larger margin justified by a minimum relevant cost | the 100,000-pair set (`h3_clock_skew`, `h3_clock_skew_latent`) | no minimum relevant cost above 0.005 BA has been stated; about 66 MB at the opening (CONJECTURE) |
| A3 | thresholds at exactly 0.90 per-class test coverage inside each resample, or recalibrate v2 and TRUTH on v3's `conformal` | the ranking-only statistic A\* | recalibration on a shared split still leaves one calibration draw's noise in the comparison (rationale D9) |
| A4 / B9 | freeze at registration; or forbid DEV fits at registered sizes; or commit DEV outputs | all three: freeze at registration, DEV cap of 2,048 rows per regime with no P/H statistic, every DEV output committed | they are complementary; a re-registration is allowed only before any v3 data exists |
| A13 / B7 (non-convergence) | A: count as a failure, not replaced; B: count as a failure or replace from pre-declared spare seeds | not replaced; it makes H2 (and for a known-good retrain, H1) NOT SUPPORTED | a spare seed reopens a choice; the modeled chance of non-convergence is about 0 |
| B7 (pre-opening failure) | proceed to the opening, or end as DOES_NOT_WIN with the salt retired | proceed to the single opening | the full record is reported, and a failure cannot become a reason to redesign with the same salt |
| B3 | require first-parent(C) = recorded HEAD; optionally C on the named tag's first-parent chain | both, in the stronger form: the named tag's target is exactly C | a squash duplicate can share the tree and the first parent; only the commit hash differs. The guarantee rests on the tag not moving (tag protection is Stephen's decision) |
| B8 | publish the cited drafting files, or re-cite public evidence | a registered rule: each is published under `ops-health/v3/evidence/` or re-cited before registration; Stephen chooses which | publishing drafting files is a remote write and a publication choice; the draft takes no remote action |
| B4 (5.2 GB) | save the `df` output under logs/, or relabel the figure | the figure is withdrawn and replaced by review B's saved measurement (12 GB free; MEASURED: `logs/verify/v3-prereg-review-b/checks_20261013.txt` §13) | this fixer runs no commands that generate evidence; the saved measurement already exists |

## Disposition of every issue

| issue | severity | fixed where |
| --- | --- | --- |
| A1 H2 false alarms understated | major | prereg §4, §6.6, §7 H2, §8 power, §11.3; rationale D5, D6, D10, §6 |
| A2 H3 power underived | major | prereg §3.2, §4, §7 H3, §8; rationale D8, D10 |
| A3 H4 does not hold coverage fixed | major | prereg §7 H4; rationale D9 |
| A4 forking paths (recipe not frozen; DEV pilot) | major | prereg §13, §15.3, §15.5, §17; rationale D13 |
| A5 re-run rule | major | prereg §11.4, §13, §15.6; rationale D13 |
| A6 failure readings | major | prereg §16; rationale D10, §3 |
| A7 bootstrap stream | minor | prereg §8, A22; rationale D11 |
| A8 float threshold band | minor | prereg §6.4, §11.2, §11.3; rationale D6 |
| A9 H2 scope and framing | minor | prereg §1, §7 H2, §9, §11.2; rationale D6, §3, §4 |
| A10 sealed-derived numbers; "keeps most" | minor | prereg §8, §17; rationale D10 |
| A11 exchangeability scope | minor | prereg §6.5, §7 H3, §9, A25; rationale D8 |
| A12 D2 shrinkage; P2 power | minor | prereg §8; rationale D2, D4, D10, §6 |
| A13 open rules | minor | prereg §6.1, §6.7, §11.2, §16; rationale D13 |
| B1 kernel contract | major | prereg §5, §6.2, §6.7, §11.1, §11.2, §11.4, §7 H2; rationale D6, D14 |
| B2 "every contender degrades" | major | prereg §17, A7; rationale §1, §5 |
| B3 origin uniqueness | major | prereg §11.4, §13, §14.4, §14.6; rationale D7 |
| B4 labels | minor | prereg §7 H1, §15.2, §15.4, A9, A29; rationale D3, D10, D13 |
| B5 t₁₉ false alarms | minor | as A1 |
| B6 salt custody | minor | prereg status block, §3.4, §13, §14.8; rationale D1 |
| B7 negative-path escape hatches | minor | prereg §6.1, §8, §13, §16; rationale D11, D13 |
| B8 public home and citations | minor | prereg status block, preamble, §13, §14.1–§14.2; rationale §0, D7 |
| B9 DEV window | minor | as A4 |

## Dated note, 2026-10-14: rows overtaken by the decisions of 2026-10-13

The rows above are kept as written. These rows no longer describe the draft:

- **B6 (salt custody).** The Stephen-held salt that B6's fix introduced was replaced by the
  beacon seal on 2026-10-13, and its timing proof was replaced on 2026-10-14 (prereg §3.4–§3.6;
  rationale D1, D15). The row's pointers are historical.
- **B7 (pre-opening failure, "with the salt retired").** The disposition stands: the opening
  proceeds whatever H1 and H2 show. There is no salt to retire; the seal round can change only
  under prereg §3.6 and §15.3.
- **B3 ("tag protection is Stephen's decision").** Decided on 2026-10-13 (item 6, prereg §18).
  Corrected on 2026-10-14: the origin guarantee and the seal do not rest on tag protection; they
  use the identities anchored in the freeze manifest (prereg §3.5, §14.4, §14.6).
- **B8 ("Stephen chooses which").** Decided on 2026-10-13 (item 7, prereg §18). The re-citations
  were made on 2026-10-14 (prereg preamble, §14.2).

## Second review: the beacon-seal amendment (v3-register-review, 2026-10-14)

Scope: the 7 issues (1 critical, 2 major, 4 minor) that the review raised against the amendment
commit `e5dbf85`. Review evidence: `logs/verify/v3-register-review/`. Evidence gathered by this
fix: `logs/verify/v3-register-fix/`, whose `sources_20261014.txt` names the public URL of every
fact. Both folders are drafting copies, outside the decision-7 publish list; every claim in the
draft that uses them also names its public source (prereg §14.2). The fix also applies the
orchestrating session's design decisions of 2026-10-14 for the timing proof. Everything is
SYNTHETIC. No v3 code, data or fit exists, and this file claims no result.

### Disputes

**None.** Every issue was checked against the draft text and the review's evidence, and every one
holds. The fix differs from a suggested fix, or from the orchestrator's design, in the details
below.

### Qualifications and adjustments

| item | suggested | adopted, and why |
| --- | --- | --- |
| Issue 1, anchors | at least 2 independent anchors, at least one outside GitHub, each at least 1 h before R; candidates: a GitHub commit comment, a Sigstore entry, an RFC 3161 token, a Software Heritage visit | both anchors outside GitHub and both required: T1, an OpenTimestamps proof (the orchestrator's choice), with a 2-hour margin on a forward median; T2, a public archive record (Software Heritage or Internet Archive), at least 1 hour before R. A commit comment can be deleted by its author or an administrator without a public trace, so it is context only. Sigstore and RFC 3161 were workable, but need a signing workflow or one authority's clock and key (rationale D15) |
| Issue 1, point (4) | SEAL_EXPOSED only when no qualifying anchor predates R | the seal check returns SEALED, PENDING or EXPOSED, once, from records fixed before R; because both anchors are required, EXPOSED also covers a missing T2 (prereg §3.5) |
| Orchestrator design (1), cutoff | the attesting block's median-time-past < R's scheduled time | ADJUSTED: the median of blocks H to H + 10, at least 2 hours before R. The attesting block's own median-time-past covers H and the 10 blocks before it, so it lags the moment H was mined by about an hour and could admit a digest made after R (MODELED, from the block time rule REPORTED at `https://developer.bitcoin.org/reference/block_chain.html`; rationale D15) |
| Orchestrator design (2), uniqueness | the protected tag ruleset makes the tag immutable; a draft PR is a pointer only | ADJUSTED: a ruleset can be changed by a repository administrator, so it does not make a tag immutable, and an OpenTimestamps proof cannot show that only one frozen state was timestamped, or stop a valid proof being withheld. T2 was ADDED for both. The draft pull request stays a pointer only (prereg §3.5) |
| Issue 2, chain link | require round R + 1's previous signature to equal round R's signature | adopted; checked on live rounds 6551371 and 6551372 at all three relays (MEASURED; drafting copy: `logs/verify/v3-register-fix/drand_chain_link_check_20261014.txt`). It adds consistency across rounds, not authenticity (rationale D15) |
| Issue 2, opening window | the opening waits, or ends ABANDONED | combined with the orchestrator's 7-day window: a miss without an external cause is a PROTOCOL_DEVIATION (DOES_NOT_WIN) and the opening still runs; ABANDONED applies only if it can never run; neither allows a new round (prereg §3.6, §16) |
| Issue 2, memory | a numeric threshold | at least 4 GiB of free physical memory. The host has 31.6 GiB, with 5.3 GiB free when measured (MEASURED 2026-10-14); the threshold itself is a judgment (CONJECTURE), informed by v2's MemoryError (prereg §13 step 9) |
| Issue 3, late freeze | close the fix window at the first R and label; or treat as a re-attempt with a new master seed | the first: no change under any round after the first registered round's scheduled time, and a label on every statement of the verdict whenever a registered round passed before the freeze was sealed. A new master seed would discard H1, H2 and every fit for no further gain (rationale D13) |
| Issue 4, relays | `drand.cloudflare.com` plus a `drand.sh` relay; disagreement means a hold; persistent conflict goes to a review under a rule fixed before R | adopted; the fixed rule: after 7 days of conflict, the value that drand's own client verifies against the registered public key is used and labeled RELAY_CONFLICT_RESOLVED; none or more than one is SEAL_COMPROMISED. That the two `drand.sh` relays share an operator stays a CONJECTURE (prereg §3.4) |
| Issue 7, pins | pin the exact CPython version, `python -I`, `PYTHONHASHSEED=0`; byte-identical re-run, for example in CI | CPython 3.12.10, the opening host's version (MEASURED 2026-10-14); the re-run is planned on a GitHub-hosted Windows runner, because math library results can differ between platforms (MODELED); the verdict is PROVISIONAL until the re-run matches (prereg §13 step 9, §15.6) |

### Disposition of every issue

| issue | severity | fixed where |
| --- | --- | --- |
| R1 the timing proof is not publicly checkable, and an expired record could force SEAL_EXPOSED after R | critical | prereg §3.3, §3.4, new §3.5, §13 steps 1, 8 and 9, §14.4, §14.6, §14.7, §14.8, §15.3, §16, §17, §18 item 9; rationale D1, D7, D13, D15, §4, §6, §7; decisions items 5 and 6 |
| R2 the 14-day rule and the preflight let the process re-roll after R | major | prereg §3.4 (relay disagreement), new §3.6, §13 step 9 (memory threshold), §15.3, §16, §18 item 10; rationale D13, D15, §6; decisions item 5 |
| R3 a late freeze gives a full rehearsal before the new round | major | prereg §15.3, §15.5, §15.6, §16, §18 item 12; rationale D13, D15, §6; decisions item 5 |
| R4 relay independence and disagreement | minor | prereg §3.3, §3.4, §16, §17, §18 item 11; rationale D15; decisions item 5 |
| R5 cited files that are not public (and the orchestrator's re-citation of `REPORT.md`, `COORDINATION.md` and `LANDING_PLAN.md`) | minor | prereg preamble, §3.4, §6.7, §11.1, §13 step 1, §14.1, §14.2, §14.5, A18, A27; rationale §0, §1, D1, D7, D14, D15; decisions items 5 and 7 |
| R6 overtaken rows in this file | minor | the dated note above |
| R7 the opening environment is not pinned | minor | prereg §13 step 9, §15.3, §15.6, §18 item 12; rationale D13, §4; decisions item 5 |
