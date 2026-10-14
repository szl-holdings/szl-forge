# v3 preregistration: decisions for Stephen's green light

Everything is SYNTHETIC. No v3 code, data or fit exists, and nothing below is a result. The
reasons are in `v3/DESIGN_RATIONALE.md`, and the review dispositions in `v3/REVIEW_DISPUTES.md`.

**Status: all eight items DECIDED on 2026-10-13.** Stephen replied "do all your recommendations"
to this list (REPORTED: the orchestrating session's record of his reply; his message as relayed
verbatim to the amending run reads "al ll your ecoemdatiosn do it"). Each item below keeps the
recommendation he approved, then the decision as recorded. Item 5 differs: Stephen gave no salt,
so the orchestrating session decided the seal under his reply (ORCHESTRATOR DECISION).
The preregistration records the same decisions in `v3/PREREGISTRATION.md` §18. It was
registered on 2026-10-14 by the commit that renamed `v3/PREREGISTRATION.DRAFT.md` to that file;
the public part of registration is the commits and tags of prereg §13 step 1 and §14.2.

**Update 2026-10-14.** An adversarial review of the beacon-seal amendment found that item 5's
timing proof (GitHub's server record of the tag push) expires and does not name the tag's
target, so it could have let the research process force a new round after seeing round R. The
orchestrating session decided a durable timing proof under the same reply, and the fixing run
adjusted two details with stated reasons. Item 5 below records the result; prereg §18 items 9–12
and `v3/REVIEW_DISPUTES.md` (second part) hold the details.

1. **Green light to register.**
   - Recommended: register by the rename commit, the registration receipt, and the public pushes
     and tags of prereg §14.
   - **DECIDED 2026-10-13:** register. The public pushes and annotated tags of prereg §14 are
     approved. Each push still logs its command, diff, tests and rollback before it runs; szl-forge
     is fetched again before every push; no force-push; no push to `main` (prereg §14.2).
2. **Win-condition design.**
   - Recommended: WIN iff P0–P3 and H1–H3 all hold, with δ = 0.005 BA, +0.002 Brier, +0.01 ECE,
     H3 margins 0.005 BA and 0.002 Brier, and H2's zero-misclassification criterion. Every
     non-inferiority condition has three readings (shown, inconclusive, cost beyond the margin).
     Modeled chance of WIN if every lever behaves as modeled: about 0.6–0.85 (CONJECTURE).
   - Alternative (not taken): relax H2 to allow at most one known-good false alarm. That removes
     H2's main risk (about 0.02–0.03, MODELED), but it weakens the claim.
   - **DECIDED 2026-10-13:** the recommended design, with strict H2: zero known-good
     misclassification (prereg §7 H2, §8, §12).
3. **Sizes.**
   - Recommended:
     - 100 band retrains, up from 20, to set the gate tolerances;
     - a separate H3 twin set of 100,000 pairs;
     - test 100,000 rows; four shift splits of 25,000 rows each;
     - 400,000 sealed rows in all, about 132 MB (CONJECTURE), generated only at the opening.
   - Alternative (not taken): keep 20 band retrains and H3 on 25,000 pairs. The modeled
     known-good false-alarm chance is then about 0.09–0.10, and H3's power at a true 0.002 is
     about 0.5–0.75 (MODELED).
   - **DECIDED 2026-10-13:** the recommended sizes (prereg §4, §11.3).
4. **Compute and disk budget.**
   - Recommended: 227 declared fits; the FRAGILE check capped at 72 hours of wall-clock time,
     with the outcome NOT_COMPLETED if it does not finish; the opening preflight requires at
     least 2 GB free. C: had 12 GB free on 2026-10-13 (MEASURED:
     `logs/verify/v3-prereg-review-b/checks_20261013.txt` §13).
   - **DECIDED 2026-10-13:** the recommended budget (prereg §6.6, §8, §13 step 9).
5. **Seal (formerly "salt custody").**
   - Recommended: Stephen generates SALT himself, outside every session and every repository,
     and gives only `sha256(SALT)` for registration and SALT only to the opening run.
   - **DECIDED 2026-10-13, replaced.** Stephen supplied no salt. The orchestrating session
     decided, under his reply, to keep the no-peek guarantee without custody: SALT is the
     randomness of a future round of the drand default mainnet chain, fixed at registration.
     - Chain hash `8990e7a9aaed2ffed73dbd7092123d6f289930540d7651336225dc172e51b2ce`, scheme
       `pedersen-bls-chained`, period 30 s, genesis_time 1595431050 (MEASURED: fetched live on
       2026-10-13 from `https://<relay>/info` and `https://<relay>/{chain hash}/info` at
       `api.drand.sh`, `drand.cloudflare.com` and `api2.drand.sh`, all identical; drafting
       copies: `logs/v3/register/`).
     - Target 2026-10-24T00:00:00Z. R = 6578966, the first round scheduled at or after the
       target; its scheduled time is exactly the target (MEASURED: computed from those
       parameters; drafting copy: `logs/v3/register/round_R_20261013.txt`).
     - Seal rule: the frozen recipe, gate tolerances, mutation suite, scorer and analyzer are
       committed and tagged publicly in szl-forge as `oac-v3-frozen`, and durable records show
       them fixed and public before R's scheduled time. Otherwise SEAL_EXPOSED: the opening does
       not proceed on round R, and the seal is re-registered with a new future round, disclosed.
     - **Timing proof (updated 2026-10-14; prereg §3.5).** Not GitHub's event or activity records,
       which expire and name no tag target. Instead, both of:
       - an OpenTimestamps proof of the freeze manifest digest M (the manifest lists the commit
         F, its tree, the tag object and the sha256 of every frozen file), submitted to at least
         two public calendars on different domains before R, with the pending proofs pushed and
         tagged `oac-v3-seal`, and upgraded to a Bitcoin block attestation. The cutoff is the
         median header time of blocks H to H + 10, at least 2 hours before R
         (2026-10-23T22:00:00Z). The orchestrator's rule compared the attesting block's own
         median-time-past with R; that median looks back and can admit a digest made about an
         hour after R, so the fixing run replaced it (ADJUSTED, reason in rationale D15);
       - a public archive record (a Software Heritage visit or Internet Archive captures), dated
         by the archive at least 1 hour before R, showing the frozen and seal tags and their
         targets. OpenTimestamps alone cannot show that only one frozen state was timestamped,
         or stop a valid proof from being withheld; the archive record does both (ADDED by the
         fixing run, reason in rationale D15).
       - The seal check is decided once, by frozen code, from these durable records only; a
         record that expires after R can never create SEAL_EXPOSED. The registration digest is
         timestamped the same way. A draft pull request quoting M is a pointer, never proof.
       - Planned: the freeze digest is submitted by 2026-10-22T00:00:00Z, 48 hours before R.
     - The opening fetches rounds R and R + 1 from all three relays, needs
       `drand.cloudflare.com` and a `drand.sh` relay to answer and agree, checks
       randomness = sha256(signature) and that round R + 1's previous signature is round R's
       signature, verifies the BLS signature if the frozen tree holds a standard-library
       verifier (else says it did not, and that anyone with the public key can), and only then
       generates the sealed splits.
     - Re-fetched on 2026-10-14 at 00:04 UTC from two relays: byte-identical (MEASURED; drafting
       copies: `logs/v3/register/recheck_info_*`).
     - A seal re-registration by choice is allowed only while R's scheduled time is still in the
       future. After it, a new round is allowed only after SEAL_EXPOSED or if the chain
       objectively never produced R (daily relay captures over 14 days); host failures, refused
       preflights, relay problems and a missed opening window never allow one (prereg §3.6,
       §15.3).
     - Opening window (2026-10-14): the opening runs within 7 days after R; a miss without an
       external cause is a PROTOCOL_DEVIATION (DOES_NOT_WIN), never a new round (prereg §3.6).
     - Fix window (2026-10-14): frozen-state fixes end for good at the first registered round's
       scheduled time, under any later round too. Runtime pins (CPython 3.12.10, `python -I`,
       `PYTHONHASHSEED=0`) and a byte-identical independent re-run decide which run is official
       (prereg §13 step 9, §15.6).
     - Timing risk (MODELED): every step up to the freeze, the 227 fits and the mutation suite
       included, must be public, and the freeze digest submitted, by about 2026-10-22, 8 days
       after 2026-10-14. If the freeze is sealed only after a registered round's time, the verdict
       carries the label "frozen after round R's randomness, a full sealed-equivalent sample, was
       public" and no fix is allowed (prereg §15.3). Missing the seal is no longer free. Since
       the registration of 2026-10-14, the target instant moves only by a disclosed seal
       re-registration while R's scheduled time is still in the future (prereg §15.3; rationale
       D15).
     - Trust model and limits: the drand threshold network (collusion of enough operators could
       compute R early, undetectably here); Bitcoin's header times under an honest majority, and
       the archive's own record date, for the timing proof; relay agreement between
       `drand.cloudflare.com` and a `drand.sh` relay, plus the BLS check only if the frozen tree
       holds a verifier. Tag protection is not relied on (rationale D15).
     - This supersedes every step in which Stephen would supply SALT or `sha256(SALT)` (prereg
       §3.2, §3.4, §13, §14.8, §15.3, §16).
6. **Tag protection** for `oac-v3-*` in szl-forge. It is a repository setting that makes moving
   a tag harder. (Corrected 2026-10-14: an earlier version said that the origin rule and the
   seal's timing proof rest on tags not being moved. A repository administrator can change a
   ruleset, so neither does: both use the identities fixed in the anchored freeze manifest, and a
   moved tag is detected; prereg §3.5, §14.4, §14.6.)
   - **DECIDED 2026-10-13:** approved. A ruleset on `refs/tags/oac-v3-*` that blocks update and
     deletion, with an empty bypass list, must be in force before `oac-v3-registered` is pushed;
     the registration receipt records its id and state (prereg §14.6, §13 step 1).
7. **Which drafting-only files are published** under `ops-health/v3/evidence/` before
   registration. The candidates were the cited excerpts of `REPORT.md`, `COORDINATION.md` and
   `LANDING_PLAN.md`, `literature/*`, `drafts/forum_thread.md`, and the two review evidence
   folders. Any file not published is re-cited to a public source instead.
   - **DECIDED 2026-10-13:** publish only `literature/*`, `logs/verify/v3-prereg-review-a/`,
     `logs/verify/v3-prereg-review-b/` and `drafts/forum_thread.md`. `REPORT.md`,
     `COORDINATION.md` and `LANDING_PLAN.md` stay internal; every citation of them in the draft
     and the rationale is re-cited to a public source, or relabeled UNKNOWN, before registration
     (prereg preamble, §14.2). Done on 2026-10-14, together with the other non-public paths that
     prereg §14.2 lists (`CONTRACT.md`, `logs/landing/…`, `logs/v3/register/*`).
8. **Size limit for per-row score files.** Above the limit, the opening commit publishes their
   digests only (prereg §14.7).
   - **DECIDED 2026-10-13:** 50 MB per file. A file of at most 50 MB is committed; a larger one
     is published as its sha256, byte size and row count, and stays regenerable from the public
     code and the public beacon round. v2's largest score file is 30,537,167 bytes for 50,000
     `test` rows (MEASURED: `v2/results/final/scores_test.jsonl`), so v3's 100,000-row `test`
     file and its two H3 files are predicted to exceed the limit (MODELED, prereg §14.7).
