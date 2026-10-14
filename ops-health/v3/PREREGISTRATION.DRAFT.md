# OAC Ops Health v3: Preregistration

<!-- status:begin -->
Status: **DRAFT. NOT registered.** This is the last draft. Stephen gave the green light on
2026-10-13: he replied "do all your recommendations" to `v3/DECISIONS_FOR_GREEN_LIGHT.md`
(REPORTED; section 18). Registration fixes this text by one commit that renames this file to
`v3/PREREGISTRATION.md` and changes nothing in it except the lines between the `status:begin`
and `status:end` markers. The registration receipt `v3/PREREGISTRATION.receipt.json` records
the registered file's sha256. The public part of registration (section 13 step 1, section 14.2)
commits the last draft and then the registered bytes, unchanged, to szl-forge. Commit ids are
recorded in the receipts, never in these lines, so that the registered bytes are the same in
every repository. The seal is fixed in the body (section 3.4: a future drand beacon round), so
no secret, and no commitment that a person supplies, enters these lines. Any other byte
difference between the last draft and the registered file is a defect that blocks registration.
<!-- status:end -->

Everything here is SYNTHETIC. No result from this protocol may be generalized to any real
transport, device, site, population, or workflow. This file contains **no v3 result**: no v3
code, no v3 data, no v3 fit and no v3 pilot exist. Every statement about what v3 will do is a
prediction and is labeled MODELED (reasoning) or CONJECTURE (weaker reasoning).

What was seen before drafting: the whole v2 record, including the v2 sealed-split results,
which are public (`v2/results/FINAL_RESULTS.md`), the v2 generator, bootstrap and kernel source
(`v2/research/generator.py`, `v2/research/bootstrap.py`, `v2/ops-health/ops_health.py`, read as
code; no data file was opened), and the literature matrix. Because v2's sealed results are known,
**no v3 decision may use any v2 split**, sealed or not. v3 is scored only on new data drawn from
new seeds (sections 3 and 4). Section 17 lists every registered number that was derived from
v2's sealed results.

Labels. Each claim about v2 carries the label it has in the cited evidence file:
**MEASURED** (the file records it as computed from saved outputs), **REPORTED** (read from the
cited file), **MODELED** (reasoning, ours or the file's), **CONJECTURE**, **UNKNOWN**. The design
reasons, the arithmetic behind every MODELED number and the literature anchors are in
`v3/DESIGN_RATIONALE.md`. Appendix A lists every v2 fact this file relies on.

Paths and citations.

- Paths are relative to the research root. In the public home (section 14.1) the research root
  is `ops-health/` in `szl-holdings/szl-forge`, so `v2/results/FINAL_RESULTS.md` there is
  `ops-health/v2/results/FINAL_RESULTS.md`.
- Some cited files exist today only in the drafting repository. Stephen decided on 2026-10-13
  (section 18) which are published:
  - **Published** before registration, byte-identically under `ops-health/v3/evidence/` at
    their research-root paths (for example `ops-health/v3/evidence/literature/anchors.json`),
    each with its sha256 in the registration receipt: `literature/*`, `drafts/forum_thread.md`,
    and the two review evidence folders `logs/verify/v3-prereg-review-a/` and
    `logs/verify/v3-prereg-review-b/`. A citation of one of them resolves under
    `ops-health/v3/evidence/`.
  - **Not published:** `REPORT.md`, `COORDINATION.md` and `LANDING_PLAN.md` are internal. On
    2026-10-14 every citation of them in this file and in the rationale was replaced by a public
    source (a szl-forge path at a pinned commit, a public pull request, or a public URL), or the
    claim was labeled UNKNOWN. The same holds for every other cited path that is not public in
    szl-forge, listed in section 14.2.
  - **Beacon and timing evidence.** The drand facts of section 3.4 are cited to the relay URLs
    that serve them, and the timing-proof facts of section 3.5 to the public services that hold
    them. The raw responses saved while drafting (`logs/v3/register/`,
    `logs/verify/v3-register-review/`, `logs/verify/v3-register-fix/`) are drafting references
    only. The registration receipt records the sha256 of each saved raw beacon response
    (section 13 step 1).

## 1. Question

v2 won its registered comparison, with measured weaknesses (Appendix A). Can a v3
operator-attention advisory that keeps v2's model class and selected hyperparameters, and adds
preregistered changes aimed at those weaknesses, do all of the following on fresh sealed
synthetic data?

- (a) Identify the weight on `seconds_since_last_success` in every known-good retrain (H1).
- (b) Let reference-relative parameter and decision gates, and not only hashes, catch the
  registered mutant classes that v2's gates missed, without flagging any known-good retrain
  (H2).
- (c) Keep its robustness to a monitoring clock that runs ahead, even though (a) makes it use
  that field (H3).
- (d) Stay non-inferior to v2 as-is on balanced accuracy and calibration, and stay superior to
  v1 as-is, the majority baseline and the rule baseline (section 8).

A fourth hypothesis asks whether v3 abstains less than v2 at the same per-class coverage, or
whether v2's abstention is already at the floor the generator imposes (H4, reported, not part of
the verdict).

## 2. Frozen facts about the data-generating process (generator truth)

v3 reuses v2's generator truth unchanged and **adds no new truth**: no new field, no new weight,
no new label rule, and no change to any v2 regime. The rule below is copied from
`v2/PREREGISTRATION.md` section 2 (REPORTED).

The eight fields are normalized as in the v1 kernel: binary fields as 0/1, and continuous fields
min-max over queue_utilization [0,1], consecutive_failures [0,20] and seconds_since_last_success
[0,86400]. Labels are binary draws from p(x) = sigmoid(z), where

```
z = -3.6 + 3.1(1-listener_running) + 1.15(1-tls_enabled) + 1.35(1-peer_allowlist_configured)
    + 4.2 queue_utilization + 2.7 consecutive_failures + 2.2 seconds_since_last_success
    + 4.4(1-ledger_integrity_ok) + 3.4(1-configuration_valid)
```

Equivalent parameterization (REPORTED, `v2/PREREGISTRATION.md` §2): intercept 9.8, weights
listener_running -3.10, tls_enabled -1.15, peer_allowlist_configured -1.35, queue_utilization
+4.20, consecutive_failures +2.70, seconds_since_last_success +2.20, ledger_integrity_ok -4.40,
configuration_valid -3.40.

On v3's representation of seconds (section 6.2, seconds divided by 900), the same truth weight is
2.2 × 900 / 86400 = **0.0229167 per 900 s** (MODELED, arithmetic). Every truth comparison in
this protocol maps fitted weights to a common scale first.

Recorded in advance (MODELED): logistic regression on these fields stays correctly specified
under every v3 change, because every v3 feature transform is linear (section 6.2).

## 3. Generator v3 (`ops-health-generator/3.0.0`)

### 3.1 What is reused

The per-row draws, their order, the rounding, the four shift regimes (`queue_saturation`,
`failure_bursts`, `clock_skew`, `missing_tls`) and the label rule are exactly those of v2's
generator `ops-health-generator/2.0.0` (REPORTED: `v2/PREREGISTRATION.md` §3,
`v2/research/generator.py`). v2's generator source sha256 is `a502d364…4384` (MEASURED:
`v2/results/STAGE2_SEARCH.md`, protocol table). Shift regimes are never used for training,
calibration, conformal calibration, threshold selection or gate tolerances.

### 3.2 What is added (none of it changes the truth)

1. **Namespace and master seed.** v3 uses seed namespace `oac-ops-health-v3` and master seed
   **M3 = 20261013**. For a non-sealed split s, the seed is
   `int.from_bytes(sha256(f"oac-ops-health-v3:{M3}:{s}".encode()).digest()[:8], "big")`.
2. **Beacon-salted seeds for sealed splits.** For a sealed split s, the seed material is
   `f"oac-ops-health-v3:{M3}:{s}:{SALT}"`, where SALT is the `randomness` field of drand round
   R = 6578966 (section 3.4), as 64 lower-case hex characters (32 bytes). Nobody holds SALT, and
   nobody supplies it. It does not exist before R's scheduled time, 2026-10-24T00:00:00Z (MODELED,
   under the trust model of section 3.4), so no sealed row can exist before then. Sealed rows are
   generated only inside the opening run (section 13 step 9).
3. **Training-design mode D** (for training splits only). Each row draws, in this order:
   listener_running, tls_enabled, peer_allowlist_configured, queue_utilization and
   consecutive_failures as nominal; then `s_nom = round(min(86400, Exp(mean 900)), 6)`; then
   `b ~ U(0,1)`; then `s_wide = round(U(0, 86400), 6)`; then ledger_integrity_ok and
   configuration_valid as nominal; then `u ~ U(0,1)`. The **D arm** uses latent seconds
   `s_wide` if `b < 0.25`, else `s_nom`. The **no-D arm** always uses `s_nom`. In each arm the
   label is `u < p(latent features of that arm)`. Both arms are written as two paired files from
   one stream, so they differ only in the rows with `b < 0.25`. Observed equals latent in both
   arms: D changes the true covariate design, not a measurement.
4. **Clock-skew twin mode** (the two sealed H3 splits only). The v2 `clock_skew` draw is
   unchanged. The generator writes the observed (skewed) row to `h3_clock_skew` and the same row
   with its latent seconds to `h3_clock_skew_latent`. Both rows carry the same label, which
   comes from the latent features as in v2. The skew value itself is never written. The pooled
   suite's `shift_clock_skew` is a plain v2 `clock_skew` draw and has no twin.
5. **v2 seal guard.** The v3 generator refuses namespace `oac-ops-health-v2` with any v2 sealed
   split name (`test` or `shift_*`). Every v3 data path refuses paths under `v2/data/sealed/`.

### 3.3 Parity and guard tests (any failure blocks every fit)

- v1 parity: the v1-parity mode reproduces the v1 public rows byte for byte, as in v2 §3.
- v2 parity, non-sealed only: with namespace `oac-ops-health-v2` and M = 20260926, the v3
  generator reproduces the digests that `v2/data/MANIFEST.json` records for `train`,
  `calibration`, `conformal` and `validation`. These rows are generated only to compare
  digests: no fit and no statistic is computed on them. No v2 sealed split is ever regenerated.
- Twin parity: on DEV seeds and within the DEV cap of section 15.5, the observed file of twin
  mode equals the v2 `clock_skew` output byte for byte.
- D-mode check: in both arms every field except seconds is identical row by row, and the no-D
  arm's seconds equal the `s_nom` draws.
- Guard tests: every refusal in item 5 above fires.
- Beacon tests, on rounds recorded before the freeze only (round R is never fetched by a test):
  the beacon client and verifier of section 3.4 accept recorded responses of at least three
  earlier rounds of the registered chain, each with the response of the next round, and refuse
  each of: a flipped signature byte; a randomness that is not sha256(signature); a round number
  other than the one requested; a chain hash or public key other than the registered one; two
  relays that disagree; a relay set without `drand.cloudflare.com` or without a `drand.sh`
  relay; and a next-round response whose previous signature is not the round's signature. The
  sealed mode of the generator accepts SALT only from the verifier's output for the registered
  chain and round R.
- Seal-check tests, on records made before the freeze only: the timing-proof checker of section
  3.5 accepts the registration digest's upgraded proof (section 13 step 1) with headers recorded
  from two explorers, and returns SEALED, EXPOSED or PENDING as section 3.5 says. It refuses each
  of: a changed byte in a proof operation; a merkle root that differs from the header's; a header
  whose hash or proof of work fails; a break in the chain of headers from H to H + 10; two
  explorers that disagree; a forward median later than the cutoff; an archive record dated after
  its cutoff; and an archive record that shows another tag target.

### 3.4 Manifest and beacon seal

Before any fit, `v3/data/MANIFEST.json` is committed with: the v3 generator source sha256; the
sha256 of every non-sealed split file; for each retrain split, its sha256 only (the files are
regenerated on demand, not stored); and the sealed-split definitions (names, regimes, sizes, seed
material template) with the beacon seal below.

**The beacon seal (no custody).** No person generates, holds or supplies SALT. SALT is the
randomness of one future round of the drand default mainnet chain, fixed here:

```
chain hash     8990e7a9aaed2ffed73dbd7092123d6f289930540d7651336225dc172e51b2ce
scheme         pedersen-bls-chained (beacon id "default")
group hash     176f93498eac9ca337150b46d21dd58673ea4e3581185f869672e59fa4cb390a
period         30 s
genesis_time   1595431050 (2020-07-22T15:17:30Z)
target         2026-10-24T00:00:00Z (unix 1792800000)
R              6578966
R scheduled    genesis_time + (R - 1) * period = 1792800000 = 2026-10-24T00:00:00Z
public key (BLS12-381 G1, compressed, 48 bytes):
868f005eb8e6e4ca0a47c8a77ceaa5309a47978a7c71bc5cce96366b5d7a569937c529eeda66c7293784a9402801af31
```

- **Source.** The chain parameters were fetched live on 2026-10-13 at 23:48 UTC (relay clock)
  from `https://api.drand.sh/info`, `https://drand.cloudflare.com/info` and
  `https://api2.drand.sh/info`, and from `https://<relay>/{chain hash}/info` on the same three
  relays. All six responses agree on every field above (MEASURED from those URLs; drafting copies
  and the computation: `logs/v3/register/`, `logs/v3/register/round_R_20261013.txt`). On
  2026-10-14 at 00:04 UTC, `/{chain hash}/info` re-fetched from `api.drand.sh` and
  `drand.cloudflare.com` was byte-identical to the first responses (MEASURED; drafting copies:
  `logs/v3/register/recheck_info_*`). Anyone can re-fetch them from the relays.
- **R.** R is the first round whose scheduled time, genesis_time + (R − 1) × period, is at or
  after the target instant (MEASURED: computed from the fetched parameters; drafting copy: same
  file). The target is fixed; the round is not chosen.
- **Hash relation and chain link, checked on live rounds.** For round 6550143
  (`https://<relay>/{chain hash}/public/6550143`), all three relays served the same round,
  signature, previous signature and randomness, and randomness = sha256(signature) (MEASURED;
  drafting copy: `logs/v3/register/round_R_20261013.txt`). On 2026-10-14 at 10:03 UTC the same
  held for round 6551371 at all three relays, and the previous signature that round 6551372
  carried was round 6551371's signature (MEASURED from the same URL form; drafting copy:
  `logs/verify/v3-register-fix/drand_chain_link_check_20261014.txt`).

**Seal rule.** The frozen state is tagged `oac-v3-frozen` in public szl-forge, and its timing
proof (section 3.5) shows it fixed and public before R's scheduled time. The frozen state is:
every frozen contender, ablation and the TRUTH reference; the gate tolerances; the mutation suite
and the kernel-relevant test list; the scorer; the analyzer (all of step 10's registered
computations, the FRAGILE check included); the beacon client and verifier with their tests and
the verification level below; the timing-proof checker of section 3.5 with its tests; and the
runtime pins of section 13 step 9. Evidence of "before" is only the durable third-party records
of section 3.5. Git author, committer and tagger dates, which the author sets, and GitHub's
expiring event and activity records are never that evidence. If the timing proof does not hold,
the outcome is SEAL_EXPOSED (section 16): the opening does not run on round R.

**Opening fetch and verification** (frozen code; section 13 step 9).

- The opening fetches `/{chain hash}/public/6578966` and `/{chain hash}/public/6578967` from all
  three relays above. It decides that round R is available from the relays' responses (section
  3.6), never from the opening host's clock. On 2026-10-13 that clock ran about 12 s behind the
  relays' `Date` headers (MEASURED; drafting copies:
  `logs/v3/register/probe_latest_*_headers_20261013.txt`).
- **Relay rule.** Answers from `drand.cloudflare.com` and from at least one of `api.drand.sh` and
  `api2.drand.sh` are required, and every relay that answers must agree with the others on round,
  randomness, signature and previous signature. Agreement of the two `drand.sh` relays alone is
  not enough: both send `Server: nginx/1.28.3`, and they may share an operator, while
  `drand.cloudflare.com` sends `Server: cloudflare` (headers MEASURED 2026-10-13 and 2026-10-14;
  drafting copies: `logs/verify/v3-register-review/live_*_headers.txt`,
  `logs/verify/v3-register-fix/drand_latest_*_headers_20261014.txt`; operators UNKNOWN).
- Each raw response is saved.
- Checks: the round is R; randomness = sha256(signature), on the bytes the hex strings encode;
  round R + 1's previous signature equals round R's signature, at every answering relay (if round
  R + 1 is not yet served, the opening waits for it); and `/{chain hash}/info`, fetched again,
  still matches every value above.
- **Verification level, fixed in the frozen tree before R.**
  - HASH_AND_RELAY_AGREEMENT is the registered minimum: the relay rule, the hash relation and the
    chain link above. The report then says that the frozen code did not verify the BLS
    signature, and that anyone holding the registered public key can verify it, for example with
    drand's own client.
  - BLS_VERIFIED applies if the frozen tree holds a standard-library BLS12-381 verifier for the
    chained scheme. It hashes sha256(previous signature ‖ R as 8 bytes big-endian) to G2, with
    the hash-to-curve suite that drand's scheme specification names for this scheme, and checks
    the signature against the registered public key. Its tests pass on at least three rounds
    recorded before the freeze, and it refuses the mutated inputs of section 3.3. The exact
    construction is settled by those recorded rounds, not by this text.
  - A standard-library BLS verifier is not required. Prediction (CONJECTURE): it is feasible, but
    it is new code with its own defect risk (`v3/DESIGN_RATIONALE.md` D15).
- SALT is set to the randomness, as 64 lower-case hex characters. Only then are the sealed splits
  generated.
- **Relay disagreement or silence.** If answering relays disagree, or a required relay does not
  answer, the opening holds and retries at least daily, saving every response. It never uses
  another round, another chain or another source. If, after 7 days, registered relays still
  serve conflicting round-R values, or a required relay is still silent, the round-R value is the
  one, among those that registered relays serve, whose signature verifies against the registered
  public key under drand's own client, run outside the frozen tree, with the tool's name,
  version, inputs and output committed. The opening then proceeds with that value, and the
  report labels the verification RELAY_CONFLICT_RESOLVED. If no value verifies, or more than one
  does, the outcome is SEAL_COMPROMISED (section 16). A relay problem never permits a new round
  (section 3.6).

**What the seal guarantees, and what it rests on** (MODELED; reasons in
`v3/DESIGN_RATIONALE.md` D15).

- Nobody, the research process included, can know SALT before R's scheduled time, so no sealed
  row can exist before then.
- If the seal check of section 3.5 is SEALED, one frozen state, identified by its commit, its tag
  object and its manifest digest, was fixed and public before SALT existed. Nothing that decides
  the verdict can change after that.
- No secret is held, so there is no custody to audit. For v2, whether anything outside the guard
  read a sealed file early is UNKNOWN (Appendix A20).
- After the opening, anyone can regenerate every sealed row and recompute every score from the
  public code and the public round, and anyone can check the timing proof from public records.
- It rests on: (i) the drand threshold network: if enough of its operators colluded, they could
  compute round R early, and this protocol cannot detect that; (ii) Bitcoin's block header times,
  under an honest majority of mining power (section 3.5); (iii) the date that the public archive
  assigns to its own record (section 3.5); (iv) the relays serving the true round: the relay
  rule and the chain link, plus the signature check if the level is BLS_VERIFIED.
- It does not rest on tag protection. A repository administrator can move a tag. The opening and
  every check use the identities fixed in the anchored manifest (section 3.5), so a moved tag is
  detected, not trusted.
- It does not prevent tuning to the distribution (section 17).

### 3.5 Timing proof of the freeze (durable records only)

**Frozen identity.** The frozen state is identified by three values, never by a tag's current
target:

- F, the full commit id that `oac-v3-frozen` targets;
- the full id of the annotated tag object `oac-v3-frozen`;
- M, the sha256 of the freeze manifest.

**Freeze manifest.** `v3/seal/FREEZE_MANIFEST.json` is canonical JSON: the bytes that
`json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()` writes,
with no trailing newline. It lists:

- the tag name, F, F's tree id and the tag object id;
- the sha256 and byte size of every file tracked at F under `ops-health/`, by path;
- the target commit and the tag object id of every earlier `oac-v3-*` tag (`oac-v3-registered`,
  `oac-v3-fit`, `oac-v3-band`, `oac-v3-tolerances` and `oac-v3-kg`);
- the beacon seal of section 3.4: the chain hash and R.

M is the sha256 of the manifest's bytes. Because every file is listed by sha256, the identity
does not rest on git's SHA-1 ids alone.

**Anchors made before R. Both are required.**

1. **T1, an OpenTimestamps proof anchored in Bitcoin.**
   - M, 32 bytes, is submitted to at least two public OpenTimestamps calendars under different
     domains: at least one of `https://a.pool.opentimestamps.org` and
     `https://b.pool.opentimestamps.org`, and at least one of `https://a.pool.eternitywall.com`
     and `https://ots.btc.catallaxy.com`. A submission sends only the digest; no file content
     leaves the host. The reference client posts the digest to `<calendar>/digest` and later
     fetches the upgrade from `<calendar>/timestamp/<commitment>` (REPORTED: the public
     reference library `opentimestamps/python-opentimestamps` at commit `3af46432`,
     `opentimestamps/calendar.py`). The calendars' operators are UNKNOWN beyond their domains.
   - Right after the submission, the manifest and the pending proofs, and nothing else, are
     committed as one commit Q on top of F. Q is pushed and tagged `oac-v3-seal` (annotated).
   - Each proof is upgraded to a Bitcoin block attestation as soon as its calendar serves the
     upgrade. Anyone can do this from the public pending proof in Q. The upgraded proofs are
     committed.
   - **Cutoff rule.** A proof attests height H when replaying its operations from M yields the
     merkle root of block H's header. Let T_fwd(H) be the median of the header times of the 11
     blocks H, H + 1, …, H + 10 (the 6th smallest of the 11). T1 holds iff, for at least one
     calendar, T_fwd(H) ≤ R's scheduled time − 2 hours, that is, at or before
     2026-10-23T22:00:00Z.
   - **Why a forward median.** The proof shows only that M existed before block H was mined. A
     header time is set by the miner: it must exceed the median of the previous 11 blocks' times,
     and nodes reject headers more than two hours in the future by their own clock (REPORTED:
     `https://developer.bitcoin.org/reference/block_chain.html`, block header field "time"). So
     block H's own time, and the median of H and the 10 blocks before it, can be earlier than
     the moment H was mined. Blocks H + 1 to H + 10 were all mined after H. With an honest
     majority of mining power, the median of H to H + 10 is at or after the moment H was mined
     (MODELED). The 2-hour margin covers clock error and small misreporting (MODELED). The
     reference client reports the attesting header's own time, which this rule does not use
     (REPORTED: same library, `opentimestamps/core/notary.py`).
   - **Headers.** Headers H to H + 10 are fetched by height from at least two public block
     explorers named in the frozen tree (planned: `https://blockstream.info/api` and
     `https://mempool.space/api`). They must agree byte for byte; each header's double sha256
     must meet the target that its `bits` field encodes; and each must name the one before it.
     On 2026-10-14 both explorers served the same header for block 971386, and a
     standard-library check of its hash and proof of work passed (MEASURED; drafting copies:
     `logs/verify/v3-register-fix/bitcoin_header_*`).
   - **Planning** (not a rule). In the week before 2026-10-14, the calendars sent a Bitcoin
     transaction every 0.71 h (alice, behind `a.pool.opentimestamps.org`), 1.7 h (bob, behind
     `b.pool.opentimestamps.org`), 6.22 h (finney, behind `a.pool.eternitywall.com`) and 8.84 h
     (`ots.btc.catallaxy.com`) on average (MEASURED: each calendar's status page, fetched
     2026-10-14; drafting copies: `logs/verify/v3-register-fix/ots_calendar_*`). The freeze
     digest is therefore planned for submission by 2026-10-22T00:00:00Z, 48 hours before R, so
     that an attestation lands well before the cutoff (MODELED). The cutoff alone decides.
2. **T2, a public archive record.** At least one record made by a public archive, dated by the
   archive itself at or before R's scheduled time − 1 hour (2026-10-23T23:00:00Z), that shows
   `refs/tags/oac-v3-frozen` pointing at the tag object recorded in the manifest, whose target
   is F, and `refs/tags/oac-v3-seal` pointing at a tag object whose target is Q, where Q's tree
   holds the manifest with digest M and the pending proofs. Accepted records:
   - a Software Heritage visit of `https://github.com/szl-holdings/szl-forge`, read from the
     anonymous API `https://archive.softwareheritage.org/api/1/`. Its snapshot maps each tag
     name to a release, which is the git tag object, and the release to its target commit. This
     mapping was MEASURED on 2026-10-14 on another public origin, where the release id equals
     the git tag object id (drafting copies: `logs/verify/v3-register-fix/swh_example_*`). The
     visit is requested with "save code now" right after Q is pushed. szl-forge had no visit and
     no save request on 2026-10-14 (MEASURED: the same API; drafting copies:
     `logs/verify/v3-register-fix/swh_*_szl-forge_20261014.json`);
   - Internet Archive captures, dated by the archive, of
     `https://api.github.com/repos/szl-holdings/szl-forge/git/ref/tags/oac-v3-frozen` and of the
     same URL for `oac-v3-seal`, showing the tag object ids. The tag objects are then read from
     any copy and checked against their ids. The Internet Archive holds captures of
     `api.github.com` URLs (MEASURED 2026-10-14 with its CDX API; drafting copy:
     `logs/verify/v3-register-fix/wayback_cdx_api_github_example_20261014.txt`).

   A tag name that any qualifying record shows with two different targets does not qualify.

**Why both.** T1 shows that M existed before R, without trusting any one service's clock. It
cannot show that M was the only manifest timestamped: a process could timestamp several frozen
variants privately, and publish one of them after seeing round R. T2 shows that the name
`oac-v3-frozen` pointed at F, and that the pending proofs were public, before R. After that, no
variant can take the name, and nobody can withhold the proofs, because anyone can upgrade them
from Q (MODELED; `v3/DESIGN_RATIONALE.md` D15).

**Not evidence of time.** Git author, committer and tagger dates; a release's `created_at`, which
equalled the commit date on the public release that the review probed (MEASURED by the review;
drafting copy: `logs/verify/v3-register-review/gh_timing_evidence_probes_20261014.txt`); GitHub's
events and activity APIs, which keep a limited window and record no tag target (on 2026-10-13
the 300 most recent szl-forge events covered 227.4 hours, and the activity API listed no tag
refs; MEASURED by the review; drafting copies: `logs/verify/v3-register-review/gh_*`); any
capture that exists only as a file that the research process committed; Hugging Face commit
dates; and any record made after its cutoff. Such records may be saved as context only. A draft
pull request from
`claude/oac-v3-research`, opened at the freeze and never merged, quotes F, the tag object id and
M. It is a pointer for readers only.

**Seal check: decided once, by frozen code, from durable records.** In the opening preflight
(section 13 step 9), the timing-proof checker in the frozen tree computes M from F, upgrades the
pending proofs in Q (or reads the committed upgraded proofs), fetches the headers and the archive
records, saves every input raw, and returns:

- **SEALED** iff T1 and T2 hold;
- **PENDING** iff a calendar, an explorer or an archive cannot be reached, the explorers
  disagree, or block H + 10 is not yet mined for a proof that could still meet the cutoff;
- **EXPOSED** otherwise: no qualifying T1, or no qualifying T2.

PENDING is retried at least daily. From R's scheduled time + 7 days on, an unreachable calendar or
archive counts as non-qualifying, so the result is SEALED or EXPOSED. The first committed result
that is not PENDING decides; an EXPOSED result decides only when a second check, at least 24
hours later, returns EXPOSED for the same reason, so that one faulty answer cannot decide it.
Every input to it was fixed by events before R's scheduled time, so
the result cannot depend on the round's value, and a record that expires or goes missing after R
never yields EXPOSED by itself (MODELED). A later run of the checker only confirms the result; a
different result is a defect under section 15.6.

**Fixes and registration.**

- A frozen-state fix (section 15.6) is a frozen state of its own: a fix tag `oac-v3-frozen-fN`,
  its seal tag `oac-v3-seal-fN`, its own manifest, and its own T1 and T2.
- At registration, the registration digest gets the same OpenTimestamps treatment (section 13
  step 1). It is the sha256 of the canonical JSON of: the registered file's sha256, the
  registration receipt's sha256, the registration commit and the `oac-v3-registered` tag object
  id. The report gives the registration's time as T_fwd of its attesting block.

### 3.6 Round availability, the opening window and new rounds

- **Available.** Round R is available once any registered relay serves a round-R response that
  passes the checks of section 3.4, or serves any later round. In the chained scheme, the
  response for round R + 1 carries round R's signature as its previous signature. Availability
  is a fact about the relays, not about the opening host.
- **Opening window.** The opening run (section 13 step 9) runs within 7 days after R's scheduled
  time. The digest of its record, `runs/v3/TEST_OPENED.json`, is submitted to OpenTimestamps as in
  section 3.5 right after the run. The window is met iff T_fwd of that proof's attesting block is
  at or before R's scheduled time + 8 days; the extra day lets the calendars attest.
- **External causes extend the window.** Each must be shown by raw captures, made at least daily
  and committed: round R not yet available (a beacon hold); a relay disagreement or silence under
  section 3.4; or a PENDING seal check. The window then ends 7 days after the last such cause
  ends. Causes on the opening host's side (the preflight, disk, memory, the local network, the
  operator) never extend it.
- **A missed window** without an external cause is a PROTOCOL_DEVIATION: the verdict is
  DOES_NOT_WIN. The opening still runs on round R, and everything is reported.
- **A new round after R's scheduled time** is allowed in two cases only (section 15.3):
  - the seal check returns EXPOSED;
  - the chain objectively never produced round R: every registered relay reports a latest round
    below R at every capture, made at least daily and committed, for 14 days after R's scheduled
    time. If the drand network later restarts and produces round R, the re-registration stands
    and round R is not used.

  No failure of the opening host, no refused preflight, no relay problem and no missed window
  allows a new round. If the opening can never run, the ending is ABANDONED (section 16).

## 4. Splits (fixed)

| split | regime or mode | rows | sealed | use |
| --- | --- | ---: | --- | --- |
| train | nominal, D arm | 32,768 | no | fit v3 |
| train_noD | nominal, no-D arm paired with train | 32,768 | no | fit the no-D ablation arms only |
| calibration | nominal | 4,096 | no | ABL-calibration's Platt fit only |
| conformal | nominal | 8,192 | no | Mondrian quantiles only |
| validation | nominal | 8,192 | no | threshold selection; ablation and conformal keep/drop decisions |
| gates | nominal | 8,192 | no | behavioral gates of section 11 only |
| train_band_001..100 | D arm | 32,768 each | no; digests only | band retrains that set gate tolerances (section 11.3) |
| train_kg_01..20, train_kg_01..20_noD | D arm and paired no-D arm | 32,768 each | no; digests only | known-good retrains for H1 and H2 |
| test | nominal | 100,000 | SEALED | opened once |
| shift_queue_saturation | v2 shift 1 | 25,000 | SEALED | pooled shift suite |
| shift_failure_bursts | v2 shift 2 | 25,000 | SEALED | pooled shift suite |
| shift_clock_skew | v2 shift 3 | 25,000 | SEALED | pooled shift suite |
| shift_missing_tls | v2 shift 4 | 25,000 | SEALED | pooled shift suite |
| h3_clock_skew | v2 shift 3, twin mode, observed rows | 100,000 | SEALED | H3 only; not in the pooled suite |
| h3_clock_skew_latent | latent twin of h3_clock_skew | 100,000 | SEALED | H3 only; not in the pooled suite |
| legacy_v1_test | v1 public test split | 240 | no (public) | secondary only |

- The pooled shift suite is the four `shift_*` splits, 100,000 rows, equal regime weights,
  resampled within regime (section 8).
- The H3 twin set is the 100,000 pairs of `h3_clock_skew` and `h3_clock_skew_latent`, matched
  by row index.
- Sealed rows in all: 400,000, about 132 MB at about 330 bytes per row (CONJECTURE), generated
  only inside the opening run.
- ABL-n uses the first 16,384 rows of `train`, and of `train_band_001..020` and of each
  known-good split for its descriptive H2 check.
- No v3 decision uses any v2 split. `legacy_v1_test` is v1's public split and is secondary only,
  as in v2.

## 5. Contenders

- **v3**: the recipe of section 6, frozen with its threshold and conformal layer before the
  opening.
- **v2 as-is**: kernel sha256 `b0a64ff3…aed9`, `model.json` `b830a5ed…977c`, receipt
  `442486a3…56f2` (MEASURED: `v2/results/STAGE2_SEARCH.md`), threshold 0.20 and its frozen
  Mondrian quantiles q̂₀ = 0.223684755436 and q̂₁ = 0.911152623366 (MEASURED: same file). It is
  scored exactly as its kernel emits.
- **v1 as-is**: the Hub kernel and model at revision `dd7d1098…`, threshold 0.16, with the score
  exactly as the kernel emits it (REPORTED: `v2/PREREGISTRATION.md` §5; `v2/AMENDMENTS.md` AM-5).
- **majority**: always NO_ALERT.
- **rule**: ALERT iff consecutive_failures > 0.
- **Ablation contenders** (section 10): ABL-D, ABL-R, ABL-DR, ABL-n, ABL-calibration and
  ABL-jitter. They are frozen before the opening and scored for confirmation only. They never
  enter the verdict. An ablation whose artifact declares a transform table other than the
  registered one (ABL-R, ABL-DR) is refused by the v3 kernel (section 6.7), so it is scored by
  the research scorer outside the kernel and is INVALID_BASELINE under precondition (i), as v2's
  ABL-optimizer was (MEASURED: `v2/results/FINAL_RESULTS.md`).
- **TRUTH reference** (descriptive only, never a contender): the score p(x) of section 2,
  with a threshold chosen by the section 6.4 rule on `validation` and Mondrian quantiles from
  `conformal`. It shows the ceiling that a correctly specified model can reach on this generator.

## 6. The v3 recipe

### 6.1 Model and fit

L2-regularized binary logistic regression on the eight features, with an unpenalized intercept,
fit by IRLS in the Python standard library with deterministic operation order. Convergence and
eligibility follow v2 §6 and AM-2: max |Δθ| < 1e-10 or 100 iterations; `CAP_REACHED` and
`LINE_SEARCH_FAILED` are non-converged, and a non-converged fit is not eligible (REPORTED:
`v2/PREREGISTRATION.md` §6, `v2/AMENDMENTS.md` AM-2). Registered consequences:

- If the main fit is not eligible, v3 does not exist, and the verdict is DOES_NOT_WIN with reason
  NO_ELIGIBLE_FIT.
- A band retrain or known-good retrain of the v3 recipe that does not converge is **not
  replaced**. It makes H2 NOT SUPPORTED, and a non-converged known-good retrain also makes H1 NOT
  SUPPORTED.
- A non-converged retrain of an ablation arm is reported and excluded from that arm's
  descriptive statistics, with the count.
- Prediction (MODELED): every fit converges, because the L2-penalized mean log loss is strictly
  convex; v2's selected trial converged in 7 iterations (MEASURED: `v2/results/STAGE2_SEARCH.md`).

**Hyperparameters are fixed, not searched**: l2 = 1e-3 (penalty λ/2·‖w‖² on the mean log loss)
and calibrator `none`. These are v2's selected trial T07 (MEASURED:
`v2/results/STAGE2_SEARCH.md`). v3 runs no hyperparameter search.

### 6.2 Representation R

- Binary fields: 0/1.
- queue_utilization: (x − 0)/1.
- consecutive_failures: (x − 0)/20.
- seconds_since_last_success: **x / 900**. The constant 900 s is the declared nominal mean of the
  generator. It is fixed here and is not estimated from data.

The artifact declares this (offset, scale) table, and the receipt binds it. The kernel refuses
any declared table that differs from these registered constants (section 6.7). The kernel's input
range checks stay those of v1 and v2, for example seconds in [0, 86400]. Every transform is
linear, so the model class stays correctly specified (MODELED, section 2).

### 6.3 Training design D

v3 is fit on `train`, the D arm of section 3.2 item 3 with π = 0.25. In about a quarter of the
training rows, the latent seconds are drawn uniformly over the field's whole declared range,
and labels follow the unchanged truth. D is not one of the four shift generators, and no sealed
split uses it.

### 6.4 Threshold

v2 §6 rule, applied on `validation`: t ∈ {0.01, …, 0.99} maximizing balanced accuracy, ties to
higher F1, then to t closer to 0.5 (REPORTED: `v2/PREREGISTRATION.md` §6). The threshold is stored
as the grid index k = 100·t, an integer in 1..99.

### 6.5 Conformal layer

v2 §6 Mondrian (label-conditional) split conformal, α = 0.10, on `conformal`. The score is
s(x,y) = 1 − p̂_y(x), and q̂_y is the ⌈(n_y+1)(1−α)⌉-th smallest class-y score. The advisory is
ALERT iff C = {1}, NO_ALERT iff C = {0}, and ABSTAIN otherwise. Keep rule (v2 ABL-conformal): keep
the layer only if validation coverage is ≥ 0.88 marginal and ≥ 0.85 for each class (REPORTED:
`v2/PREREGISTRATION.md` §6, §9).

**Scope of the guarantee**, as in v2: class-conditional coverage of at least 1 − α holds only
under exchangeability with the nominal conformal split. It holds on nominal data only. Coverage
under any shift, and on `legacy_v1_test`, is descriptive, with no guarantee (REPORTED:
`v2/PREREGISTRATION.md` §6).

### 6.6 Fits, receipts and retrains

- Every declared fit runs in one bounded sequence, in three committed segments (section 13
  steps 4, 5 and 6). The cap is the number of declared fits, 227: the main fit (1); the
  ablation fits of section 10 (6: ABL-D, ABL-R, ABL-DR, ABL-n, ABL-jitter and ABL-calibration's
  Platt fit); 100 band retrains; 20 known-good retrains for each of the four H1 arms (80); and
  ABL-n's descriptive retrains on the first 16,384 rows of `train_band_001..020` and of
  `train_kg_01..20` (40). The only stop reasons are SEQUENCE_EXHAUSTED and CAP_REACHED.
- Each fit emits one governed receipt. Receipts attest integrity and origin, never quality.
- A **known-good retrain** (the v3 form of v2 AM-4) re-runs the recipe end to end: l2 = 1e-3, no
  calibrator, the threshold re-selected on `validation`, and Mondrian on `conformal`. Only the
  train split changes: `train_kg_k` for the D arms, and `train_kg_k_noD` for the no-D arms.
  A **band retrain** is the same, on `train_band_k`.

### 6.7 Kernel contract (frozen at registration)

The v3 kernel keeps every fail-closed property of the v2 kernel and adds one check. The v2 items
are REPORTED from the v2 kernel source `v2/ops-health/ops_health.py` (its imports, its
duplicate-key hook, `AUTHORITY_MAP` and the prohibited-key screen), the v2 tests
(`v2/tests/test_determinism.py`, `v2/tests/test_failclosed_probes.py`,
`v2/tests/test_review_hardening.py`) and `v2/AMENDMENTS.md` AM-6, all public in szl-forge under
`ops-health/`.

1. **Dependency-free.** Python standard library only. A test refuses any other import.
2. **Inputs.** Exactly v2's eight input fields, under v2's names. Missing fields, extra fields,
   non-finite values (NaN, +inf, −inf), out-of-range values and values of the wrong type are
   refused.
3. **Prohibited keys.** v2's prohibited-key screen, with its key normalization and Unicode
   folding, runs before the schema check, so a prohibited name is refused as prohibited and not
   only as an extra field.
4. **Strict JSON.** Duplicate keys are refused, at every level, in inputs and in the artifact.
   Non-finite JSON constants are refused.
5. **Authority map.** The output authority map is v2's five-key, all-false map, unchanged. It is
   defined by reference to the constant `AUTHORITY_MAP` in `v2/ops-health/ops_health.py`, and
   every outcome, ABSTAIN included, carries it.
6. **Outcomes.** ALERT, NO_ALERT and ABSTAIN, with the reason BOTH or EMPTY for ABSTAIN, as in v2.
7. **Transform table.** The kernel refuses, at load, any artifact whose declared (offset, scale)
   table differs from the registered constants of section 6.2 in any entry, has a missing or an
   extra entry, or has a non-finite or non-positive scale.
8. **Receipt.** The receipt pins the source commit and the step tag (section 14.4), and the
   model and kernel hashes.
9. **Determinism.** Training and freezing twice from the same seeds gives a byte-identical
   `model.json`.

**Transform-table probes.** These artifact-level probes join v2's 66 input probes in
`g_failclosed` (section 11.2), 76 probes in all. Each must be refused at load:

- TT1: seconds scale 86400 in place of 900;
- TT2: seconds scale 0;
- TT3: seconds scale −900;
- TT4: seconds scale NaN;
- TT5: seconds scale +inf;
- TT6: seconds offset 1 in place of 0;
- TT7: the tls_enabled scale 2 in place of 1;
- TT8: the seconds entry missing;
- TT9: an extra entry for a field outside the eight;
- TT10: a duplicate key inside the table.

**Required tests** (the v3 suite fails if any is missing or fails): the 76 probes;
duplicate-key refusal in inputs and in the artifact; the all-false authority map on ALERT,
NO_ALERT and ABSTAIN; the standard-library-only import check; determinism; receipt mismatch;
the broken baselines of section 11.1; the mutation suite; metric recomputation. The probe
classes of v2's 66 probes include no duplicate-key class and no authority-map check (REPORTED:
`v2/research/gates.py` docstring); v2 tests those separately (REPORTED:
`v2/tests/test_failclosed_probes.py`, `v2/tests/test_review_hardening.py`), and v3 does the same.

## 7. Hypotheses

Each hypothesis names the v2 weakness it targets, its lever, its exact statistic, its success
criterion, when it is evaluated, and a prediction made now (MODELED unless marked).

### H1: the staleness weight is identified

- **v2 weakness.** v2's weight on seconds_since_last_success is +0.036548, against a truth of
  +2.2 (weight REPORTED: `v2/results/MUTATION_READINESS.md`; truth REPORTED:
  `v2/PREREGISTRATION.md` §2). Three of five known-good retrains have a negative weight
  (−0.003455, −0.009359, −0.013441; weights REPORTED from each `model.json` in
  `v2/results/MUTATION_READINESS.md`), and each of the three fails `g_truth_sign` and
  `g_monotone` (MEASURED: same file). v2 §2 predicted this weak identification in advance
  (MODELED there; REPORTED here).
- **Lever.** D (section 6.3) adds information about the seconds coefficient. R (section 6.2)
  removes the L2 shrinkage that the [0, 86400] min-max scale puts on it.
- **Statistic.** For each fixture f, w_s(f) is the fitted seconds weight on the R scale, and
  r(f) = w_s(f) / 0.0229167. For the no-R arms, the weight is mapped to the R scale first.
- **Fixtures.** For each arm, the arm's frozen fit plus its 20 known-good retrains: 21 fixtures.
- **SUPPORTED iff**, for the v3 arm (D and R):
  1. w_s(f) > 0 for all 21 fixtures;
  2. the median of r over the 21 fixtures lies in [0.75, 1.25];
  3. the sample standard deviation of r over the 21 fixtures is ≤ 0.10;
  4. `g_truth_sign` = 1 and `g_monotone` = 1 for all 21 fixtures;
  5. all 21 fits converged (section 6.1).
- **Ablation contrast** (reported, not part of the criterion): the same statistics for
  ABL-D (R, no D), ABL-R (D, no R) and ABL-DR (neither; the v2 recipe on v3 data).
- **When.** After all known-good retrains, before any mutant is generated and before the opening.
  The result is committed.
- **Predictions (MODELED, `v3/DESIGN_RATIONALE.md` D3, D4).**
  - v3: 21 of 21 positive, median r ≈ 1, standard deviation of r ≈ 0.03.
  - ABL-R: 21 of 21 positive, median r ≈ 0.87, from the remaining shrinkage.
  - ABL-D: about 2 of 21 negative; r is unbiased but spread, with a standard deviation ≈ 0.8.
  - ABL-DR: about 2 of 21 negative, median r ≈ 0.01.
  - So rescaling alone does not identify the sign. If ABL-D shows ≥ 19 of 21 positive and a
    standard deviation of r ≤ 0.10, this account is wrong, and the report must say so.

### H2: reference-relative parameter and decision gates catch the registered blind-spot classes

- **v2 weakness.** 11 BLIND_SPOT mutants (MEASURED: `v2/results/MUTATION_READINESS.md`):
  - weight scaling ×0.5 and ×1.5 on tls_enabled, queue_utilization, consecutive_failures and
    seconds_since_last_success;
  - threshold ±0.1;
  - the supplementary receipt mutant that re-points `source.commit` to the parent commit, whose
    five bound blobs are identical.

  v2's `g_validation` and `g_calibration` are absolute tolerances (BA ≥ 0.70, ECE ≤ 0.05), not
  distances from v2. Every readiness aggregator had a negative margin (MEASURED: same file). With
  the registered gates, the integrity layer, not a behavioral gate, detected a modified artifact
  (MODELED: same file).
- **Lever.** Three reference-relative gates: two in parameter space (`g_coef_band`,
  `g_threshold_band`) and one on outputs (`g_ba_rel`). Their tolerances are set by a fixed rule
  from 100 known-good band retrains before any mutant exists (section 11.3). Also the kernel
  contract (section 6.7) and the v3 origin binding (section 14.4).
- **Scope.** The registered classes are the ones that c, the band count and the training size
  were sized against (`v3/DESIGN_RATIONALE.md` D5, D6). H2 therefore shows what these gates do on
  these classes. Smaller changes are covered only by the descriptive mutants of section 11.4,
  whose outcomes are predicted below and reported, but do not enter H2.
- **SUPPORTED iff all hold:**
  1. Every one of the 36 registered v2-class artifact mutants of v3 (section 11.4) is caught in
     pass 2 by at least one gate of section 11.2 other than `g_integrity`. These include the ten
     artifact members of v2's blind-spot classes.
  2. The registered transform-table artifact mutant (section 11.4) is refused by the kernel in
     pass 2, and `g_coef_band` also catches it.
  3. Every one of the 7 registered receipt mutants is caught by the integrity layer, including
     both re-pointing mutants (to the first parent and to a descendant) and the squash-style
     duplicate.
  4. Every one of the 8 registered kernel-source mutants is caught by integrity or by the
     kernel-relevant tests, whose list is committed before step 7 (section 13).
  5. AND aggregation at its cut (section 12) marks all 21 known-good fixtures (v3 and
     `kg_01..20`) READY and all 37 integrity-bypassed artifact mutants NOT_READY: 0
     misclassifications.
  6. All 100 band retrains and all 20 known-good retrains of the v3 arm converged (section 6.1).
- **When.** After the tolerances are committed and after H1, before the opening.
- **Prediction (MODELED, `v3/DESIGN_RATIONALE.md` D5, D6).** SUPPORTED. The main risk is a
  known-good false alarm, not a detection miss:
  - The chance of at least one known-good false alarm is about 0.02–0.03: about 0.0185 from the
    nine parameter bands, about 0.001–0.01 from the threshold band, and little from the BA band
    (MODELED: `logs/verify/v3-prereg-review-a/modeled_arithmetic_20261013.txt` §1). The retained
    absolute gates add about 0 if v3's ECE on `gates` lies near v2's known-good range
    0.027–0.035 (MEASURED for v2: `v2/results/MUTATION_READINESS.md`,
    `v2/results/STAGE2_SEARCH.md`); v3's own level is UNKNOWN before the fit.
  - With 20 band retrains instead of 100, the same chance would be about 0.09–0.10 (MODELED:
    same file §1), because each tolerance would use an SD estimated with 19 degrees of freedom.
  - Detection margins: tls_enabled scaling is caught by `g_coef_band` (ρ ≈ 5.4) and by
    `g_ba_rel`; consecutive_failures scaling by `g_coef_band` (ρ ≈ 7.1). With 100 band
    retrains, the chance that an SD estimate is large enough to hide either is about 0.
  - Seconds scaling and consecutive_failures ×1.5 are predicted to be caught only by
    parameter-space gates. v2's BA drops for the seconds mutants were 0.0007 and 0.0000
    (MEASURED: `v2/results/MUTATION_READINESS.md`).
- **Predicted detection floor per feature (MODELED, ρ table of `v3/DESIGN_RATIONALE.md` D6).**
  At c = 4, `g_coef_band` catches a scaling ×k of feature j when |1 − k| > 2/ρ_j, where ρ_j is
  the detection ratio of a ×0.5 change:

  | feature | ρ_j at 32,768 rows | smallest caught \|1 − k\| | ×0.75 / ×1.25 by the band |
  | --- | ---: | ---: | --- |
  | listener_running | 14.4 | 0.14 | caught |
  | tls_enabled | 5.4 | 0.37 | missed |
  | peer_allowlist_configured | 6.2 | 0.32 | missed |
  | queue_utilization | 12.1 | 0.17 | caught |
  | consecutive_failures | 7.1 | 0.28 | missed |
  | seconds_since_last_success (with D) | 11.4 | 0.18 | caught |
  | ledger_integrity_ok | 11.8 | 0.17 | caught |
  | configuration_valid | 14.1 | 0.14 | caught |

  The ×0.75 and ×1.25 mutants of the three features that the band misses (six mutants) are
  predicted to escape `g_ba_rel` too, because their BA drops should fall below its 0.005 floor
  (CONJECTURE). Threshold ±0.05 (5 grid
  steps) is predicted caught by `g_threshold_band` if its tolerance is at most 4 steps (predicted
  3; MODELED, rough); −0.05 is likely also caught by `g_ba_rel`, and +0.05 is borderline
  (CONJECTURE).

### H3: clock-skew robustness is kept

- **v2 weakness.** On `shift_clock_skew`, v2 and v1 cannot be told apart: v2 0.7564
  [0.7455, 0.7665], v1 0.7498 [0.7399, 0.7595], paired v2 − v1 BA +0.0066 [−0.0013, +0.0146]
  (MEASURED: `v2/results/FINAL_RESULTS.md`).
- **Reading (MODELED).** Both models put almost no weight on seconds; v2's is +0.036548
  (REPORTED: `v2/results/MUTATION_READINESS.md`). The skew therefore barely moves either model,
  and 12,500 rows give wide intervals. H1 makes v3 use the field that the skew corrupts. So the
  v3 question is whether identifying the weight costs robustness.
- **Lever.** None aimed at the sealed shift. v3 does not train on any skew-shaped perturbation
  (`v3/DESIGN_RATIONALE.md` D8). A dedicated twin set measures the cost with power.
- **Statistics, on the H3 twin set (100,000 pairs, resampled as pairs; section 8):**
  - ΔBA_skew(m) = BA(m, h3_clock_skew_latent) − BA(m, h3_clock_skew);
  - ΔBrier_skew(m) = Brier(m, h3_clock_skew) − Brier(m, h3_clock_skew_latent).
- **Standard error (MODELED, `v3/DESIGN_RATIONALE.md` D8).** For a pair i of class y, let
  D_i(m) = 1[m correct on the latent twin] − 1[m correct on the skewed twin]. Then
  ΔBA_skew(m) = Σ_y Σ_{i in y} D_i(m) / (2 n_y), and the H3 statistic
  ΔBA_skew(v3) − ΔBA_skew(v2) has standard error ≈ ½·√(φ₁/n₁ + φ₀/n₀), where φ_y is the
  class-y rate at which v3's decision differs between the twins (v2's decisions almost never
  differ). At a flip rate of 1.8–3.5%, with 20% of flips in class 1, the standard error is
  0.00113–0.00156 on 25,000 pairs and **0.00057–0.00078 on 100,000 pairs**. The Brier arm's
  standard error is about 4.9e-5 on 25,000 pairs and about 2.5e-5 on 100,000 pairs (MODELED:
  `logs/verify/v3-prereg-review-a/modeled_arithmetic_20261013.txt` §3).
- **SUPPORTED iff:**
  - the upper bound of the paired 95% interval of ΔBA_skew(v3) − ΔBA_skew(v2) is ≤ 0.005; and
  - the upper bound of the paired 95% interval of ΔBrier_skew(v3) − ΔBrier_skew(v2) is ≤ 0.002.
- **When.** At the opening.
- **Prediction (MODELED).** SUPPORTED.
  - ΔBA_skew(v3) is about +0.0007 to +0.0011 from the second-order term, plus up to
    ±0.0012–0.0018 if the selected threshold sits about 0.1 logit from the BA optimum: about
    −0.0011 to +0.0029 in all.
  - ΔBrier_skew(v3) ≈ 0.0002.
  - Class-0 conformal coverage of v3 on `h3_clock_skew` is about one percentage point below its
    coverage on `h3_clock_skew_latent` (rough). This is a descriptive prediction: under the skew
    the conformal guarantee does not apply (section 6.5).

### H4: abstention at fixed per-class coverage (reported; not part of the verdict)

- **v2 weakness.** On nominal test, the conformal layer abstains on 0.4143 of rows, all of them
  BOTH sets, with class coverage 0.9003 (class 0) and 0.9061 (class 1) (MEASURED:
  `v2/results/FINAL_RESULTS.md`).
- **Recorded before any v3 data (MODELED, exact algebra; `v3/DESIGN_RATIONALE.md` D9).** A
  strictly increasing recalibration of p̂ leaves every Mondrian prediction set of section 6.5
  unchanged. Better calibration therefore cannot reduce abstention at the same α. Only a better
  ranking can, and the TRUTH reference bounds what any ranking can do.
- **Statistic A\*(m), which depends on ranking only.** It is computed on `test` and recomputed
  inside every bootstrap resample from that resample's counts. Let N_y be the class-y count of
  the resample, counting multiplicity.
  - a₁(m) is the ⌈0.90·N₁⌉-th largest p̂_m among class-1 rows; a₀(m) is the ⌈0.90·N₀⌉-th
    smallest p̂_m among class-0 rows.
  - The fixed-coverage set of a row is {1 if p̂_m ≥ a₁(m)} ∪ {0 if p̂_m ≤ a₀(m)}.
  - A\*(m) is the share of rows whose set is not a single class (BOTH or EMPTY).
  - Each class's coverage is then at least 0.90 by construction, for every model, so the three
    models are compared at the same per-class coverage. A\* is unchanged by any strictly
    increasing map of p̂ (MODELED, exact). It is an evaluation statistic that uses test labels
    to place the cut-offs. It is not a deployable rule.
- **H4a, reduction.** SUPPORTED iff the upper bound of the paired 95% interval of
  A\*(v3) − A\*(v2) is < −0.02. Prediction (MODELED): NOT SUPPORTED, because v3 keeps v2's model
  class and l2, and the seconds signal on nominal rows is too small to change the ranking
  materially.
- **H4b, floor.** SUPPORTED iff the upper bound of the paired 95% interval of
  A\*(m) − A\*(TRUTH) is ≤ 0.02, for m = v2 and for m = v3. Prediction (CONJECTURE): SUPPORTED.
  If it holds, the abstention that 90% per-class coverage costs (about 41% for v2's operational
  layer) is a property of this generator's class overlap, not a defect of v2.
- **Descriptive only: the operational Mondrian abstention.** Each model's own layer (v2 with its
  frozen quantiles from n₀ = 3,346 and n₁ = 750 calibration rows, MEASURED:
  `v2/results/STAGE2_SEARCH.md`; v3 and TRUTH with quantiles from `conformal`) is reported, with
  its realized per-class coverage. The calibration draw alone gives the v3 − v2 difference of this
  abstention a standard deviation of about 3.3–4.1 percentage points, which the bootstrap
  interval does not contain, because it holds every quantile fixed (MODELED:
  `logs/verify/v3-prereg-review-a/modeled_arithmetic_20261013.txt` §4). That difference is
  therefore never read as a reduction.
- **When.** At the opening.

## 8. Primary metric and win condition

**Primary metric.** As in v2 §7, the primary metric is the balanced accuracy (BA) of each
contender's binary decision: for v3, the section 6.4 threshold decision, not the three-way
advisory.

**Uncertainty: the bootstrap, fully specified.**

- 2,000 paired resamples. Percentile intervals, type 7: linear interpolation between order
  statistics at the 2.5th and 97.5th percentiles.
- **One stream per evaluation set.** Each evaluation set below has its own stream, a fresh
  `random.Random(20261014)`. This is v2's layout: one count-vector stream per evaluation set,
  driven by one `random.Random(seed)` (REPORTED: `v2/results/FINAL_RESULTS.md`, bootstrap line;
  `v2/research/bootstrap.py`).
- **Draws.** In resample b, for each stratum of the set in its listed order, m row indices are
  drawn with replacement from the stratum's m rows as `int(random() * m)`. The draws become one
  count vector per resample, which every contender and every statistic on that set shares.
- **The evaluation sets and their strata:**
  1. `test`: one stratum of 100,000 rows. P0, P1, P3 and H4 use it.
  2. The pooled shift suite: four strata of 25,000 rows each, in this order:
     `shift_queue_saturation`, `shift_failure_bursts`, `shift_clock_skew`, `shift_missing_tls`.
     P0 and P2 use it. The secondary per-regime intervals reuse each stratum's counts from the
     pooled resample.
  3. The H3 twin set: one stratum of 100,000 pair indices. Each drawn index gives its count to
     both rows of the pair. H3 uses it.
  4. `legacy_v1_test`: one stratum of 240 rows. Secondary only.
- A resample on which a statistic is undefined is excluded and counted.
- H4's statistic A\* recomputes its cut-offs inside each resample (section 7).

v2 used this implementation and disclosed it as deviations 8 and 9 (REPORTED:
`v2/results/COMPLIANCE.md`). v3 registers it.

**Conditions** (each computed on the sealed splits at the single opening, except H1 and H2):

| id | condition | criterion |
| --- | --- | --- |
| P0 | superiority over the trivial and prior contenders | for c in {v1, majority, rule}, the lower bound of the paired 95% interval of BA(v3) − BA(c) is > 0, on `test` and on the pooled shift suite |
| P1 | nominal non-inferiority to v2 | the lower bound of the paired 95% interval of BA(v3) − BA(v2) on `test` is > −0.005 |
| P2 | shift non-inferiority to v2 | the same on the pooled shift suite, > −0.005 |
| P3 | calibration non-inferiority to v2 | on `test`: the upper bound of the paired 95% interval of Brier(v3) − Brier(v2) is < +0.002, and ECE(v3) ≤ ECE(v2) + 0.01 as point estimates (10 equal-width bins, last bin closed, count-weighted) |
| H1 | section 7 | SUPPORTED |
| H2 | section 7 | SUPPORTED |
| H3 | section 7 | SUPPORTED |

**Verdict.**

- v3 **WIN** iff P0, P1, P2, P3, H1, H2 and H3 all hold. Otherwise v3 **DOES_NOT_WIN**, and the
  report lists every failing condition with its numbers and its reading (section 16).
- H4a and H4b are reported with their own outcome and never change the verdict.
- A WIN means only this: on this synthetic benchmark, v3 removes the targeted weaknesses without
  losing more than the stated margins against v2, and it stays ahead of v1, majority and rule.
- A WIN never licenses calling v3 more accurate than v2. No surface may say so.

**Resample-stream check (registered).** The verdict is defined by the streams above. After the
registered analysis is committed, the opening analysis repeats P0–P3 and H3 with 100 further
stream sets: set i uses `random.Random(20261014 + i)`, i = 1..100, in the same per-set layout.

- If any of these conditions changes outcome in more than 5 of the 100 stream sets, the verdict
  is labeled **FRAGILE** wherever it appears. The label does not change the verdict.
- The check is memory-light and checkpointed per stream set. Its budget is 72 hours of wall-clock
  time, with one retry per stream set after a memory failure. v2's 100-stream check of W1 alone
  stopped twice on MemoryError on this host (REPORTED: `v2/results/FINAL_RESULTS.md`,
  post-opening correction 5).
- If it cannot finish within its budget, the report gives the number of stream sets completed
  and the number of changes among them. The label is FRAGILE if more than 5 changes are already
  recorded, and otherwise "FRAGILE check NOT_COMPLETED (k of 100)". It never abandons the verdict.

**Why this design** (full reasons and arithmetic: `v3/DESIGN_RATIONALE.md` D10):

- **v2's non-overlap rule does not fit v3.** v2 used non-overlap of marginal intervals. v3 and
  v2 are correctly specified models of the same 9-parameter family. v2's 12 validation trials
  spread over about 0.012 BA, and the top two differed by 0.0005, which v2's own record calls
  far inside the sampling noise (MEASURED values and MODELED reading:
  `v2/results/STAGE2_SEARCH.md`). An expected nominal BA difference near 0 makes a superiority
  test on BA pointless (MODELED).
- **The paired difference is the natural test.** v2's record notes that non-overlap of two
  marginal intervals is stricter than a paired interval excluding zero (MODELED:
  `v2/results/FINAL_RESULTS.md`, robustness reading).
- **What the margin preserves.** δ = 0.005 is below the lower bound of v2's nominal gain over v1,
  +0.0104 [0.0066, 0.0142] (MEASURED: same file). A v3 exactly at the margin keeps 52% of v2's
  point gain over v1 and 24% of its lower bound (MODELED, arithmetic). P0 is the direct check of
  v3 against v1.
- **No multiplicity adjustment is needed for the verdict.** It is an intersection: WIN needs
  every condition. If any one condition is truly false, a false WIN is no more likely than that
  condition's own false pass (MODELED).

**Power at the planned sizes (MODELED; derivations in `v3/DESIGN_RATIONALE.md` D6, D8, D10).**

| condition | assumption | power |
| --- | --- | ---: |
| P1 | true difference 0; v3 and v2 disagree on 5% of rows | ≈ 1.00 |
| P1 | true difference 0; disagreement 11% (the v2–v1 level) | ≈ 0.96 |
| P1 | true difference 0; disagreement 15% | ≈ 0.88 |
| P1 | true difference −0.002; disagreement 11% | ≈ 0.60 |
| P2 | true difference 0; standard error 0.0007–0.00106 | ≈ 1.00–0.997 |
| P2 | true pooled difference −0.002; same range of standard error | ≈ 0.99–0.81 |
| P2 | true pooled difference −0.003; same range of standard error | ≈ 0.82–0.47 |
| P0 | v3 − v1 ≈ +0.010, as for v2 | ≈ 1.00 |
| P3 | Brier difference near 0 | ≈ 1.00 |
| H1 | the identification account of D3 holds | ≈ 1.00 |
| H2 | known-good false alarms ≈ 0.02–0.03; detection as in D6 | ≈ 0.9–0.97 (CONJECTURE for implementation defects) |
| H3 | true ΔBA_skew difference 0.002; standard error 0.00057–0.00078 | ≈ 0.97–1.00 |
| H3 | true difference 0.0029 (threshold off the BA optimum) | ≈ 0.77–0.96 |

The chance that v3 wins, if every lever behaves as modeled, is about 0.6–0.85 (CONJECTURE). It is
at the low end if v3 shifts systematically against v2 under queue saturation (P2) or if v3's
selected threshold sits off the BA optimum (H3). For the earlier draft's design (20 band retrains,
H3 on 25,000 pairs) the same estimate was about 0.55–0.8 (CONJECTURE:
`logs/verify/v3-prereg-review-a/modeled_arithmetic_20261013.txt` §9).

## 9. Secondary metrics (reported, never part of the verdict)

On `test`, each shift split, the pooled suite, the H3 twin set and `legacy_v1_test`, for every
contender, ablation and the TRUTH reference:

- Everything in v2 §8:
  - Brier, ECE with a reliability table, log loss and ROC AUC;
  - precision, recall, specificity, F1 and false-alert share;
  - conformal marginal and per-class coverage, abstention split into BOTH and EMPTY, and
    selective BA;
  - paired v3 − v2 and v3 − v1 differences;
  - disaggregation by regime, and on `test` by tls_enabled and by listener_running;
  - generator-truth recovery (sign agreement k/8, rank correlation, relative L2 error).
  Per-class rates are withheld under INVALID_BASELINE as in v2 §10.
- Conformal coverage on every set other than `test` and `conformal` is descriptive. The
  coverage guarantee holds only under exchangeability with the nominal conformal split (section
  6.5), so no shift-coverage number carries a guarantee.
- Each fitted weight's ratio to truth, after mapping to a common scale.
- Sensitivity only:
  - BCa bootstrap intervals for the P and H statistics;
  - equal-mass-bin ECE next to the registered equal-width ECE;
  - calibration slope and intercept (logistic recalibration fit);
  - the risk–coverage curve.
- The reject-option baseline: thresholding max(p̂, 1 − p̂) at the conformal layer's abstention
  rate, so that abstention gains are not overstated.
- The worst regime per contender.
- Class-0 coverage under each shift (descriptive).
- The v2-style marginal non-overlap comparison of v3 against each contender, descriptive only,
  for continuity with v2.
- The H1 ablation table and the readiness metrics of every aggregator (section 12).
- For every mutant, the gates that catch it, split into parameter-space gates (`g_coef_band`,
  `g_threshold_band`, `g_truth_sign`) and output-based gates (`g_ba_rel`, `g_validation`,
  `g_calibration`, `g_monotone`), and the list of mutants that only parameter-space gates catch.

## 10. Ablations

All ablations use the same splits and seeds. Keep/drop decisions are made on `validation` or
`gates` only. Test values are reported as confirmation and never enter the verdict.

| id | change against v3 | isolates | used by |
| --- | --- | --- | --- |
| ABL-D | no D (`train_noD`), R kept | the training design | H1 contrast |
| ABL-R | v1 min-max seconds, D kept | the representation | H1 contrast |
| ABL-DR | neither D nor R: v2's recipe re-fit on v3 data | both | H1 contrast; baseline for the v3 − v2 gap that is not due to data |
| ABL-n | the first 16,384 rows of `train` | training size | H2 margins, through a descriptive gate run with ABL-n as the reference |
| ABL-G | v2's seven gates and absolute tolerances, on v3's fixtures and mutants | the gate set | H2 contrast (blind spots and false alarms, both ways) |
| ABL-origin | v2's `verify_origin` rule on v3's receipt mutants | the origin binding | H2 contrast |
| ABL-calibration | Platt calibrator (v2 §6 definition) fit on `calibration` | calibration | P3 context; H4 invariance check |
| ABL-jitter | exploratory: in 25% of the D-arm training rows, observed seconds = clip(latent + U(−1800, +1800), 0, 86400), with the label from latent | symmetric timestamp noise, which is not the sealed one-sided skew | H3 context only |
| ABL-conformal | the layer kept or dropped by the section 6.5 rule | abstention | as v2 |
| ABL-conformal-size | quantiles from the first 4,096 rows of `conformal` | conformal split size | H4 context |

## 11. Validity gates, tolerances, mutation suite and broken baselines

### 11.1 Baseline validity

The INVALID_BASELINE rule of v2 §10 applies unchanged: (i) the receipt verifies, (ii) at least 30
rows of each class, (iii) non-constant decisions, (iv) every fail-closed probe refused (REPORTED:
`v2/PREREGISTRATION.md` §10). For v3 the probes of (iv) are the 76 of section 6.7. The
constant-ALERT fixture, a fail-open v3 kernel and majority must each come out INVALID_BASELINE.

For v2 as-is, precondition (i) is evaluated at the kernel level. Its research-side origin check
cannot run from a public checkout, because its source commit is unpublished: the v2 receipt
names `92872f88193242b3fc0c1f604321d889d6988358` (REPORTED: `v2/ops-health/artifact_receipt.json`
in szl-forge at main `c9a03e46`), and on 2026-10-14 the public API
`https://api.github.com/repos/szl-holdings/szl-forge/commits/<id>`, asked for that id, answered
422, "No commit found for SHA" (MEASURED; drafting copy:
`logs/verify/v3-register-fix/sources_20261014.txt`).

### 11.2 Gate set (definitions frozen at registration)

All gates are evaluated on `gates` (opened with purpose GATES). The reference is the frozen v3.
A gate **catches** a fixture iff the fixture's value is strictly below the reference's own value,
as in v2. An artifact that the kernel refuses to load scores 0 on every gate that needs kernel
output (fail closed).

- **Kept from v2 §10:**
  - `g_integrity`;
  - `g_failclosed`, on the 76 probes of section 6.7 (v2's 66 input probes and TT1–TT10);
  - `g_truth_sign` (k/8);
  - `g_monotone`, v2's metamorphic checks;
  - `g_validation` = min(1, BA/0.70);
  - `g_calibration` = 1 if ECE ≤ 0.05, else max(0, 1 − (ECE − 0.05)/0.10);
  - `g_baseline_valid`.
- **New, each in {0, 1}:**
  - `g_coef_band` = 1 iff |θ_j(f) − θ_j(ref)| ≤ tol_j for every parameter j: the intercept and
    the 8 weights. Each θ is first mapped to the registered representation R of section 6.2,
    using the artifact's declared table, so a change of the declared table alone moves θ on
    this scale.
  - `g_threshold_band` = 1 iff |k(f) − k(ref)| ≤ tol_k, where k is the integer grid index of
    section 6.4 and tol_k is an integer number of grid steps. The comparison is between integers.
  - `g_ba_rel` = 1 iff BA_gates(f) ≥ BA_gates(ref) − tol_BA.
- **Gate types**, for the report of section 9: `g_coef_band`, `g_threshold_band` and
  `g_truth_sign` act in parameter space; `g_ba_rel`, `g_validation`, `g_calibration` and
  `g_monotone` act on outputs.

### 11.3 Tolerance rule (frozen at registration; values computed before any mutant exists)

- For each parameter θ_0..θ_8 (on the R scale) and for BA_gates:

  > tol_T = max(floor_T, c × √2 × SD_T), with **c = 4**.

- For the threshold, in integer grid steps:

  > tol_k = max(2, ⌈c × √2 × SD_k⌉), with SD_k the standard deviation of the grid index k.

- SD is the sample standard deviation (n − 1) over the **100 band retrains**.
- The floors are: 0 for the parameters, 2 grid steps for the threshold, and 0.005 for BA.
- √2 is there because a known-good retrain and the reference are two independent draws of the
  same recipe.
- **False-alarm target.** c and the band count are set so that the modeled chance of at least one
  known-good false alarm, over the 20 known-good fixtures and the 11 band statistics, with the
  reference and the SD estimates shared by all fixtures, is at most 0.05. At c = 4 with 100 band
  retrains it is about 0.02–0.03 (MODELED, section 7 H2). The retained absolute gates count
  against the same budget.

The tolerance vector and its sha256 are committed before any mutant is generated. The mutation
code refuses to run unless that digest is in a commit that precedes it. The band retrains are
never used as known-good evaluation fixtures.

### 11.4 Mutation suite

All mutants are applied to the frozen v3.

**Registered artifact mutants (37).** The v2 §10 classes (36):

- weight ×0.5 and ×1.5 per feature (16);
- a sign flip per feature (8);
- each feature dropped (8);
- intercept ±1 (2);
- threshold ±0.1 (2).

And one transform-table mutant: the declared seconds scale changed from 900 to 86400, with every
weight unchanged and a consistent receipt.

**Registered receipt mutants (7):**

- a model-hash byte flip;
- a wrong schema;
- an altered (non-existent) source commit;
- a metrics block added (the v2 AM-3 form);
- `source.commit` re-pointed to the claimed commit's first parent, whose bound blobs are
  identical;
- `source.commit` re-pointed to a descendant whose bound blobs are identical;
- `source.commit` re-pointed to a squash-style duplicate S of the claimed commit C: the same
  tree and the same first parent as C, with different commit metadata, built in a scratch clone.

**Registered kernel-source mutants (8):**

- finiteness check removed;
- range check removed;
- prohibited-key screening removed;
- model-hash comparison inverted;
- self-hash comparison inverted;
- transform-table check bypassed: the kernel applies whatever table the artifact declares;
- duplicate-key refusal removed;
- one value of the authority map set to true.

A kernel-source mutant is caught iff integrity refuses it, or at least one test in the
**kernel-relevant test list** fails with it in place. That list (test ids and the sha256 of the
test files) is committed before step 7 (section 13) and never changed afterwards.

**Descriptive mutants** (sensitivity profile; not in H2; predictions in section 7 H2):

- weight ×0.75 and ×1.25 per feature;
- threshold ±0.05;
- recipe substitution: the ABL-D, ABL-R and ABL-DR models, each presented with a consistent
  receipt.

**Passes, as in v2 §10.** Pass 1 pairs each artifact mutant with the frozen receipt. Pass 2
regenerates the receipt consistently, which bypasses integrity, and evaluates every gate.
BLIND_SPOT means an artifact mutant that no gate other than `g_integrity` catches in pass 2 (for
the transform-table mutant: one that the kernel accepts or `g_coef_band` misses), a receipt mutant
that the integrity layer accepts, or a kernel mutant that neither integrity nor the kernel-relevant
tests catch. The test suite is also run once with each BLIND_SPOT mutant in place of the frozen
pair, and the outcome is reported. This makes v2's supplementary in-place run (REPORTED:
`v2/results/MUTATION_READINESS.md`) a registered step.

## 12. Readiness aggregation

- **Fixtures.** Known-good: v3 and `kg_01..20`, 21 in all. Known-broken: the 37
  integrity-bypassed artifact mutants.
- **Aggregators.**
  - **AND** is the registered rule: READY iff every gate = 1.
  - **min**, with cut ≥ 0.99, is the secondary aggregator.
  - **mean** is reported descriptively only.
- **Separation measures, as in v2 §11:** misclassifications at the cut, the margin
  min(good) − max(broken), and the AUROC of good against broken.
- **Λ is not an aggregator in v3.** v2 REJECTED it under its registered rule, and on every
  fixture Λ lay between min and mean (MEASURED: `v2/results/MUTATION_READINESS.md`). No new
  argument for Λ exists. Re-admitting it would change a frozen item (section 15.3), so it would
  need a new registration. No Λ property is claimed.

## 13. Order of operations

Each step is committed, and then pushed and tagged under section 14, before the next step
starts. Every fit runs from a clean tracked tree; the run refuses otherwise.

1. **Registration** (green light given by Stephen on 2026-10-13; section 18).
   - Preconditions, each checked and logged before the first push:
     - the tag protection ruleset for `oac-v3-*` is in force in szl-forge (section 14.6);
     - the evidence files of the preamble are published, and every cited path is public in
       szl-forge, published under `ops-health/v3/evidence/`, or handled by the list of section
       14.2 (the internal-file citations were replaced on 2026-10-14);
     - `/{chain hash}/info`, fetched again from `drand.cloudflare.com` and a `drand.sh` relay,
       still matches section 3.4. A difference blocks registration and needs a re-registration
       (section 15.3);
     - the same relays report a latest round below R, so R is still in the future. Otherwise
       the seal is re-registered first (section 15.3).
   - The public commit that adds the last draft unchanged, then the rename commit, then the
     registration receipt `v3/PREREGISTRATION.receipt.json`.
   - The receipt records: the sha256 of the preregistration; the public commit of the last draft
     (and the drafting commit, as information only); the beacon seal of section 3.4 (chain hash,
     public key, period, genesis time, R, R's scheduled time, the relay list and the relay rule);
     the URL and sha256 of each raw beacon response saved while drafting
     (`logs/v3/register/*`); the calendars, explorers and archives of section 3.5; the runtime
     pins of step 9; the sha256 of v2's generator, kernel, model and receipt; the sha256 of each
     drafting-only file published under `ops-health/v3/evidence/`; the id and enforcement state
     of the tag protection ruleset, as the GitHub API returns them; v3 code exists: false; v3
     data exists: false; pilot runs: 0; DEV runs: 0; and that it attests integrity and ordering
     only, not quality.
   - The commits are pushed and the annotated tag `oac-v3-registered` is placed. Then the
     registration digest (section 3.5) is submitted to at least two OpenTimestamps calendars as
     in section 3.5, and its pending proofs are committed and pushed. The upgraded proofs are
     committed when the calendars serve them. A Software Heritage save of szl-forge is requested
     at the same time; this also tests the archive path of section 3.5 before the freeze.
   - The registration's T_fwd (section 3.5) must be earlier than R's scheduled time. Otherwise
     the seal is re-registered before step 3 (section 15.3).
2. **v3 code and tests on DEV data,** under the DEV rule of section 15.5: generator, guards,
   kernel v3 with its contract (section 6.7), gates v3, mutation v3, and bootstrap reuse. The
   parity and guard tests of section 3.3 and the required kernel tests pass. Every DEV run's
   output is committed.
3. **Non-sealed splits generated;** `v3/data/MANIFEST.json` committed.
4. **Main and ablation fits,** the first segment of the bounded sequence. The fit trace records
   the git HEAD P at run time and is committed directly on P, as commit C. The tag `oac-v3-fit`
   is placed on C before any merge. The receipts, which name C and the tag, follow. v3 is frozen.
5. **100 band retrains,** the second segment, under the same trace rule (tag `oac-v3-band`). Then
   the tolerance vector is committed (tag `oac-v3-tolerances`).
6. **Known-good retrains** for every H1 arm, the third segment (tag `oac-v3-kg`). H1 is evaluated
   and committed.
7. **The kernel-relevant test list is committed** (section 11.4). Then the mutation suite is
   generated and run. H2 is evaluated and committed.
8. **Freeze.** Every contender, ablation and the TRUTH reference is frozen. The opening code is
   committed: the scorer; the analyzer; the beacon client and verifier with their tests and the
   verification level (section 3.4); the timing-proof checker with its tests (section 3.5); and
   the runtime pins (step 9). Then, in this order (section 3.5):
   - the commit F is pushed and tagged `oac-v3-frozen` (annotated);
   - the freeze manifest is written, and M is submitted to at least two OpenTimestamps calendars;
   - the commit Q, holding the manifest and the pending proofs, is pushed and tagged
     `oac-v3-seal` (annotated);
   - a Software Heritage save of szl-forge and Internet Archive captures of the two tag refs are
     requested;
   - the draft pull request that quotes F, the tag object id and M is opened (a pointer only);
   - the upgraded proofs are committed as the calendars serve them.

   All of this is planned to finish by 2026-10-22T00:00:00Z, 48 hours before R. Whether the
   freeze came before R is decided only by the seal check of section 3.5.
9. **The single opening,** never before round R is available, and within the opening window
   (section 3.6). It goes ahead whatever H1 and H2 showed (section 16).
   - **Preflight,** run by the frozen code:
     - a clean tree at the anchored F (or at the anchored commit of the last sealed fix, section
       15.6), checked file by file against the manifest whose digest is M;
     - every earlier `oac-v3-*` tag's target equal to the one that the manifest records; a moved
       tag is reported, and the manifest's record is used;
     - the seal check of section 3.5: SEALED proceeds; PENDING waits and retries; EXPOSED ends in
       SEAL_EXPOSED (section 16), and nothing is opened on round R;
     - at least 2 GB of free disk;
     - at least 4 GiB of free physical memory, as the operating system reports it. The host has
       31.6 GiB in all (MEASURED 2026-10-14; drafting copy:
       `logs/verify/v3-register-fix/host_memory_python_20261014.txt`);
     - the runtime pins below.

     A refused preflight opens nothing and is retried. It never permits a new round (section
     3.6).
   - **Runtime pins** (frozen at registration): CPython 3.12.10, run as `python -I` with
     `PYTHONHASHSEED=0`, for every fit and retrain, the mutation suite, the opening and the
     analysis. The opening host runs CPython 3.12.10 (MEASURED 2026-10-14; same file).
   - The run fetches and verifies round R (section 3.4) and sets SALT to its randomness. Then it
     generates the sealed splits, scores every contender once, writes `runs/v3/TEST_OPENED.json`
     (with F, M, the seal-check result, the digest of each saved beacon response and the
     verification level), saves the per-row scores, and submits the digest of
     `TEST_OPENED.json` to OpenTimestamps (section 3.6).
   - **The official run and the re-run.** The official run is the one recorded in the opening
     commit. One independent re-run at F, on a separate machine with the same pins, regenerates
     the sealed splits and every score file (planned: a GitHub-hosted Windows runner in
     szl-forge, so that the platform math library matches the opening host's; MODELED reason:
     floating-point library results can differ between platforms). The verdict is PROVISIONAL
     until the sha256 of every score file of the re-run equals the official run's. A mismatch is
     a defect under section 15.6. With the pins, the frozen code's output is fixed by F and round
     R, so a private run before the official one gives no choice (MODELED); the re-run checks it.
10. **Analysis** by the frozen analyzer, from the saved scores only: `v3/results/FINAL_RESULTS.md`;
    then the FRAGILE check (section 8); then a compliance audit, a datasheet-style description of
    the v3 synthetic splits (generator version and digest, seeds, the beacon round, sizes, label
    rule, SYNTHETIC scope), and the report.

## 14. Public provenance plan (approved 2026-10-13)

1. **Home.** The canonical home is a public branch of `szl-holdings/szl-forge`,
   `claude/oac-v3-research`, under `ops-health/v3/`. `ops-health/` mirrors the research root, so
   every path in this file resolves under it. `ops-health/` entered szl-forge through the public
   pull request szl-forge#570, from the branch `claude/oac-ops-health-v2` (MEASURED: the public
   pull request record, 2026-10-14). Which lane owns which paths is a working rule of the
   drafting process with no public source (UNKNOWN publicly); v3 follows it (item 9). The
   drafting repository keeps the drafting history. After registration, the public branch is
   authoritative.
2. **Registration in public.**
   - Before registration, every file that this file or the rationale cites is checked against
     the public branch. Stephen decided on 2026-10-13 (section 18):
     - `literature/*`, `drafts/forum_thread.md`, `logs/verify/v3-prereg-review-a/` and
       `logs/verify/v3-prereg-review-b/` are added byte-identically under
       `ops-health/v3/evidence/`, each with its sha256 in the registration receipt;
     - `REPORT.md`, `COORDINATION.md` and `LANDING_PLAN.md` are not published. Each of their
       citations was replaced by a public source, or the claim labeled UNKNOWN, on 2026-10-14;
     - nothing else from the drafting repository is published as evidence.
   - **Every other cited path that is not public** in szl-forge at registration is re-cited to
     a public source, or its claim is labeled UNKNOWN, or it is marked as a drafting copy of a
     public source that the citation also names. This covers this file, the rationale,
     `v3/DECISIONS_FOR_GREEN_LIGHT.md` and `v3/REVIEW_DISPUTES.md`. On 2026-10-14 these paths
     were:
     - `CONTRACT.md`: re-cited to the public v2 kernel, v2 tests and `v2/AMENDMENTS.md`
       (section 6.7, A27);
     - `logs/landing/pr3a_final_r3_sim_py312_20261006.txt`: re-cited to the public API and the
       public v2 tests (sections 11.1 and 14.5, A18);
     - `logs/v3/register/*`, `logs/verify/v3-register-review/*` and
       `logs/verify/v3-register-fix/*`: drafting copies only; each claim names the relay URL or
       public service it was read from, and the receipt records the URL and sha256 of each raw
       beacon response (section 13 step 1).

     Before registration the check is repeated, and any new such path is handled the same way.
   - One commit adds the last draft unchanged. The rename commit follows, then the registration
     receipt.
   - They are pushed, and the annotated tag `oac-v3-registered` is placed.
   - Every commit hash that the status block, the registration receipt or any v3 receipt records
     is a public szl-forge commit. Drafting-repository commits are named only as information.
   - Stephen approved the public pushes and annotated tags of this plan on 2026-10-13
     (section 18). Each push still writes its command, diff, tests and rollback to the drafting
     log before it runs (this protocol's remote-write rule). Before each push, szl-forge is
     fetched again. No push is a force-push, and none goes to `main`.
   - The OpenTimestamps submissions (digests only) and the archive requests of section 3.5 are
     part of this plan. They send no file content (section 18).
3. **Everything a receipt names is public before it is used.**
   - Generator, kernel, gates, mutation code, the `math/` adapters that v3 loads, the manifest,
     fit traces and the tolerance vector are all committed on the branch.
   - The commits that receipts name are pushed and tagged `oac-v3-fit`, `oac-v3-band`,
     `oac-v3-kg`, `oac-v3-tolerances` and `oac-v3-frozen`.
   - This addresses v2's deviation 11: three `math/` modules were untracked at v2's opening
     (MEASURED: `v2/results/COMPLIANCE.md`).
4. **Origin binding (v3 `verify_origin`).** A receipt names a commit C and a step tag T. All of
   these must hold:
   - (a) T is a public annotated `oac-v3-*` tag, and its target is exactly C. From the freeze
     on, T's target is read from the anchored freeze manifest (section 3.5), not from the live
     tag.
   - (b) At C, the kernel, generator, manifest, preregistration and amendments blobs match the
     receipt's digests (v2's rule).
   - (c) At C, the trace that the receipt binds (the fit trace, or a retrain's trace) matches the
     receipt's trace digest, and the receipts directory as it stands at C (the receipts of
     earlier steps) matches the receipt's digest of it.
   - (d) At C's first parent, that trace is absent or different.
   - (e) The trace records the git HEAD P at run time, and C's first parent is P.

   What this guarantees: the receipt is bound to one commit, the target of T, which introduced
   the trace directly on top of the commit whose code ran. Re-pointing to a parent fails (a) and
   (d); re-pointing to a descendant fails (a) and (e); a squash, merge, cherry-pick or rebase
   duplicate has another commit hash, so it fails (a) even when it has the same tree and the same
   first parent. Tag protection makes moving T harder but not impossible (item 6). The guarantee
   therefore does not rest on the live tag: each tag's target is recorded in the next step's
   commit, the anchored freeze manifest records every earlier tag's target and tag object, and a
   receipt names C itself, so a moved tag is detected. v2's rule bound only blobs that a parent
   shared, which is the blind spot `R-source-commit-other` (MEASURED:
   `v2/results/MUTATION_READINESS.md`).
5. **Publicly verifiable origin, unlike v2.** A fresh clone of the public tag must run every
   origin test with 0 skipped. A skip counts as a failure, not as UNAVAILABLE. For contrast, v2's
   public tests skip 7 origin test sites in szl-forge, because its source commit is absent there:
   `ops-health/v2/tests/_support.py` defines `ORIGIN_SKIP_REASON`, and
   `test_final_analyze.py` (1), `test_final_score.py` (1) and `test_mutation_suite.py` (5) skip
   on it (REPORTED: those files at szl-forge main `c9a03e46`; the absent commit MEASURED, section
   11.1).
6. **Merge safety.** Each step's tag is placed before any merge of that step into `main`. Tags
   keep the receipt commits reachable even if a later squash merge leaves the branch history out
   of `main`. The branch is not deleted while a tag depends on it.
   - **Tag protection** (approved by Stephen on 2026-10-13; section 18). A szl-forge ruleset
     that targets `refs/tags/oac-v3-*`, blocks the update and the deletion of a matching tag, and
     has an empty bypass list is in force before `oac-v3-registered` is pushed. The registration
     receipt records its id and enforcement state as the GitHub API returns them.
   - The ruleset blocks updates and deletions by anyone without a bypass, but a repository
     administrator can still change the ruleset, so it does not make a tag immutable. Nothing in
     this protocol trusts a live tag's target: the seal check and the opening use the identities
     anchored by section 3.5, and each tag's target is also recorded in the next step's commit and
     in the freeze manifest. A moved tag is detected and reported; it changes nothing.
7. **Data.**
   - Non-sealed JSONL files are not committed. They are regenerable, and the manifest holds their
     digests.
   - Sealed splits are generated only inside the opening run.
   - The opening commit publishes the raw round-R and round-(R + 1) responses of every relay
     used, SALT (the round's randomness), the verification level, the seal-check result with its
     raw inputs, and every sealed digest.
   - **Per-row score files: 50 MB per file** (decided 2026-10-13; section 18). A score file of at
     most 50 MB is committed. A larger one is not committed; its sha256, byte size and row count
     are. Anyone can regenerate it from the public code and the public round R (section 3.4).
   - Prediction (MODELED): the `test` file and the two H3 files exceed 50 MB and are published as
     digests only; the four `shift_*` files stay below the limit. Basis: v2's `test` score file
     holds 50,000 rows in 30,537,167 bytes, about 611 bytes per row, with fewer contenders than v3
     scores (MEASURED: `v2/results/final/scores_test.jsonl`).
8. **Seal timing, not custody.** No person holds SALT (section 3.4). The guarantee is an ordering
   that anyone can check from durable public records: the OpenTimestamps proof of the freeze
   manifest, checked against Bitcoin block headers (T1), and a public archive's record of the
   tags (T2), both made before R's scheduled time (section 3.5).
   - Neither record depends on a retention window that the research process could wait out. An
     upgraded proof can be checked offline against any copy of the Bitcoin headers. The archives
     keep their records with no stated retention window (CONJECTURE about their future policy).
     The upgraded proofs and every raw archive response are also committed.
   - Git dates, release dates, GitHub's events and activity APIs and captures that exist only as
     committed files are never evidence of time (section 3.5).
   - SEAL_EXPOSED is decided only by the seal check of section 3.5, once, from those records. A
     record that expires or goes missing after R never yields SEAL_EXPOSED by itself.
   - For v2, whether anyone looked at sealed rows early is UNKNOWN (UNKNOWN:
     `v2/results/COMPLIANCE.md`, "What this audit cannot show"). For v3 the question becomes
     checkable for the sealed sample: before round R no sealed row can exist, and if the seal
     check is SEALED, one frozen state was fixed and public before then. Whether any unrecorded
     run used self-generated rows of the same distribution stays UNKNOWN: the DEV rule forbids
     it (section 15.5), and no seal can detect it (section 17).
9. **Lane discipline.**
   - One writer per repository.
   - No push while another OAC change to the same paths is open.
   - No Hugging Face write in v3 until a verdict exists and a separate card change is approved.

## 15. Amendments, DEV data and defects

1. **Before registration.** The draft could change freely, and every change was a commit.
2. **After registration.**
   - The registered file is pinned by sha256 in the v3 kernel and research registry and is never
     edited.
   - Amendments go into an append-only `v3/AMENDMENTS.md`. Each is dated and committed before the
     step it affects (section 13). v2's AM-1..AM-6 were all committed before any search
     (MEASURED: `v2/results/COMPLIANCE.md` check C, which checks commit order).
   - Requests that Stephen has not yet decided go into `v3/AMENDMENT_REQUESTS.md`.
3. **Items frozen at registration:**
   - the question; the generator truth; generator v3's modes, with π = 0.25 and the twin mode;
     the namespace, master seed and seed derivation;
   - the beacon seal of section 3.4: the chain, its public key, period and genesis time, the
     target instant, R, the relays and the relay rule, the seal rule, the checks, the
     verification-level rule and the disagreement rule;
   - the timing proof of section 3.5: the frozen identity, the manifest format, the calendar
     rule, the cutoff rule with its forward median and margins, the explorer rule, the archive
     records accepted, the list of what is not evidence, and the seal-check procedure;
   - the availability, opening-window and new-round rules of section 3.6;
   - the runtime pins, the memory and disk thresholds, and the re-run rule (section 13 step 9);
   - every split's definition, regime, size and role (section 4);
   - the recipe: model class, IRLS settings and the eligibility and non-convergence rules
     (section 6.1); l2 = 1e-3; calibrator `none`; representation R with the 900 s constant and the
     declared table; train size 32,768; the threshold rule and grid; the conformal method, α and
     keep rule;
   - the kernel contract and its probes and required tests (section 6.7);
   - the contenders, their frozen identities, the TRUTH reference and the ablations;
   - the counts: 100 band retrains, 20 known-good retrains per H1 arm, 227 declared fits;
   - the primary metric, the margins (0.005 BA, +0.002 Brier, +0.01 ECE, 0.005 and 0.002 for H3,
     0.02 for H4), the P and H criteria, the verdict rule and the readings of section 16;
   - the gate definitions, c, the floors and the tolerance rule (section 11.3);
   - the mutation classes and counts, and the rule for the kernel-relevant test list;
   - the origin rule (a)–(e) of section 14.4;
   - the bootstrap of section 8: resamples, seed, stream layout, interval type, the undefined
     rule; the FRAGILE rule and its budget;
   - H4's statistic A\*.

   A change to any of these after registration is one of three things:
   - a **re-registration**, allowed only before step 3, while no v3 non-sealed data, fit or
     retrain exists. It gets a new tag `oac-v3-registered-rN`, and its digest gets an
     OpenTimestamps proof as in section 3.5. It keeps the beacon seal if its T_fwd is earlier than
     R's scheduled time, and otherwise takes a new round by the rule of the next item. It lists
     every DEV-run output committed so far, and the report discloses every earlier
     registration;
   - a **seal re-registration**. It is allowed by choice only while R's scheduled time is still
     in the future: the re-registration's digest gets an OpenTimestamps proof as in section 3.5,
     and its T_fwd must be earlier than R's scheduled time. After R's scheduled time it is
     allowed only in the two cases of section 3.6: the seal check returned EXPOSED, or the chain
     objectively never produced round R. In every other case, abandoning round R after its
     scheduled time is a PROTOCOL_DEVIATION: its randomness is public then, so a choice to drop it
     could depend on the sealed rows it generates (MODELED). No failure of the opening host, no
     refused preflight, no relay problem and no missed opening window permits one (section 3.6).
     - It changes only the beacon round, and the chain only if the registered chain stopped;
       every other frozen item stays as registered.
     - Its target instant is a 00:00:00Z instant at least 72 hours after the re-registration is
       pushed, and the new round R′ is derived from it by the rule of section 3.4, from chain
       parameters fetched live from the relays under the relay rule. Its own T_fwd must be at
       least 2 hours before R′'s scheduled time. It gets a new tag `oac-v3-registered-rN`.
     - From then on, R in this protocol means R′, with one exception: **the window for
       frozen-state fixes closed for good at the first registered round's scheduled time**
       (section 15.6). An opening on R′ uses the frozen state exactly as it stood: the anchored
       F of `oac-v3-frozen`, or of the last fix whose own seal check is SEALED against the first
       registered round. No other fix, under any round, is allowed. If `oac-v3-frozen` does not
       exist yet, it may still be created once, and it must be sealed before R′.
     - If any registered round's scheduled time passed before the frozen state used for the
       opening was sealed, every statement of the verdict carries this label: "frozen after round
       R's randomness, a full sealed-equivalent sample, was public", naming that round, the
       freeze's T1 and T2 times and R′'s scheduled time.
     - The report discloses every round, the reason for each change, and that the randomness of
       each abandoned round was public from its scheduled time. The choice of a target instant
       carries no information about any future randomness (MODELED); or
   - a **PROTOCOL_DEVIATION**, which makes the verdict DOES_NOT_WIN.
4. **Clarifications.** These are allowed before the step they affect, only for details that item
   3 does not fix, and each says why it changes no frozen item. v2's amendments file states that
   none of its six entries changes the question, the primary metric, the win condition, the
   splits, the seeds, the search space or the shift regimes (REPORTED: `v2/AMENDMENTS.md`). One of
   them, AM-2, added the eligibility rule for non-converged fits, which v3 registers directly in
   section 6.1.
5. **DEV data rule.** Between registration and step 3, and afterwards, v3 code is exercised only
   on DEV data:
   - DEV rows are generated only under the split name `dev`, at most 2,048 rows per regime per
     DEV run;
   - no fit uses more than 2,048 rows of DEV data;
   - no P or H statistic, gate tolerance, detection ratio, power or WIN-chance quantity is
     computed on DEV data;
   - every DEV run's output (logs, fitted parameters, test results) is committed;
   - the v2 parity tests of section 3.3 generate rows only to compare digests;
   - no run generates rows from the sealed seed material of any registered round, an abandoned
     round included, except the official opening run and its re-run for the round that the
     opening uses. This rule can be stated but not enforced by any seal (section 17).
6. **Defects and re-runs.**
   - A code defect found before a step's registered statistic (the tolerance vector, H1, H2) is
     computed may be fixed and the step re-run. Both runs are reported and the defect is
     disclosed.
   - Once a step's registered statistic is computed and committed, that first run stands. A
     corrected run is reported next to it. The condition counts as met only if both runs meet
     it, or a recheck shows that no deciding number changed.
   - **Frozen-state fixes.** A defect found after step 8 and before the first registered round's
     scheduled time may be fixed. The fix is committed and tagged `oac-v3-frozen-f2`, then `-f3`
     and so on, never by moving a tag, and each fix gets its own manifest, seal tag and timing
     proof (section 3.5), checked against that first round. The opening uses the last fix whose
     seal check is SEALED, and the report discloses every fix. **From the first registered round's
     scheduled time on, no change to the frozen state is allowed under any round,** including
     after a seal re-registration (section 15.3); any change is a deviation.
   - **The official opening run decides the verdict** (section 13 step 9). Its score files must
     equal, byte for byte, those of one independent re-run at F with the runtime pins; until they
     do, the verdict is PROVISIONAL. A mismatch, or a code fix after the opening, is a defect: the
     verdict is WIN only if both the official run and the corrected or re-run give WIN, or if a
     recheck in the style of v2's shows that no deciding number changed. v2's post-opening recheck
     compared 3,884 numeric fields and found all 3,884 identical (MEASURED:
     `v2/results/FINAL_RESULTS.md`, post-opening corrections).
   - A defect in a definition is a deviation. A deviation that touches any item in 3 makes the
     verdict DOES_NOT_WIN, with reason PROTOCOL_DEVIATION.
7. **No tuning after a result.** No tolerance, margin, mutant class or criterion is changed after
   any result it governs is known.

## 16. Endings, negative results and readings

- **The outcome.** v3 DOES_NOT_WIN if any of P0–P3 or H1–H3 fails. Then v2 stays the program's
  exemplar, and the report says plainly that v3 did not win. A negative v3 gets the same full
  write-up as a positive one: FINAL_RESULTS, MUTATION_READINESS and COMPLIANCE, with every
  number.
- **The opening goes ahead whatever H1 and H2 show.** A pre-opening failure of H1 or H2 already
  fixes the verdict as DOES_NOT_WIN. The single opening still runs, so that P0–P3, H3 and H4 are
  measured and reported.
- **Every condition already committed is reported with its outcome in any ending.**
- **Three readings for each non-inferiority condition** (P1, P2, the Brier arm of P3, and both
  arms of H3), registered now:
  - **shown:** the interval bound lies inside the margin (the condition holds);
  - **inconclusive:** the interval contains the margin. This is reported as "non-inferiority not
    shown", never as a cost or a trade-off;
  - **cost beyond the margin:** the whole interval lies beyond the margin (for P1 and P2, the
    upper bound is < −0.005; for the Brier arm of P3, the lower bound is > +0.002; for H3, the
    lower bound is > 0.005 or > 0.002). Only this reading may be called a cost or a trade-off.

  The ECE arm of P3 is a point estimate; if it fails, the report says that ECE(v3) exceeded
  ECE(v2) + 0.01 as a point estimate, without an interval reading.
- **What other failures would mean** (pre-registered readings, MODELED):
  - **H1 fails in the v3 arm.** The identification account of `v3/DESIGN_RATIONALE.md` D3 is
    wrong. If, in addition, ABL-D succeeds, rescaling was the issue after all.
  - **H2 fails.** At the registered tolerances, reference-relative parameter and decision gates
    do not separate these mutants from known-good retrains. The escaping mutants or false alarms
    are listed, and nothing is retuned.
  - **P0 fails.** A v3 that does not beat v1 or the trivial baselines is reported as such.
- **Endings without a full verdict on the hypotheses:**
  - **NO_ELIGIBLE_FIT:** the main fit does not converge (section 6.1).
  - **SEAL_EXPOSED:** the seal check of section 3.5 returns EXPOSED: no OpenTimestamps proof of
    the freeze manifest meets the cutoff, or no public archive record shows the frozen and seal
    tags before their cutoff. It is decided once, by the frozen code, from durable records only;
    a record that expires or goes missing after R never yields it by itself. The opening must
    not proceed on round R. This ends the seal, not v3: the seal is re-registered with a new
    future round (section 15.3), and the opening on that round uses the frozen state exactly as
    it stood, with no fix (section 15.6) and with the label of section 15.3. The report
    discloses round R, the T1 and T2 evidence found, R's scheduled time and the reason. Stopping
    v3 instead of continuing on the new round is ABANDONED, and a later registration must
    disclose it.
  - **SEAL_COMPROMISED:** evidence that the randomness of round R was available before its
    scheduled time; a sealed row generated from anything other than the verified round-R
    randomness; or a persistent relay conflict that drand's own client cannot resolve to one
    valid signature (section 3.4). The verdict cannot be WIN. It never permits a new round.
  - **Beacon hold** (not an ending): round R is not yet available (section 3.6), or the relays
    disagree or a required relay is silent (section 3.4). The opening waits and retries, and
    never substitutes another round, chain or source. A new round is allowed only if the chain
    objectively never produced round R, shown by daily captures over 14 days (section 3.6).
  - **Late opening:** the opening window of section 3.6 is missed without an external cause. It
    is a PROTOCOL_DEVIATION: the verdict is DOES_NOT_WIN. The opening still runs on round R, and
    everything is reported.
  - **ABANDONED:** the protocol cannot be completed as registered, for example because the host
    runs out of disk or memory and the opening can never run. ABANDONED covers only the
    conditions not yet evaluated, which are reported as "not run", never as evidence. Conditions
    already committed, such as H1 and H2, are reported as recorded. The verdict is DOES_NOT_WIN.
    It never permits a new round.
  - A FRAGILE check that cannot finish never abandons the verdict (section 8).
- **No re-attempt.** v3 is not attempted again without a new registration, with a new master
  seed and a new future beacon round, which discloses this one and its outcome. A seal
  re-registration (section 15.3) is not a re-attempt: nothing has been opened, and it keeps every
  frozen item and the frozen state.
- **H4 never makes v3 negative.**

## 17. What v3 cannot show

- **Nothing about any real system.** The data, the labels and the shifts are synthetic, and the
  model class matches the generator exactly (MODELED, section 2).
- **Other shifts.** The four shift regimes are synthetic perturbations reused from v2. Other
  shifts are not tested.
- **The queue-saturation weakness.** Queue saturation is the worst regime for v2 (BA 0.5928,
  specificity 0.2269, class-0 coverage 0.2994), for v1 (0.5311), for ABL-data (0.5842) and for
  ABL-optimizer (0.5693). The rule's lowest shift regime is clock skew (0.5336), and its nominal
  BA (0.5317) is lower still; majority is 0.5000 in every regime (MEASURED:
  `v2/results/FINAL_RESULTS.md`). v3 measures this regime again but makes no change aimed at it
  (`v3/DESIGN_RATIONALE.md` §5).
- **Designed with v2's results in view.** v3's hypotheses were chosen knowing v2's sealed
  results. Fresh seeds and a beacon-salted seal prevent reuse of the test **sample**. They do
  not prevent tuning to the **distribution**, which is the same generator as v2's. Predictions
  written in advance limit the consequences, but they cannot remove the choice of which weaknesses
  to target.
- **The seal's own assumptions** (sections 3.4 and 3.5). The beacon seal shows an ordering: one
  frozen state was fixed and public before round R's randomness existed. It cannot show that the
  drand threshold network did not collude to compute round R early. Its timing evidence is as good
  as Bitcoin's header times under an honest majority of mining power, with the 2-hour margin, and
  as the date a public archive gives its own record. It does not rest on GitHub's records or on
  tag protection. If the verification level is HASH_AND_RELAY_AGREEMENT, the frozen code did not
  check the round's signature against the registered public key, and the randomness is trusted to
  the agreeing relays, `drand.cloudflare.com` among them, and to the chain link; anyone with the
  public key can still check the signature.
- **Self-generated samples.** The generator is public and deterministic, so anyone can draw rows
  of the same distribution with any salt at any time. The DEV rule forbids such runs at scale
  (section 15.5), but no seal can detect one. The seal protects the sealed **sample**; it does not
  show that no unrecorded run looked at the **distribution**.
- **Numbers registered from v2's sealed results.** v2's sealed splits amount to a 100,000-row
  pilot of the same distributions. These registered numbers were derived from them:

  | registered number | derived from (v2 sealed result) |
  | --- | --- |
  | δ_BA = 0.005 (P1, P2) | v2 − v1 nominal paired BA +0.0104 [0.0066, 0.0142] (MEASURED: `v2/results/FINAL_RESULTS.md`) |
  | Brier margin +0.002 (P3) | v2 − v1 nominal paired Brier −0.0144 (MEASURED: same file) |
  | disagreement d ≈ 0.11, hence the test size 100,000 | the same paired BA interval and class counts 9,096 / 40,904 (MEASURED: same file) |
  | P2 standard error ≈ 0.00106 | the pooled-shift paired interval [0.0498, 0.0557] (MEASURED: same file) |
  | w̄ ≈ 0.10, hence the ρ table, π's margin and the train size 32,768 | v2's nominal-test Brier 0.1013 (MEASURED: same file) |
  | H3 predictions and the H3 set size | v2's reliability table (class-0 density, g = 0.25–0.40 per logit) and the skew half-widths 0.0080 / 0.0038 (MEASURED: same file) |
  | H4 margin 0.02 and H4 predictions | v2's test abstention 0.4143 and the BOTH-band row count (MEASURED: same file) |
  | the shift split size 25,000 | v2's skew half-width 0.0080 on 12,500 rows (MEASURED: same file) |

- **Receipts.** They attest integrity and origin, not quality, and they are unsigned, as v2's
  were (REPORTED: `v2/results/STAGE2_SEARCH.md`, `signed: false`).

## 18. Decisions recorded at the green light (2026-10-13)

On 2026-10-13 Stephen replied "do all your recommendations" to the list in
`v3/DECISIONS_FOR_GREEN_LIGHT.md` (REPORTED: the orchestrating session's record of his reply; his
message as relayed verbatim to the amending run reads "al ll your ecoemdatiosn do it"). Every
item there is DECIDED, and each is in force in this file. Item 5 is a decision of the
orchestrating session under that reply, not a choice Stephen made himself:

| # | decision | where in this file |
| ---: | --- | --- |
| 1 | Register v3. The public pushes and annotated tags of section 14 are approved. | status block; §13 step 1; §14.2 |
| 2 | The recommended win condition: WIN iff P0–P3 and H1–H3 all hold; δ = 0.005 BA, +0.002 Brier, +0.01 ECE; H3 margins 0.005 BA and 0.002 Brier. H2 stays strict: zero known-good misclassification. The relaxation to one known-good false alarm is rejected. | §7 H2 criterion 5; §8; §12; §16 |
| 3 | Sizes: 100 band retrains; an H3 twin set of 100,000 pairs; `test` 100,000 rows; four shift splits of 25,000 rows each. | §4; §6.6; §11.3 |
| 4 | Budget: 227 declared fits; the FRAGILE check capped at 72 hours of wall-clock time, with NOT_COMPLETED if it does not finish; the opening preflight requires at least 2 GB free. | §6.6; §8; §13 step 9 |
| 5 | Seal: Stephen supplied no salt. The orchestrating session replaced the held salt with the randomness of a future drand round, R = 6578966 of the default mainnet chain, scheduled 2026-10-24T00:00:00Z, fixed at registration. The frozen state must be public before it, or SEAL_EXPOSED. This supersedes every step in which Stephen would have supplied SALT or sha256(SALT). The timing proof was changed on 2026-10-14 (below). | §3.2; §3.4–§3.6; §13 steps 1, 8, 9; §14.8; §15.3; §16 |
| 6 | Tag protection for `oac-v3-*` in szl-forge. It hardens the tags; the seal and the origin rule do not rely on it (§14.6). | §14.6; §13 step 1 |
| 7 | Published evidence: `literature/*`, the two v3 review evidence folders and `drafts/forum_thread.md` only. `REPORT.md`, `COORDINATION.md` and `LANDING_PLAN.md` stay internal; their citations were re-cited to public sources on 2026-10-14, as were the other non-public paths of §14.2. | preamble; §14.2 |
| 8 | Per-row score files: at most 50 MB per file is committed; above that, digests only. | §14.7 |

**Later decisions under the same reply (2026-10-14).** An adversarial review of the beacon-seal
amendment found that the draft's timing evidence, GitHub's events and activity records, expires
and does not name the tag's target, so the research process could have forced SEAL_EXPOSED after
seeing round R (`v3/REVIEW_DISPUTES.md`, second part). The orchestrating session decided, under
Stephen's reply, and the fixing run adjusted with stated reasons:

| # | decision | where in this file |
| ---: | --- | --- |
| 9 | Timing proof: an OpenTimestamps proof of the freeze manifest digest, submitted to at least two public calendars before R and upgraded to a Bitcoin attestation (orchestrator). The cutoff uses the forward median of blocks H to H + 10 and a 2-hour margin, not the attesting block's own median-time-past, and a public archive record of the frozen and seal tags is also required (fixing run; reasons in rationale D15). The same mechanism timestamps the registration digest. SEAL_EXPOSED is decided once, by frozen code, from durable records only. | §3.5; §13 steps 1, 8; §14.8; §16 |
| 10 | Opening window: 7 days after R's scheduled time; a miss without an external cause is a PROTOCOL_DEVIATION, never a new round. A new round after R only after EXPOSED, or if the chain objectively never produced R. | §3.6; §15.3; §16 |
| 11 | Round R from at least two relays, `drand.cloudflare.com` among them, with randomness = sha256(signature) and the R + 1 chain link; a standard-library BLS check is not required, and anyone with the public key can run one. | §3.4 |
| 12 | The window for frozen-state fixes closes for good at the first registered round's scheduled time; runtime pins (CPython 3.12.10, `python -I`, `PYTHONHASHSEED=0`) and a byte-identical independent re-run; a numeric memory threshold. | §13 step 9; §15.3; §15.6 |

The OpenTimestamps submissions send only 32-byte digests, and the archive requests ask public
archives to copy the public repository. Both are part of the provenance plan that Stephen
approved (item 1); neither sends file content.

## Appendix A. v2 evidence this file relies on

| # | fact | label | file |
| ---: | --- | --- | --- |
| A1 | W1 against v1 held by 0.000319 BA (v2 lower 0.759159, v1 upper 0.758840) | MEASURED | `v2/results/FINAL_RESULTS.md` |
| A2 | Nominal test: v2 BA 0.7644 [0.7592, 0.7695], v1 0.7540 [0.7494, 0.7588]; paired v2 − v1 +0.0104 [0.0066, 0.0142] | MEASURED | `v2/results/FINAL_RESULTS.md` |
| A3 | Pooled shift: v2 0.7581 [0.7542, 0.7618], v1 0.7053 [0.7019, 0.7087]; paired +0.0528 [0.0498, 0.0557] | MEASURED | `v2/results/FINAL_RESULTS.md` |
| A4 | shift_clock_skew paired v2 − v1 BA +0.0066 [−0.0013, +0.0146]; v2 0.7564, v1 0.7498 | MEASURED | `v2/results/FINAL_RESULTS.md` |
| A5 | Nominal test Brier v2 0.101330, v1 0.115765; ECE v2 0.030599, v1 0.084775; positives 9,096 of 50,000 | MEASURED | `v2/results/FINAL_RESULTS.md` |
| A6 | Conformal on test: abstention 0.4143 (all BOTH), coverage 0.9003 and 0.9061; selective BA 0.8447 | MEASURED | `v2/results/FINAL_RESULTS.md` |
| A7 | Queue saturation: v2 BA 0.5928, specificity 0.2269, class-0 coverage 0.2994; worst regime for v2, v1 (0.5311), ABL-data (0.5842) and ABL-optimizer (0.5693); rule 0.5317 nominal, 0.5373 queue saturation, 0.5336 clock skew; majority 0.5000 everywhere | MEASURED | `v2/results/FINAL_RESULTS.md` |
| A8 | v2 weights: listener −2.539293, tls −0.802793, peer −1.038998, queue +2.705005, failures +1.718150, seconds +0.036548, ledger −2.851449, config −2.821703; intercept 7.132311; threshold 0.20 | REPORTED | `v2/results/MUTATION_READINESS.md` |
| A9 | Known-good k1–k3: seconds weights −0.003455, −0.009359, −0.013441 (REPORTED from each model.json); each fails g_truth_sign and g_monotone, 3,200 of 19,600 checks, all raise:seconds (MEASURED) | MEASURED / REPORTED | `v2/results/MUTATION_READINESS.md` |
| A10 | 11 BLIND_SPOT (10 artifact, 1 receipt); 10 under a literal "any layer" reading | MEASURED | `v2/results/MUTATION_READINESS.md` |
| A11 | Validation BA of the blind-spot mutants (v2 0.768680): tls ×0.5 0.754758, ×1.5 0.745493; queue ×0.5 0.737230, ×1.5 0.753756; failures ×0.5 0.762420, ×1.5 0.766326; seconds ×0.5 0.768029, ×1.5 0.768680; threshold +0.1 0.738431, −0.1 0.708534 | MEASURED | `v2/results/MUTATION_READINESS.md` |
| A12 | Readiness: all aggregator margins negative; AND and min 13 misclassified, mean and Λ 14; Λ REJECTED; Λ between min and mean on every fixture | MEASURED | `v2/results/MUTATION_READINESS.md` |
| A13 | `R-source-commit-other`: re-pointing to parent `064481f`, with the five bound blobs identical, passes kernel and verify_origin | MEASURED | `v2/results/MUTATION_READINESS.md` |
| A14 | Known-good thresholds 0.19, 0.18, 0.19, 0.19, 0.20; known-good validation BA 0.766026, 0.765625, 0.766276, 0.764623, 0.768129 | MEASURED | `v2/results/STAGE2_SEARCH.md`; `v2/results/MUTATION_READINESS.md` |
| A15 | Selected T07 (l2 1e-3, none, t 0.20, validation BA 0.768680, CONVERGED in 7 iterations); top two trials 0.0005 apart, "far inside the sampling noise" | MEASURED / MODELED | `v2/results/STAGE2_SEARCH.md` |
| A16 | Frozen Mondrian quantiles q̂₀ 0.223684755436, q̂₁ 0.911152623366 (n₀ 3,346, n₁ 750) | MEASURED | `v2/results/STAGE2_SEARCH.md` |
| A17 | Frozen digests: kernel `b0a64ff3…`, model `b830a5ed…`, receipt `442486a3…`, generator `a502d364…` | MEASURED | `v2/results/STAGE2_SEARCH.md` |
| A18 | Receipt origin layer unavailable publicly: source commit `92872f88…` absent from szl-forge (API 422 on 2026-10-14); 7 origin test sites skip on `ORIGIN_SKIP_REASON` in the public v2 tests | MEASURED (absence) / REPORTED (skip sites, from code) | `https://api.github.com/repos/szl-holdings/szl-forge/commits/<id>`; `v2/ops-health/artifact_receipt.json`; `v2/tests/_support.py`, `test_final_analyze.py`, `test_final_score.py`, `test_mutation_suite.py` at szl-forge main `c9a03e46` |
| A19 | Three `math/` modules untracked at the opening, committed byte-identically later (`d9c5fb3`) | MEASURED | `v2/results/COMPLIANCE.md` |
| A20 | Whether any process outside the guard read a sealed file before the opening | UNKNOWN | `v2/results/COMPLIANCE.md` |
| A21 | Nominal seconds ~ Exp(mean 900 s) on 0–86400; weak identifiability predicted | REPORTED (MODELED there) | `v2/PREREGISTRATION.md` §2; `v2/research/generator.py` |
| A22 | Bootstrap implementation: percentile type 7, one `random.Random(seed)` stream per evaluation set, shared count vectors, strata in a fixed order; undefined resamples excluded and counted | REPORTED | `v2/results/COMPLIANCE.md` deviations 8 and 9; `v2/results/FINAL_RESULTS.md`; `v2/research/bootstrap.py` |
| A23 | W1 held in 100 of 100 other resample streams (post hoc, not registered) | MEASURED | `v2/results/FINAL_RESULTS.md` |
| A24 | Search receipts are unsigned (`signed: false`) | REPORTED | `v2/results/STAGE2_SEARCH.md` |
| A25 | The conformal guarantee holds under exchangeability with the nominal conformal split only; coverage under shift is descriptive | REPORTED | `v2/PREREGISTRATION.md` §6 |
| A26 | Post-opening recheck: 3,884 of 3,884 numeric fields identical; the 100-stream check stopped twice on MemoryError | MEASURED / REPORTED | `v2/results/FINAL_RESULTS.md`, post-opening corrections |
| A27 | v2 kernel: standard library only, duplicate JSON keys refused, five-key all-false authority map (`AUTHORITY_MAP`), prohibited-key screen before the schema check, a determinism test | REPORTED | `v2/ops-health/ops_health.py` (imports, pairs hook, `AUTHORITY_MAP`, prohibited-key screen); `v2/tests/test_determinism.py`; `v2/AMENDMENTS.md` AM-6 |
| A28 | Known-good validation ECE 0.027431, 0.027913, 0.026614, 0.028354, 0.030538; v2's 0.034658 | MEASURED | `v2/results/MUTATION_READINESS.md`; `v2/results/STAGE2_SEARCH.md` |
| A29 | v2's six amendments committed in one commit before any search (check C); their content | MEASURED (order) / REPORTED (content) | `v2/results/COMPLIANCE.md`; `v2/AMENDMENTS.md` |
