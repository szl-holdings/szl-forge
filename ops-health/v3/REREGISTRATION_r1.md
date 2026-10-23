# OAC Ops Health v3: seal re-registration r1

Everything here is SYNTHETIC. This record contains no v3 result. No v3 code, v3 data, fit,
retrain or DEV run exists (DEV runs: 0; pilot runs: 0).

## What this is

This is a **seal re-registration** under the registered protocol `ops-health/v3/PREREGISTRATION.md`.
Its registration commit is `970ce6ce533f77c2c679f27d31ba86b7c43bb7a4`, with annotated tag
`oac-v3-registered`, tag object `4fc95253d81b4847e054849d9c874fba90c0fb12`. The rules come from
§15.3, which allows one by choice only while the registered round's scheduled time is still in
the future, and from §3.6. It changes **only the beacon round**. The chain, and every other
registered item, stay exactly as registered.

| | registered | this re-registration |
| --- | --- | --- |
| chain | drand default mainnet `8990e7a9…b2ce` | unchanged |
| round | R = 6578966 | R′ = **6659606** |
| scheduled | 2026-10-24T00:00:00Z | **2026-11-21T00:00:00Z** |

From here on, R in the protocol means R′ = 6659606, with the one exception §15.3 names: the
window for frozen-state fixes closed for good at 2026-10-24T00:00:00Z, the first registered
round's scheduled time.

## Why

The v3 build (protocol §13, steps 2 to 8) had not started. It covers the code, 227 fits, 100
band retrains, the tolerances, the mutation suite, the freeze, and the freeze's OpenTimestamps
proof with T_fwd at least 2 hours before R. It could not be done before 2026-10-24T00:00:00Z.
The program's sessions were repeatedly stopped by account usage limits between 2026-10-14 and
2026-10-22 (REPORTED: orchestrator). Re-registering before R's scheduled time is the protocol's
own path for this case (§15.3). Missing R instead would have been SEAL_EXPOSED.

## Conditions of §15.3, checked

- **Before R's scheduled time.** At the capture used here, every registered relay reported
  latest round 6576422, scheduled 2026-10-23T02:48:00Z. That is below R = 6578966, so R's
  randomness did not yet exist (MEASURED: `logs/v3/rereg_r1/latest_*_20261023.json`).
- **Chain parameters fetched live, under the relay rule.** `drand.cloudflare.com`,
  `api.drand.sh` and `api2.drand.sh` each served `/info` and `/{hash}/info`. All six responses
  agree on the hash, public key, period (30 s), genesis time (1595431050), scheme
  (`pedersen-bls-chained`) and group hash, and match the registered values. On each relay's
  latest round, randomness equals sha256(signature) (MEASURED:
  `logs/v3/rereg_r1/round_Rprime_20261023.txt`).
- **R′ by the rule of §3.4.** R′ is the first round whose scheduled time, genesis_time +
  (R′ − 1) × period, is at or after the target instant. The registered script
  `scripts/v3_beacon_round.py` gives 6659606. An independent one-line calculation gives the
  same (MEASURED: the same log). That script reads files named with its registration stamp, so
  the 2026-10-23 captures were copied under that stamp for the run. The originals keep their
  `_20261023` names.
- **Target at least 72 hours after the push.** The target is 2026-11-21T00:00:00Z, about 28 days
  after this record is pushed.
- **Its own T_fwd at least 2 hours before R′'s scheduled time, and, as a seal re-registration
  by choice, earlier than the first registered round's scheduled time (2026-10-24T00:00:00Z).**
  This record's digest is submitted to OpenTimestamps right after its tag is pushed.
  `ops-health/v3/timestamps/` holds the proofs. The checker decides T_fwd from the upgraded
  proofs, by §3.5's forward-median rule.
- **New tag** `oac-v3-registered-r1`, annotated, on the commit that adds this file. It is
  protected by the `oac-v3-*` tag ruleset.

**Digest of this record**, defined like the registration digest of §3.5: the sha256 of the
canonical JSON of the sha256 of this file, the sha256 of
`logs/v3/rereg_r1/round_Rprime_20261023.txt`, the commit adding them, and the
`oac-v3-registered-r1` tag object id.

## Disclosures

- **The registration's own timestamp proof is late.** §13 step 1 required the registration
  digest to be submitted to OpenTimestamps right after `oac-v3-registered` was placed. It was not
  submitted then. The publishing run stopped at a usage limit after the tag push (REPORTED:
  `progress/v3-register-publish.md`). It is submitted now, together with this record. The
  registration's time in every report is T_fwd of that late proof. The registration remains
  public since the tag object's push on 2026-10-14 (REPORTED: GitHub). Its T_fwd must still be
  earlier than 2026-10-24T00:00:00Z (§13 step 1). If it is not, the report labels the
  registration as dated only by its own late proof.
- **Old round R = 6578966 will be public before any v3 freeze.** Its randomness is public from
  2026-10-24T00:00:00Z. Under §15.3, because a registered round's scheduled time passes before
  the frozen state is sealed, **every statement of the v3 verdict carries this label**: "frozen
  after round R's randomness, a full sealed-equivalent sample, was public". The label names
  round 6578966, the freeze's T1 and T2 times, and R′'s scheduled time. That sample is not the
  sample sealed under R′. The choice of target instant carries no information about any future
  randomness (MODELED).
- **No other change.** The splits, sizes, seeds and seed derivation, the recipe, the hypotheses,
  the win condition, the gates, the tolerance rule, the mutation suite and the analysis plan are
  unchanged.
