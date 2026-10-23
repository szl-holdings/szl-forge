# MSQ650 source-only corpus and preflight contract

This is a **new, non-authorizing source profile**, not an exception to the v4
exclusive-GPU worker. No runtime launcher, inference call, VM probe, reservation
writer, signer, training framework, dependency installation or publication is
implemented here. Every aggregate result is **MODELED / BLOCKED** with
`training_admitted = execution_authority = publication_eligible = false`.

## Closed candidate

`corpus_contract.observe_corpus` accepts six captured immutable byte buffers:
manifest, train, dev, units, rights and acquisition. Its versioned artifact pins
are fixed in source, not supplied by the caller. It checks byte sizes/hashes,
closed schemas, chain consistency, counts, message roles, provenance labels,
unique tasks/prompts and group separation. A different corpus needs a new
reviewed profile. Builder/verifier digests are metadata, **not observed tool
bytes or proof that derivation was replayed**.

| Split | Math | Science | Quant | Total |
| --- | ---: | ---: | ---: | ---: |
| Train | 192 | 128 | 188 | **508** |
| Development | 48 | 32 | 62 | **142** |
| Total | 240 | 160 | 250 | **650** |

There are 252 groups. Development rows must not enter gradients or training
mounts. There is no `FULL_650_ONCE` recipe. Development is not sealed final
evaluation. Semantic held-out contamination clearance remains **UNKNOWN**;
sealed evaluation, base revision, tokenizer and recipe qualification remain
**UNAVAILABLE**. This module does not open any held-out content.

The profile references original Apache-2.0 mathematical generation, a numerical
QUDT CC-BY-4.0 projection excluding UCUM properties, and historical Treasury
records as retrieved. Rights metadata remains **DECLARED**, not legal authority
or clearance for raw-source redistribution. The actual dataset and raw sources
are not included in this source change. Corpus answers retain SIMULATED or
MODELED labels; byte/structure verification alone is MEASURED, never licensing,
correctness, independent replay or model qualification.

## Three pure observations

1. `corpus_contract.observe_corpus(**six_buffers)` identifies this exact candidate.
2. `coexistence.observe_coexistence(plan, before, after, expected_plan_sha256=...,
   now_epoch=...)` checks two **pre-launch** snapshots against a pinned plan:
   host, boot, GPU UUID, PID/start ticks, executable/model digests, sole incumbent
   GPU process, idle/healthy service, freshness and positive capacity reserves.
   Snapshots must be strictly ordered, at most 30 seconds old, and in the same
   current lease. Plans have a maximum 1,200-second slice; that does not enlarge
   the outer campaign. Capacity and identity inputs are MODELED comparisons.
3. `campaign_budget.observe_reservation(ledger, request, context)` recomputes an
   integer hash-chain ledger against independently supplied genesis/tip/sequence
   expectations. Maximum **200,000 cents** and **180,000 seconds**; lower caps
   are supported. Both absolute elapsed campaign time (including idle gaps) and
   cumulative reserved wall time count. Rate costs round **up** to cents per
   reservation and include declared overhead. Failed, completed and unknown
   attempts retain their full reservations. No refunds or clock resets exist.

`preflight.observe_preflight(...)` runs all three on captured bytes and binds the
fixed manifest, campaign, lease, clock and full charged time window. The proposed
reservation must start at the comparison clock and end at lease expiry. Its
`wall_seconds` must cover setup, worker, termination, drain and receipt writing,
not optimizer time alone. Consistency is **not admission**. Results are frozen;
Boolean coercion raises. Closed input shapes reject injected qualification flags.

The budget API's proposed record is an immutable **unsigned proposal**, not a
persisted reservation. Hashes do not prevent rollback without an authentic prior
tip. Two concurrent feasibility checks can both pass; neither obtains a lease.
Rates, clocks and supplied expectations are unauthenticated comparison inputs.
Unknown storage/network/fixed charges must be conservatively accounted for by
the future authorized custodian before any real reservation.

## Bonsai-preserving runtime remains BLOCKED

Ample VRAM, sampled utilization, a CUDA allocator fraction, CPU cgroups, or a
matching PID list do **not** establish enforced shared-GPU isolation or protect
Bonsai's latency and availability. No caller flag or test fixture can clear
`ISOLATION_BACKEND_UNAVAILABLE`. The incumbent must never be signaled, restarted,
reloaded or evicted. A future worker stop path must identify only its exact owned
job using an independently verified process/unit identity, not PID alone.

The existing `qwen35-receiptagent-v4-json` exclusive-GPU checks and source/data
identity are unchanged. This contract cannot be substituted for its admission.

## Next gates, in order

1. Dedicated Security Review and exact-head required CI for this source slice.
2. Source derivation and rights-scope review, signed curriculum and sealed
   held-out commitments/decontamination, separately pinned base/tokenizer/recipe.
3. Native dependency/runtime closure and externally trusted owner signing policy.
4. Independently qualified Bonsai-preserving isolation and continuous monitoring,
   including fault tests proving termination can affect only the owned worker.
   If the hardware/backend cannot provide this, coexistence stays BLOCKED.
5. Authenticated durable single-writer budget custodian, retained anti-rollback
   tip, atomic single-use reservation and trusted clock/price boundaries.
6. Only after all gates: fresh authorized workload/capacity observation, bounded
   smoke with stop/drain reserve, incumbent readback, then a separately admitted
   training run. Source tests and a merge do not qualify runtime or release.

## Bounded offline tests

No external dataset, GPU, network or new package is needed for CI tests. Fixtures
are SIMULATED; aggregate tests mock corpus success solely to exercise bindings.
Actual pinned-candidate byte replay is a separate local evidence check.

```text
python -I -S -B -m unittest discover -s tests -p test_msq650_*.py -v
python -I -S -B -O -m unittest discover -s tests -p test_msq650_*.py -v
ruff check frontier/msq650_admission tests/test_msq650_*.py --select E9,F63,F7,F82
```

CI's existing Python 3.11/3.12 base test workflow discovers the root tests.
Keep the original failure and successful retry evidence from prior PRs; this
profile does not claim to repair the unrelated Windows sampler timeout.
