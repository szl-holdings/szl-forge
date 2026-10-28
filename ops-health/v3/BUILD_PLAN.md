# OAC Ops Health v3: build plan for protocol steps 2 to 8

Everything here is SYNTHETIC. This file holds no v3 result. When it was written (2026-10-23, ws
`v3a-setup`), no v3 code, v3 data, fit, retrain or DEV run existed (DEV runs: 0).

**Revision 1 (2026-10-28, ws `v3a-setup`).** The host facts, the rule for VM outputs and the
schedule are brought up to date (sections 1, 3, 4, 5, 6, 8, 9 and 11). The step map (section
7), the tags and the freeze-submission planning date (2026-11-19T00:00:00Z) are unchanged.
Section 12 lists every change. At this revision, step-2 code exists in part (ws `v3a-code`), and
no v3 data, fit or retrain exists.

This is a working plan, not a registered item. It changes nothing in the registered protocol
`v3/PREREGISTRATION.md` or in the seal re-registration `v3/REREGISTRATION_r1.md`. Where this plan
and the protocol differ, the protocol wins, and the difference is a defect of this plan. A genuine
problem with the protocol goes to `v3/AMENDMENT_REQUESTS.md`, and work continues on the registered
text. Section numbers (§) refer to `v3/PREREGISTRATION.md`. Labels: MEASURED, REPORTED, MODELED,
CONJECTURE and UNKNOWN, as in the protocol. Dates and durations in the schedule are plans
(MODELED or CONJECTURE), never evidence of time (§3.5).

## 1. Binding inputs

| item | value | label |
| --- | --- | --- |
| protocol | `ops-health/v3/PREREGISTRATION.md`, sha256 `24c239e7…a3e1d0e`, registered at `970ce6ce533f77c2c679f27d31ba86b7c43bb7a4`, tag `oac-v3-registered` (tag object `4fc95253d81b4847e054849d9c874fba90c0fb12`) | MEASURED: `git ls-remote`, sha256 of the public bytes |
| seal re-registration | `ops-health/v3/REREGISTRATION_r1.md`, sha256 `6c5f83ae…3ac3`, at `80eeaf12be7dc4d73e5f81e2bda9c05b6257ab23`, tag `oac-v3-registered-r1` (tag object `7a036f228886b8d29b501aee3277dc1983c21175`) | MEASURED: same |
| research branch head when this plan started | `6b2fe85f318587e013b48a510245a8f2868bc650` | MEASURED: same |
| research branch head at revision 1 | `cd62f1441f563ef1c5767d3b0d4a7c060fbc11cf` (step-2 code, part 1, ws `v3a-code`); the v3 documents, timestamp files and the 2 tools equal their drafting copies, except that the drafting copy of `AMENDMENT_REQUESTS.md` holds one request not yet published (the public bytes are its exact prefix) | MEASURED 2026-10-28: `git ls-remote`, sha256; drafting log `logs/v3/build/resume_verify_20261028T0234Z.txt` |
| drafting copies equal to the public bytes | all 6 v3 documents, the 4 timestamp files and the 2 tools | MEASURED: drafting log `logs/v3/build/v3_files_equal_20261023.txt` |
| beacon round | R′ = 6659606 of the drand default mainnet chain `8990e7a9…b2ce`, scheduled 2026-11-21T00:00:00Z. From r1 on, "R" in the protocol means R′ | REPORTED: r1 |
| exception | the window for frozen-state fixes closed for good at 2026-10-24T00:00:00Z, the scheduled time of the first registered round 6578966 (§15.3, §15.6) | REPORTED: r1 |
| label on every v3 verdict | "frozen after round R's randomness, a full sealed-equivalent sample, was public", naming round 6578966, the freeze's T1 and T2 times and R′'s scheduled time (§15.3) | REPORTED: r1 |

## 2. Fixed instants

Computed as genesis_time + (round − 1) × 30 s and from the protocol's offsets (MEASURED:
arithmetic, 2026-10-23).

| instant (UTC) | what | source |
| --- | --- | --- |
| 2026-10-24T00:00:00Z | round 6578966 public; frozen-state-fix window closed for good; the registration and r1 digests' T_fwd must be earlier | §13 step 1, §15.3, r1 |
| **2026-11-19T00:00:00Z** | **freeze-submission planning date** (R′ − 48 h): step 8 planned to be finished, the freeze manifest digest M submitted | §3.5 planning, §13 step 8 |
| 2026-11-20T22:00:00Z | T1 cutoff: T_fwd of an attesting block must be at or before this, for at least one calendar | §3.5 |
| 2026-11-20T23:00:00Z | T2 cutoff: a public archive record of the frozen and seal tags, dated by the archive, at or before this | §3.5 |
| 2026-11-21T00:00:00Z | R′ scheduled; earliest opening, once the relays serve R′ (and R′ + 1 for the chain link) | §3.4, §3.6 |
| 2026-11-28T00:00:00Z | end of the opening window (7 days), unless an external cause extends it | §3.6 |
| 2026-11-29T00:00:00Z | T_fwd of the `runs/v3/TEST_OPENED.json` proof must be at or before this | §3.6 |

## 3. What the registered text forces on the build (MODELED readings)

1. **One freeze, no fix.** No frozen state exists before 2026-10-24T00:00:00Z, so no
   `oac-v3-frozen-fN` can ever be made. `oac-v3-frozen` is created exactly once and must be
   sealed before R′ (§15.3). After F, any change to the frozen state is a deviation (§15.6).
   The whole frozen state (contenders, ablations, TRUTH, tolerances, mutation suite and test
   list, scorer, analyzer, beacon client and verifier, timing-proof checker, runtime pins) must
   therefore be reviewed and dry-run on DEV data before F. This is the main risk of the build.
2. **Re-registration closes at step 3.** A change to a frozen item is possible only as a
   re-registration before step 3 (§15.3). Every protocol problem found in step 2 must be raised
   in `v3/AMENDMENT_REQUESTS.md` and decided before step 3 starts.
3. **The §3.3 tests block every fit.** "Any failure blocks every fit" covers the v1, v2, twin and
   D-mode parity tests, the guard tests, the beacon tests on recorded rounds and the seal-check
   tests on the registration digest's upgraded proof. So the beacon client and verifier and the
   timing-proof checker, with their tests, are step 2 work, not step 8 work, and the upgraded
   registration proof must be committed before step 4.
4. **Runtime pins.** CPython 3.12.10, run as `python -I` with `PYTHONHASHSEED=0`, for every fit
   and retrain, the mutation suite, the opening and the analysis (§13 step 9). Each entry point
   refuses to run otherwise (version, `sys.flags.isolated`, `sys.flags.hash_randomization`).
   Under `-I` the script directory is not on `sys.path`, so entry points add the research root
   themselves. At revision 1, one open request about these pins is held in the drafting copy of
   `v3/AMENDMENT_REQUESTS.md` for the orchestrator. It must be decided before step 3 (gate G3),
   because a change to a frozen item is possible only as a re-registration before step 3 (§15.3).
5. **One step at a time.** Each step is committed, pushed and tagged before the next starts
   (§13). Tags are placed by the orchestrator only, never by a lane. Each step's first commit
   records the previous tag's target and tag object id (`v3/results/TAGS.json`, append-only), so
   a moved tag is detected (§14.6).
6. **Clean tree and origin.** Every fit runs from a clean tracked tree at a pushed commit P. The
   fit trace records P and is committed directly on P as C, which the orchestrator tags. The
   receipts follow and name C and the tag (§13 step 4, §14.4).

## 4. Hosts

| host | role | facts | label |
| --- | --- | --- | --- |
| **H-local**, the drafting host (Windows 11, 16 logical processors, 31.59 GiB) | every registered computation: steps 3 to 7, the freeze tooling, the opening (step 9) and the analysis (step 10). It is the protocol's opening host | CPython 3.12.10 (`tags/v3.12.10:0cc8128`, MSC v.1943, 64 bit), `python.exe` sha256 `4d6f5f81a4bca11191c4c7c6b43632694d0a4ce74e068619d8fdc161d469859a`, `python312.dll` sha256 `9a0e3435…2120`; free physical memory 7.70 GiB; C: free 25,587 MB at 03:05Z and 23,122 MB at 03:14Z on 2026-10-23 (it is volatile and has reached 0). At revision 1 (2026-10-28T02:33Z to 02:39Z): same interpreter and sha256; C: free 20,736 MB; free physical memory **2,201 MB** of 32,346 MB, below the opening preflight's 4 GiB (CP-4) | MEASURED: `logs/v3/build/local_python_20261023.txt`, `logs/v3/build/resume_verify_20261028T0234Z.txt` |
| **H-vm**, the CPU execution VM (Ubuntu 22.04, kernel 6.8.0-107, 6 vCPU Xeon E5-2699 v4) | DEV runs, the v3 test suite on a second platform, the cross-host determinism check; no registered computation while its interpreter misses the pin | work dir `/home/Ubuntu/oac-v3-20261023T030756Z-68976ccc`, claim `.oac-v3.claim.json` sha256 `e5e49e06…44dd`; interpreter miniconda CPython **3.12.4** (Anaconda build, GCC 11.2.0, glibc 2.35), sha256 `f299eeefdd0297384ef80076fa9b984469cc1a666e7e24872e5773ecd9bb1611`; libm sha256 `3dd5511a…dfee6`; MemTotal 94.39 GiB, MemAvailable 78.82 GiB; 136,754 MB free on `/`; git 2.34.1; every other `python3.12` found is the same 3.12.4 (three venv copies with the same sha256, and conda's cached `python-3.12.4` package). At revision 1 (2026-10-28T02:37Z): claim unchanged; base interpreter and libm unchanged; the work dir also holds `py31210/`, a conda-forge CPython **3.12.10** environment (packaged by conda-forge, built Apr 10 2025, GCC 13.3.0, glibc 2.35; `bin/python3.12` sha256 `da02ac288c3d1b6d0efa4fab1cab08f7c91a02c05d739beedf45b35ce88269f5`, the `python` package plus conda's default dependencies), made 2026-10-24; MemAvailable 76.00 GiB; 92,434 MB free on `/` | MEASURED: `logs/v3/build/vm_census_20261023.txt`, `logs/v3/build/vm_python_search_20261023.txt`, `logs/v3/build/resume_verify_20261028T0234Z.txt`; the environment's maker is ws `v3a-code` (REPORTED: drafting records) |
| **second re-check host** | planned host for re-checking VM outputs | not in use at revision 1: its SSH host key is not confirmed, so no lane connects to it until the orchestrator says so; until then every re-check runs on H-local | REPORTED: orchestrator, 2026-10-28 |
| **GitHub-hosted Windows runner** in szl-forge | the independent re-run of step 9 with the same pins | planned by §13 step 9; a workflow is needed (section 6, step 8) | REPORTED: §13 step 9 |

**The VM does not meet the pin.** Its only CPython 3.12 is 3.12.4. The Anaconda `defaults`
channel, the only channel the VM's conda lists, offers no 3.12.10; conda-forge does (MEASURED:
`logs/v3/build/conda_python312_versions_20261023.txt`). Decision: the build does not depend on the
VM. Every registered computation runs on H-local. A python-only conda-forge `python=3.12.10`
environment inside the VM work dir is an option for the orchestrator to approve. It would let the
VM carry step 10's FRAGILE check (72-hour budget; v2's check stopped twice on MemoryError on
H-local, REPORTED: `v2/results/FINAL_RESULTS.md`). It never moves a registered fit to the VM
unless the determinism rule of section 5 also passes there.

At revision 1 such an environment exists (`py31210/`, above). The decision stands: no registered
computation depends on the VM.

**Every VM output is re-checked on H-local (decision "VM + local re-check", 2026-10-23).** The
VM is not under this program's sole control. Every file computed on the VM is recomputed on
H-local with `py -3.12`, at the same commit and from the same inputs, and is committed only if
its sha256 is identical. Both sha256 values go to `logs/v3/build/recheck_<ws>.txt` in the
drafting repository. A difference is a blocker: both copies are kept in
`logs/v3/build/quarantine/<ws>/<UTCstamp>/` with a `MISMATCH.json` (paths, both sha256 values,
sizes, commit, exact commands, interpreter versions and binary sha256), neither copy is committed
as a result, nothing is rewritten, re-derived or retried silently, and neither the protocol nor
`v3/AMENDMENTS.md` is changed to force agreement. Before each local recomputation H-local must
have C: free of at least 15,360 MB plus twice the expected output size and at least 2,048 MB of
free physical memory, and it runs one heavy process at a time; otherwise the lane stops and
reports. Nothing secret is ever staged on the VM. Two consequences (MODELED): a VM job saves no
H-local compute, it only adds a second-platform observation; and a VM file that records host
facts (a run record naming its interpreter, its stdout, or any digest over such a file) cannot
equal its local recomputation, so under this rule it is not committed. How such a file is kept is
open decision D-1 (section 5).

## 5. Determinism: what must match, and the DEV check

What the protocol requires to be byte-identical (REPORTED, with the reading MODELED):

- the same recipe trained and frozen twice from the same seeds gives the same `model.json`
  (§6.7 item 9), which is a same-host requirement;
- the official opening and the independent re-run give the same sha256 for every score file,
  with the same pins (§13 step 9, §15.6). The re-run is planned on a Windows runner because
  floating-point library results can differ between platforms (MODELED there);
- the v1 parity rows (byte for byte) and the v2 non-sealed digests in `v2/data/MANIFEST.json`
  must be reproduced on any host that runs the §3.3 tests (§3.3);
- every regenerated non-sealed or retrain split must match its digest in `v3/data/MANIFEST.json`
  (§3.4).

Byte-identity of DEV outputs between Windows and Linux is not required by the protocol. Its
status is UNKNOWN, because no v3 code exists yet. Possible sources of difference (MODELED):
`math.exp` and `math.log` come from the platform math library (Windows UCRT on H-local, glibc 2.35
libm on H-vm). The generator's `s_nom` draw (`Exp(mean 900)`, then rounded to 6 decimals) and the
label draw `u < p` use them, so they can differ only in rare boundary cases. IRLS evaluates the
sigmoid on every row of every iteration, so fitted weights can differ in their last digits.
Addition, multiplication, division and square root give the same results when the operation
order is the same.

**DEV check (ws `v3b-dev`, before step 3).** At one pushed commit, on H-local (pinned) and on
H-vm, with the same DEV seed and at most 2,048 rows per regime: generate the DEV splits, run one
DEV fit (at most 2,048 rows), score the DEV rows with the kernel, and run the gates' plumbing.
Then compare the sha256 of every output. The result goes to
`logs/v3/dev/<run_id>/xplat_compare.json`, with both interpreters' versions and sha256. Rule,
fixed now and before any DEV run: (a) every registered computation runs on H-local, whatever the
check shows; (b) a VM job may compute step 10's FRAGILE check only if the VM has a pinned 3.12.10
and the check shows identical bytes for the bootstrap layer; (c) a difference in the
generator's DEV bytes across hosts is recorded as a finding, and step 3 runs on H-local in any
case, after the §3.3 parity tests pass there.

**Status at revision 1 (open).** The first cross-host DEV comparison ran at `cd62f144` on
2026-10-23, with the VM's base 3.12.4 (not the pin) and H-local's pinned 3.12.10, and 125 of 125
tests passing on each. Of the 8 files compared, `dev_fit/model.json`, `dev_fit/diagnostics.json`,
`tests.json` and `advisories.jsonl` were byte-identical. `dev_fit/artifact_receipt.json` (field
`fit.trace_sha256`), `dev_outputs.json` (field `dev_fit.receipt_sha256`), `run.json` and
`stdout.txt` were not (MEASURED: drafting record
`logs/v3/build/quarantine/v3a-code/20261023T085156Z/MISMATCH.json`). Under the re-check rule
(section 4) this is a blocker of ws `v3a-code`, and neither copy is committed as a result. A
second VM DEV run at `cd62f144`, with `py31210/` 3.12.10 (2026-10-24), has only its run record in
the drafting repository and no local recomputation record (REPORTED: drafting records). The
DEV check of this section is therefore not done. Rules (a) to (c) are unchanged.

**Open decision D-1 (for the orchestrator, before more VM DEV runs).** §15.5 says every DEV run's
output is committed; the re-check rule forbids committing a VM file that differs from its local
recomputation. At revision 1, three DEV runs have no committed output: the two VM runs above and
the local recomputation `20261023T085057Z-local-4d68cde8` (MEASURED: the research branch at
`cd62f144` holds only `20261023T084621Z-local-0a662cff` and `20261023T084715Z-local-3ca5bddb`).
Options (MODELED): (i) commit such files under a separate path as quarantined evidence, labelled
"VM copy, not a result", next to their `MISMATCH.json`; (ii) leave them uncommitted and disclose
each such run in the compliance audit and the report; (iii) before any further VM DEV run, make
the compared set exclude every host-fact file by design, so that only deterministic files are
compared and committed. The plan takes no option itself.

## 6. Code layout (planned; paths under the research root, `ops-health/` in public)

```
v3/__init__.py
v3/research/              standard library only; imported as v3.research.*
  registry.py             registered constants: namespace, M3 = 20261013, seed rule, split table,
                          counts (227, 100, 20), representation table, c = 4 and floors, runtime
                          pins, sha256 of PREREGISTRATION.md and REREGISTRATION_r1.md, R' = 6659606
  generator.py            ops-health-generator/3.0.0: v2 draws reused; modes nominal, v1-parity,
                          v2-parity (non-sealed only), D (paired arms), twin; sealed mode accepts
                          SALT only from beacon.verify() for the registered chain and round
  guards.py               v2 seal guard; refusal of paths under v2/data/sealed/; DEV cap
                          (split name `dev`, at most 2,048 rows per regime per run)
  make_data.py            step 3: writes non-sealed splits; hashes retrain splits; MANIFEST.json
  representation.py       table R (seconds / 900) and the v1 min-max table; mapping to the R scale
  lr.py                   IRLS reused from v2/research/lr.py (same stop reasons, AM-2 rule)
  pipeline.py             one recipe run: fit -> threshold index k on validation -> Mondrian on
                          conformal -> canonical model.json
  segments.py             the bounded sequence: declared fit list (227), segments 1 to 3, cap,
                          stop reasons; pin and clean-tree refusal; runs the §3.3 tests first;
                          trace with HEAD P
  freeze.py               canonical artifacts, receipts, verify_origin rules (a) to (e) of §14.4
  metrics.py, gates.py    §11.2 gates (kept and new), g_failclosed with 76 probes
  tolerances.py           §11.3 rule over the 100 band retrains; vector and sha256
  h1.py                   H1 statistics and criterion; ablation contrasts
  mutation.py             37 artifact, 7 receipt, 8 kernel-source and the descriptive mutants;
                          passes 1 and 2; refuses to run unless the tolerance digest is in an
                          earlier commit
  h2.py, readiness.py     H2 criterion; AND, min and mean aggregation; separation measures
  bootstrap.py            §8 bootstrap, reusing v2/research/bootstrap.py's stream layout
  beacon.py               drand client and verifier: relay rule, randomness = sha256(signature),
                          chain link to R' + 1, /info re-check; level HASH_AND_RELAY_AGREEMENT
  seal_check.py           §3.5 timing-proof checker: OpenTimestamps proof replay, headers from two
                          explorers (hash, proof of work, chain H..H+10, forward median), Software
                          Heritage and Internet Archive records; SEALED / PENDING / EXPOSED
  score.py, analyze.py    opening scorer; analyzer with every step-10 computation and FRAGILE
  opening.py              step 9 preflight and the single opening; re-run mode
v3/ops-health/ops_health.py   kernel v3: the v2 contract plus the transform-table check (§6.7)
v3/tests/                 unittest suite, DEV data only; recorded beacon and timestamp fixtures
v3/tools/                 v3_beacon_round.py, v3_ots_stamp.py (exist); run_v3_tests.py, dev_run.py
v3/data/MANIFEST.json     step 3
v3/fits/                  per-segment fit traces and model.json files (steps 4 to 6)
v3/results/               TAGS.json, H1, tolerances, kernel-relevant test list, H2,
                          MUTATION_READINESS, later FINAL_RESULTS
v3/seal/FREEZE_MANIFEST.json  step 8
runs/v3/receipts/, runs/v3/TEST_OPENED.json
logs/v3/dev/<run_id>/     every DEV run's output (§15.5)
```

The kernel's public path is `ops-health/v3/ops-health/ops_health.py`. The doubled
`ops-health/` is intended: it mirrors v2's `ops-health/v2/ops-health/ops_health.py` (MEASURED:
both tracked at `cd62f144`; note added in revision 1).

The only use of the banned word in new v3 code is the frozen v1 authority-map key, which the v3
kernel takes from `AUTHORITY_MAP` in `v2/ops-health/ops_health.py` (§6.7 item 5). Verification
level: HASH_AND_RELAY_AGREEMENT, the registered minimum. A standard-library BLS12-381 verifier is
not built, because a defect in it could not be fixed after F (§3.4 allows either; decision of
this plan).

## 7. Step map (§13 steps 2 to 8)

### Step 2: v3 code and tests on DEV data

| sub-lane (ws id) | builds | must pass before step 3 |
| --- | --- | --- |
| `v3b-gen` | registry, generator, guards, representation | v1 parity; v2 non-sealed digests; twin parity on DEV; D-mode check; every guard refusal (§3.3) |
| `v3b-kernel` | kernel v3 | the 76 probes (66 v2 input probes, TT1 to TT10); duplicate keys in inputs and artifact; all-false authority map on ALERT, NO_ALERT and ABSTAIN; import check; receipt mismatch (§6.7) |
| `v3b-recipe` | lr, pipeline, segments, freeze, receipts, verify_origin | determinism (twice, byte-identical `model.json`); refusals (pins, dirty tree, cap); origin rules (a) to (e) on a fresh clone with 0 skipped (§14.5); a test clone fetches enough history for first parents and descendants |
| `v3b-gates` | metrics, gates, tolerances, readiness | broken baselines (constant-ALERT, fail-open kernel, majority all INVALID_BASELINE, §11.1); metric recomputation; tolerance code on hand-made numbers only (no tolerance on DEV data, §15.5) |
| `v3b-mut` | mutation | each mutant class is generated and applied, both passes execute, the tolerance-digest refusal fires; no catch rate is computed on DEV data |
| `v3b-beacon` | beacon client and verifier | accepts at least 3 recorded earlier rounds, each with its next round, from all three relays; refuses the seven mutated cases of §3.3; never fetches R′ |
| `v3b-seal` | timing-proof checker | accepts the registration digest's upgraded proof with headers recorded from two explorers; returns SEALED, EXPOSED or PENDING as §3.5 says; refuses the eight mutated cases of §3.3 |
| `v3b-open` | bootstrap reuse, scorer, analyzer, opening and re-run mode | bootstrap stream layout equal to v2's on DEV inputs; the rest may continue until step 8 |
| `v3b-dev` | DEV runs on H-local and H-vm | the cross-host check of section 5; every output committed under `logs/v3/dev/` |
| `v3b-review` | independent adversarial review of the step 2 code against §3.3, §6, §11, §14.4 | findings fixed or filed; open protocol problems decided (AR-1 to AR-3 and any new one) |

Host: H-local for every pinned DEV run; H-vm job VM-1 (section 8). No tag. Outputs: code and
tests under `v3/`, DEV outputs under `logs/v3/dev/`, all committed and pushed.

### Step 3: non-sealed splits and the data manifest

- Script: `v3/research/make_data.py` at a pushed commit P3, on H-local, pinned, after the §3.3
  suite passes at P3.
- Writes (not committed, §14.7): `train`, `train_noD`, `calibration`, `conformal`, `validation`,
  `gates` (94,208 rows, about 31 MB at about 330 bytes per row, CONJECTURE).
- Hashes, generating and discarding: `train_band_001..100`, `train_kg_01..20`,
  `train_kg_01..20_noD` (140 splits of 32,768 rows).
- Commits `v3/data/MANIFEST.json`: generator source sha256; every non-sealed digest; every
  retrain digest; the sealed-split definitions with the beacon seal (chain hash, R′ = 6659606,
  and the r1 reference). No sealed row is generated.
- Check `df -B1M /c` first (more than 50 MB is written). No tag.

### Step 4: main and ablation fits, segment 1 (7 fits)

- Script: `v3/research/segments.py --segment 1` at pushed commit P4, H-local, pinned, clean tree.
  Order as in AR-1.
- Commit C4 (fit trace and the 7 `model.json` files, directly on P4); push; the orchestrator tags
  **`oac-v3-fit`** on C4; then the receipts naming C4 and the tag; push. v3 is frozen.
- NO_ELIGIBLE_FIT ends v3 here if the main fit is not eligible (§6.1).

### Step 5: band retrains, segment 2 (120 fits), then the tolerances

- `segments.py --segment 2` at P5: 100 band retrains, then ABL-n's 20 band-head retrains (AR-1).
- C5 on P5; push; tag **`oac-v3-band`**; receipts.
- `v3/research/tolerances.py` computes the vector from the 100 band retrains (θ on the R scale,
  BA on `gates`, k) and commits it with its sha256; push; tag **`oac-v3-tolerances`**. No mutant
  exists before this commit.

### Step 6: known-good retrains, segment 3 (100 fits), then H1

- `segments.py --segment 3` at P6: 4 H1 arms × 20, then ABL-n's 20 kg-head retrains (AR-1).
- C6 on P6; push; tag **`oac-v3-kg`**; receipts.
- `v3/research/h1.py` evaluates H1 and the ablation contrast; commit and push before any mutant.

### Step 7: kernel-relevant test list, mutation suite, H2

- Commit `v3/results/KERNEL_RELEVANT_TESTS.json` (test ids and sha256 of the test files); push.
  Never changed afterwards.
- `v3/research/mutation.py`, H-local, pinned: all mutants, passes 1 and 2, the in-place test run
  for each BLIND_SPOT; ABL-G and ABL-origin contrasts. Kernel-source mutants run in a temporary
  copy, never in the tracked tree. The squash-style duplicate is built in a scratch clone.
- `v3/research/h2.py` and `readiness.py`: H2 and the readiness metrics; commit and push. No tag
  is named by §13 for this step.

### Step 8: freeze, seal and archive records

1. Freeze candidate. The TRUTH reference is frozen (its threshold by §6.4 on `validation`, its
   quantiles from `conformal`). The v1 Hub kernel and model at revision `dd7d1098…` are frozen
   as tracked files, reusing v2's staged copies if they are byte-identical. Commit the opening
   code (scorer, analyzer, beacon, seal checker, opening and re-run mode, runtime pins).
2. Dry runs on DEV data before F: the opening in DEV mode against recorded beacon responses; the
   scorer and the analyzer end to end on DEV scores; and the re-run workflow on the Windows runner
   on the same DEV inputs, with byte-identical score files to H-local's. The workflow is planned
   at `.github/workflows/oac-ops-health-v3.yml` on the research branch, triggered by push. It has
   no secrets, CPython 3.12.10 by `setup-python`, and its actions are pinned by commit as the
   existing `oac-ops-health.yml` does. It sits outside `ops-health/`, so it is not in the freeze
   manifest, and it runs the frozen code at F. The existing v2 workflow runs only on pull
   requests and pushes to `main`.
3. A pre-freeze review of the whole frozen state against §3.4 (the seal rule's list) and §14.4.
   Its findings are fixed before F, because no fix is possible after.
4. Then, in the order of §13 step 8: push F, and the orchestrator tags **`oac-v3-frozen`**;
   write `v3/seal/FREEZE_MANIFEST.json` (canonical JSON; every file under `ops-health/` at F by
   sha256 and size; every earlier `oac-v3-*` tag, `oac-v3-registered-r1` included per AR-3; the
   chain hash and R′); submit M to all four calendars (at least one from each group, §3.5) with
   `v3/tools/v3_ots_stamp.py`; commit Q (the manifest and the pending proofs only) on F; push,
   and the orchestrator tags **`oac-v3-seal`**; request a Software Heritage save of szl-forge and
   Internet Archive captures of the two `api.github.com` tag-ref URLs; open the draft pointer
   pull request quoting F, the tag object id and M; commit the upgraded proofs as the calendars
   serve them.
5. Until R′, a daily read-only context check (`ots info` of the upgraded proofs; the archive
   APIs) is logged as context only. It is not the seal check. The seal check is decided only by
   the frozen checker in the opening preflight (§3.5).

## 8. VM jobs

Common rules: the work dir above and nothing else; the claim file is kept; CPU only; at most 4
processes, each under `nice -n 10`; no credential or token is ever on the VM; llama-server,
cloudflared and every process this lane did not start are left alone. Code comes only from the
public repository at an exact pushed commit:

```
W=/home/Ubuntu/oac-v3-20261023T030756Z-68976ccc
cd "$W" && git clone --filter=blob:none --sparse https://github.com/szl-holdings/szl-forge.git src-<sha8>
cd src-<sha8> && git sparse-checkout set ops-health && git checkout --detach <sha>
test "$(git rev-parse HEAD)" = "<sha>"
PYTHONHASHSEED=0 nice -n 10 /home/Ubuntu/miniconda3/bin/python3.12 -I ops-health/v3/tools/<entry>.py ... > "$W/out/<run_id>/..."
```

Results return by `scp` to the drafting repository (`logs/v3/build/vm_return/<run_id>/` at
revision 1). Each file is then recomputed on H-local and committed, under
`ops-health/logs/v3/dev/`, only if its sha256 equals the local recomputation (section 4); the
comparison goes to `logs/v3/build/recheck_<ws>.txt`. With the work dir's 3.12.10, the
interpreter path in the template above is `"$W"/py31210/bin/python3.12`. Before any VM write over
1 GB, `df` on the VM is checked; its free space fell from 136,754 MB to 92,434 MB between
2026-10-23 and 2026-10-28 through another lane (MEASURED).

| job | stage | what | interpreter |
| --- | --- | --- | --- |
| VM-0 | `v3a-setup` (done) | work dir, claim, census; census repeated at revision 1 | 3.12.4 (census only) |
| VM-1 | `v3b-dev` | the v3 test suite and the DEV runs of section 5 at each step-2 code commit that the DEV check names; at most 4 processes; outputs committed only after the local re-check, and not before D-1 is decided | 3.12.4, labeled "not the runtime pin" in every output, or `py31210/` 3.12.10 |
| VM-2 | step 10, optional | the FRAGILE check (100 stream sets, checkpointed per set), only with the pinned 3.12.10 in the work dir and rule (b) of section 5. Under the re-check rule its output must also be recomputed on H-local, so VM-2 no longer takes the FRAGILE load off H-local (MODELED); the FRAGILE check is planned on H-local within its 72-hour budget | `py31210/` 3.12.10 only |

No VM job generates sealed rows, opens any v2 sealed file, or computes a registered statistic.

## 9. Schedule with buffers

The program's sessions were stopped by usage limits for most of 2026-10-14 to 2026-10-22
(REPORTED: r1). After that, the research branch did not move from `cd62f144` (2026-10-23) to
2026-10-28, and the registration and r1 proofs were not upgraded (MEASURED 2026-10-28; the cause
is UNKNOWN to this lane). Revision 1 therefore re-baselines the schedule from 2026-10-28. The
planning date and the cutoffs do not move. The buffer between the planned freeze and the planning date shrinks
from 7 days to 4 days, with a further 1.9 days before the T1 cutoff, and steps 3 to 8 keep a
latest margin of 2 to 4 days each. Each lane keeps `progress/<ws>.md`, appends "done: …; next: …"
after each sub-step, and commits and pushes small increments.

Revision 0's targets (2026-10-23) stay in this file's history at `66a9cb3f`. Target and latest
dates are plans (MODELED). The status column is as of 2026-10-28T02:40Z (MEASURED unless
marked).

| milestone | ws id | target done (UTC) | latest done | status at revision 1 / note |
| --- | --- | --- | --- | --- |
| setup | `v3a-setup` | 2026-10-23 | — | DONE 2026-10-23 (plan public at `66a9cb3f`); census repeated and this revision on 2026-10-28 |
| AR-1 to AR-3 decided | orchestrator | — | — | DONE: AC-1 to AC-3, public at `808edc3e` |
| registration and r1 proofs upgraded and committed; T_fwd checked against 2026-10-24T00:00Z | orchestrator | 2026-10-29 | 2026-10-30 | LATE (revision-0 latest was 2026-10-26): each drafting proof still holds 4 pending attestations and no Bitcoin attestation. Needed by `v3b-seal` (§3.3) and by §13 step 1 |
| archive-path pretest (Software Heritage save, Internet Archive capture of the two registered tag refs) | orchestrator | 2026-10-29 | 2026-10-31 | no record found at revision 1 (revision-0 latest was 2026-10-28). §13 step 1 planned this test of the T2 path; it measures the archive delay before the freeze |
| step 2 part 1: generator 3.0.0, guards, representation, kernel v3, gates, tolerance code, mutation catalogue, bootstrap reuse, pipeline, freeze | `v3a-code` | — | — | PUSHED at `cd62f144` (2026-10-23); DEV runs `0a662cff` (122 of 125) and `3ca5bddb` (125 of 125) committed. Open: the cross-host blocker and D-1 (section 5) |
| step 2 beacon client and verifier | `v3b-beacon` | 2026-10-31 | 2026-11-02 | no `beacon` module at `cd62f144` (MEASURED); listed as pending by the DEV run records (REPORTED) |
| step 2 recipe remainder (segment runner, origin rules (a) to (e)), seal checker | `v3b-recipe`, `v3b-seal` | 2026-11-01 | 2026-11-03 | no `segments` or `seal_check` module at `cd62f144`; the IRLS fit sits in `pipeline` (MEASURED); `v3b-seal` needs the upgraded registration proof first |
| step 2 mutation passes 1 and 2 | `v3b-mut` | 2026-11-01 | 2026-11-03 | catalogue exists; passes listed as pending by the DEV run records (REPORTED) |
| DEV runs and cross-host check, with the local re-check | `v3b-dev` | 2026-11-02 | 2026-11-04 | first comparison blocked (section 5); D-1 first. VM-1 |
| step-2 review; open requests decided; last chance for a re-registration | `v3b-review`, orchestrator | 2026-11-04 | 2026-11-06 | gate G3: step 3 does not start before this |
| step 3 splits and manifest | `v3c-data` | 2026-11-05 | 2026-11-07 | |
| step 4 segment 1; `oac-v3-fit` | `v3d-fit` | 2026-11-06 | 2026-11-08 | |
| step 5 segment 2; `oac-v3-band`; tolerances; `oac-v3-tolerances` | `v3e-band` | 2026-11-07 | 2026-11-10 | |
| step 6 segment 3; `oac-v3-kg`; H1 | `v3f-kg` | 2026-11-08 | 2026-11-11 | |
| step 7 test list; mutation suite; H2 | `v3g-mut` | 2026-11-10 | 2026-11-13 | |
| step 8 freeze candidate, DEV dry runs, Windows-runner check, pre-freeze review | `v3b-open`, `v3h-freeze` | 2026-11-13 | 2026-11-17 | opening code is built from 2026-10-31 on, in parallel |
| **step 8 F, `oac-v3-frozen`, M submitted, Q, `oac-v3-seal`, archive requests, pointer pull request** | `v3h-freeze`, orchestrator | **2026-11-15T00:00Z** | **2026-11-19T00:00Z** (planning date) | hard limits 2026-11-20T22:00Z (T1) and 23:00Z (T2) |
| upgraded freeze proofs committed; T1 and T2 seen (context only) | `v3h-freeze` | 2026-11-16 | 2026-11-20T12:00Z | |
| step 9 opening on H-local; `TEST_OPENED.json` proof; Windows re-run | orchestrator | 2026-11-21 | 2026-11-27 | window ends 2026-11-28T00:00Z |
| step 10 analysis, FRAGILE (72 h budget), compliance, datasheet, report | later ws | 2026-11-26 | — | |

**Contingency decisions (fixed now; CP-0 is new and CP-1 is moved in revision 1).**

- **CP-0, 2026-11-06T00:00:00Z.** If gate G3 has not passed by then (step 2 reviewed, every open
  request decided), the orchestrator decides between continuing on the compressed schedule and
  a second seal re-registration by choice, as CP-1 describes. Deciding here leaves the most time
  for a later round. A re-registration that changes anything other than the beacon round is
  possible only before step 3 (§15.3).
- **CP-1, 2026-11-14T00:00:00Z** (revision 0: 2026-11-15). If H2 is not committed by then, the
  orchestrator decides between finishing the freeze by the planning date and a second seal
  re-registration by choice (§15.3), pushed by 2026-11-17T00:00:00Z at the latest: target a
  00:00:00Z instant at least 72 hours after its push, R″ by the §3.4 rule from live relay
  parameters, its own OpenTimestamps proof with T_fwd before R′'s scheduled time and at least 2
  hours before R″'s, and a new tag `oac-v3-registered-r2`. The §15.3 label already applies, and
  the report would disclose both abandoned rounds.
- **CP-2.** A defect found after F cannot be fixed (section 3 item 1). It is disclosed, and the
  opening runs on the frozen state as it stands. This is why steps 3 to 8 carry a latest margin
  of 2 to 4 days each and the freeze has a pre-freeze review.
- **CP-3.** If no calendar has attested the freeze by 2026-11-20T12:00Z, nothing may be re-frozen.
  The seal check then decides, and EXPOSED leads to a new round under §3.6.
- **CP-4, H-local resources.** The opening preflight needs at least 2 GB free disk and at least
  4 GiB free physical memory (§13 step 9). At revision 1, H-local had 2,201 MB of free physical
  memory, below that floor and just above the 2,048 MB floor for a local re-check (section 4)
  (MEASURED 2026-10-28). Freeing memory is a host-owner action (closing applications), not a lane
  action; no lane installs, cleans or deletes anything to make room. C: is volatile, so every
  pinned lane (steps 3 to 7, the re-checks and the opening) checks `df -B1M /c` and memory at its
  start and stops and reports below its floor, and the orchestrator keeps at least 10 GB free on
  C: from 2026-11-19 on. A refused preflight is retried, and host-side causes never extend the
  window (§3.6).

## 10. Compute and size estimates (CONJECTURE)

- v2's 12 search trials on 16,384 training rows took 1.4 s to 75.4 s each, 203 s in all
  (MEASURED: `v2/results/search_trace.json`, `duration_ms`). At 32,768 rows, a v3 recipe run
  (regenerate the split, fit, threshold, conformal) is estimated at 10 s to 60 s. Segment 1 then
  takes under 15 minutes, segment 2 30 to 150 minutes and segment 3 25 to 120 minutes, run as one
  sequential process per segment.
- Gate evaluation per fixture (76 probes, the metamorphic grid, BA on 8,192 rows) and the
  mutation suite (52 registered mutants, two passes, the kernel-source mutants each running the
  kernel-relevant tests) are estimated at 1 to 4 hours in all.
- Disk: about 31 MB of non-sealed splits in step 3. At the opening, about 132 MB of sealed rows
  (§4) plus the score files, of which the `test` file and the two H3 files are predicted above
  50 MB and published as digests only (§14.7).

## 11. Rules every lane follows

- Push only to `claude/oac-v3-research`, never to `main` and never with force. Fetch before
  every push. If another lane changed `ops-health/` on the branch since the lane's base, stop and
  report. Lanes may build in parallel in their own sparse clones under `scratch/`, but only one
  lane pushes at a time (§14.9).
- Commits are signed, stored with LF line endings (`core.autocrlf=false` in each clone), and end
  with the co-author line. No lane creates an `oac-v3-*` tag.
- Standard library only in the kernel and the research code. The existing stamping tool uses
  the public `opentimestamps` library; the frozen checker does not.
- DEV rule (§15.5): split name `dev` only; at most 2,048 rows per regime per DEV run; no fit on
  more than 2,048 DEV rows; no P or H statistic, tolerance, detection ratio, power or WIN-chance
  quantity on DEV data; every DEV run's output committed. The v2 parity tests generate v2
  non-sealed rows only to compare digests.
- No run generates rows from the sealed seed material of any registered round, abandoned rounds
  included. No v2 sealed file is opened; v3 paths refuse `v2/data/sealed/`.
- Check `df -B1M /c` before any write over 50 MB, and keep every clone sparse.
- Every VM-computed file is recomputed on H-local and committed only if its sha256 is identical;
  a difference is quarantined and reported, never retried silently (section 4). The second
  re-check host is not contacted until the orchestrator confirms it.
- Every claim in a v3 file carries its label, and every metric is SYNTHETIC.

## 12. Revision record

| revision | date (UTC) | ws | public commit | what changed |
| --- | --- | --- | --- | --- |
| 0 | 2026-10-23 | `v3a-setup` | `66a9cb3f` | first version (sha256 `c7db116e3e46818830463812c572c05844c321da1b37eff701339c0d2ccfe6e1`) |
| 1 | 2026-10-28 | `v3a-setup` | this commit | revision note at the top; section 1: branch head at revision 1; section 3 item 4: an open request on the runtime pins must be decided before step 3; section 4: H-local and H-vm facts of 2026-10-28 (H-local free memory below the opening floor; the VM work dir's conda-forge 3.12.10), the second re-check host (not in use), and the rule that every VM output is re-checked on H-local; section 5: status of the cross-host check (blocked) and open decision D-1; section 6: a note that the kernel's doubled `ops-health/` path is intended; section 8: return path, re-check, interpreter path, VM disk, VM-2 reading; section 9: schedule re-baselined from 2026-10-28 with a status column, CP-0 added, CP-1 moved from 2026-11-15 to 2026-11-14, CP-2 and CP-4 updated; section 11: the re-check rule. Unchanged: the binding inputs, the fixed instants, the step map, the tags, the planned code layout, and the planning date 2026-11-19T00:00:00Z |
