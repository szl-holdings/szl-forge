# OAC Ops Health v3: amendments (clarifications under section 15.4)

Everything here is SYNTHETIC. The registered files are not edited:
- `v3/PREREGISTRATION.md`, sha256 `24c239e7…1d0e`, registered at `970ce6ce`, tag
  `oac-v3-registered`;
- `v3/REREGISTRATION_r1.md`, at `80eeaf12`, tag `oac-v3-registered-r1`.

Each entry decides a request in `v3/AMENDMENT_REQUESTS.md`. Each is a clarification under
section 15.4: it fixes a detail the registered text leaves open and changes no frozen item. Each
is decided and committed before the first protocol step it affects. The decider is the
orchestrating session. Stephen approved publishing this record by name ("O1 Push AMENDMENTS",
2026-10-23). The decision date is 2026-10-23, before section 13 step 4. No v3 fit, band
retrain, known-good retrain or non-sealed split exists at this decision.

## AC-1 (decides AR-1): segments and order of the 227 declared fits

ACCEPTED as requested.

- **Segment 1 (step 4), 7 fits:** main, ABL-D, ABL-R, ABL-DR, ABL-n, ABL-jitter, and
  ABL-calibration's Platt fit.
- **Segment 2 (step 5), 120 fits:**
  - the 100 band retrains `train_band_001..100`, in index order;
  - then ABL-n's 20 retrains on the first 16,384 rows of `train_band_001..020`, in index order.
- **Segment 3 (step 6), 100 fits:**
  - for k = 01..20 in order, the four H1 arms: v3 (`train_kg_k`), ABL-D (`train_kg_k_noD`),
    ABL-R (`train_kg_k`) and ABL-DR (`train_kg_k_noD`);
  - then ABL-n's 20 retrains on the first 16,384 rows of `train_kg_01..20`.

The total stays 227 and the cap stays 227. ABL-n's retrains are descriptive. They enter neither
the tolerance vector nor H1.

## AC-2 (decides AR-2): a segment interrupted by a host-side failure

ACCEPTED as requested.

- **Recording.** A run stopped by a cause outside the code (a host restart, a full disk, a killed
  session) gets stop reason `INTERRUPTED`. Its partial trace is committed as evidence. It counts
  as no segment result.
- **Re-run.** The whole segment re-runs from its first fit, at the same commit P, from a clean
  tree, with the runtime pins.
- **Reproduction check.** Every fit that completed in the interrupted attempt must give a
  byte-identical `model.json` in the re-run. A difference is a defect under section 15.6.
- **Reporting.** Every attempt is reported. The cap of 227 applies to each attempt.

The segment's result is its first complete run. That run is fixed by P and the seeds, so it
cannot be chosen against an earlier attempt.

## AC-3 (decides AR-3): the freeze manifest lists every earlier `oac-v3-*` tag

ACCEPTED as requested. The freeze manifest (section 3.5) lists every `oac-v3-*` tag that exists
before `oac-v3-frozen`: `oac-v3-registered-r1`, any later `oac-v3-registered-rN`, and the
section 3.5 tags. The manifest format (canonical JSON, the fields and the digest rule) is
unchanged.

## AC-4 (decides AR-4): the runtime pin under `python -I`

ACCEPTED as requested. Decided on 2026-10-28, before section 13 step 3. Stephen approved
publishing it by name ("A4 Publish AR-4 + AC-4", 2026-10-28).

- **When a run meets the pin.** A run meets the runtime pin of section 13 step 9 when:
  - the interpreter is CPython 3.12.10;
  - it runs isolated, with `sys.flags.isolated` equal to 1 (from `-I`);
  - the process environment holds `PYTHONHASHSEED=0`.
  This is the registered invocation, taken literally.
- **What the pin does not do.** `-I` implies `-E`, so CPython ignores `PYTHONHASHSEED`, and
  string hashing is randomized in every run. Each run record states the effective
  `sys.flags.hash_randomization` and `sys.flags.ignore_environment` as facts.
- **No v3 output may depend on the hash seed.** This covers string hashing, `hash()` and the
  iteration order of a set of strings. Two checks enforce it:
  - the determinism test of section 6.7 item 9;
  - the cross-host DEV check, which requires byte-identical deterministic outputs on two hosts.
  A dependency found later is a defect under section 15.6.
- **No frozen item changes.** The interpreter, the invocation and the environment variable stay
  exactly as registered.

## AC-5 (decides AR-5): the paired no-D files share their D-arm split's stream

ACCEPTED as requested. Decided on 2026-10-28, before section 13 step 3. Stephen approved
publishing it by name ("A5 Publish AR-5 + AC-5", 2026-10-28).

- **The rule.** Each paired no-D training file (`train_noD`, `train_kg_01_noD` …
  `train_kg_20_noD`) is drawn from the same stream and seed as its D-arm split (`train`,
  `train_kg_k`), as section 3.2 item 3 requires ("two paired files from one stream").
- **Where item 1 applies.** Section 3.2 item 1's seed formula is applied to the D-arm split's
  name. It is not applied separately to the no-D file's name.
- **The record.** The step-3 MANIFEST lists, for every file, the split name, the seed actually
  used and the partner it is paired with.
- **No frozen item changes.** The seed formula, namespace, master seed, split names, sizes and
  design-mode draws stay exactly as registered.
