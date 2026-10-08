# OAC Ops Health v2: mutation suite, broken baselines and readiness aggregation

Everything here is **SYNTHETIC**: synthetic data, synthetic mutants, and nothing that says anything about a real system. Every number was **MEASURED** by `PYTHONUTF8=1 py -3.12 -B -m v2.research.mutation run` (log `logs/v2/mutation_run.txt`, results `v2/results/mutation_readiness.json`), on the registered **validation** split only (opened with purpose GATES). `runs/TEST_OPENED.json` did not exist before or after the run (MEASURED), and no code path of the suite opens a sealed split (MODELED, from the source). This report makes no statement about v2 versus v1; that comparison exists only after the single final opening.

**Post-run corrections (wording, render and test fixes only).** The mutation run was recorded in `c46a42c`, before the sealed splits were opened. The single final opening happened afterwards (`runs/TEST_OPENED.json`, opened 2026-09-29T21:13:47Z; REPORTED; the v2-versus-v1 comparison is in `v2/results/FINAL_RESULTS.md`, not here). After that opening, following the stage's adversarial verification, these corrections were made:

- The Λ implementation `math/lambda_stdlib.py` was committed, byte-identical, in `d9c5fb3`; it had not been in version control. `v2/tests/test_readiness.py` now pins its sha256 to the recorded one. See "Λ implementation provenance".
- The test-layer wording was corrected. The recorded run justified the artifact and receipt rules by a false premise, that the test suite never reads the frozen pair. See "Test layer for artifact and receipt mutants". A supplementary in-place test run, outside any verdict, is reported there.
- Render wording: this page now states which later edits touched `mutation.py`, and `v2/tests/test_readiness.py` checks that this page equals the render of the JSON.

None of this changes a recorded number, a status, a gate definition or the cut. `v2/results/mutation_readiness.json` is unchanged since `c46a42c`, and this page is re-rendered from it; the run itself was not repeated.

Labels: **MEASURED** (ran here; output in the cited file), **REPORTED** (read from the cited file), **MODELED** (reasoning, no measurement behind it).

## Summary

- Artifact mutants: 36. Integrity caught 36/36 (pass 1). With the receipt regenerated (integrity bypassed), the behavioral gates caught 26/36 (pass 2). (MEASURED)
- Receipt mutants: 5 (4 registered + 1 supplementary). The integrity layer caught 4/5. (MEASURED)
- Kernel-source mutants: 5 (4 registered + 1 supplementary). Integrity caught 4/5; the kernel-relevant tests caught 5/5. (MEASURED)
- **BLIND_SPOT: 11** (listed below). (MEASURED) The pass-2 status of an artifact mutant counts the behavioral gates only (section 10: "they must then be caught by a behavioral gate"); how the test suite reacts with a blind-spot mutant in place is reported separately below.
- Broken baselines: constant-ALERT, the fail-open kernel and majority all give `INVALID_BASELINE` with no per-class rate: True. (MEASURED)
- **Λ readiness aggregation: REJECTED** (numbers below). (MEASURED)

## Setup (MEASURED unless marked)

| item | value |
| --- | --- |
| git HEAD at run / guarded tree clean | `bb8d56f5d73cfbcce6ab8dfa9322531c5a3418ee` / True |
| PREREGISTRATION.md sha256 | `145497789c7ad78a80a94cfd7f2b031ca9a20df49c06e0b3412af9070fa1ef4b` |
| AMENDMENTS.md sha256 | `5152cd7f816c41e810861bbcd6489dee1d2598da89a36f6b418ef4a9867f181d` |
| kernel sha256 | `b0a64ff3f26ea284b0588de351ed7795a6089203b35fded0e7d82cbd4871aed9` |
| frozen v2 model.json / receipt sha256 | `b830a5edca271d667ab09b378dd5d3ab505d71a7e3451de8d9ca2bacfeed977c` / `442486a3b451f0ad765aac253830cdc5455bad3a34186b4cf73388cf207056f2` |
| v2 receipt source commit | `92872f88193242b3fc0c1f604321d889d6988358` |
| validation split (purpose GATES) | 4096 rows, 768 positive, rows sha256 `d0a37356784dcd60a4dbea68d760c55d56a7bd69b3a6f4dc833cfdbb38a951f3` |
| mutation.py / gates.py / lambda_stdlib.py sha256 | `b4dd3922f865ead1…` / `959d243a63fa0045…` / `2b6d353e4475f2ba…` |
| python | 3.12.10 |

The run used `v2/research/mutation.py` as committed at the git HEAD above (sha256 in the table). This page is rendered afterwards from the recorded JSON by `python -m v2.research.mutation render`. Later edits to `mutation.py` touch only the module docstring, the report rendering (the render function, its helpers and constants) and the test-layer wording that `run` writes; `git diff bb8d56f -- v2/research/mutation.py` lists them. None of them changes a recorded number or status, and `v2/tests/test_readiness.py` recomputes the aggregation from the JSON and checks that this page equals the render of the JSON.

**Λ implementation provenance.** `math/lambda_stdlib.py` (the Λ implementation that section 11 names) and `math/lambda_reference.md` were not in version control at the run, nor in any of this stage's commits (`bb8d56f`, `c46a42c`, `929677d`). The implementation was then pinned only by the sha256 recorded in the JSON, and a clean checkout of those commits could not run `v2/tests/test_readiness.py` (REPORTED from the stage's verification). Both files were committed afterwards, byte-identical, in `d9c5fb3` (after the opening). `v2/tests/test_readiness.py` now asserts that the file's sha256 equals the recorded `2b6d353e4475f2ba…`.

Rules, all fixed in `v2/research/mutation.py` before the first run:

- A behavioral gate **catches** a fixture iff its value is strictly below v2's own value of that gate (section 10: known-good reference = v2). The gates are the ones in `v2/research/gates.py`, whose only commit is the stage-1 commit `974159c` (REPORTED from git log; its sha256 is in the table).
- An artifact mutant is a **BLIND_SPOT** iff no behavioral gate catches it in pass 2 (section 10: pass-2 mutants "must then be caught by a behavioral gate"). The pass-2 status counts the behavioral gates only: integrity is bypassed there, and the v2 test suite was not run with the mutant in place of the frozen pair, because each mutant was evaluated on a temporary copy. (Corrected wording: see "Test layer for artifact and receipt mutants".)
- A receipt mutant is a **BLIND_SPOT** iff the kernel and `freeze.verify_origin` both accept it (integrity layer only; the test suite was not run with the mutant receipt in place either).
- A kernel mutant is a **BLIND_SPOT** iff neither integrity nor the kernel-relevant tests catch it.

## Known-good fixtures: gate vectors on validation (MEASURED)

| fixture | integrity | failclosed | truth_sign | monotone | validation | calibration | baseline_valid | BA_val | ECE_val |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.768680 | 0.034658 |
| k1 | 1.0000 | 1.0000 | 0.8750 | 0.8367 | 1.0000 | 1.0000 | 1.0000 | 0.766026 | 0.027431 |
| k2 | 1.0000 | 1.0000 | 0.8750 | 0.8367 | 1.0000 | 1.0000 | 1.0000 | 0.765625 | 0.027913 |
| k3 | 1.0000 | 1.0000 | 0.8750 | 0.8367 | 1.0000 | 1.0000 | 1.0000 | 0.766276 | 0.026614 |
| k4 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.764623 | 0.028354 |
| k5 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.768129 | 0.030538 |

## Artifact mutants (36): pass 1 integrity, pass 2 behavior (MEASURED)

Pass 1 pairs the mutant with the frozen v2 receipt. Pass 2 regenerates the receipt consistently (only `model_sha256` changes), so the kernel and `verify_origin` accept it, and then evaluates the seven gates on validation. "gates below v2" lists the gates that caught the mutant.

| mutant | change | pass 1 integrity | pass 2 gates below v2 | BA_val | ECE_val | status |
| --- | --- | --- | --- | ---: | ---: | --- |
| A-scale0.5-listener_running | -2.539293238803 → -1.269646619401 | CAUGHT | validation, calibration | 0.579577 | 0.172924 | **CAUGHT** |
| A-scale0.5-tls_enabled | -0.802793259351 → -0.401396629675 | CAUGHT | none | 0.754758 | 0.049898 | **BLIND_SPOT** |
| A-scale0.5-peer_allowlist_configured | -1.038997837438 → -0.519498918719 | CAUGHT | calibration | 0.748798 | 0.062905 | **CAUGHT** |
| A-scale0.5-queue_utilization | 2.705004607740 → 1.352502303870 | CAUGHT | none | 0.737230 | 0.042812 | **BLIND_SPOT** |
| A-scale0.5-consecutive_failures | 1.718149829576 → 0.859074914788 | CAUGHT | none | 0.762420 | 0.034320 | **BLIND_SPOT** |
| A-scale0.5-seconds_since_last_success | 0.036548130400 → 0.018274065200 | CAUGHT | none | 0.768029 | 0.034263 | **BLIND_SPOT** |
| A-scale0.5-ledger_integrity_ok | -2.851449125324 → -1.425724562662 | CAUGHT | validation, calibration | 0.529497 | 0.213815 | **CAUGHT** |
| A-scale0.5-configuration_valid | -2.821703311766 → -1.410851655883 | CAUGHT | validation, calibration | 0.532652 | 0.206634 | **CAUGHT** |
| A-scale1.5-listener_running | -2.539293238803 → -3.808939858204 | CAUGHT | calibration | 0.710136 | 0.095273 | **CAUGHT** |
| A-scale1.5-tls_enabled | -0.802793259351 → -1.204189889026 | CAUGHT | none | 0.745493 | 0.042713 | **BLIND_SPOT** |
| A-scale1.5-peer_allowlist_configured | -1.038997837438 → -1.558496756157 | CAUGHT | calibration | 0.751302 | 0.053497 | **CAUGHT** |
| A-scale1.5-queue_utilization | 2.705004607740 → 4.057506911610 | CAUGHT | none | 0.753756 | 0.048128 | **BLIND_SPOT** |
| A-scale1.5-consecutive_failures | 1.718149829576 → 2.577224744364 | CAUGHT | none | 0.766326 | 0.031506 | **BLIND_SPOT** |
| A-scale1.5-seconds_since_last_success | 0.036548130400 → 0.054822195600 | CAUGHT | none | 0.768680 | 0.034662 | **BLIND_SPOT** |
| A-scale1.5-ledger_integrity_ok | -2.851449125324 → -4.277173687986 | CAUGHT | validation, calibration | 0.678486 | 0.115337 | **CAUGHT** |
| A-scale1.5-configuration_valid | -2.821703311766 → -4.232554967649 | CAUGHT | validation, calibration | 0.682141 | 0.108906 | **CAUGHT** |
| A-signflip-listener_running | -2.539293238803 → 2.539293238803 | CAUGHT | truth_sign, monotone, validation, calibration, baseline_valid | 0.500000 | 0.750374 | **CAUGHT** |
| A-signflip-tls_enabled | -0.802793259351 → 0.802793259351 | CAUGHT | truth_sign, monotone, validation, calibration | 0.507762 | 0.220818 | **CAUGHT** |
| A-signflip-peer_allowlist_configured | -1.038997837438 → 1.038997837438 | CAUGHT | truth_sign, monotone, validation, calibration | 0.504207 | 0.327505 | **CAUGHT** |
| A-signflip-queue_utilization | 2.705004607740 → -2.705004607740 | CAUGHT | truth_sign, monotone, validation, calibration | 0.680489 | 0.108048 | **CAUGHT** |
| A-signflip-consecutive_failures | 1.718149829576 → -1.718149829576 | CAUGHT | truth_sign, monotone | 0.746945 | 0.028596 | **CAUGHT** |
| A-signflip-seconds_since_last_success | 0.036548130400 → -0.036548130400 | CAUGHT | truth_sign, monotone | 0.768179 | 0.034349 | **CAUGHT** |
| A-signflip-ledger_integrity_ok | -2.851449125324 → 2.851449125324 | CAUGHT | truth_sign, monotone, validation, calibration, baseline_valid | 0.500000 | 0.784562 | **CAUGHT** |
| A-signflip-configuration_valid | -2.821703311766 → 2.821703311766 | CAUGHT | truth_sign, monotone, validation, calibration, baseline_valid | 0.500000 | 0.775990 | **CAUGHT** |
| A-drop-listener_running | -2.539293238803 → 0.000000000000 | CAUGHT | truth_sign, validation, calibration, baseline_valid | 0.500000 | 0.428454 | **CAUGHT** |
| A-drop-tls_enabled | -0.802793259351 → 0.000000000000 | CAUGHT | truth_sign, calibration | 0.716897 | 0.090330 | **CAUGHT** |
| A-drop-peer_allowlist_configured | -1.038997837438 → 0.000000000000 | CAUGHT | truth_sign, validation, calibration | 0.658403 | 0.130685 | **CAUGHT** |
| A-drop-queue_utilization | 2.705004607740 → 0.000000000000 | CAUGHT | truth_sign, calibration | 0.711689 | 0.070499 | **CAUGHT** |
| A-drop-consecutive_failures | 1.718149829576 → 0.000000000000 | CAUGHT | truth_sign | 0.757011 | 0.032929 | **CAUGHT** |
| A-drop-seconds_since_last_success | 0.036548130400 → 0.000000000000 | CAUGHT | truth_sign | 0.768179 | 0.034357 | **CAUGHT** |
| A-drop-ledger_integrity_ok | -2.851449125324 → 0.000000000000 | CAUGHT | truth_sign, validation, calibration, baseline_valid | 0.500000 | 0.506327 | **CAUGHT** |
| A-drop-configuration_valid | -2.821703311766 → 0.000000000000 | CAUGHT | truth_sign, validation, calibration, baseline_valid | 0.500000 | 0.494688 | **CAUGHT** |
| A-intercept+1 | 7.132311089958 → 8.132311089958 | CAUGHT | validation, calibration | 0.664864 | 0.138627 | **CAUGHT** |
| A-intercept-1 | 7.132311089958 → 6.132311089958 | CAUGHT | calibration | 0.717448 | 0.095899 | **CAUGHT** |
| A-threshold+0.1 | 0.200000000000 → 0.300000000000 | CAUGHT | none | 0.738431 | 0.034658 | **BLIND_SPOT** |
| A-threshold-0.1 | 0.200000000000 → 0.100000000000 | CAUGHT | none | 0.708534 | 0.034658 | **BLIND_SPOT** |

Pass 1 refusal message (every artifact mutant): `ModelArtifactError: model artifact hash does not match receipt`.

## Receipt mutants (MEASURED)

| mutant | change | kernel | verify_origin | status |
| --- | --- | --- | --- | --- |
| R-model-hash-flip | model_sha256 first hex digit 'b' -> 'a' (low bit flipped) | refuses: `ModelArtifactError: model artifact hash does not match receipt` | accepts | **CAUGHT** |
| R-wrong-schema | schema -> 'szl-oac/ops-health-artifact-receipt/v1' | refuses: `ModelArtifactError: artifact receipt schema mismatch` | refuses: `ModelArtifactError: artifact receipt schema mismatch` | **CAUGHT** |
| R-source-commit-altered | source.commit last hex digit '8' -> '9' (low bit flipped) | accepts | refuses: `OriginMismatch: source commit 92872f88193242b3fc0c1f604321d889d6988359 is not in the repository` | **CAUGHT** |
| R-source-commit-other | SUPPLEMENTARY: source.commit -> 064481f, the parent of the recorded 92872f8 (a real commit of this repository) | accepts | accepts | **BLIND_SPOT** |
| R-metrics-added | AM-3: a metrics block added to the receipt | refuses: `ModelArtifactError: artifact receipt fields mismatch: missing=[], unexpected=['metrics']` | refuses: `ModelArtifactError: artifact receipt fields mismatch: missing=[], unexpected=['metrics']` | **CAUGHT** |

## Kernel-source mutants (MEASURED)

Control: an unmodified copy (sha256 identical to the shipped kernel: True) loads the frozen v2 pair (True) and passes the kernel-relevant tests (44 run, ok True). Test layer valid: True. Tests: `v2.tests.test_failclosed_probes`, `v2.tests.test_receipt.ReceiptTest`, `v2.tests.test_review_hardening.ArtifactHardeningTest`, `v2.tests.test_review_hardening.InputHardeningTest`, `v2.tests.test_review_hardening.EchoBoundTest`, `v2.tests.test_gates.GateTest.test_integrity_gate`.

| mutant | integrity (frozen v2 pair) | kernel-relevant tests | descriptive: receipt re-bound to the copy's sha256 | status |
| --- | --- | --- | --- | --- |
| K-no-finiteness | CAUGHT: `ModelArtifactError: kernel sha256 does not match receipt` | FAILED: 3 of 44 tests failed (5 failures, 0 errors, subtests counted) | loads; probes refused 65/66; escaped: nan_continuous | **CAUGHT** |
| K-no-range | CAUGHT: `ModelArtifactError: kernel sha256 does not match receipt` | FAILED: 3 of 44 tests failed (7 failures, 0 errors, subtests counted) | loads; probes refused 63/66; escaped: out_of_range_high, out_of_range_low, out_of_range_failures | **CAUGHT** |
| K-no-prohibited-screen | CAUGHT: `ModelArtifactError: kernel sha256 does not match receipt` | FAILED: 3 of 44 tests failed (50 failures, 0 errors, subtests counted) | loads; probes refused 66/66 | **CAUGHT** |
| K-inverted-model-hash | CAUGHT: `ModelArtifactError: kernel sha256 does not match receipt` | FAILED: 16 of 44 tests failed (23 failures, 2 errors, subtests counted) | refuses: `ModelArtifactError: model artifact hash does not match receipt` | **CAUGHT** |
| K-inverted-self-hash (supplementary) | missed (loads) | FAILED: 16 of 44 tests failed (23 failures, 2 errors, subtests counted) | refuses: `ModelArtifactError: kernel sha256 does not match receipt` | **CAUGHT** |

Failing tests per mutant, with the failing subtests of each (full output in `logs/v2/mutation_kernel_tests/<mutant>.txt`):

- **K-no-finiteness**: 3 failing test(s):
  - `test_failclosed_probes.FailClosedProbeTest.test_cli_refuses_with_exit_code_2`
  - `test_failclosed_probes.FailClosedProbeTest.test_every_probe_is_refused_with_expected_message` (subtests: probe='inf_continuous', probe='nan_continuous', probe='neg_inf_continuous')
  - `test_failclosed_probes.FailClosedProbeTest.test_gate_counts_every_probe`
- **K-no-range**: 3 failing test(s):
  - `test_failclosed_probes.FailClosedProbeTest.test_every_probe_is_refused_with_expected_message` (subtests: probe='out_of_range_failures', probe='out_of_range_high', probe='out_of_range_low')
  - `test_failclosed_probes.FailClosedProbeTest.test_gate_counts_every_probe`
  - `test_review_hardening.InputHardeningTest.test_numeric_edge_cases` (subtests: case='-1e308', case='1e308', case='negative_subnormal')
- **K-no-prohibited-screen**: 3 failing test(s):
  - `test_failclosed_probes.FailClosedProbeTest.test_every_probe_is_refused_with_expected_message` (43 failing subtests; listed in the log)
  - `test_failclosed_probes.FailClosedProbeTest.test_prohibited_screen_runs_before_schema_check` (6 failing subtests; listed in the log)
  - `test_review_hardening.InputHardeningTest.test_unicode_folded_prohibited_keys`
- **K-inverted-model-hash**: 16 failing test(s):
  - `test_failclosed_probes.FailClosedProbeTest.test_cli_refuses_with_exit_code_2`
  - `test_gates.GateTest.test_integrity_gate`
  - `test_receipt.ReceiptTest.test_cli_refuses_tampered_model`
  - `test_receipt.ReceiptTest.test_consistent_receipt_for_invalid_model_is_still_refused`
  - `test_receipt.ReceiptTest.test_generator_digest_must_match_model`
  - `test_receipt.ReceiptTest.test_kernel_byte_flip_is_refused`
  - `test_receipt.ReceiptTest.test_model_byte_flip_is_refused` (subtests: offset=0, offset=1823, offset=3645)
  - `test_receipt.ReceiptTest.test_valid_pair_loads_and_binds_everything`
  - `test_review_hardening.ArtifactHardeningTest.test_unhashable_stop_reason_is_refused_cleanly` (subtests: stop_reason=7, stop_reason=None, stop_reason=['CONVERGED'], stop_reason={'x': 1})
  - `test_review_hardening.EchoBoundTest.test_calibrator_kind_value_is_not_echoed` (subtests: kind='big_list', kind='long_text', kind='short_text')
  - `test_review_hardening.EchoBoundTest.test_deep_path_of_long_keys_is_bounded` (subtests: case='depth', case='text')
  - `test_review_hardening.EchoBoundTest.test_many_extra_keys_are_counted_not_listed`
  - `test_review_hardening.EchoBoundTest.test_model_extra_long_key_with_consistent_receipt`
  - `test_review_hardening.EchoBoundTest.test_nested_artifact_extra_long_keys` (subtests: label='calibrator.params', label='weights')
  - `test_review_hardening.InputHardeningTest.test_bom_deep_nesting_and_size_are_refused`
  - `test_review_hardening.InputHardeningTest.test_negative_zero_is_canonical`
- **K-inverted-self-hash**: 16 failing test(s):
  - `test_failclosed_probes.FailClosedProbeTest.test_cli_refuses_with_exit_code_2`
  - `test_gates.GateTest.test_integrity_gate`
  - `test_receipt.ReceiptTest.test_cli_refuses_tampered_model`
  - `test_receipt.ReceiptTest.test_consistent_receipt_for_invalid_model_is_still_refused`
  - `test_receipt.ReceiptTest.test_generator_digest_must_match_model`
  - `test_receipt.ReceiptTest.test_kernel_byte_flip_is_refused`
  - `test_receipt.ReceiptTest.test_model_byte_flip_is_refused` (subtests: offset=0, offset=1823, offset=3645)
  - `test_receipt.ReceiptTest.test_valid_pair_loads_and_binds_everything`
  - `test_review_hardening.ArtifactHardeningTest.test_unhashable_stop_reason_is_refused_cleanly` (subtests: stop_reason=7, stop_reason=None, stop_reason=['CONVERGED'], stop_reason={'x': 1})
  - `test_review_hardening.EchoBoundTest.test_calibrator_kind_value_is_not_echoed` (subtests: kind='big_list', kind='long_text', kind='short_text')
  - `test_review_hardening.EchoBoundTest.test_deep_path_of_long_keys_is_bounded` (subtests: case='depth', case='text')
  - `test_review_hardening.EchoBoundTest.test_many_extra_keys_are_counted_not_listed`
  - `test_review_hardening.EchoBoundTest.test_model_extra_long_key_with_consistent_receipt`
  - `test_review_hardening.EchoBoundTest.test_nested_artifact_extra_long_keys` (subtests: label='calibrator.params', label='weights')
  - `test_review_hardening.InputHardeningTest.test_bom_deep_nesting_and_size_are_refused`
  - `test_review_hardening.InputHardeningTest.test_negative_zero_is_canonical`

## BLIND_SPOT list (MEASURED)

- `A-scale0.5-tls_enabled` (artifact): no catch by behavioral gates, integrity bypassed (pass 2).
- `A-scale0.5-queue_utilization` (artifact): no catch by behavioral gates, integrity bypassed (pass 2).
- `A-scale0.5-consecutive_failures` (artifact): no catch by behavioral gates, integrity bypassed (pass 2).
- `A-scale0.5-seconds_since_last_success` (artifact): no catch by behavioral gates, integrity bypassed (pass 2).
- `A-scale1.5-tls_enabled` (artifact): no catch by behavioral gates, integrity bypassed (pass 2).
- `A-scale1.5-queue_utilization` (artifact): no catch by behavioral gates, integrity bypassed (pass 2).
- `A-scale1.5-consecutive_failures` (artifact): no catch by behavioral gates, integrity bypassed (pass 2).
- `A-scale1.5-seconds_since_last_success` (artifact): no catch by behavioral gates, integrity bypassed (pass 2).
- `A-threshold+0.1` (artifact): no catch by behavioral gates, integrity bypassed (pass 2).
- `A-threshold-0.1` (artifact): no catch by behavioral gates, integrity bypassed (pass 2).
- `R-source-commit-other` (supplementary) (receipt): no catch by integrity (kernel + verify_origin).

## Test layer for artifact and receipt mutants (post-run correction)

The recorded field `caught_by_tests` of all 36 artifact and 5 receipt mutants reads "NOT_APPLICABLE: the v2 test suite runs on DEV fixtures and never reads the frozen v2 artifact or receipt; this mutant is applied to a temporary copy" (REPORTED from the JSON). The reason it gives is false. `v2/tests/test_mutation_suite.py`, committed with the suite in `bb8d56f` before the run, reads `v2/ops-health/model.json` and `artifact_receipt.json` (REPORTED from the source). The actual reason the test layer did not count is that the v2 test suite was never run with an artifact or receipt mutant in place of the frozen pair, because each mutant was evaluated on a temporary copy. `mutation.py` now writes "NOT_RUN: the status counts the behavioral gates (artifact pass 2) or the integrity layer (receipt) only; this mutant was evaluated on a temporary copy and the v2 test suite was not run with it in place of the frozen pair". The recorded JSON is left byte-identical as the record of the run.

PREREGISTRATION section 10 says "A mutant is CAUGHT if the integrity layer or the test suite fails". For the integrity-bypassed pass, it says artifact mutants "must then be caught by a behavioral gate". The recorded statuses follow the second sentence for pass 2, as `mutation.py` declared before the run.

**Supplementary in-place run** (MEASURED; post-run and after the opening; not part of any verdict). Script `logs/v2/mutation_inplace_tests.py`, results `logs/v2/mutation_inplace_tests.json`, per-variant output in `logs/v2/mutation_inplace_tests/`. Each recorded BLIND_SPOT mutant was put in place of the frozen pair in a copy of the tree at the stage's last commit `929677d`. The copy held no sealed file and no split file (checked on the archive member list: True). Artifact mutants were placed with their consistently regenerated pass-2 receipt. `v2.tests.test_mutation_suite` then ran against the copy. By a source scan (hits recorded in the JSON), it is the only test module at that commit that reads the frozen pair: `v2/tests/test_mutation_suite.py`. That no other module reads the pair indirectly is MODELED. Control, with the unmodified pair: 14 tests, OK.

| mutant | kind | tests run | outcome | extra failures vs control |
| --- | --- | ---: | --- | --- |
| A-scale0.5-consecutive_failures | artifact, pass 2 (consistent receipt) | 14 | OK | none |
| A-scale0.5-queue_utilization | artifact, pass 2 (consistent receipt) | 14 | OK | none |
| A-scale0.5-seconds_since_last_success | artifact, pass 2 (consistent receipt) | 14 | OK | none |
| A-scale0.5-tls_enabled | artifact, pass 2 (consistent receipt) | 14 | OK | none |
| A-scale1.5-consecutive_failures | artifact, pass 2 (consistent receipt) | 14 | OK | none |
| A-scale1.5-queue_utilization | artifact, pass 2 (consistent receipt) | 14 | OK | none |
| A-scale1.5-seconds_since_last_success | artifact, pass 2 (consistent receipt) | 14 | OK | none |
| A-scale1.5-tls_enabled | artifact, pass 2 (consistent receipt) | 14 | OK | none |
| A-threshold+0.1 | artifact, pass 2 (consistent receipt) | 14 | OK | none |
| A-threshold-0.1 | artifact, pass 2 (consistent receipt) | 14 | FAILED (errors=2) | `test_mutation_suite.CatalogueTest.test_artifact_catalogue_on_frozen_v2`, `test_mutation_suite.IntegrityLayerTest.test_artifact_mutant_pass1_refused_pass2_loads` |
| R-source-commit-other | receipt | 14 | OK | none |

Under the registered pass-2 reading, the BLIND_SPOT count is 11. Under a literal "any layer" reading of section 10's first sentence, `A-threshold-0.1` would count as CAUGHT and the count would be 10. (MEASURED)

The `A-threshold-0.1` failures are incidental (MODELED, from the traceback in `logs/v2/mutation_inplace_tests/A-threshold-0.1.txt`). The test builds the registered mutant catalogue from whatever pair is in place, and at threshold 0.10 the -0.1 step leaves the 0.01..0.99 grid (`MutationRefused`). No test checks the decision behavior that changed.

The stage's verification independently ran the full suite at `929677d` with three of these mutants in place: `R-source-commit-other`, `A-scale0.5-tls_enabled` and `A-threshold-0.1` (pass 2). The first two gave no extra failure. `A-threshold-0.1` gave the same two `test_mutation_suite` errors, plus one error that the host's paging file caused (REPORTED; `logs/verify/v2-mutation/r_suite929.txt`, not committed). Other lanes added tests after this stage that also read the frozen pair (`v2/tests/test_final_score.py`, `f9c5588`). They are not part of this stage's test layer and were not run here.

## Broken baselines (section 10; MEASURED on validation)

| fixture | status | failed preconditions | BA | per-class rates |
| --- | --- | --- | ---: | --- |
| constant_alert | INVALID_BASELINE | iii_decisions_not_constant | 0.500000 | INVALID_BASELINE |
| fail_open_kernel | INVALID_BASELINE | i_receipt_verifies, iv_refused_all_probes | 0.768680 | INVALID_BASELINE |
| fail_open_kernel, precondition (i) granted | INVALID_BASELINE | iv_refused_all_probes | 0.768680 | INVALID_BASELINE |
| majority | INVALID_BASELINE | i_receipt_verifies, iii_decisions_not_constant, iv_refused_all_probes | 0.500000 | INVALID_BASELINE |

## Readiness aggregation (section 11; MEASURED)

Known-good: v2 and the five AM-4 retrains (6). Known-broken: every integrity-bypassed artifact mutant (36). Cut: aggregate ≥ 0.99 (AND: every gate = 1). Λ = `lambda_stdlib.lambda_aggregate` with the reference defaults: uniform weights (`weights=None`) and `semantics="kernel"`. Λ has no other parameter; the `lambda_gate` default threshold of 0.5 is not used.

| aggregator | misclassified at cut | broken READY | good NOT_READY | min(good) | max(broken) | margin | AUROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AND | 13 | 10 | 3 | 0.000000 | 1.000000 | -1.000000 | 0.611111 |
| min | 13 | 10 | 3 | 0.836735 | 1.000000 | -0.163265 | 0.722222 |
| mean | 14 | 11 | 3 | 0.958819 | 1.000000 | -0.041181 | 0.722222 |
| Λ | 14 | 11 | 3 | 0.956437 | 1.000000 | -0.043563 | 0.722222 |

**Λ verdict: REJECTED.** Rule: KEEP only if Lambda's margin is strictly greater than every baseline's AND its misclassifications at the cut are no more than every baseline's (section 11). Strictly better margin than: AND True, mean False, min True. No more misclassifications than: AND False, mean True, min False.

Aggregates of the known-good fixtures, and of every known-broken fixture that at least one aggregator marks READY (MEASURED):

| fixture | group | AND | min | mean | Λ | READY under |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| v2 | good | 1.0000 | 1.0000 | 1.0000 | 1.0000 | AND, min, mean, Λ |
| k1 | good | 0.0000 | 0.8367 | 0.9588 | 0.9564 | none |
| k2 | good | 0.0000 | 0.8367 | 0.9588 | 0.9564 | none |
| k3 | good | 0.0000 | 0.8367 | 0.9588 | 0.9564 | none |
| k4 | good | 1.0000 | 1.0000 | 1.0000 | 1.0000 | AND, min, mean, Λ |
| k5 | good | 1.0000 | 1.0000 | 1.0000 | 1.0000 | AND, min, mean, Λ |
| A-scale0.5-tls_enabled | broken | 1.0000 | 1.0000 | 1.0000 | 1.0000 | AND, min, mean, Λ |
| A-scale0.5-queue_utilization | broken | 1.0000 | 1.0000 | 1.0000 | 1.0000 | AND, min, mean, Λ |
| A-scale0.5-consecutive_failures | broken | 1.0000 | 1.0000 | 1.0000 | 1.0000 | AND, min, mean, Λ |
| A-scale0.5-seconds_since_last_success | broken | 1.0000 | 1.0000 | 1.0000 | 1.0000 | AND, min, mean, Λ |
| A-scale1.5-tls_enabled | broken | 1.0000 | 1.0000 | 1.0000 | 1.0000 | AND, min, mean, Λ |
| A-scale1.5-peer_allowlist_configured | broken | 0.0000 | 0.9650 | 0.9950 | 0.9949 | mean, Λ |
| A-scale1.5-queue_utilization | broken | 1.0000 | 1.0000 | 1.0000 | 1.0000 | AND, min, mean, Λ |
| A-scale1.5-consecutive_failures | broken | 1.0000 | 1.0000 | 1.0000 | 1.0000 | AND, min, mean, Λ |
| A-scale1.5-seconds_since_last_success | broken | 1.0000 | 1.0000 | 1.0000 | 1.0000 | AND, min, mean, Λ |
| A-threshold+0.1 | broken | 1.0000 | 1.0000 | 1.0000 | 1.0000 | AND, min, mean, Λ |
| A-threshold-0.1 | broken | 1.0000 | 1.0000 | 1.0000 | 1.0000 | AND, min, mean, Λ |

## Reading the results

- **Known-good fixtures fail registered gates.** 3 of 6 known-good fixtures have a gate below 1: k1 (sign disagreement: seconds_since_last_success -0.003455; monotonicity checks failed 3200/19600, by kind {'raise:seconds_since_last_success': 3200}); k2 (sign disagreement: seconds_since_last_success -0.009359; monotonicity checks failed 3200/19600, by kind {'raise:seconds_since_last_success': 3200}); k3 (sign disagreement: seconds_since_last_success -0.013441; monotonicity checks failed 3200/19600, by kind {'raise:seconds_since_last_success': 3200}). (MEASURED; weights REPORTED from each model.json.) v2's own seconds_since_last_success weight is +0.036548 (REPORTED). PREREGISTRATION §2 recorded in advance that this coordinate is weakly identifiable, because nominal seconds_since_last_success is near 0 on its normalized scale (REPORTED; labeled MODELED there). A retrain of the selected recipe can therefore land on either side of zero (MODELED), and g_truth_sign and g_monotone then fail. Every known-good fixture still has g_validation = 1 and g_calibration = 1 (MEASURED). Every aggregator marks the failing fixtures NOT_READY at the 0.99 cut (MEASURED).
- **Behavioral blind spots.** 10 integrity-bypassed artifact mutants pass every gate: `A-scale0.5-tls_enabled`, `A-scale0.5-queue_utilization`, `A-scale0.5-consecutive_failures`, `A-scale0.5-seconds_since_last_success`, `A-scale1.5-tls_enabled`, `A-scale1.5-queue_utilization`, `A-scale1.5-consecutive_failures`, `A-scale1.5-seconds_since_last_success`, `A-threshold+0.1`, `A-threshold-0.1`. They keep every weight sign and monotonicity, so g_truth_sign and g_monotone stay at 1. g_validation and g_calibration are absolute tolerances (BA ≥ 0.70 and ECE ≤ 0.05), not distances from v2. The largest validation BA loss among the blind spots is 0.060146 (`A-threshold-0.1`, BA 0.708534 against v2's 0.768680). The largest ECE is 0.049898 (`A-scale0.5-tls_enabled`). (MEASURED.) The threshold mutants change no score, so only g_validation could see them, and their BA stays above 0.70 (MEASURED). Every one of these mutants was caught by integrity in pass 1 (36/36, MEASURED). With the gates and cut as registered, the integrity layer, not the behavioral gates, is what detects a modified artifact (MODELED).
- **Aggregators.** No aggregator separates known-good from known-broken: every margin is negative (max(broken) = 1.0 under every aggregator, from the blind spots, while some known-good fixture is below 1). Misclassifications at the cut range from 13 to 14 of 42 fixtures. Mean and Λ also mark `A-scale1.5-peer_allowlist_configured` READY, where min does not: its only failing gate is slightly below 1. Λ's margin is strictly better than AND's and min's but not mean's, and Λ misclassifies more fixtures than AND and min. Under the registered rule Λ is therefore **REJECTED**. (MEASURED numbers; the rule is fixed in section 11.) On every fixture, Λ lies between min and mean (MEASURED), as `math/lambda_reference.md` §7 describes for uniform weights.
- **Receipt-origin blind spot (`R-source-commit-other`, supplementary).** The kernel only checks that source.commit is 40 lower-case hex digits; it cannot run git. `freeze.verify_origin` checks that the committed kernel, generator, manifest, preregistration and amendments blobs at the claimed commit hash to the receipt's digests. At the parent of the recorded freeze commit, those five blobs are identical, so re-pointing source.commit there passes both layers (MEASURED). The model bytes and every bound digest remain protected. What goes undetected is which of several commits with identical bound blobs the receipt names (MODELED). This is a limit of the research-side origin check, not a defect of the frozen kernel. Possible remedies are to also bind the committed search trace at the claimed commit, or to sign receipts. Neither is applied here: both would change the integrity layer after its result is known.
- **Kernel mutants.** With prohibited-key screening removed, the kernel still refuses 66/66 probes, because the exact eight-field schema check rejects the extra names. g_failclosed alone would not notice the mutation. The tests catch it: they check the refusal reason and that screening runs before the schema check (MEASURED). The inverted self-hash mutant passes the integrity layer (MEASURED): a kernel that verifies its own hash cannot detect a change to that verification (MODELED). The tests catch it too. For every kernel mutant, the kernel-relevant tests failed, and the unmodified control copy passed all of them (MEASURED).
- **Defects.** No mutant revealed a defect in the frozen v2 kernel. The shipped kernel passes the control, and every kernel-source mutant is caught (MEASURED). The blind spots above are properties of the registered behavioral gates, the registered cut and the research-side origin check. Per the protocol, none of them was tuned after these results.

No Λ property is called proven here (CONTRACT §5.5; `math/lambda_reference.md` §5).

