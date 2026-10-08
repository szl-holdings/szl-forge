# OAC Ops Health v2: stage 1 report

**Status:** stage 1 (build, test, adversarial review) is complete. Stage 2 has not started: no
registered search has run, the registered validation split has never been opened, the sealed
splits are unopened, and no v2 model has been selected or evaluated. Nothing here is a result
about v2 versus v1. Everything is SYNTHETIC, and nothing was pushed, published or posted.

Every claim carries one label:

- **MEASURED:** the command ran and its raw output is in the cited log.
- **REPORTED:** the value comes from the cited file at the cited revision.
- **MODELED:** reasoning that no measurement here backs.

## Commits (local repository, unpublished)

| commit | content |
| --- | --- |
| `bca9225` | preregistration, registered before any v2 code or data (sha256 `145497789c7a…1ef4b`) |
| `2fe8181` | line-ending conversion disabled, so hash-bound artifacts stay byte-exact |
| `852381b` | M1: byte-exact legacy v1 splits, with a verified source record |
| `00e9d19` | M2: `ops-health-generator/2.0.0`, with byte-exact v1 parity |
| `7c7226a` | M3: v2 splits generated and sealed before any training |
| `974159c` | M4–M6: v2 library, fail-closed kernel, 94-test suite |
| `b75785c` | the stage-1 review fixes (kernel, metrics, origin verifier) and 17 regression tests |
| `ec7874b` | review evidence, bootstrap benchmark log, `README_DEV.md`, first version of this report |
| `efde7ff` | fixer round: bounded refusal echoes in the kernel (completes A5), 8 regression tests, extended attack harness |
| the commit that updates this report | fixer-round evidence and documentation (see `git log -1 -- v2/STAGE1_REPORT.md`) |

## Tests

- **MEASURED, 974159c:** 94 tests OK (py 3.12.10), re-run by the orchestrator
  (`logs/v2/stage1_tests_orchestrator.txt`).
- **MEASURED, b75785c:** 111 tests OK on a clean tree (py 3.12.10, and python 3.11.9 in the
  CONTRACT §6 form). Those logs were replaced by the `efde7ff` runs below. The `b75785c`
  versions remain in git at `ec7874b`.
- **MEASURED, efde7ff:** **119 tests OK** on a clean tree (py 3.12.10,
  `unittest discover -s v2/tests -t .`) in `logs/v2/stage1_tests_after_review.txt`. The same
  119 pass in the CONTRACT §6 form (python 3.11.9, `unittest discover -s v2/tests`) in
  `logs/v2/stage1_tests_after_review_py311.txt`.

The suite covers what CONTRACT §6 requires:

- fail-closed probes: 66 in `gates.FAILCLOSED_PROBES` plus the review's hostile inputs;
- receipt and self-hash mismatch;
- the broken baselines: constant-ALERT, and a fail-open kernel mutant with a regenerated receipt,
  both giving INVALID_BASELINE;
- the parts of the mutation suite that stage 1 can exercise: model and kernel byte flips, a
  sign-flipped artifact, a decreasing calibrator, and the fail-open source mutant. The full §10
  suite is stage-2 work;
- metric recomputation against v1's receipt counts;
- determinism: training and freezing twice gives a byte-identical `model.json`.

## Parity and reproduction

- **MEASURED (this session, within the 119):** generator parity holds.
  `test_v1_parity_byte_for_byte` reproduces the v1 public train, validation and test splits byte
  for byte from seed 2500. The original M2 log is `logs/v2/m2_generator_parity.txt` (REPORTED:
  2026-09-26 run).
- **REPORTED (`logs/v2/m1_legacy_source_write.txt`, 2026-09-26):** the legacy v1 split copies are
  byte-identical to the committed upstream blobs and match v1's dataset receipt.
- **REPORTED (`logs/v2/m3_make_data_verify.txt`, 2026-09-26):** a hash-only verify of all 9
  registered splits shows disk equal to regenerated. MANIFEST sha256 is `c0d2fa0a…7252`
  (MEASURED unchanged today).
- **MEASURED:** the v1 GD reproduction matches at 12 decimals. `v2.research.v1_gd --reproduce`
  gives `all_equal_at_12dp: true` on py 3.12.10 and python 3.11.9. The max unrounded difference
  from the published v1 parameters is 4.61e-13 and 4.60e-13
  (`logs/v2/review/v1_gd_reproduce_py312.txt`, `…_py311.txt`). The published values are
  REPORTED from Hub revision `dd7d1098…`, `model.json` sha256 `f111b7fc…9557`.
- **MEASURED:** the research arithmetic equals independent numpy arithmetic. Normalization is
  exact, and the raw-score difference is ≤ 3.3e-16 (`logs/v2/review/stats_reference_after.txt`).

## Adversarial review (full detail in `logs/v2/review/STAGE1_REVIEW.md`)

There were three lenses: kernel security, statistics and sealing. The review found eight defects,
all MEASURED. Seven are fixed, each with regression tests: six in `b75785c`, and A5 partly in
`b75785c` and completed in `efde7ff` after verification found the first fix incomplete. One
(A7) is filed as an amendment request.

| id | severity | defect | status |
| --- | --- | --- | --- |
| A1 | HIGH | An unhashable `training.stop_reason` crashed the CLI with a stack trace (exit 1). | fixed: clean refusal, plus a CLI catch-all refusal (exit 2) |
| A2 | MEDIUM | alpha ≠ 0.10 and off-grid l2 were accepted. | fixed: both pinned |
| A3 | MEDIUM | The preregistration digest, manifest digest and source commit in the receipt were unchecked. | fixed: the prereg digest is pinned in the kernel; `freeze.verify_origin` checks the commit, kernel, generator, manifest and prereg blobs |
| A4 | LOW | Look-alike prohibited keys were refused only by the schema check. | fixed: a Unicode-folded screen (NFKC, Cf characters removed, casefolded) |
| A5 | LOW | Keys were echoed in full in refusal messages. **Correction:** the `b75785c` fix covered only input keys. Model and receipt keys, and the calibrator kind value, were still echoed in full: a 100,000-character receipt key gave 100,089 bytes of stderr (MEASURED, `logs/v2/review/kernel_attack_echo_before.txt`). Exit 2 held and no traceback appeared. | fixed in `efde7ff`: every echoed key is capped at 64 characters; at most 8 unexpected keys are listed; the calibrator kind is not shown; input paths are capped at 256 characters; the CLI refusal is one ASCII JSON line of ≤ 1,024 bytes. MEASURED: the maximum refusal stderr over 98 probes is 355 bytes |
| A6 | LOW | `-0.0` produced different output bytes, and negative-zero contributions appeared. | fixed: canonical +0.0 |
| A7 | MEDIUM | The §6-banned word appears in the upstream v1 path recorded in `PREREGISTRATION.receipt.json`. | open: AR-1 in `v2/AMENDMENT_REQUESTS.md`, because a registration record is not edited |
| B1 | MEDIUM | An INVALID_BASELINE report still exposed accuracy, from which sensitivity and specificity can be solved exactly. | fixed: accuracy and confusion counts are withheld for invalid fixtures |

Results after the fixes (all MEASURED):

- **Kernel attacks (`logs/v2/review/kernel_attack_after.txt`, kernel `efde7ff`):** 98/98 probes
  behave as expected. There are 69 raw CLI inputs: the §2.6 classes, duplicate keys, 1e308, 1e400,
  -0.0, numbers-as-strings, look-alike and whitespace keys, deep nesting, oversize input, BOM,
  encodings, and 4 echo-bound probes. There are 29 artifact and receipt attacks, 8 of them
  echo-bound probes. No stack traces appear, and every refusal's stderr is ≤ 1,100 bytes (the
  largest is 355). Before the `b75785c` fixes, 79 of the first 86 probes passed. Against the
  `b75785c` kernel, 87/98 passed, with 11 echo findings (`kernel_attack_echo_before.txt`).
- **Statistics (`logs/v2/review/stats_reference_after.txt`):** 34/34 independent numpy/scipy checks
  pass. These cover:
  - IRLS vs scipy (≤ 6.7e-8) and Platt vs scipy (6.8e-9);
  - isotonic vs scipy PAV with ties (≤ 4.9e-13);
  - ECE edges;
  - AUROC vs Mann-Whitney with ties;
  - bootstrap draws, pairing and the type-7 percentile over 2,000 resamples;
  - the Mondrian rank and coverage;
  - INVALID_BASELINE;
  - threshold selection.
- **Sealing (`logs/v2/review/sealing_audit_after.txt`):** an audit hook over the in-process suite
  recorded 0 opens of sealed files and 0 opens of the registered validation split (119 tests,
  re-run at `efde7ff`). The preregistration sha256 is unchanged. `v2/data` is unchanged since
  `7c7226a`.

## Bootstrap benchmark (DEV timing only; the intervals are not results)

The run is MEASURED in `logs/v2/bootstrap_bench.txt` (py 3.12.10, HEAD `b75785c`, clean tree,
exit 0). Its setup:

- 50,000 DEV rows;
- 12 statistics per resample: BA for 4 contenders, plus Brier, log loss, ECE and AUROC for 2
  scored contenders;
- 2,000 resamples, seed 20260927.

| layout | seconds | ms per resample | undefined values |
| --- | ---: | ---: | ---: |
| unstratified | 82.47 | 41.24 | 0 |
| stratified 4 × 12,500 (the pooled-shift layout) | 93.76 | 46.88 | 0 |

- Total wall time was 179 s, of which preparation took 1.23 s.
- The weighted statistics at unit weights equal `metrics.py` exactly for all 5 spot checks.
- The run completed and replaces the log that a kill had truncated. That log is kept at
  `logs/v2/review/bootstrap_bench_truncated_previous.txt`.
- Other sessions' Python processes were running on the machine at the same time (MEASURED,
  `ps -W`), so these timings were taken under shared load.

MODELED: at this rate, stage 2's evaluation fits in minutes. It needs two suites of 50,000 rows
(nominal test and pooled shift), each run once with 2,000 resamples.

## Stage 2 readiness

- **MODELED:** the pieces stage 2 needs exist and are tested on DEV data: the candidate builder,
  freeze, verify_origin, the sealed guard, metrics, bootstrap, gates and v1_gd.
- **Not built yet:** the registered 12-trial search loop and its receipts, the scorer for v1 as-is
  through the Hub kernel file, the retrains, the full §10 mutation runner, the Λ aggregation and
  REPORT.md. `v2/README_DEV.md` lists these in order.
- **Open decisions for Stephen:** AR-1 (the banned word in the registration receipt), AR-2
  (`LINE_SEARCH_FAILED`) and AR-3 ("tampered metrics"). None blocks stage 2 from starting, but
  AR-1 should be decided before any v2 artifact is described as ready.
