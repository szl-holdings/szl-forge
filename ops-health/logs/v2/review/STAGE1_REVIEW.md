# OAC Ops Health v2: stage-1 adversarial review

- **Workstream:** F, stage 1 (review and finish).
- **Reviewer:** the v2-stage1-review agent.
- **Date:** 2026-09-29.
- **Reviewed:** commit `974159c`, the stage-1 code and tests.
- **Fixes:** commit `b75785c`; fixer round (verified issue on A5 scope): commit `efde7ff`.

Everything here is SYNTHETIC and uses DEV data only (`generator.dev_rows`, split name `dev`). No
registered split was read, no sealed file was opened or parsed, and nothing was pushed or
published.

Every claim below carries one label, with the evidence file in brackets. MEASURED means the check
ran in this session and its raw output is saved under `logs/v2/review/`. REPORTED means the value
comes from a named file at a named revision.

## Evidence files (all under `logs/v2/review/` unless noted)

| file | what | interpreter |
| --- | --- | --- |
| `kernel_attack.py` | lens (a) harness: builds a DEV artifact in a temp dir, then runs raw-input probes and artifact/receipt attacks through the CLI in fresh subprocesses. The first review ran 65 + 21; the fixer round added 4 + 8 `echo_` probes and a check that every refusal's stderr is ≤ 1,100 bytes (69 + 29 = 98) | py 3.12.10 |
| `kernel_attack_before.txt` | the 86-probe harness against the reviewed kernel (sha256 `1247c9ff…6d92`) | py 3.12.10 |
| `kernel_attack_echo_before.txt` | fixer round: the 98-probe harness against the `b75785c`/`ec7874b` kernel (sha256 `b0215162…17e5`) | py 3.12.10 |
| `kernel_attack_after.txt` | the 98-probe harness against the fixed kernel (sha256 `bb95f517…4bb8`, commit `efde7ff`). The 86-probe run against `b0215162…17e5` is the version of this file committed in `ec7874b` | py 3.12.10 |
| `echo_tests_on_prefix_kernel.txt` | fixer round: the 8 new `EchoBoundTest` tests run against the `ec7874b` kernel; all 8 fail there (9 failures, 11 errors) | py 3.12.10 |
| `stats_reference.py` | lens (b): independent numpy/scipy references (plus one sklearn cross-check) | python 3.11.9, numpy 2.4.6, scipy 1.17.1 |
| `stats_reference_before.txt`, `stats_reference_after.txt` | 34 checks: 33/34 before, 34/34 after | python 3.11.9 |
| `v1_gd_reproduce_py312.txt`, `v1_gd_reproduce_py311.txt` | output of `-m v2.research.v1_gd --reproduce` | both |
| `sealing_audit.py` | lens (c): a `sys.addaudithook` open-hook around the in-process test suite | py 3.12.10 |
| `sealing_audit_before.txt`, `sealing_audit_after.txt` | 94 tests before; 119 after (re-run at `efde7ff`); 0 sealed opens and 0 validation-split opens both times | py 3.12.10 |
| `../stage1_tests_after_review.txt` | full suite at clean HEAD `efde7ff`: 119 tests OK (111 at `b75785c`) | py 3.12.10 |
| `../stage1_tests_after_review_py311.txt` | the same suite in the CONTRACT §6 command form at `efde7ff`: 119 tests OK | python 3.11.9 |

## Findings and fixes

Severity reflects impact on the fail-closed and truth-boundary contract.

| id | sev | finding (evidence) | fix (commit `b75785c`) | regression test (`v2/tests/test_review_hardening.py`) |
| --- | --- | --- | --- | --- |
| A1 | HIGH | MEASURED: when a model's `training.stop_reason` is a JSON list or object and the receipt was regenerated consistently, the kernel crashes. It raises `TypeError: unhashable type` from the `frozenset` membership test, and the CLI exits **1 with a Python stack trace** (`kernel_attack_before.txt`: `stop_reason_list`, `stop_reason_dict`). | The kernel type-checks `stop_reason` as `str` before the membership test. `main()` also turns any unexpected exception into `{"ok": false, "error": "internal error (<Type>); refusing"}` with exit 2 and no trace. | `test_unhashable_stop_reason_is_refused_cleanly`, `test_cli_subprocess_never_prints_a_traceback`, `test_unexpected_internal_error_is_a_refusal_without_traceback` |
| A2 | MEDIUM | MEASURED: registered constants were not pinned. An artifact with `conformal.alpha` 0.5, or with `training.l2` 0.5 (off the registered grid), was accepted and scored (`kernel_attack_before.txt`: `alpha_0_5_not_registered`, `l2_0_5_not_on_grid`, exit 0). | `validate_artifact` requires alpha == 0.10. `validate_training` requires l2 ∈ {0, 1e-4, 1e-3, 1e-2}. `conformal.fit_mondrian` stays generic, so the rank tests at other alphas still work. | `test_alpha_must_be_registered`, `test_l2_must_be_on_registered_grid` |
| A3 | MEDIUM | MEASURED: the kernel accepted receipts whose `preregistration_sha256`, `data_manifest_sha256` or `source.commit` were altered to other well-formed values (exit 0). No layer checked the origin claim. §10's "altered source commit" mutant would therefore have been a BLIND_SPOT. | The kernel pins `PREREGISTRATION_SHA256` (equal to `registry.PREREG_SHA256` and to the file digest). New `freeze.verify_origin(receipt)` is the research-side integrity layer: the commit must exist, and its committed blobs (`git cat-file blob`) of the kernel, generator, manifest and preregistration must hash to the receipt digests. The standalone kernel cannot run git, so manifest and commit tampering are accepted at the kernel **by design** and caught by `verify_origin` (`kernel_attack_after.txt` marks both `origin`). | `test_receipt_preregistration_digest_is_pinned`, `OriginVerificationTest` (hermetic temp git repo: accept, altered commit, drifted generator, tampered manifest digest; the DEV fake commit is refused) |
| A4 | LOW | MEASURED: Unicode look-alike keys were refused only by the exact eight-field schema check, not as prohibited: full-width `ｐａｔｉｅｎｔ`, `pat<U+200B>ient`, `m<U+00AD>rn`. The fail-closed outcome held for all 16 look-alike probes (exit 2). | The screen also matches the Unicode-folded key: NFKC, category-Cf characters removed, casefolded, whitespace and hyphen runs turned into `_`. These keys are now refused as `(… , unicode-folded match)`. Cross-script homoglyphs (Cyrillic) do not fold. They remain refused by the schema check, which is a documented residual. | `test_unicode_folded_prohibited_keys`, `test_look_alike_and_padded_feature_names_are_refused` |
| A5 | LOW | MEASURED: refusal messages echoed keys in full (a 1,000-character key was echoed whole). Keys are JSON-escaped: no raw ESC bytes reached stderr (`key_ansi_escape`). **Correction (fixer round, verified issue):** the `b75785c` fix covered only inference-input keys. `_exact_keys` still echoed model and receipt keys in full, and the calibrator `kind` *value* was echoed via `{kind!r}`, so the earlier claims that every key was capped and that values were never echoed were false. With a consistent receipt, `kernel_attack_echo_before.txt` measured stderr of 100,089 bytes (receipt key of 100,000 characters), 100,087 (model key), 100,058 (calibrator kind) and 104,056 (kind as a large list). It also measured 600,064 bytes (60,000 extra short input keys) and 2,637 bytes (a deep path of long keys). The receipt is not hash-bound, so anyone able to write it could trigger this. Fail-closed held throughout: exit 2 and no traceback. | `b75785c`: `_shown_key` caps an input key at 64 characters plus `...(+N chars)`. **`efde7ff`:** at most 8 unexpected keys are listed (`...(+N more)`), each capped at 64 characters, in `_exact_keys` and in the input schema check. An unsupported calibrator kind is refused without being shown (`unsupported calibrator kind (not shown); expected one of [...]`). An echoed input path is capped at 256 characters. `_refusal_line` makes the CLI refusal one ASCII JSON line of ≤ 1,024 bytes, whatever the message holds. Short keys are unchanged, so the existing probe fragments still match. After the fix, the maximum stderr over all 98 probes is 355 bytes (`kernel_attack_after.txt`). | `test_long_keys_are_truncated_in_messages`; `EchoBoundTest` (8 tests: receipt key, model key with a consistent receipt, `weights`/`calibrator.params` keys, calibrator kind as long text, large list and short text, many extra keys, deep path, `_refusal_line` byte bound, and a CLI subprocess byte count) |
| A6 | LOW | MEASURED: an input of `-0.0` was accepted as 0 (correct), but the advisory bytes differed from those for `0.0`. Normal outputs also carried `-0.0` contributions whenever a negative weight met a zero feature (`neg_zero_in_output` ×2 before, ×0 after). | `_normalize_value` returns `x + 0.0`, and reported contributions are `round(c, 12) + 0.0`. Scores and model bytes are unchanged: the DEV artifact sha256 is `0a578f03…d182` both before and after. | `test_negative_zero_is_canonical` |
| A7 | MEDIUM | MEASURED: `grep -ri` for the CONTRACT §6 banned word over `v2/` finds three hits. Two are allowed: v1's authority-map key in the kernel, and the test asserting it. The third is not: the `v1_generator_source` upstream path in `v2/PREREGISTRATION.receipt.json`, which is registration evidence committed in `bca9225`. | Not edited, because it is a registration record. Filed as **AR-1** in `v2/AMENDMENT_REQUESTS.md` for Stephen's decision. | `ForbiddenWordTest` (whitelists exactly that line, pending AR-1) |
| B1 | MEDIUM | MEASURED: an INVALID_BASELINE report withheld the four per-class rates, but still reported BA, **accuracy** and prevalence. Solving acc = prev·sens + (1−prev)·spec together with BA = (sens+spec)/2 recovered sensitivity 0.324390911580 (true value 0.324390911580) and specificity 0.734691380681 for a DEV fixture that failed precondition (iv) (`stats_reference_before.txt`, `invalid_baseline_rates_not_derivable`). This defeats §10's "never reported". | `metrics.binary_report` now withholds accuracy and the confusion counts unless the fixture is valid. BA and prevalence stay (one equation, two unknowns). | `test_invalid_report_withholds_accuracy` |

## Lens (a): kernel security

The results come from `kernel_attack_after.txt` (MEASURED, kernel `efde7ff`): **98/98 probes behaved as expected, 0 findings, and the largest refusal stderr was 355 bytes**. Before
the first fixes, the 86-probe run gave 79/86, with 7 findings: A1 ×2, A2 ×2, A3 ×3. After A3, the
harness was updated to expect `origin` for the manifest-digest and commit probes: the kernel
accepts both by design, and `freeze.verify_origin` refuses them. In the fixer round, the 98-probe
harness against the `b75785c` kernel gave 87/98, with 11 echo-bound findings
(`kernel_attack_echo_before.txt`; see A5).

- **CONTRACT §2.6 classes** (MEASURED): missing field, extra field, NaN, +Inf and −Inf literals,
  out-of-range, string-for-number, and field names `patient`, `hl7`, `fhir`, `specimen`, `order`,
  `result` and `mrn`. Every one was refused with exit 2, a one-line JSON error on stderr, empty
  stdout and no traceback. The in-process suite additionally runs 66 probes through
  `gates.FAILCLOSED_PROBES`, with the expected message fragment checked for each.
- **Duplicate JSON keys** (MEASURED): refused in the features object (same and conflicting
  values), at the wrapper, nested, in model.json and in the receipt. The mechanism is
  `object_pairs_hook`. v1 accepts duplicate keys (REPORTED, workstream B audit).
- **Numbers** (MEASURED):
  - Refused: `1e308` and `-1e308` (range), a `1e400` literal (finite check), a 5,001-digit
    integer (the parser's digit limit), a 401-digit integer (overflow check), `-5e-324`
    (range), `-0.0` and `1.0` for a binary field, numbers as strings, and `null`.
  - Accepted as 0: `-0.0` for a continuous field, now with byte-identical output (A6).
- **Look-alike and whitespace keys** (MEASURED): full-width keys, zero-width space or joiner, word
  joiner, soft hyphen, BOM and NBSP inside prohibited names are all refused as prohibited after
  A4. A trailing space, leading space, zero-width character, full-width letter or upper case in
  a feature name is refused by the exact schema check.
- **Structure and encoding** (MEASURED):
  - Nesting: 20,000 nested lists are refused by the parser's recursion guard. 200 nested
    objects are refused at the 32-level screen.
  - Size: a 1,000,000-byte input is accepted; 1,000,001 bytes and 5 MB are refused.
  - Encoding: a UTF-8 BOM, UTF-16, invalid UTF-8, an empty file, `null`, a list, a number, a
    string, trailing garbage, two concatenated documents, a comment, single quotes and an
    unwrapped payload are all refused.
- **Self-hash binding** (MEASURED): a receipt whose `kernel_sha256` does not match the kernel file
  is refused (`kernel sha256 does not match receipt`). The existing `test_kernel_byte_flip_is_refused`
  also passes.
- **Strict receipt** (MEASURED): an added `metrics` key, a duplicate key, a BOM, a list-typed
  `selection.stop_reason`, and (after A3) a foreign preregistration digest are all refused.
- **Authority map** (MEASURED by reading both files): the kernel's `AUTHORITY_MAP` is exactly v1's
  five keys, all false. It matches `hub/oac-v1/oac_operational_health.py:287-293` (REPORTED:
  Hub revision `dd7d1098…`), and every advisory emits it (`kernel_attack_after.txt`,
  `control_valid`).
- **Banned word** (MEASURED): only the authority-map key in the kernel and in the tests, plus AR-1.

## Lens (b): statistics

The results come from `stats_reference_after.txt` (MEASURED, python 3.11.9): **34/34 checks pass**. Every value is a
DEV value, used only to compare implementations.

| check | result |
| --- | --- |
| normalization parity (numpy min-max from the raw DEV dicts vs the kernel) | exact over 32,768 values |
| IRLS vs `scipy.optimize.minimize(trust-exact)` at l2 = 0, 1e-4, 1e-3, 1e-2 | max\|Δθ\| = 6.7e-8, 4.1e-11, 5.3e-15, 2.4e-13, all < 1e-6. v2 converged in 7/7/7/6 iterations. At the v2 optimum the gradient is ≤ 1.2e-16; scipy's is ≤ 5.9e-11. |
| Platt vs scipy | max\|Δ(a,b)\| = 6.8e-9 (v2 rounds to 12 dp) |
| isotonic vs `scipy.optimize.isotonic_regression` and a textbook PAV, with ties pooled | ≤ 4.9e-13 on raw DEV scores (2,048 tie blocks), on 2-dp-rounded scores (92 tie blocks) and on a hand case. Step prediction between keys matches. |
| ECE bin edges | 33 edge and nextafter scores give the same bins: `0.3` goes to bin 3, `nextafter(0.3, −)` to bin 2, `1.0` to bin 9 (last bin closed). ECE values agree to < 1e-15. |
| AUROC vs scipy Mann-Whitney U / (P·N), and sklearn | agree to < 1e-12 with no ties and with 19,902 tied values. The count-weighted AUROC equals MWU on the materialized rows for 5 random count vectors. |
| bootstrap pairing | both contenders saw identical count vectors in all 2,000 resamples |
| bootstrap draws | all 2,000 resamples equal `random.Random(20260927).choices(range(n), k=n)`, and the BA values match exactly |
| bootstrap percentile | equals `numpy.percentile(..., method="linear")`, which is type 7. Lower bound: h = 1999·0.025 = 49.975, between order statistics #50 and #51. Upper bound: h = 1949.025, between #1950 and #1951. |
| strata | 50 stratified resamples (4 × 5,000) equal per-stratum `choices`, and every stratum keeps its size |
| Mondrian | k = ⌈(n+1)(0.9)⌉ gives 3015 for n = 3348 and 675 for n = 748. This equals `np.quantile(..., method="inverted_cdf")`. The stored q̂ is ≥ the exact q̂ and within 1e-12 of it. DEV-eval (20,000 rows) per-class coverage is 0.9038–0.9110, with none, Platt and isotonic all ≥ 0.90. Coverage from independent numpy sets matches. |
| INVALID_BASELINE vs §10 | (i), (ii), (iii) and (iv) each trigger. The constant-ALERT fixture fails (iii). Majority gives BA = 0.5 and `INVALID_BASELINE`. The rates are no longer derivable (B1 fixed). |
| threshold selection | a numpy brute force with the registered tie rules picks the same k = 21 |
| v1_gd 12-dp reproduction | `all_equal_at_12dp: true` on 3.12.10 and on 3.11.9. Max unrounded \|Δ\| is 4.61e-13 and 4.60e-13 against the published model (REPORTED: Hub `dd7d1098…`, `model.json` sha256 `f111b7fc…9557`). Train sha256 is `b868acce…b1f6`. |

## Lens (c): sealing

- **Preregistration** (MEASURED): the sha256 of `v2/PREREGISTRATION.md` is
  `145497789c7ad78a80a94cfd7f2b031ca9a20df49c06e0b3412af9070fa1ef4b`, unchanged. Its only commit
  is `bca9225`.
- **Data** (MEASURED): `git diff --stat 7c7226a HEAD -- v2/data v2/PREREGISTRATION.md` is empty,
  and `git status --porcelain -- v2/data` is empty. The sha256 of `v2/data/MANIFEST.json` is
  `c0d2fa0a…7252`.
- **Static** (MEASURED, grep): outside `sealed_guard.py`, `make_data.py`, `registry.py` and
  `dataio.py`, no v2 module names a sealed path or constant. `make_data.verify_all` hashes sealed
  bytes without parsing them. No module opens the registered `validation` split: the only
  `open_split` calls are in tests, against temp fixture directories, or with a wrong purpose that
  is refused before any read.
- **Runtime** (MEASURED): during the in-process suite (94 tests before; 111 after `b75785c`; 119
  after `efde7ff`, the current `sealing_audit_after.txt`), the audit hook
  recorded **0 opens under `v2/data/sealed/` and 0 opens of `v2/data/validation.jsonl`**. The
  only `v2/data` opens were `MANIFEST.json` and `legacy_v1/*` (the v1 public splits, for parity
  and `v1_gd`). Limitation: subprocesses spawned by tests (the CLI subprocess and the determinism
  re-run) are not hooked. By source inspection they use only DEV rows.

## Observations that are not defects (carried to stage 2)

1. **Origin check in stage 2** (MODELED): the stage-2 integrity layer must call
   `freeze.verify_origin` on every frozen artifact. The kernel alone cannot catch the manifest or
   commit mutants (A3).
2. **Probe gate counts crashes as refusals** (MEASURED from source): `gates.run_failclosed_probes`
   counts any exception as a refusal. That follows §10's "fraction refused", but a crash is a
   refusal only by accident. Stage 2 should report the exception types it records.
3. **Integrity gate lets unexpected exceptions escape** (MEASURED from source): `gates.g_integrity`
   catches only `OperationalModelError`. After A1, no known artifact reaches another exception
   type.
4. **Rates must go through the report gate** (MODELED): `bootstrap.BinaryStats.rates` computes
   every rate. Stage-2 reporting must route baseline per-class rates through
   `metrics.binary_report` gating.
5. **Protocol clarifications** (REPORTED): AR-2 (`LINE_SEARCH_FAILED` stop reason) and AR-3 ("tampered
   metrics" when the receipt carries no metrics) are filed in `v2/AMENDMENT_REQUESTS.md`.
