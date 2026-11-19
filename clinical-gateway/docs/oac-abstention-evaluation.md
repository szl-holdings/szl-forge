# oac-system-health-v1: certified missed-alert threshold, abstention band, receipt-bound verifier (2026-11-10)

Prototype, unpushed. It ran on szl-vm2 against `szl-forge` main model files (byte-identical to GitHub main: model.json
`f111b7fc…d9557`, oac_operational_health.py `2fd8ffd3…92669`) and the committed synthetic split
`ops-health/v2/data/legacy_v1` (validation `4557220a…`, test `7fc6ea63…`). Stdlib only; the model's own scorer;
synthetic operations telemetry only. Labels: MEASURED (synthetic) unless stated.

## 1. Reproduction gate: passed
Test AUC 0.830169 and confusion at the card threshold 0.16 (TP 40, FP 57, TN 132, FN 11) match the model card exactly.

## 2. What the card threshold does and what a certified threshold does (test split, n = 240, 51 positives)
| rule (chosen on validation, n = 192, 33 positives) | missed-alert rate | alert rate | precision | accuracy |
|---|---|---|---|---|
| always negative (majority) | 100% | 0% | — | 0.7875 |
| consecutive_failures > 0 | 70.6% | 26.7% | 0.23 | 0.6458 |
| card threshold 0.16 (balanced-accuracy pick, **no guarantee**) | **21.6%** (11/51) | 40.4% | 0.41 | 0.7167 |
| **Learn-then-Test, α = 0.20, δ = 0.10 → 0.1377** | **5.9%** (3/51) | 64.2% | 0.31 | 0.5458 |
| Learn-then-Test, α = 0.10, δ = 0.10 → 0.1206 | 0% | 95.8% | 0.22 | 0.2542 |

- Certificate: 3 misses among 33 validation positives; binomial tail p = 0.0808 ≤ δ.
- **Validity (2,000 resampled calibration sets, population = pooled validation+test):** the LTT threshold's true missed-alert rate
  exceeded α in **6.25%** of sets (95% CI 5.27–7.40%), within δ = 10%. Naive point-estimate selection exceeded it in **54.75%** (52.56–56.92%).
- Reading: the model's ranking is weak enough that a missed-alert guarantee costs a high alert rate. A guarantee of α = 0.10 makes
  it alert on almost everything. The honest operating point is α = 0.20 at a 64% alert rate.

## 3. Abstention (defer to a human) does better than any single threshold
LTT over 325 candidate bands (Bonferroni), joint risk P(answer ∧ wrong) ≤ 0.10 at δ = 0.10, maximising coverage:
band [0.1514, 0.5331] → on test it answers **135/240 (56.3%) with 94.1% accuracy** (8 wrong; joint risk 0.033) and defers 105 to a person.

## 4. Validation→test exchangeability (permutation, B = 2000)
| statistic | value | p |
|---|---|---|
| KS on model scores | 0.1385 | **0.0305** |
| label rate difference | 0.0406 | 0.3243 |
| consecutive_failures mean difference | 0.1344 | 0.5472 |
Uncorrected, the score distributions differ at the 5% level. Holm-adjusted over the three tests, the smallest p is 0.092. The realised
test missed-alert rate (5.9%) still met α = 0.20 here, but a verifier cannot know that in advance.

## 5. Receipt-bound verifier `oac_rbcg.py` (sha256 73379a3d…d544)
The receipt (sha256 cd587508…728e, deterministic) binds the hashes of model.json, artifact_receipt.json, the scorer and the calibration
file; α, δ, the threshold, miss count and p-value; and a self-seal. `verify(receipt, deployment_batch)` is fail-closed:
REFUSE:RECEIPT_TAMPERED · REFUSE:HASH · REFUSE:CERT (recomputed misses differ or p > δ) · REFUSE:SHIFT (KS permutation p < 0.05) · else ACCEPT.
- On real data: deployment = validation → **ACCEPT**; deployment = committed test split → **REFUSE:SHIFT**.
- Selftest **9/9**: issue reproduces LTT; binomial KAT; accept exchangeable; refuse real shift; refuse tampered calibration (one label flipped);
  refuse edited threshold; refuse resealed threshold with the wrong miss count; refuse an internally consistent forgery that fails δ; refuse a strong synthetic shift.
- Mutants **8/8 caught**: shift check removed, hash check removed, δ ignored in verify (initially SURVIVED → closed by the consistent-forgery case),
  seal ignored, naive LTT, binomial tail off by one, KS always zero, calibration hash unbound.

## 6. Limits
- n = 33 validation positives makes every certificate coarse. More calibration positives would buy a much lower alert rate at the same α.
- The KS test checks marginal score shift only; conditional (label-given-score) shift is not detected.
- The data is synthetic; the guarantee is about this generator, not any real system.
- Novelty of the receipt-bound guarantee is UNKNOWN (no CrossRef/OpenAlex pass yet). The sibling prototype RBCG-1/V1 (session 0dc4549b, synthetic TRIAGE-X) is the closest internal work.

## 7. Re-verification for this PR (2026-11-14)
- **Path binding change:** both tools now resolve the repository relative to their own file (`clinical-gateway/tools/` → repo root)
  instead of the original hard-coded szl-vm2 path. This edit changes the file hashes from the VM-run prototypes
  (`oac_rbcg.py` 73379a3d…d544, per §5) to the in-repo versions: `oac_rbcg.py` `10129d36e672b9759524aebbd95b77c85ff2cd0fc20917ba0d4f6f09702cf94a`,
  `oac_ltt.py` `1435965e8846b15933470dd87668fb599229b096bb6a79a28040c9a41eb96d33`.
- **Re-run on szl-vm2 (2026-11-14)** against model files byte-identical to szl-forge main (model.json f111b7fc…d9557,
  artifact_receipt.json 23239bb3…f95a1, oac_operational_health.py 2fd8ffd3…92669; split validation 4557220a…, test 7fc6ea63…):
  - selftest **9/9 PASS** (all nine cases of §5, on the edited in-repo file);
  - mutant suite **8/8 caught** (same eight mutants as §5, each failing exactly its intended check; runner `mutants_runner.py`);
  - `verify` on validation → **ACCEPT**; on the committed test split → **REFUSE:SHIFT** (unchanged);
  - `oac_ltt.py` re-run reproduces every published number: gate AUC 0.830169 with TP 40 / FP 57 / TN 132 / FN 11;
    LTT thresholds 0.120629 (α = 0.10) and 0.137721 (α = 0.20); validity simulation 6.25% LTT violations vs 54.75% naive;
    shift KS 0.1385, p 0.0305.
- **Receipt determinism:** re-issuing with `oac_rbcg.py issue` reproduces the receipt **byte-identically**
  (receipt file sha256 `cd587508815655ba1d3485111d9f1e311483a9ae4c6cb099dcb2670441c4728e`, inner seal `8e3f5ee3…04ce`).
- Owner context: approved as OAC-ABSTAIN-PR (2026-11-10, reaffirmed 2026-11-14). DCO `Signed-off-by` waived by owner direction
  (solo-builder estate, recorded in estate decisions 2026-11-14). Model card README and the Hugging Face Hub are NOT changed by this PR.

## 8. Review hardening of `oac_rbcg.py` / `oac_ltt.py` (2026-11-19)
An independent review (YUH 2) of the earlier draft found a receipt-policy bypass; the same bug class was reproduced against this file on main
(`fe646e98`) before the fix: `verify` took `alpha`, `delta` and `shift_alpha` from the receipt with no policy check, so a **resealed** receipt
with `shift_alpha` 0, -1 or 1e-300 flipped the test split from `REFUSE:SHIFT` to `ACCEPT` (a lower `shift_alpha` is more lenient), and a resealed
`alpha` 0.99 / threshold 0 receipt also returned `ACCEPT`. The seal is an integrity hash, not a signature, so the receipt cannot choose its own policy.
- `REFUSE:POLICY`: `0 < alpha <= 0.20`, `0 < delta <= 0.10`, `0.05 <= shift_alpha <= 1`, and `threshold` finite (NaN, bool, non-number refused).
- `REFUSE:UNPINNED_KERNEL`: the receipt's kernel hash must be in a pinned set (in-repo kernel `5e06ddff…202e`; HF `ed0a3bbd` kernel `e2aa6399…930d`).
  Both tools also check the on-disk kernel before importing it. `--allow-unpinned-kernel` is a deliberate override. A kernel change now requires updating the pin.
- `REFUSE:MALFORMED:<type>` replaces tracebacks for huge integers, missing files, bad JSON and non-object receipts. None of these ever returned `ACCEPT`.
- Selftest 9 → 14 cases; four ad-hoc mutants (policy check off, shift lower bound flipped, pin off, malformed handler off) each fail the selftest. Not in CI.
- Unchanged: LTT 0.137721 (3 misses of 33, p 0.0808), defer band 56.25% answered at 94.07% accuracy (135 answered, 8 wrong), and the
  `ACCEPT` (validation) / `REFUSE:SHIFT` (test) verdicts. `ACCEPT` on validation only checks the receipt against its own calibration split.
  Remaining limits: low shift-test power, 33 validation positives, and the selftest and mutants are not run in CI.

## 9. Review fixes (YUH 2 re-review of #771, 2026-11-19)
Results below are **MEASURED (synthetic)**; nothing here is a production-readiness claim.
- **CLI exit status.** `oac_rbcg.py verify RECEIPT DEPLOY.jsonl` exits 0 only for `ACCEPT`. Every `REFUSE:*` (including `SHIFT`, `POLICY`, `MALFORMED`) exits 1; a bad
  argument list exits 2; `issue` over an unpinned kernel exits 3. Unreadable or non-JSON receipts print `REFUSE:MALFORMED:<type>` and exit 1.
- **Minimum deployment rows.** `REFUSE:INSUFFICIENT_DEPLOYMENT` below `POLICY_MIN_DEPLOY = 30` rows (explicit; zero rows no longer depends on a `ZeroDivisionError`).
  30 is the floor #723 (`8a233c4b`) used and #734 dropped; it is a policy constant, not receipt-controlled. It is a conventional minimum-evidence rule, not a
  calibrated power threshold. Measured (100 reps, permutation B = 500): against the strong synthetic shift of the selftest (the 120 highest-scored test rows) the
  shift test already refuses 70% of 3-row, 100% of 10-, 30-, 60- and 120-row batches; on same-distribution validation subsamples it falsely refuses 8% / 4% / 1% / 0% / 0%
  at n = 3 / 10 / 30 / 60 / 120. So 30 is conservative for strong shifts and unmeasured for mild ones. Selftest covers 0, 1, 3, 10, 29 (refused) and 30 (not refused).
- **File pins.** `model.json`, `artifact_receipt.json`, the kernel and the calibration split must now equal one pinned *profile* exactly (`repo-main`, or HF
  `SZLHOLDINGS/oac-system-health-v1 @ ed0a3bbd`; mixing hashes from the two is refused). Pinned values: model.json `f111b7fc…d9557` (both); artifact_receipt.json
  `53180b90…b685` (repo) / `c5b58aa1…0a27` (HF); kernel `5e06ddff…202e` (repo) / `e2aa6399…930d` (HF); calibration (`validation.jsonl`) `4557220a…8f5abf` (both, identical bytes).
  Reasons: `REFUSE:UNPINNED_KERNEL`, `REFUSE:UNPINNED_FILE:<name>`. `issue` refuses to emit a receipt over unpinned files. `--allow-unpinned-kernel` (alias `--allow-unpinned`)
  is the deliberate escape and now disables **all** these file pins, not just the kernel's; it is recorded on stderr. **Claim:** ACCEPT means "internally consistent and
  bound to a pinned profile", not "issued by a trusted party" (the seal is not a signature). The `--model-dir`/`--calibration` API arguments can no longer swap in unpinned files.
- **Kernel execution order.** The kernel bytes are read once, hashed, checked against the pin, and only then compiled from those same bytes. Importing `oac_rbcg.py` or
  `oac_ltt.py` now does nothing (no argv parsing, no `sys.exit`, no kernel execution); flags are parsed in `main()`.
- **MALFORMED.** `RecursionError` and `MemoryError` are now `REFUSE:MALFORMED:<type>` (deeply nested receipt, deployment row or CLI input).
- **Threshold floor.** Scores are sigmoid probabilities, so `threshold` must be finite and in `[0, 1]` (`REFUSE:POLICY`), and it must be one of the calibration scores (the only
  values `issue` emits; otherwise `REFUSE:CERT`). A resealed `-1e9` or `1e-9` threshold with `cal_misses` 0 is refused.
- **Tests.** Selftest 14 → 29 cases (adds the CLI exit-code and bad-JSON cases, the on-disk kernel pin with a marker-file probe, the exact pinned sets with no all-zero hash,
  profile mixing, check order seal → policy → pin → hash, and the import-has-no-side-effects subprocess check). `oac_rbcg_mutants.py` applies 32 mutants (covering every mutant class from the review, including the
  three that survived) to a scratch copy; each must fail the selftest (**32/32 killed**). Not in CI.
- **Unchanged (MEASURED (synthetic)):** LTT 0.137721 (3 misses of 33, p 0.0808); defer band 135 / 240 = 56.25% answered at 94.07% accuracy (8 wrong);
  validation `ACCEPT`, test `REFUSE:SHIFT`; `oac_ltt.py` writes a byte-identical `oac_ltt.json` before and after this change.
- **Still open:** the selftest and mutants are not in CI (no workflow change in this PR); low shift-test power; 33 validation positives; the `oac_rbcg_receipt.json`
  files from #723 are not part of this PR.

## Proposed next steps (each needs the owner's approval by name)
- A model-card section reporting the 21.6% missed-alert rate of the 0.16 threshold next to the certified α = 0.20 operating point and the abstention band.
- Collect more calibration positives before any guarantee is advertised.
