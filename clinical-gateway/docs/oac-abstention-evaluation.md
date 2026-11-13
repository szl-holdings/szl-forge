# oac-system-health-v1: certified missed-alert threshold, defer band and receipt-bound gate

Status: **research only, synthetic data, not a qualification.** Advisory tooling;
it does not change the published model, its card threshold (0.16) or any
authority boundary (`advisory_only_no_ack_release_or_clinical_authority`).

Rebuilt from scratch on 2026-11-13 (UTC) because the original files from an
earlier session were lost. Every number below is labeled:

- **MEASURED**: produced by this run, from the named output file and revisions.
- **UNVERIFIED**: reported by the earlier (lost) session; no artifact exists to check it.
- **MODELED / HYPOTHESIS / UNKNOWN**: reasoning, not measurement.

## 1. Inputs (MEASURED)

| Input | Revision | sha256 |
|---|---|---|
| HF `SZLHOLDINGS/oac-system-health-v1` | `ed0a3bbd1611dd0d051df30931dee2bd548f363f` (HF main at run time) | — |
| `oac_operational_health.py` (kernel) | HF ed0a3bbd, git blob `6f448bc4` | `e2aa63991b94596a16d69a288510b4b10af88932e075c055d97d1dac4d9a930d` (= `artifact_receipt.kernel_sha256`) |
| `model.json` | HF ed0a3bbd, git blob `2c65ae1c` | `f111b7fc65db561c80763b25e538366a19bed18526ca881915777487935d9557` |
| `artifact_receipt.json` | HF ed0a3bbd | `c5b58aa1f85c67ec6ce64a5516cbe5d3e72f1063eed768b4bfe997f5a18c0a27` |
| HF dataset `SZLHOLDINGS/oac-clinical-transport-observability-synthetic` | `f7ab6170bf78138b187b8cb707d374a25ad85375` (revision pinned by the card; dataset main is `1ed34ff6`, and its test/validation bytes are identical) | — |
| `data/train.jsonl` (768 rows) | blob `58a6d40a` (same blob as szl-forge main) | `b868acce6a2e6a8f6a5e5edb3f5b45c4f2e7e7b33c1bc56d22128620aab8b1f6` |
| `data/validation.jsonl` (192 rows, 33 positives) | blob `9f8dd2d3` | `4557220a73b7f8efbb59ab41d043785cebaff71eac4d9fef8c59800a4f8f5abf` |
| `data/test.jsonl` (240 rows, 51 positives) | blob `8ea667b4` | `7fc6ea635fcc501d56eed6a8b9e4f5f3a9057d0aa04505502759197150d79679` |

The split sha256 values equal `dataset_receipt.json` at f7ab6170.

**Kernel cross-check (MEASURED).** Three kernel versions exist (census report
`abstention-hatun-card-gaps-2026-11-13.md` §1c): HF ed0a3bbd (blob `6f448bc4`),
szl-forge main c40e9ae7 (blob `081083c4`, sha256 `2fd8ffd3…2669`) and draft
PR #722 head a6c004ce (blob `b3f8a04d`, sha256 `5e06ddff…b685`). They differ
only in input/receipt hardening; the scoring math is the same. `oac_ltt.py`
produced byte-identical reproduction, LTT, operating-point, defer-band and
validity sections with all three kernels, and `oac_rbcg.py selftest` passes
14/14 with the main kernel as well. So none of the numbers below depend on which
kernel becomes the source of truth.

## 2. Card reproduction (MEASURED, `tools/oac_ltt_results.json`)

**Reproduced exactly.** All metrics for train, validation and test match
`artifact_receipt.json` to 1e-9, with zero differences. Test at threshold 0.16:
ROC AUC **0.830169104679**, TP **40**, FP **57**, TN **132**, FN **11** (240 rows, 51 positives).
Validation ROC AUC 0.832094530208 (192 rows, 33 positives). These match the
earlier session's numbers (UNVERIFIED: AUC 0.830169, 40/57/132/11).

## 3. Learn-then-Test missed-alert threshold (MEASURED, validation split only)

Method: Learn-then-Test (Angelopoulos, Bates, Candès, Jordan, Lei, 2021,
arXiv:2110.01052). Risk = missed-alert rate (FNR, conditional on a true alert);
alert iff `operator_attention_score >= λ`. Fixed, data-independent grid
λ ∈ {0, 0.0005, …, 1} (2001 points). Because FNR does not decrease as λ rises,
fixed-sequence testing proceeds upward from λ = 0 and stops at the first
non-rejection, which controls family-wise error at δ. H0(λ): FNR(λ) > α.
Primary p-value: exact binomial tail P(Bin(n_pos, α) ≤ misses). The
Hoeffding–Bentkus p-value is also reported. α = 0.20, δ = 0.10.

| | Binomial (primary) | Hoeffding–Bentkus |
|---|---|---|
| Certified λ | **0.1375** (3 misses / 33, p = 0.0808) | 0.1305 (2/33, p = 0.0728) |
| First non-rejected λ | 0.1380 (4/33, p = 0.1821) | 0.1310 (3/33, p = 0.2197) |

Earlier session (UNVERIFIED): certified λ 0.1377. This run's grid step is 0.0005,
so 0.1375 is the matching grid point. The earlier grid is UNKNOWN.

## 4. Test realization against baselines (MEASURED, test split, used once)

| Operating point | λ | TP | FP | TN | FN | Missed-alert rate | Alert rate | Precision |
|---|---|---|---|---|---|---|---|---|
| Card threshold | 0.16 | 40 | 57 | 132 | 11 | **21.6%** | 40.4% | 41.2% |
| LTT certified (binomial) | 0.1375 | 48 | 108 | 81 | 3 | **5.9%** | **65.0%** | 30.8% |
| LTT certified (HB) | 0.1305 | 50 | 135 | 54 | 1 | 2.0% | 77.1% | 27.0% |
| Naive (largest λ with validation FNR ≤ 0.20) | 0.1525 | 43 | 64 | 125 | 8 | 15.7% | 44.6% | 40.2% |
| Rule `consecutive_failures > 0` | — | 15 | 49 | 140 | 36 | 70.6% | 26.7% | 23.4% |
| Always negative | — | 0 | 0 | 189 | 51 | 100% | 0% | — |

Against the earlier session (UNVERIFIED): test FNR 5.9% vs 21.6% at the card
threshold matches. The alert rate is **65.0%** here vs **64.2%** earlier. The likely
cause is the different grid point (0.1375 vs 0.1377), but that is a HYPOTHESIS.
The cost is plain: getting missed alerts down to 5.9% means alerting on 65% of
rows, and 108 of the 156 alerts are false.

## 5. Defer-to-human band (MEASURED)

Design used in this rebuild: auto-clear if score < λ_low = 0.1375 (FNR-certified
above). Auto-alert if score ≥ λ_high, where λ_high is the smallest λ certified by
the same LTT procedure (descending fixed sequence) for false-alarm rate
FPR ≤ β = 0.10 at δ = 0.10 on validation negatives: **λ_high = 0.218** (10/159
false alarms, p = 0.0703). Rows in between are deferred.

| Split | Answered | Deferred | Accuracy on answered | Missed alerts auto-cleared | False alarms auto-alerted | Positives deferred |
|---|---|---|---|---|---|---|
| Validation | 112/192 (58.3%) | 80 | 88.4% | 3 | 10/27 | 13 |
| Test | 127/240 (**52.9%**) | 113 | **83.5%** | 3 | 18/43 | 23 |

The earlier session reported (UNVERIFIED) "answering 56% at 94.1% accuracy". **This
rebuild does not reproduce that.** The earlier band's design is UNKNOWN. HYPOTHESIS:
94.1% equals 48/51, the test *recall* at λ = 0.1375, so the earlier figure may
have been mislabeled. That can't be checked because the files are lost.

## 6. Resampling validity check (MEASURED, 2000 reps, seed 20261112)

Population = pooled validation + test (432 rows, 84 positives). Each rep draws a
192-row calibration set, certifies λ, and measures FNR.

| Variant | Certified exceeds α | Naive exceeds α |
|---|---|---|
| (a) bootstrap from the pooled population; true risk = population FNR | **6.65%** (≤ δ = 10%) | **51.3%** |
| (b) random 192/240 partition; held-out empirical FNR (includes evaluation noise) | 11.25% | 52.3% |

For (a), the certified FNR has median 0.107 and 90th percentile 0.167. Earlier session
(UNVERIFIED): 6.3% vs naive 54.8%. Variant (a) is the clean check and stays
under δ. Variant (b) runs above δ because the held-out FNR is itself a noisy
estimate on about 47 positives. It is reported here so the result isn't overstated.

## 7. Receipt-bound certified gate (MEASURED, `tools/oac_rbcg.py`)

The receipt (`tools/oac_rbcg_receipt.json`, `receipt_sha256 81f1bc40…479a`) binds
the sha256 of `model.json`, `artifact_receipt.json`, the kernel and the
calibration split, plus a digest of the calibration scores, α, δ, the grid, the
threshold 0.1375 and the certificate (n_pos 33, misses 3, p 0.0808). The digest
is a sha256 integrity check, **not a signature**. Anyone who can edit the receipt
can recompute it. Protection against forgery comes from the verifier
recomputing everything: hashes, scores, certificate, the fixed sequence up to λ,
policy pins (α ≤ 0.20, δ ≤ 0.10, binomial p-value, grid step, shift level ≤ 0.10,
≥ 30 deployment rows), then the shift tests.

Shift test: two-sample KS, Mann–Whitney U, and a Fisher test on the alert rate
at λ, each comparing calibration scores with deployment scores. Bonferroni ×3,
refusing when the corrected p < 0.10.

| Deployment | Verdict | KS p | MW p | Alert-rate p | Bonferroni p |
|---|---|---|---|---|---|
| Validation (self) | **ACCEPT** | 1.0 | 1.0 | 1.0 | 1.0 |
| Test split | **REFUSE:SHIFT** | 0.0300 | 0.0174 | 0.0594 | **0.0523** |

Earlier (UNVERIFIED): validation ACCEPT; test REFUSE:SHIFT with p = 0.031, corrected
0.092. The KS p (0.030) is close to the earlier 0.031. The corrected p differs because
Mann–Whitney gives the minimum here. **Sensitivity:** at a 0.05 shift level, the test
split would be ACCEPTed (0.0523 > 0.05), so this refusal is borderline. The
validation-to-test score shift is real but small (alert rate 55.7% → 65.0%).

Cross-kernel (MEASURED): verifying this receipt against the szl-forge main
kernel or the PR #722 kernel returns `REFUSE:HASH_MISMATCH:artifact_receipt.json`.
That is the binding working as designed. A receipt has to be re-issued once the
kernel source of truth is chosen.

**Selftest: 14/14 pass** (`tools/oac_rbcg_selftest.json`): genuine ACCEPT;
certificate known-answer pin; receipt edited without reseal → RECEIPT_DIGEST;
model.json tampered → HASH_MISMATCH:model.json; kernel tampered →
HASH_MISMATCH:oac_operational_health.py; calibration row dropped →
HASH_MISMATCH:calibration; threshold raised and resealed → CERTIFICATE;
**consistent forgery at λ = 0.16 (certificate recomputed honestly, resealed) →
DELTA**; the same forgery with δ relaxed to 0.9 → POLICY; shifted deployment
(consecutive_failures + 5) → SHIFT; 10-row deployment →
INSUFFICIENT_DEPLOYMENT; Bonferroni arithmetic; KS known answer; deterministic issue.

**Mutants: 8/8 caught, unmutated control passes** (`tools/oac_rbcg_mutants.json`):
M1 skip receipt digest, M2 skip model-file hashes, M3 skip calibration hash, M4
delta check disabled, M5 policy δ bound disabled, M6 no Bonferroni, M7 binomial
CDF off by one, M8 minimum deployment rows ignored. Each mutant is applied to a
copy of the file, and its selftest must exit non-zero.

## 8. Limits

- **Few positives:** 33 validation positives. The certificate moves one grid step
  per miss (3 → 4 misses flips p from 0.08 to 0.18). Test has 51 positives.
- **The shift test only sees score distributions.** It cannot see label shift at
  equal scores, concept drift, or feature changes that leave scores unchanged.
  The 0.10 level is a policy choice, and the test-split verdict is borderline.
- **Synthetic data only** (fixed seed 2500, one generator). The test split was used
  during development (card §Training), so it is not a fresh holdout. Nothing here
  says anything about real telemetry.
- FNR control is marginal over the calibration distribution, under
  exchangeability. It is not conditional per row, and it does not survive shift. That's the reason for the gate.
- The receipt digest is not a signature, and the receipt has no signer identity.
- Novelty: **UNKNOWN.** LTT, selective classification and KS/MW shift tests are
  standard. No claim is made that the combination is new.
- Stdlib only (Python ≥ 3.10; run on 3.13.5). No third-party dependencies.

## 9. Reproduce

```
python3 -B clinical-gateway/tools/oac_ltt.py                     # writes tools/oac_ltt_results.json
python3 -B clinical-gateway/tools/oac_rbcg.py issue  --output clinical-gateway/tools/oac_rbcg_receipt.json
python3 -B clinical-gateway/tools/oac_rbcg.py verify --receipt clinical-gateway/tools/oac_rbcg_receipt.json \
    --deployment clinical-gateway/huggingface/dataset/oac-clinical-transport-observability-synthetic/data/test.jsonl
python3 -B clinical-gateway/tools/oac_rbcg.py selftest
python3 -B clinical-gateway/tools/oac_rbcg.py mutants
```

Both scripts default to `../huggingface/model/oac-system-health-v1` and
`../huggingface/dataset/oac-clinical-transport-observability-synthetic/data`
relative to their own file, and accept `--model-dir` / `--data-dir`. In szl-forge,
the default model dir holds the main-branch kernel (blob 081083c4). The numbers are
identical there (§1), but the committed receipt binds the HF ed0a3bbd bytes,
so `verify` refuses with HASH_MISMATCH until kernel parity lands, or until a receipt is
re-issued against the in-repo files.

## 10. Rollback

The change is additive: two tools, this doc and six JSON evidence files, with no edits
to existing files, the model, the card or CI. Rollback: close the draft PR and
delete its branch. After a merge, `git revert <merge-commit>`. Nothing is
published to Hugging Face.
