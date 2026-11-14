# oac-system-health-v1: certified missed-alert threshold, abstention band, receipt-bound verifier (2026-11-10)

Prototype, unpushed. It ran on szl-vm2 against `szl-forge` main model files (byte-identical to GitHub main: model.json
`f111b7fc…d9557`, oac_operational_health.py `2fd8ffd3…92669`) and the committed synthetic split
`ops-health/v2/data/legacy_v1` (validation `4557220a…`, test `7fc6ea63…`). Stdlib only; the model's own scorer;
synthetic operations telemetry only. Labels: MEASURED unless stated.

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

## Proposed next steps (each needs the owner's approval by name)
- A model-card section reporting the 21.6% missed-alert rate of the 0.16 threshold next to the certified α = 0.20 operating point and the abstention band.
- Collect more calibration positives before any guarantee is advertised.
