# RBCG v1 — method, assumptions, validity sketches and references

**Scope.** In-silico research prototype by SZL Holdings. Synthetic data only. Not for clinical use; it must never receive clinical or patient data. Testbed: SZL's TRIAGE-X simulator and its frozen logistic-regression toxidrome advisory `szl-triagex-toxidrome-lr-v0` (weights sha256 `5f11addc…`), used unchanged.

Labels: **REPORTED** = stated by a cited source whose text or metadata was fetched and checked here; **INFERRED** = our own reasoning (a proof sketch, not a peer-reviewed proof); **MEASURED** = computed in this build; **UNKNOWN** = not established.

## 1. What changed from v0
| part | v0 | v1 |
|---|---|---|
| LTT p-value | row-level exact binomial, BinomCDF(k; n, α) | scene-level betting e-value on the linearized selective loss (primary); Hoeffding–Bentkus on the same scene means (secondary) |
| multiple testing | fixed sequence over τ = 0.99 … 0.30 | unchanged |
| label audit | Clopper–Pearson at δ·6/(π²j²) per look (union bound), rows as i.i.d. | anytime-valid betting confidence sequence on the same scene-level loss |
| monitor | one CTM, score mean(1−pmax), δ_s = 0.01 | two CTMs: primary OOD count per scene (δ_A = 0.0075), secondary mean(1−pmax) (δ_B = 0.0025); alarm when either crosses |
| anchoring | none | RFC 9162-style Merkle log; signed tree heads (DSSE, ECDSA P-256 / RFC 6979); verifier checks inclusion of the guarantee and of the monitor-chain head, and STH consistency |
| pre-registration | none | signed protocol receipt binding code hashes; its digest prefix is embedded in every evaluation seed family |

## 2. Assumptions
- **A1 Scene exchangeability.** Calibration and deployment *scenes* are exchangeable (in the testbed: i.i.d. draws for a given `env_profile`). Rows within a scene may be dependent (they share night/smoke/indoor flags). This is the only exchangeability v1 needs; v0 additionally treated rows as i.i.d. inside its binomial test and its audit.
- **A2 Known loss bound.** Every scene has at most n_max = 24 rows (bound in the receipt; the verifier refuses larger units), so the scene loss below lies in [0, 1] deterministically.
- **A3 Pre-specified design.** Grid and order, α, δ, c, n_cal, the monitor split and the audit level are fixed in the signed protocol before evaluation data exist.
- **A4 Labels.** Audit labels are correct. A scene enters the audit only if every accepted row in it is labelled; skipping is assumed independent of the scene's errors (missing completely at random at the scene level).
- **A5 Tie-break uniforms.** θ_j are derived by SHA-256 from a seed fixed at issuance and are treated as independent U(0, 1] of the data (computational assumption, INFERRED adequate because deployment data are generated after the seed exists).
- **A6 Cryptography and the log.** SHA-256 is collision and second-preimage resistant; ECDSA P-256 is unforgeable; the log key is not compromised. A single verifier cannot detect a split view (two inconsistent histories shown to different parties) unless signed tree heads are exchanged; inclusion and consistency proofs only bind what one verifier sees.

## 3. Scene-level linearized selective loss (INFERRED, elementary)
For threshold τ and scene s, let a_s(τ) be the accepted rows and e_s(τ) the errors among them. Define
L_s(τ) = (e_s − α·a_s)/n_max ∈ [−α, 1 − α] and X_s = L_s + α ∈ [0, 1], computed exactly as an integer numerator over n_max·10⁶.
With R(τ) = E[e_s]/E[a_s] (the pooled selective risk, i.e. P(wrong | accepted) for a row drawn from a random scene when sizes are fixed), E[X] − α = E[a]·(R − α)/n_max. Hence **H0(τ): E[X] ≥ α ⇔ R(τ) ≥ α** whenever E[a] > 0; if E[a] = 0 then E[X] = α and H0 cannot be rejected (fail closed).

## 4. Primary LTT p-value: one-sided predictable plug-in betting (REPORTED method, INFERRED specialization)
For calibration scenes in their fixed order, with μ̂_t = (1/2 + Σ_{i≤t}X_i)/(t+1) and σ̂²_t = (1/4 + Σ_{i≤t}(X_i − μ̂_i)²)/(t+1):
λ_t = min( √(2 log(1/δ) / (σ̂²_{t−1} · t · log(1+t))), c/(1 − α) ),  K_t = Π_{i≤t}(1 + λ_i(α − X_i)),  **p = min(1, 1/max_{t≤n} K_t)**.
- REPORTED (Waudby-Smith & Ramdas, betting_martingales.tex of arXiv:2010.09686, fetched): the hedged capital process K^±(m) = max{θK⁺(m), (1−θ)K⁻(m)} with λ⁺ = |λ̃| ∧ c/m, λ⁻ = |λ̃| ∧ c/(1−m), c ∈ [0, 1) (defaults 1/2 or 3/4), and the predictable plug-in λ̃_t = √(2 log(2/α)/(σ̂²_{t−1} t log(t+1))) with the μ̂, σ̂² above; {m : K^±_t(m) < 1/α} is a (1−α) confidence sequence.
- INFERRED specialization: for the one-sided null we bet only on the side that profits when E[X] < α (θ = 0, the K⁻ branch at m = α) and tune λ̃ with log(1/δ). Validity does not depend on the tuning: any predictable λ_t ∈ [0, c/(1−α)] works.
- **Validity sketch (INFERRED).** Under H0 with i.i.d. scenes, E[1 + λ_t(α − X_t) | F_{t−1}] = 1 + λ_t(α − μ) ≤ 1 and every factor is ≥ 1 − c > 0, so (K_t) is a nonnegative supermartingale with K_0 = 1. Ville's inequality (REPORTED via Ramdas, Grünwald, Vovk & Shafer 2023; Shafer, Shen, Vereshchagin & Vovk 2011) gives P(max_t K_t ≥ 1/u) ≤ u, so P(p ≤ u) ≤ u. The stored ceil(p·10⁹) is conservative.
- **Power cost (INFERRED; MEASURED in the design pilot).** Because a scene could in principle be all-wrong (X = 1), the bet is capped at c/(1−α) = 0.833 for c = 3/4. The per-scene log-growth is then at most about 0.833·g with g = coverage·(α − R), and at the first grid point (τ = 0.99, coverage ≈ 0.12) g ≈ 0.012, so roughly 230–350 scenes are needed before anything can be certified. At v0's 80 scenes v1 issues nothing (fail closed). The design pilot fixed n_cal = 320 and c = 3/4 (protocol `pilot` block).

## 5. Fixed-sequence testing (REPORTED)
Test τ = 0.99, 0.98, … in order, reject while p(τ) ≤ δ, stop at the first non-rejection; τ* is the last rejection. LTT (main.tex, "Fixed sequence testing controls FWER", citing Bauer 1991) states this controls the family-wise error rate at δ for any order fixed without looking at the calibration data. Consequence (INFERRED): P(R(τ*) ≥ α) ≤ δ.

## 6. Secondary p-value: Hoeffding–Bentkus (REPORTED)
p_HB = min( exp(−n·h₁(μ̂ ∧ α, α)), e·P(Bin(n, α) ≤ ⌈n μ̂⌉) ), h₁(a, b) = a log(a/b) + (1−a) log((1−a)/(1−b)), with n = number of scenes and μ̂ = mean X (LTT main.tex Prop. "Hoeffding–Bentkus inequality p-values", combining Hoeffding 1963 and Bentkus 2004 as in Bates et al. 2021). ⌈n μ̂⌉ is computed exactly in integers. Expected to have almost no power here because it does not adapt to the small variance of X (INFERRED; MEASURED as the HB issuance rate).

## 7. Label audit: anytime-valid betting confidence sequence (REPORTED method, INFERRED specialization)
For labelled deployment scenes at τ*: K_t(m) = Π_{i≤t}(1 + min(λ̃_i, c/m)(X_i − m)), λ̃ the predictable plug-in with log(1/δ_audit), CS_t = {m : K_t(m) < 1/δ_audit}. **VIOLATED** at the first look with K_t(α) ≥ 1/δ_audit, i.e. α ∉ CS_t, i.e. the lower confidence bound for E[L] = E[X] − α is at least 0. δ_audit = 0.05, c = 3/4.
- **Validity sketch (INFERRED).** If the guarantee holds on deployment (E[X] ≤ α, scenes i.i.d.), K_t(α) is a nonnegative supermartingale (λ ∈ [0, c/α], factors ≥ 1 − c > 0), so by Ville P(ever VIOLATED) ≤ δ_audit over any number of looks: no union bound is needed. Each factor is positive and non-increasing in m, so K_t(m) is non-increasing and CS_t = [l_t, 1]; the reported l_t is the lower end of a 60-step bisection (conservative).
- Nearest prior art for label-based deployment monitoring with confidence sequences: Podkopaev & Ramdas (arXiv:2110.06177, metadata fetched).

## 8. Monitor: two conformal test martingales (REPORTED components, INFERRED combination)
Each score is a function of one scene: primary = number of rows with the advisory's OOD condition; secondary = floor(mean(1 − pmax)) (v0). Each has smoothed conformal p-values over calibration scenes then deployment scenes (θ streams domain-separated) and a uniform mixture of power martingales over ε ∈ {0.1, …, 0.9}, betting from the first deployment scene. REPORTED: smoothed conformal p-values are i.i.d. U(0,1) under exchangeability and power-martingale mixtures are test martingales (Vovk, Gammerman & Shafer, "Testing Exchangeability", ALRW 2nd ed. 2022; Vovk 2021, "Testing Randomness Online"; Vovk et al., arXiv:2102.10439).
- **Validity sketch (INFERRED).** Each CTM is a test martingale in its own filtration, so P(M^A ever ≥ 1/δ_A) ≤ δ_A and P(M^B ever ≥ 1/δ_B) ≤ δ_B; by the union bound P(any alarm) ≤ δ_A + δ_B = 0.01 = v0's δ_s. We did **not** average the two martingales: they are martingales with respect to different filtrations, and an average is not guaranteed to be a supermartingale for a common one.
- Limitations kept from v0: a score-only monitor cannot see shifts that leave both score distributions unchanged; shifted scenes enter the reference and dilute later p-values; the OOD-count score is heavily tied (θ resolves ties validly but costs power).

## 9. Transparency-log anchoring (REPORTED algorithms, MEASURED cross-checks)
Implemented from RFC 9162 §2.1 text (rfc9162.txt sha256 `a0e432ce…`): MTH with leaf = SHA-256(0x00‖d), node = SHA-256(0x01‖l‖r) and empty root SHA-256(""); PATH (§2.1.3.1), inclusion verification (§2.1.3.2), PROOF/SUBPROOF (§2.1.4.1), consistency verification (§2.1.4.2). Outside 0 < first < second we follow the conventions encoded in the transparency-dev test data (first = second needs an empty proof and equal roots; first = 0 fails). Leaf data = the canonical bytes of a receipt envelope. A signed tree head is a DSSE envelope over {schema, log_id, tree_size, root_hash, timestamp_utc}, log_id = keyid of the log key, signed with the same ECDSA P-256 / RFC 6979 code as the receipts.
Verifier checks (REFUSED codes): STH signatures (BAD_SIGNATURE); STH log id equals the log named in the guarantee (NOT_INCLUDED); guarantee inclusion in the issuance STH (NOT_INCLUDED); proof sizes match the STHs and the final STH is an append-only extension of the issuance STH (INCONSISTENT_STH); monitor-chain head inclusion in the final STH (NOT_INCLUDED); head appended after the issuance STH (ORDER). Because each monitor receipt commits to its predecessor, the head's inclusion anchors the whole chain.
Cross-checks (MEASURED in the suite): RFC 9162 §2.1.5 worked example (all four inclusion proofs and three consistency proofs stated in the RFC); transparency-dev/merkle test data at commit `c39fbf350a405f68571bf4c42a6a518eb0a46a4b` (tarball sha256 `ed0c1017…`, Apache-2.0, data only, no code copied): root hashes for sizes 0–8 and every inclusion and consistency probe, positive and negative.
Not proven: the truth of timestamps, freshness, absence of split views without STH gossip, honesty of the log beyond its signature.

## 10. Verdicts
VALID; VIOLATED; or REFUSED:{MISSING_INPUT, SCHEMA, BAD_SIGNATURE, HASH_MISMATCH, RECOMPUTE_MISMATCH, CLAIM_UNSUPPORTED, CHAIN_BROKEN, ORDER, LEAKAGE, NOT_INCLUDED, INCONSISTENT_STH, SHIFT_ALARM, INTERNAL}. Integrity refusals (including the two log codes) always win. Otherwise the first event wins; an alarm at scene j is evaluated before scene j's labels, so a same-scene alarm and violation report SHIFT_ALARM. The verifier never raises; floats, NaN/Infinity, duplicate keys, missing or extra fields and wrong types are refused in every document. Per-scene counts, scores, p-values and audit losses are recomputed by direct definitions; the floating-point kernels are shared and are checked against independent restatements in the tests.

## 11. Pre-registration
`rbcg1_protocol.py` signs a protocol receipt (DSSE, throwaway key, private key never written) that fixes hypotheses, endpoints, families, R and the analysis plan, and binds the sha256 of every file that generates or analyses evaluation data. The experiment refuses to run unless the protocol verifies and the code hashes match. P16 = first 16 hex of the protocol digest is embedded in every evaluation seed family; every shard's log has the protocol envelope as leaf 0; every v1 guarantee binds the full protocol digest.

## 12. References (verified 2026-11-03/04 from szl-vm2)
Metadata were fetched from the CrossRef REST API and the arXiv API; formula statements marked "text" were read in the arXiv e-print LaTeX sources.
- Waudby-Smith, Ramdas. Estimating means of bounded random variables by betting. JRSS-B 86(1):1–27, DOI 10.1093/jrsssb/qkad009 (CrossRef); arXiv:2010.09686 (API; e-print text: hedged capital theorem and predictable plug-in).
- Angelopoulos, Bates, Candès, Jordan, Lei. Learn then Test. Ann. Appl. Stat. 19(2):1641–1662 (2025), DOI 10.1214/24-aoas1998 (CrossRef); arXiv:2110.01052 (API; e-print text: HB p-value and fixed-sequence FWER).
- Bates, Angelopoulos, Lei, Malik, Jordan. Distribution-free, risk-controlling prediction sets. J. ACM 68(6) (2021), DOI 10.1145/3478535 (CrossRef); arXiv:2101.02703 (API).
- Hoeffding. Probability inequalities for sums of bounded random variables. JASA 58(301):13–30 (1963), DOI 10.1080/01621459.1963.10500830 (CrossRef).
- Bentkus. On Hoeffding's inequalities. Ann. Probab. 32(2):1650–1673 (2004), DOI 10.1214/009117904000000360 (CrossRef).
- Ramdas, Grünwald, Vovk, Shafer. Game-theoretic statistics and safe anytime-valid inference. Stat. Sci. 38(4):576–601 (2023), DOI 10.1214/23-sts894 (CrossRef); arXiv:2210.01948 (API).
- Shafer, Shen, Vereshchagin, Vovk. Test martingales, Bayes factors and p-values. Stat. Sci., DOI 10.1214/10-sts347 (CrossRef).
- Howard, Ramdas, McAuliffe, Sekhon. Time-uniform, nonparametric, nonasymptotic confidence sequences. Ann. Statist. 49(2):1055–1080 (2021), DOI 10.1214/20-aos1991 (CrossRef); arXiv:1810.08240 (API).
- Vovk. Testing randomness online. Stat. Sci. 36(4):595–611 (2021), DOI 10.1214/20-sts817 (CrossRef); arXiv:1906.09256 (API).
- Vovk, Gammerman, Shafer. Testing exchangeability. In Algorithmic Learning in a Random World (2022), DOI 10.1007/978-3-031-06649-8_8 (CrossRef).
- Vovk, Petej, Nouretdinov, Ahlberg, Carlsson, Gammerman. Retrain or not retrain: conformal test martingales for change-point detection. arXiv:2102.10439 (API).
- Podkopaev, Ramdas. Tracking the risk of a deployed model and detecting harmful distribution shifts. arXiv:2110.06177 (API).
- Laurie, Messeri, Stradling. RFC 9162, Certificate Transparency Version 2.0 (2021), DOI 10.17487/rfc9162 (CrossRef); text https://www.rfc-editor.org/rfc/rfc9162.txt, sha256 a0e432ce7580c99fcc8faf0cf3f6191f9ce6b4864f949158e020d4ecb295081f.
- Laurie, Langley, Kasper. RFC 6962, Certificate Transparency (2013), DOI 10.17487/rfc6962 (CrossRef).
- transparency-dev/merkle test data, commit c39fbf350a405f68571bf4c42a6a518eb0a46a4b, tarball sha256 ed0c1017ea66fc7e88e67998f7d6b9cc1eef66d30b97b58a1dbfc1aa3a1edd09, LICENSE (Apache-2.0) sha256 cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30.
- Carried from v0 and not re-checked here: RFC 6979, FIPS 186-4, DSSE v1.0.2, the a11oy canonical-JSON convention. Still UNVERIFIED: Vovk, Nouretdinov & Gammerman (2003) "Testing exchangeability on-line" and Ville (1939) originals (not found in CrossRef); Ville's inequality is cited through the 2011 and 2023 sources above.

**Novelty: UNKNOWN.** A CrossRef and arXiv title/keyword pass (results in reports/novelty.json) found no record combining scene-level betting LTT, a betting-CS label audit, a conformal monitor and transparency-log anchoring of signed receipts. Nearest titles (contents not read): arXiv:2608.17176 (durable policy-decision receipts for AI audit evidence), CrossRef 10.52202/079017-1920 (active, anytime-valid risk-controlling prediction sets, 2024), arXiv:2608.14639 (per-field selective risk control). Titles are not contents; we do not claim to be first. **Licence: UNDECIDED (owner choice).**
