# SZL mathematics × Edge AI — source-bound mathematical evolution

**Status:** original numerical decision proposal, offline-only; no Lean theorem, empirical benchmark, energy measurement or deployed model was created by this source.

## Source authority: reuse, do not rebrand open results as proven

- [szl-formulas](https://github.com/szl-holdings/szl-formulas/tree/6e6c18ade01acec627eed04645e6e9c661916cb4) exact main; locked-reference Git blob `61eca5b6d531ad758ff557b881ff49ec2b9af3ea`.
- [lutar-lean](https://github.com/szl-holdings/lutar-lean/tree/80e35f859801d80fce7cf81e4d9b0c8bdf91cfea) and [szl-papers](https://github.com/szl-holdings/szl-papers/tree/39c228dc3a29c6cfd831d43be6a8531e718692ac) exact source revisions.
- The locked source names **F1, F4, F7, F11, F12, F18, F19, F22** as Lean-carried statements, with explicit limitations: no Python refinement proof, no automatic informal semantic proof, no deployment authority, no independent fresh Lean4checker run. Λ uniqueness remains a conjecture **refuted as stated**; do not claim it as a theorem. MC1, F24 and F25 remain candidates, F26 empirical conjecture, and F23 retains its original Λ identity.
- The [Free-Energy Certification preprint](https://doi.org/10.5281/zenodo.24025918) proposes a conditional PAC-Bayes model certificate; the code here **does not apply PAC-Bayes** without a defined prior/posterior and a valid KL term. It uses a conservative one-sided Hoeffding *formula*, whose coverage requires independently justified IID and pre-registration assumptions.

## The SZL-specific math we implement

Given fixed heldout sample size `n`, baseline `b`, candidate `c`, and metrics measured on identical authorized scenarios, define:

`U_r = min(1, errors_c/n + sqrt(log(2/δ)/(2n)))`.

`U_s = min(1, violations_c/n + sqrt(log(2/δ)/(2n)))`.

The two terms are *conditional* Hoeffding upper bounds with Bonferroni split `δ/2`; they are **not** verified guarantees for adaptively selected, dependent or distribution-shifted samples. The safety bound remains positive when zero violations were observed.

Define strict nonregression as the componentwise partial order over `(errors, flash, RAM, p99 latency, joules/decision)`:

`c ≼ b ⇔ ∀k: c_k ≤ b_k`; improvement requires `∃k: c_k < b_k`.

A candidate is rejected if a required rights, receipt, source, data, physical-target or safety gate fails, regardless of its score. No favorable scalarization overrides a hard constraint. A risk upper bound must be below the pre-registered threshold. Flash, RAM, p99 and energy must meet their own ceilings.

Only **after** all gates, calculate an **advisory** five-axis canonical Λ weighted geometric mean:

`Λ_w(x) = exp(Σ_i w_i log(x_i))`, `w_i > 0`, `Σ_i w_i = 1`, with exact zero-veto.

Here axes are normalized accuracy and four resource headrooms; uniform weights are **a declared policy choice**, not learned or a theorem. Λ is not used as the deployment authorization. A more rigorous future model would use uncertainty-aware constrained Pareto optimization, independent multiple-comparison control, and actual device energy measurements.

## Where the original SZL formulas connect

- **F1 replay hash determinism:** deterministic canonical JSON hash for a *claimed* input; hash alone is not signature or independently verified receipt.
- **F4 Khipu DAG:** future evidence lineage must use strictly backward edges and explicit parent references. This source does not implement DAG validation.
- **F7 Chaski FIFO:** future asynchronous board test and artifact delivery must preserve order; not implemented in this calculator.
- **F19 budget monotonicity:** motivates explicit hard resource budgets; does not prove this Python resource estimator.
- **F22 append-only emission:** future admission receipts must be append-only; not implemented in this calculator.
- **F11, F12, F18:** retained in source authority but not asserted as relevant to this first numerical candidate.

## Integration plan

`szl-forge` PR #762 estimates MLP MAC/flash/RAM/latency from supplied specs; this new source is **separate** and requires actual measured candidate/baseline metrics, rights, and independent receipt claims. The two must be linked only through an exact versioned contract after both pass review. No ad hoc cross-repo source copy.

`szl-energy-attest` owns measured device-total joules and UNAVAILABLE abstention. `szl-quant-witness` owns quantization comparisons and weight identity. `szl-papers` and `lutar-lean` own formal statement/proof evidence. `szl-forge` owns model-lab admission. `a11oy` may consume an admitted, signed artifact only after runtime and rights gates. HF, Cloudflare and PyPI are downstream projections, never proof authorities merely by publishing.

**Next verified gates:** independent fixture-to-device benchmark, target compiler binary SHA, actual float32/int8/int4 accuracy and calibration, matched source/weights, independent evaluator and rights, physical power/latency measurements, fixed protocol with statistical correction, cross-repo receipt signatures, and owner-reviewed source-to-runtime deployment.

The module returns `STRUCTURALLY_ADMISSIBLE_UNVERIFIED` even for a positive synthetic fixture. It does **not** certify that external evidence flags are true, that the Hoeffding assumptions hold, or that any SZL formula was formally proved anew.
