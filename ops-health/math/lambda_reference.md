# Λ (Lambda) aggregator: reference for OAC System Health v2

Workstream E, item 3. Owner: E1-math. Date: 2026-09-26.
Implementation: `math/lambda_stdlib.py` (ours, standard library only).
Cross-check tool: `math/tools/lambda_crosscheck.py`. Evidence: `logs/E/lambda_crosscheck.txt`.

Each claim below carries one label: MEASURED (ran in this session, log cited), REPORTED
(source file and revision cited), MODELED (our static analysis or reasoning), CONJECTURE, or UNKNOWN.
All numeric inputs used to test Λ were generated from a seed and are SYNTHETIC.

## 0. Sources (REPORTED)

| Role | Repository @ revision | File |
|---|---|---|
| Kernel semantics (the default here) | szl-holdings/szl-lambda-gate @ 99bc2cf1 (kernel `__version__` 0.2.0) | `torch-ext/szl_lambda_gate/_lambda.py` |
| Python reference (kept for bit-for-bit comparison only) | szl-lambda-gate @ 99bc2cf1 | `tests/lambda_aggregator_source.py` |
| Formal definition | szl-holdings/lutar-lean @ c2de9d76 | `Lutar/Invariant.lean:26` (Λ), `Lutar/Axioms.lean:88-135` (A1 to A5) |
| Conjecture 1 status | lutar-lean @ c2de9d76 | `Lutar/Uniqueness/Identifiability.lean:29-33, 150-161`; README lines 77-90 |

## 1. API (MEASURED: the function exists and was exercised by the cross-check)

```python
from lambda_stdlib import lambda_aggregate, lambda_gate
lambda_aggregate(values, weights=None, **params) -> float   # params: semantics="kernel" | "python_ref" | "lean"
lambda_gate(values, weights=None, threshold=0.5, **params) -> (float, bool)   # ADVISORY
```

An unknown keyword parameter raises `TypeError`. An unknown `semantics` value raises `ValueError`.
v2 must use `semantics="kernel"`, which is the default. The other two modes exist only so the
cross-check can reproduce the Python reference and the Lean definition.

## 2. Formula and domain

**Kernel** (REPORTED from `_lambda.py:160-185`, mirrored in `lambda_stdlib._kernel`):

    w_i  = w_i / Σ_j w_j                  (uniform 1/k when weights is None)
    Λ_w(x) = 0                            if any x_i is non-finite (NaN, ±Inf) or clamp(x_i,0,1) <= 0
           = clamp( exp( Σ_i w_i · log(clamp(x_i, 0, 1)) ), 0, 1 )   otherwise

This is a weighted geometric mean of axis scores clamped to [0, 1]. It is non-compensatory:
one failing axis sets Λ to 0. It fails closed on garbage input: +Inf does not count as a
perfect axis, and NaN does not propagate.

- Domain: any finite or non-finite real is accepted and routed as above. Range: [0, 1].
- Weights: length k, every weight finite and strictly positive; they are then normalised to
  sum to 1. Anything else raises `ValueError` (REPORTED `_lambda.py:100-118`).
- k = 0 raises `ValueError` (REPORTED `_lambda.py:80`).
- dtype (torch op only): float16 and bfloat16 are computed in float32, float32 in float32,
  float64 in float64 (REPORTED `_lambda.py:51-58`). The stdlib version always computes in
  Python float, which is IEEE binary64.
- Gate: `threshold` must be finite and in [0, 1], otherwise `ValueError` (REPORTED
  `_lambda.py:220-230`). The gate is advisory only.

**Lean definition** (REPORTED `Lutar/Invariant.lean:26-29`):
`Λ k x = if k = 0 then 0 else (∏ i, x i) ^ (1/k)` over `NNReal`. Weights are always uniform.
There is no clamp to [0, 1] and no special handling of non-finite values, because `NNReal`
has none.

**Python reference** (REPORTED, `tests/lambda_aggregator_source.py`): k = 0 returns 0.0; it only
requires the weights to have a positive sum; NaN propagates to NaN; +Inf is clamped to 1.
**Do not use it in v2.**

## 3. Edge cases

"Kernel" is both the stdlib default and the torch op. Rows are MEASURED by the cross-check
unless marked otherwise.

| Input | kernel | python_ref | lean mode |
|---|---|---|---|
| empty (k = 0) | `ValueError` | 0.0 | 0.0 (the `k = 0` branch) |
| k = 1, x = [a] | clamp(a, 0, 1) | same | a (not clamped) |
| any x_i = 0 | 0.0 | 0.0 | 0.0 |
| any x_i < 0 | 0.0 (fail closed) | 0.0 | `ValueError` (outside NNReal) |
| any x_i > 1 (finite) | treated as 1 | treated as 1 | not clamped; Λ can exceed 1 (MODELED from the definition) |
| any x_i = NaN | 0.0 (fail closed) | NaN | `ValueError` |
| any x_i = +Inf | 0.0 (fail closed) | treated as 1 | `ValueError` |
| all x_i = c in [0, 1] | c (up to rounding) | c | c |
| weights=None | uniform 1/k | uniform 1/k | uniform (the only mode) |
| a zero weight | `ValueError` | accepted | `ValueError` (no weights allowed) |
| a negative weight | `ValueError` | accepted if the sum is > 0 | `ValueError` |
| a NaN weight | `ValueError` | NaN | `ValueError` |
| wrong weights length | `ValueError` | `ValueError` | `ValueError` |

The run found 14 inputs where kernel and python_ref disagree. All 14 are in the non-finite,
out-of-range, empty or bad-weight rows above (`logs/E/lambda_crosscheck.txt`, section [S]).
This is intended: python_ref does not fail closed, which is why v2 does not use it.

## 4. Numerical cross-check (MEASURED, SYNTHETIC inputs)

Command: `PYTHONUTF8=1 py -3.12 -B math/tools/lambda_crosscheck.py src/szl-lambda-gate`.
Environment: torch 2.10.0+cu130, Python 3.12.10, seed 20260926, 241 cases including edge cases.
Log: `logs/E/lambda_crosscheck.txt`, exit code 0.

| Check | Result |
|---|---|
| [A] stdlib kernel vs the repo torch op (float64) | 236 numeric comparisons, **max abs diff 1.665e-16**; 5 error-parity cases (both raised the same exception type); 0 mismatches |
| [B] stdlib python_ref vs the repo `lambda_aggregator_source` (exact, NaN included) | 240 comparisons, 0 mismatches |
| [C] stdlib (float64) vs torch op on float32 inputs | max abs diff 6.710e-08 (informational only) |
| [D] torch batched 64×9 (weighted, with zero, NaN and Inf rows) vs stdlib per row | max abs diff 1.665e-16 |
| [E] stdlib kernel vs stdlib lean mode, uniform weights, x in (0, 1] | 200 draws, max abs diff 1.110e-16 |
| [P] empirical axiom checks on the kernel, 500 draws, tolerance 1e-12 | 0 violations of A1 (monotone), A2 (homogeneous, t in [0, 1]), A3 (diagonal), A4 (≤ max), and min ≤ Λ |

Verdict in the log: AGREE. Both implementations ran, so the agreement is MEASURED.
Check [P] is a sampling check, not a proof.

## 5. Claimed properties and proof status

**Proof-status rule (CONTRACT §5.5).** We may write "proven" only for a closed lutar-lean
theorem with zero sorries in its dependency cone, backed by pasted `lake build` output.
Lean, lake and elan are not installed on this machine (MEASURED, `logs/E/env_probe.txt`),
so the local build status is **UNKNOWN**. For that reason **this document calls nothing "proven"**.

What we do have:
- **MODELED:** a static dependency-cone scan, `math/tools/lean_claims_scan.py`, output in
  `logs/E/lean_scan_raw.json`.
- **REPORTED CI evidence:** lutar-lean workflow "Lake build (gate + numbers)", run 36239702518
  at c2de9d76, conclusion success. Its log has `Build completed successfully.` at line 1172 of
  `logs/E/ci/lake-build_job_108397777717.log`.

The `#print axioms` column is REPORTED from that CI log. "std3" means only `propext`,
`Classical.choice` and `Quot.sound`, with no `sorryAx`.

| Property (Lean, uniform-weight Λ on NNReal) | Declaration @ file:line | Static class (MODELED) | CI `#print axioms` |
|---|---|---|---|
| A1 monotone | `Lutar.lambda_isMonotone` @ Lutar/Uniqueness.lean:63 | CLOSED-IN-SOURCE (built) | not printed |
| A2 positively homogeneous (degree 1) | `Lutar.lambda_isHomogeneous` @ Uniqueness.lean:72 | CLOSED-IN-SOURCE (built) | not printed |
| A3 diagonal: Λ(c,…,c) = c | `Lutar.lambda_isEgyptianExact` @ Uniqueness.lean:85; `a3_normalize_proof` @ Invariant.lean:54 | CLOSED-IN-SOURCE (built) | not printed |
| A4 Λ ≤ max | `Lutar.Λ_le_max` @ Bound.lean:36; `lambda_isBounded` @ Uniqueness.lean:91 | CLOSED-IN-SOURCE (built) | not printed |
| min ≤ Λ | `Lutar.min_le_Λ` @ Bound.lean:78 | CLOSED-IN-SOURCE (built) | not printed |
| A5 permutation invariant | `Lutar.lambda_isPermutationInvariant` @ Uniqueness.lean:109; `lambda_perm_invariant` @ LambdaPermInvariant.lean:20 | CLOSED-IN-SOURCE (built) | not printed |
| Λ satisfies A1 to A5 | `Lutar.lambda_satisfiesAxioms` @ Uniqueness.lean:113 | CLOSED-IN-SOURCE (built) | not printed |
| Λ ≤ arithmetic mean | `Wave5.MathlibCore.w5_1_lambda_le_arith_mean` @ Wave5/MathlibCore.lean:48 | CLOSED-IN-SOURCE (built) | std3 |
| Schur-concave, 2 axes | `Lutar.Lambda.lambda_two_axis_schur_concave` @ Lambda/SchurConcave.lean:132 | CLOSED-IN-SOURCE (built) | not printed |
| Schur-concave, n axes | `Lutar.Lambda.lambda_schur_concave_n_axis` @ SchurConcave.lean:193 | **CONJECTURE**: stated as an in-repo `axiom` | n/a |
| Conditional uniqueness (Theorem U: modulo ≈Λ under the Identifiability Assumptions) | `Lutar.Uniqueness.TheoremU_LambdaUnique` @ Uniqueness/TheoremU.lean:67; strict `=` only if Anchored: `TheoremU_LambdaUnique_eq` @ :77 | CLOSED-IN-SOURCE (built) | std3 |
| **Unconditional uniqueness under A1 to A5 (Conjecture 1)** | `Lutar.Uniqueness.Conjecture1_LambdaUnique` @ Uniqueness/Identifiability.lean:159 (a `Prop`, with no proof) | **CONJECTURE**, and false as stated (next row) | n/a |
| Conjecture 1 as stated is false: max-aggregator ≠ Λ₂ | `Lutar.Wave4.BlockConsistency.unconditional_lambda_is_false` @ Wave4/LambdaBlockConsistency.lean:192; `Lutar.Round13.maxAgg_ne_Lambda` @ Round13/Lambda_Uniqueness.lean:188 | CLOSED-IN-SOURCE (built) | std3 (the Wave4 one) |
| "Λ is the geometric mean" (A1 to A5 force Λ) | `Lutar.lutar_is_geomean` @ Uniqueness.lean:187; `Lutar.Round13.lambda_unique` @ Round13/Lambda_Uniqueness.lean:232 | **OPEN** (`sorry` in the body; the CI log warns "declaration uses 'sorry'" at log lines 504 and 594) | n/a |
| `Lutar.lutar_unique` | Uniqueness.lean:218 | **CONJECTURE** (depends on the `sorry` in `lutar_is_geomean`) | n/a |

Plain summary (MODELED from the rows above):
- **Λ satisfies A1 to A5, A4, min ≤ Λ ≤ max, and Λ ≤ arithmetic mean.** These are closed in
  source for uniform weights on NNReal, and CI reports a successful lake build.
  Local build: UNKNOWN.
- **Unconditional uniqueness is Conjecture 1.** The repository itself documents it as
  machine-checked false as stated, because the max aggregator also satisfies A1 to A5.
  Nothing about Λ being *the* right aggregator follows from these axioms.
  Theorem U gives only a conditional uniqueness, under extra assumptions.
- `w5_1_lambda_le_arith_mean` is the weighted AM-GM inequality on ℝ: `∏ z_i^w_i ≤ Σ w_i z_i`
  for `w ≥ 0` with `Σ w = 1`. So it also covers the *weighted* kernel's bound Λ_w ≤ weighted mean,
  on the unclamped domain (MODELED reading of the statement).
- **The in-repo axioms are false (MODELED; derived by hand, not machine-checked).** Two in-repo
  axioms, `Wave4.BlockConsistency.A6'_block_consistent` (built) and `Puriq.F23.A6_bisymmetric`
  (not built), claim that *every* aggregator satisfying A1 to A5 factors as a power product.
  `maxAgg` satisfies A1 to A5 but is not Λ 2 (`maxAgg_ne_Lambda`), so these axioms contradict
  the repository's own theorems. As a result, `lambda_unique_under_block` proves nothing, even
  though CI built it (REPORTED; its `#print axioms` at CI log line 622 lists A6′). The full
  per-claim record is in `math/lean_claims.json`.

## 6. What transfers to the v2 kernel (MODELED)

The Lean results are about uniform weights, on NNReal, with no clamping and no fail-closed routing.
- **Uniform weights and finite x in (0, 1]:** the kernel equals the Lean Λ up to float rounding
  (check [E], 1.1e-16). The Lean properties above apply in this region only.
- **Weighted kernel:**
  - A3, A4 and min ≤ Λ still hold for any normalised positive weights (standard facts about
    weighted geometric means; MODELED, not checked in Lean).
  - **A5 does not hold** unless the weights are permuted together with the axes.
  - A2 holds only while c·x_i ≤ 1, because the output is clamped.
- **Fail-closed routing** (NaN, Inf and x ≤ 0 all give 0) is a design choice of the kernel.
  It is not part of any Lean theorem.

## 7. Implications for v2 readiness aggregation (CONTRACT §5.2)

Λ is an advisory, non-compensatory roll-up: in the kernel semantics a single failing axis forces
Λ = 0 (MODELED, from §2 and §3). It is ordered between the minimum and the arithmetic mean:
min ≤ Λ ≤ AM for uniform weights (`Lutar.min_le_Λ` @ Bound.lean:78,
`w5_1_lambda_le_arith_mean` @ Wave5/MathlibCore.lean:48): CLOSED-IN-SOURCE (MODELED),
CI-built at c2de9d76 (REPORTED), local build UNKNOWN. This is not a "proven" claim under
CONTRACT §5.5.

Whether Λ separates known-good and known-broken fixtures better than logical AND, min, or a
weighted mean is an **empirical question for workstream F**. It is UNKNOWN here. If the
mutation suite shows no measurable improvement, Λ must be recorded as REJECTED together with
the numbers.
