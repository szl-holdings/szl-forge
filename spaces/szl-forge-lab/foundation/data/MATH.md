# Source dependence and the price of confirmation

The original contribution in this build is its particular implementation, controlled comparison and retained evidence. The Bayesian identities, source-bias model and value-of-information decision rule are established ideas; the build makes no claim of inventing them.

Let four unknown target labels be `X = (X0, X1, X2, X3)`, with each label in the integers modulo 16. Let `B` be one persistent cheap-source offset. A cheap observation of group `g` has response `Y = Xg + B + E (mod 16)`. A reference observation has response `Y = Xg + E (mod 16)`. The assumed fresh error is zero with probability 0.98 and each nonzero offset has probability `0.02/15`.

Repeated cheap reports share `B`. Without a reference, adding the same offset to all target labels while subtracting it from `B` can preserve cheap observations. Repetition reduces uncertainty due to fresh noise, but this confounding remains. A reference can constrain `Xg` without using the cheap-source offset and thereby revise beliefs about other groups through the shared `B`.

The prior assigns probability 0.8 to zero bias and distributes the remaining probability uniformly over fifteen nonzero biases. It mixes two world components: 0.8 mass on 76 supplied unchanged/global/local patterns and 0.2 mass on independent group labels. Each label in the independent component is zero with probability 0.8. All four-label combinations have support, but that does not mean the system discovers their structure.

The compact posterior stores `W[h,b]` for the structured component and `Q[b] × product_g U[b,g,xg]` for the independent component. Conditioning a group observation multiplies its relevant likelihood, updates the shared bias weights and updates the selected group's conditional label distribution. The other groups become dependent through the bias mixture even though their conditional product representation remains compact.

For target weights `w_g`, the current optimal expected classification accuracy is:

`A = sum_g w_g max_x P(Xg = x | history)`.

For affordable action `a`, the exact one-step acquisition utility is:

`V(a) = sum_y sum_g w_g max_x P(Xg = x, Ya = y | history) - A - price(a)`.

Using the joint distribution folds the predictive response probability into the expression. The analytic policy selects the largest positive utility; otherwise it stops. It does not solve a multi-step optimal experiment-design problem. The dense review enumerates all `16^5 = 1,048,576` target/bias states to check the compact computation.

The independent-bias comparison uses the same initial per-reading likelihood averaged over the same bias prior, but it redraws the bias for each cheap reading. Thus the comparison changes dependence assumptions without substituting a different one-reading error rate. Its final predictor differs as well as its acquisition decisions; the learned-versus-independent-bias comparison is a comparison of full systems.

The neural selector receives 208 raw public features and predicts eight action utilities. Its supervised targets come from the source-aware analytic controller. Its final label predictions still come from the explicit Bayesian engine. Comparing it directly with the analytic source-aware controller measures the consequences of replacing analytic acquisition with this learned approximation. A learned model's victory over a weaker controller would not establish a victory over its own teacher or the strongest available control.

For evaluation, utility is **realized** target-weighted correctness minus paid price. This differs from the model's expected utility when its likelihood assumptions are wrong. Reference outliers and changing bias deliberately test that gap. Confidence, coverage and confidently wrong mass are separate measurements; a lower confidence score alone is not a successful correction.

The research motivations and precise source links are in `research/GLOBAL_RESEARCH.md` and `research/AGENCY_RESEARCH.md`. Those publications did not specify or endorse this exact simulator.
