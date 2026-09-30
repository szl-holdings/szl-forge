# Measured results — Foundation Confirmation v0.4

**The shared-fault gain passed, but the overall registered gate failed because clean-sensor utility regressed beyond its allowed limit.** The learned model did not establish an advantage over the strongest control.

Completed 5,184 policy rollouts across 576 distinct synthetic worlds: 96 per challenge family, six deterministic/random controls and three separately trained learned selectors. All outcomes below come from this frozen evaluation, not the interactive interface.

## Registered result

Against the independent-bias system, learned shared-fault utility improved by **0.879378** (paired 95% bootstrap interval **[0.825416, 0.921410]**). This exceeds the registered 0.03 gain threshold and positive lower confidence bound.

Clean-sensor utility changed by **-0.019675** (95% interval [-0.033224, -0.007540]). The allowed mean regression was 0.01. Therefore `overall_pass=false`.

The strongest observed shared-fault control was **paid reference only**. Learned utility minus that control was **-0.009558**, interval **[-0.035228, 0.010462]**. The strongest control is reselected inside each bootstrap replicate. Against source-aware analytic acquisition, the learned difference was -0.001827, interval [-0.019120, 0.013454]. Neither comparison demonstrates learned superiority.

## Shared-fault condition

Accuracy and confident error are target-weighted proportions. Utility subtracts incremental observation cost; it is not a percentage. Confidence threshold is 0.8. Learned rows average three model seeds on the same 96 worlds.

| Policy | Accuracy | Utility | Paid cost | Confidently wrong | Mean paid references |
|---|---:|---:|---:|---:|---:|
| No extra evidence | 33.43% | 0.334287 | 0.000000 | 2.22% | 0.000 |
| Independent bias | 2.17% | 0.019598 | 0.002083 | 97.77% | 0.042 |
| Paid cheap only | 32.29% | 0.311979 | 0.010938 | 2.08% | 0.000 |
| Random mixed | 88.14% | 0.781437 | 0.100000 | 2.51% | 1.135 |
| Source-aware analytic | 96.38% | 0.900802 | 0.063021 | 2.90% | 1.219 |
| Paid reference only | 97.36% | 0.908533 | 0.065104 | 1.92% | 1.302 |
| Learned selector | 95.90% | 0.898975 | 0.060069 | 3.12% | 1.201 |

The poor independent-bias result is a targeted model-mismatch result: all four inherited reports can share the same incorrect offset. It does not represent a general comparison with deployed language models. The learned and source-aware systems share a programmed predictor, while the independent-bias system changes both inference and action selection.

## Every challenge family

| Condition | Learned accuracy | Learned utility | Analytic utility | Best control utility | Best control |
|---|---:|---:|---:|---:|---|
| clean | 99.16% | 0.963280 | 0.977715 | 0.992558 | Paid cheap only |
| shared_bias | 95.90% | 0.898975 | 0.900802 | 0.908533 | Paid reference only |
| independent_outliers | 88.85% | 0.849032 | 0.930931 | 0.958905 | Independent bias |
| reference_outliers | 84.99% | 0.784946 | 0.795119 | 0.795119 | Source-aware analytic |
| bias_drift | 95.78% | 0.896905 | 0.914319 | 0.914319 | Source-aware analytic |
| compositional | 97.70% | 0.911456 | 0.932717 | 0.933395 | Paid reference only |

These are descriptive secondary comparisons. The learned selector underperforms the independent-bias control on independent cheap outliers and the reference-only control on compositional changes. When references have 25% outlier probability, learned confident error rises to 7.34%. A separate source is not guaranteed correct.

## Actual training

Three 26,792-parameter models trained from fresh weights on CPU using 4,096 synthetic states and 768 development states. Each ran 1,600 optimizer updates; only development decision regret selected checkpoints. The initial and selected SafeTensors, hashes and fingerprints are retained. All 26,792 selected parameters changed for each seed.

| Seed | Selected step | Initial development regret | Selected development regret |
|---|---:|---:|---:|
| 17 | 400 | 0.098513 | 0.004565 |
| 23 | 400 | 0.099349 | 0.004561 |
| 41 | 200 | 0.025896 | 0.004456 |

Evaluation completed in one uninterrupted attempt: no committed evaluation prefix was resumed. Post-evaluation in-memory fingerprints matched the admitted checkpoints. Full per-episode observations, chosen utilities, predictions, probabilities and hidden scoring labels are retained in `evaluation.jsonl`; hidden labels are used only outside the policy for scoring.

## Interpretation and next hypothesis

This supports the bounded mechanism: modeling persistent source dependence and buying reference observations can correct shared errors in these synthetic worlds. It also shows why the next experiment should target unnecessary verification and robustness to independently corrupted observations. That future change needs a newly registered protocol and fresh evaluation data. This release is not retuned after seeing its challenge outcomes.

The priors, likelihoods, finite world support and acquisition teacher are supplied by the implementation. Only the acquisition approximation is learned. Results do not demonstrate AGI, open-domain factual reliability, a new foundation language model, or global novelty. Bootstrap intervals resample 96 episode clusters, averaging learned seeds within each episode; they do not establish general population guarantees.

![Frozen synthetic benchmark](evidence/benchmark.png)

Protocol SHA-256: `dd0c469401a71c73d196320260d27f58d2e9cf7a8be7bd78ece3cad1a6cca10e`.
Evaluation SHA-256: `6a35a3473bcac52a2fd473dbb08f9682042a6fa3b8fc2be1c32223ea74ccef7c`.
Training summary SHA-256: `c4e40e509c637fc209fb3ee7b5943557dcba3db13c8a277538bc5f4e60ebf9a1`.
