# Cosmos 3 × SZL Forge — physical-world research and admission

**Status: research + source-only evaluation admission; no checkpoint copied, model trained, synthetic dataset generated, or physical system controlled.** Recorded 2026-11-18 UTC.

## Research and reproducible upstream references

- [NVIDIA Cosmos3 collection](https://huggingface.co/collections/nvidia/cosmos3), [Cosmos code](https://github.com/NVIDIA/cosmos), [Cosmos Framework](https://github.com/NVIDIA/cosmos-framework), [technical report](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf): mixture-of-transformers (causal text/reasoning plus diffusion image/video/audio/action pathways), 3D multimodal RoPE, shared multimodal attention; Nano ~16B, Super ~65B, Edge ~4B. Model cards and OpenMDW-1.1 rights must be reviewed per asset, including derivative/data terms.
- [Interactive Cosmos Simulation](https://research.nvidia.com/labs/cosmos-lab/cosmos-simulation/technical-report.pdf): 2026-11-04 framework for streaming, action-conditioned virtual worlds. Evaluate against an **external** simulator's independent physical state; generated video alone is not a physical law oracle.
- [Meta V-JEPA 2](https://ai.meta.com/research/publications/v-jepa-2-self-supervised-video-models-enable-understanding-prediction-and-planning/): action-conditioned latent predictions and planning with limited robot interaction data. Learn *representation-space consistency* instead of always rendering every candidate future.
- [Google DeepMind Genie 4](https://deepmind.google/blog/genie-4-a-new-generation-of-world-simulation/): real-time, steerable multimodal simulation, but closed research preview and acknowledged long-horizon/object permanence limitations. Learn the interactive evaluation protocol, not an unavailable implementation.
- [DreamerV3](https://github.com/danijar/dreamerv3): actor-critic on imagined latent rollouts. Counterfactual policies need calibration against heldout real transitions; imaginary reward alone cannot qualify a robot.
- [Physical Intelligence OpenPI](https://github.com/Physical-Intelligence/openpi): π0/π0.5 flow-action policies and relative action representation; require embodiment calibration and human safety interlocks before physical deployment.

## SZL-specific synthesis: a *counterfactual evidence lattice*

Rather than importing a monolithic Cosmos model into the existing Forge and declaring success, keep **three independent, pinned lanes**:

1. **Observe:** immutable real/synthetic scene data, consent/rights, camera geometry, time synchronization, embodiment and uncertainty. Never silently relabel generated clips as observed reality.
2. **Imagine:** candidate action-conditioned futures from Cosmos3 or another world model; pin exact base, adapter, dataset, inference code, simulator version, seed, units, frame rate, compute, model license and provenance. No automatic execution.
3. **Falsify:** compare with heldout ground-truth trajectories, external physical constraints and an independent model family (JEPA-style latent consistency); record counterfactual disagreement, physical violation, out-of-distribution abstention, latency and cost.

**Proposed ranking (research hypothesis, not a measured model):**

`J(a) = E[R(a)] - λ_phys E[E_physics(a)] - λ_unc U(a) - λ_cost C(a)`.

Here `E_physics` must use independent measured physical state and unit-calibrated error, `U` is ensemble/distributional uncertainty and `C` is measured compute or energy cost. Thresholds and lambdas require prospective calibration on an immutable heldout set. Never optimize generated video realism as a proxy for actual physical safety.

**Candidate acceptance** requires paired scenario IDs/seeds/split, no leakage, baseline comparison, nonregression in normalized physics error and task success, zero observed safety violations, minimum sample coverage, rights and independently reviewed evaluation. Passing a synthetic offline fixture only establishes that the checker executes. Zero violations among 30 trials does **not** statistically prove zero risk; a binomial upper confidence bound and domain-specific safety validation remain required. No autonomous physical action without separate operator authority.

## Hardware and implementation

The existing SZL RTX A6000 48 GB nodes are **not assumed capable** of running full Cosmos3 Nano 16B video inference or Super 65B. Edge ~4B may be a future candidate only after official card hardware requirements, available VRAM, model weights rights, pinned CUDA/Triton environment and small offline inference canary are verified. Do not displace existing Nemotron or active training or allocate SaladCloud replicas without budget and owner lease.

The additive `tools/cosmos3_world_eval_gate.py` is a **standard-library-only, offline** admission verifier for externally supplied normalized, paired physical-error and success measurements. It never downloads or executes a model, fabricates results, creates training jobs, changes Cloudflare or publishes HF assets. Its `STRUCTURALLY_ADMISSIBLE_UNVERIFIED` state is *not* `TRAINED`, `DEPLOYED` or `QUALIFIED`. Synthetic fixture results must not be presented as experimental findings.

**Next verified production path:** source/rights manifest → pinned data and weights → owner GPU capacity and budget preflight → independent real/synthetic evaluation with external simulator → measured calibration and uncertainty → fail-closed benchmark and review → separate deployment and safety acceptance. Publish only sanitized, signed and independently read-back evidence.

## Limitations

The current evaluator checks scalar normalized physical error and binary success, not detailed trajectory geometry, force/contact units, object permanence, calibration, confidence intervals, physics law compliance or real-world robot safety. These are mandatory next gates for any physical-world qualification. No claim of improved SOTA is made.
