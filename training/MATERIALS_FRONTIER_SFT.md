# SZL Materials Frontier — five challenge-linked scientific SFT lanes

**Status: original synthetic curriculum candidate, not a trained model, not
independent evaluation, not a scientific discovery and not a contest entry.**

## Grounded source and benchmark boundaries

The user supplied a November 2026 Discovered Materials Grand Challenges
page and an Astra/Sol/Fable capability scoreboard. The five problem themes
and the **official criteria** were checked at:

- [01 Cuprate superconductivity](https://grandchallenges.discoveredmaterials.com/cuprate-superconductivity/):
  independent Tc(p), d-wave gaps, pseudogap, magnetic and transport
  predictions from one microscopic framework across cuprate families.
- [02 Interface electrical properties](https://grandchallenges.discoveredmaterials.com/interfacial-electrical-properties/):
  geometry, process history, oxide thickness/composition, energy-resolved
  interface-trap density Dit(E), and fixed charge Qf across multiple materials.
- [03 Quantum chemistry](https://grandchallenges.discoveredmaterials.com/speed-accuracy-barrier/):
  genuine exchange-correlation functionals, GMTKN55 and independent reaction
  errors, with reproducible timing and accuracy-vs-cost.
- [04 Light-driven materials](https://grandchallenges.discoveredmaterials.com/tuning-materials-with-light/):
  nonequilibrium electron-electron, electron-phonon and phonon-phonon
  couplings; model limits and experimentally measured material traces.
- [05 Thermal interfaces](https://grandchallenges.discoveredmaterials.com/interfacial-heat-transport/):
  predictions of G(T) and RK(T), transfer across solid interfaces, AMM/DMM
  baselines and truly unseen materials.

The provided comparison screenshot names ten **capability targets**. It
does not provide benchmark-data rights or independently replicated scores:

| Training-adjacent ability | Benchmark names from screenshot |
| --- | --- |
| Scientific terminal agency | Terminal-Bench-Science; Agents' Last Exam |
| Scientific method | Vals Science Index |
| Source-grade coding | FrontierCode 2.0; ModelDial |
| CAD and spatial design | BenchCAD 2.1 Core; AA-3D |
| Broad reasoning | Artificial Analysis Intelligence Index |
| Legal reasoning | Harvey LAB-AA |
| Business workflows | GDPval v2 |

**No benchmark questions, answers, hidden tests, or screenshot score labels
are ingested by this curriculum.** A future comparison must separately
obtain lawful benchmark access, effort-level parity, task isolation, model
revision, sample size, cost and honest uncertainty.

## The original cross-domain mechanism

Each lane combines three procedural skills, all with original *toy*
mathematics and an exact standard-library oracle:

1. Calculation with explicit quantities and units.
2. Conservation, antisymmetry or cycle-closure invariants.
3. Evidence-bound refusal of unsupported scientific conclusions.

Cuprate arithmetic computes a synthetic d-wave gap. Semiconductor
arithmetic solves the toy positive root of x^2 + A x = B t. Quantum
chemistry arithmetic checks reaction-energy differences and Hess cycles.
Ultrafast problems perform simple energy accounting, **not** NEGF.
Thermal examples use DeltaT=q/G and inverse resistance, **not**
a predictive phonon-transfer model. A correct answer cannot establish
the actual grand challenge success criteria.

This is a falsifiability and uncertainty-literacy training track,
not a newly discovered physical theory. Its objective is to prevent the
model from confusing algebra, fitted curves and genuine out-of-sample
experimental predictions.

## Actual private training input now materializable

The deterministic generator is training/materials_frontier_seed.py.
It produces five lanes times three skills with disjoint synthetic
parameter instances:

| Split | Rows | Role |
| --- | ---: | --- |
| train | 240 | Candidate chat SFT training examples |
| dev | 60 | Developer-only diagnostic cases |
| diagnostic | 60 | Additional original procedural diagnostics |

Both dev and diagnostic share generator templates with training.
**They are not independently sealed holdouts** and cannot establish
scientific generalization. Train/dev inputs are checked for duplicates
and overlap with the existing Forge validator,
tools/validate_sft_dataset.py. No proprietary experimental values,
published challenge answers or literature datasets are embedded.

From a clean source checkout with Python 3.10+ and pytest installed,
the reproducible, credential-free commands are:

    python -m pytest -q tests/test_materials_frontier_seed.py
    python training/materials_frontier_seed.py --output-dir /new/private/materials-review

The output directory must not already exist and must be outside the
source checkout. The generator writes train.jsonl, dev.jsonl,
diagnostic.jsonl, and manifest.json with byte SHA-256 and source
identity. It does not load a model, contact HF/SaladCloud, launch GPU
training or publish any content. The output remains private until a
source, data-rights and model admission permits wider circulation.

## Model-training admission (existing Forge authority unchanged)

Before passing the train.jsonl candidate to existing
training/launch_contract.py and training/train_wave.py, require:

- A reviewed source merge revision plus exact generated-code SHA-256,
  and an audited first-party synthetic data-rights statement.
- An approved base model/student with immutable Hub revision, license,
  declared tokenizer and independently sealed material-family holdouts.
- Source-overlap and cross-family decontamination, quantitative
  scientific accuracy baselines, units, calibration, negative controls,
  honest abstention and measurement uncertainty.
- Explicit GPU owner and cost budget: Massed for admitted inference,
  separately controlled SaladCloud for interruptible training/eval.
  Do not compete with active jobs, reset optimizer state or treat
  desired replicas as evidence of running trainers.
- Existing Forge SFT launch contract including 32-example floor,
  heldout and admission receipt; training checkpoints/opaque-byte
  integrity, durable backup, evaluation and protected publisher review.
- Genuinely external validation on held-out published experiments
  under their allowed licenses; benchmark confidence intervals,
  runtime, latency and energy readings before a promotional claim.
- No HF model or PyPI publication, Cloudflare monetization, public
  GPU routing, or competition submission until separately approved.

The authorized Massed provider readback on 2026-11-18 01:23 UTC
reported Gate1 STOPPED, with CUDA still serving 45/45 verified model
placements over the private operator SSH/loopback path. That earlier
status is not a guarantee of future capacity and does not authorize
a new SFT job. The existing Salad snapshot had incomplete project/
group coverage. These are reasons to stage the data before scheduling
training rather than creating an unowned paid experiment.

## Competition rights, review, and submission

The [Discovered Materials official rules](https://discoveredmaterials.com/grand-challenges/official-rules)
were last updated 2026-11-05. They say entrants retain their IP but
submission grants the sponsor an extremely broad perpetual,
irrevocable, royalty-free, sublicensable license with a covenant
not to sue; submissions are not confidential. Independent technical,
patent and licensing review is required before disclosing original
proprietary breakthroughs or submitting. No email is sent.

Prize eligibility, contest success, benchmark performance, scientific
discovery, model quality and production deployment are NOT established
by the new toy generator or its unit tests.
