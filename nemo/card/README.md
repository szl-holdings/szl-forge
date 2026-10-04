---
license: apache-2.0
tags:
- software
- recipe-conformance
- historical
- szl-holdings
- not-a-checkpoint
---

<!-- SZL-CARD-PRESENTATION:v1 -->
<p><a href="https://huggingface.co/spaces/SZLHOLDINGS/szl-command-lab"><img src="https://raw.githubusercontent.com/szl-holdings/.github/main/profile/assets/szl/logos/szl_mark_holographic.svg" alt="SZL Holdings" width="112" /></a></p>

# szl-nemo

Inspect historical recipe-conformance scripts and receipt records.

**Artifact:** Historical software and receipt mirror · **Stage:** Scorer quarantined

[Explore in Command Lab](https://huggingface.co/spaces/SZLHOLDINGS/szl-command-lab) · [Build](https://github.com/szl-holdings/szl-nemo) · [Evidence](https://github.com/szl-holdings/szl-forge/blob/2b4d7a0f69e68d60cb0f35f09c9bac837d66acdc/nemo/card/README.md)

## Before you use it

- The scorer and its generator remain quarantined; no approved loadable checkpoint or qualified kernel-loading path is established here.
- Newer deterministic witness APIs remain source-only until their separate source-bound Hub publication is verified.
- Historical synthetic fidelity and paraphrase figures do not establish a current runtime, generative model, or safety guarantee.

<details>
<summary>Technical details and evidence</summary>

<!-- SZL-CARD-TECHNICAL:v1:START -->

<p align="center"><img src="https://raw.githubusercontent.com/szl-holdings/szl-forge/main/nemo/card/holo-banner.svg" alt="Historical recipe-conformance records illustration" width="100%"/></p>

# szl-nemo

**Historical recipe-conformance records; the scorer and its generator remain quarantined.**

This model-style Hub listing is a historical scripts-and-receipt mirror. It
contains no approved loadable checkpoint, Nemotron weights, generative model,
or qualified kernel loading path. No `joblib.load`, retraining, or future scorer
publication is recommended by this card.

## Artifact and source identity

- Reviewed [Hub snapshot](https://huggingface.co/SZLHOLDINGS/szl-nemo/tree/56c429886bf5ea98826bf99b27313626d70f01cc): `56c429886bf5ea98826bf99b27313626d70f01cc`.
- Reviewed [canonical source](https://github.com/szl-holdings/szl-nemo/blob/3860c3cb92cf5dee7f7f15635e878713330b219c/README.md): `szl-holdings/szl-nemo@3860c3cb92cf5dee7f7f15635e878713330b219c`.
- Card authoring for this mirror is a separate Forge publisher input: `nemo/card/README.md`. A card-only change does not ship the complete newer canonical package.

The historical TF-IDF/logistic-regression scorer and generator are quarantined.
Historical scripts, rule-check context, `Modelfile` prompt text, and receipt
records remain inspectable. An upstream Ollama-tag observation in
`BASE_MODEL_MANIFEST.json` is not a weight artifact or an immutable base binding.

Current canonical development describes deterministic R1–R5 doctrine checks
and E1–E10 envelope witnesses. Those newer APIs, package files, and test claims
must remain source-only until an exact source-bound Hub publication is verified.
This older mirror does not establish complete current-package parity.

## Retained historical evidence

The unsigned [TRAINING_RECEIPT.json](https://huggingface.co/SZLHOLDINGS/szl-nemo/blob/56c429886bf5ea98826bf99b27313626d70f01cc/TRAINING_RECEIPT.json)
reports the following historical scorer experiment:

| Field | Receipt-reported value and scope |
|---|---|
| Training time | `2026-07-21T02:52:42Z` |
| Host/software | Replit 2-vCPU; scikit-learn `1.9.0` |
| Seed | `20260721` |
| Checker-labelled rows | `5620`: `2638` conforming and `2982` violating; 80/20 stratified split |
| Fidelity against rule checker | `1.0`; reported recipe-conformance experiment |
| Unseen paraphrases | `0.8333`, `N=12`; small historical sample |
| Named scorer digest | `model.joblib`: `d3f0cd7bebbb73fedbc9a0f098148f46f5834bf9184b43cd29b07f286a77ff5b` |
| Scorer artifact in reviewed mirror | **ABSENT / UNAVAILABLE**; the named digest does not establish a published load path |

These are preserved receipt-reported results, not newly measured model quality,
an LLM evaluation, a Nemotron benchmark, an independent benchmark, or a current
latency result. The scorer cannot be replayed from the reviewed Hub snapshot
because the named artifact is absent. Its historical digest must not be
interpreted as permission to recover, retrain, or load that quarantined scorer.

The [PROMOTION_READINESS_AUDIT.json](https://huggingface.co/SZLHOLDINGS/szl-nemo/blob/56c429886bf5ea98826bf99b27313626d70f01cc/PROMOTION_READINESS_AUDIT.json)
is retained at the same reviewed revision. This documentation correction makes
no new promotion, runtime qualification, release approval, or deployment claim.

## Intended use and limits

Inspect historical recipe-conformance evidence and separately reviewed canonical
software. Keep a deterministic policy decision authoritative outside any
historical learned scorer.

- No approved joblib scorer, generative checkpoint, hosted model, or `from_pretrained` path is established here.
- No 10 ms latency, comparative capability, safety guarantee, or ecosystem-wide novelty claim is made.
- Receipt-reported fidelity `1.0` and paraphrase result `0.8333` on `N=12` remain historical context.
- Hashes identify records or named artifact bytes; they do not authenticate authorship, measurement, or correctness.
- The September 30, 2026 review read immutable text and records. No weights were downloaded, no generator/scorer or inference was executed, and no signature verification or runtime test was repeated.
- A card-only publisher changes the README and its banner; it does not publish the newer deterministic witness package or remove its release gates.

## License

Apache-2.0 is declared for the repository's SZL files. A [LICENSE](https://huggingface.co/SZLHOLDINGS/szl-nemo/blob/56c429886bf5ea98826bf99b27313626d70f01cc/LICENSE)
is present in the reviewed mirror; this review does not establish upstream or
downstream license coverage. No NVIDIA weights are republished by this card.

Lambda uniqueness remains Conjecture 1, open and advisory. Historical evidence
and failed outcomes remain intact.

<!-- SZL-CARD-TECHNICAL:v1:END -->
</details>
