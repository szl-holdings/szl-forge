---
license: apache-2.0
title: SZL Forge Lab
emoji: "⚒️"
colorFrom: green
colorTo: gray
sdk: static
app_file: index.html
pinned: false
short_description: Evidence console for SZL Forge models and formulas.
---


<div align="center">
<p>

[![governed](https://img.shields.io/badge/governed-SZL%20Holdings-3af4c8?style=flat-square)](https://huggingface.co/SZLHOLDINGS)
[![Λ](https://img.shields.io/badge/Λ-Conjecture%201%20advisory-d7b96b?style=flat-square)](https://a-11-oy.com)
[![license](https://img.shields.io/badge/license-apache--2.0-7e8aa3?style=flat-square)](https://huggingface.co/spaces/SZLHOLDINGS/szl-forge-lab)

</p>
</div>
<!-- SZL-ESTATE-CARD:v2:START -->
<p align="center"><a href="https://a-11-oy.com/"><img src="https://huggingface.co/spaces/SZLHOLDINGS/README/resolve/main/assets/estate-banner-v2.svg" alt="SZL Holdings — governed, receipted, verifiable" width="100%"></a></p>
<p align="center">
  <a href="https://github.com/szl-holdings/.github/tree/main/doctrine"><img src="https://img.shields.io/badge/doctrine-v11%20LOCKED-0B1F3A?style=flat-square" alt="doctrine v11"></a>
  <a href="https://a-11-oy.com/"><img src="https://img.shields.io/badge/evidence%20wall-LIVE%20%C2%B7%20verify%20in%20browser-3AF4C8?style=flat-square" alt="live evidence wall"></a>
  <a href="https://huggingface.co/datasets/SZLHOLDINGS/szl-lake"><img src="https://img.shields.io/badge/szl--lake-offline%20verifiable-C9B787?style=flat-square" alt="szl-lake offline verifiable"></a>
  <a href="https://huggingface.co/spaces/SZLHOLDINGS/holographic"><img src="https://img.shields.io/badge/estate%20map-holographic-5B8DEE?style=flat-square" alt="holographic estate map"></a>
</p>
<p align="center"><sub>Part of the <a href="https://huggingface.co/SZLHOLDINGS">SZL Holdings</a> governed estate — claims are designed to carry checkable receipts. Verification proves integrity &amp; origin, never accuracy or performance.</sub></p>
<!-- SZL-ESTATE-CARD:v2:END -->

# SZL Forge Lab

SZL Forge Lab is the read-only evidence surface for the SZL sovereign model and
formula Forge. It exposes independently named views for system state, artifact
integrity, fixture evaluation, receipts, formula metadata, scientific-source
policy, and the governed curriculum blueprint.

The static console also renders the canonical
`szl.model-kernel-portfolio/v1` registry. It separates two trained fine-tunes,
one quantized derivative, one learned kernel, and eleven software
kernels/recipes/cards. No Hub card is promoted to a trained-model claim merely
because it is published under the model API.

## Evidence boundary

- `REACHABLE` describes transport availability only.
- `SNAPSHOT` identifies packaged evidence, not live training or provider state.
- A measured local QLoRA run completed on 61 owned doctrine records; the weights remain local and unpublished.
- The raw-model contract result is 1/12. The deterministic governed runtime result is 12/12. Neither is a broad capability benchmark.
- Formula statuses are registry metadata and are not independently re-proven by this Space.
- The curriculum is `BLUEPRINT_NOT_TRAINED`.
- Promotion remains blocked pending independent evaluator, model-owner, and security-reviewer approvals.
- No Space endpoint trains, publishes, promotes, deploys, downloads data, or mutates external state.

## Machine-readable evidence

This Space is now a **static frontend** (`index.html`): your browser fetches the
packaged evidence files, recomputes every artifact SHA-256 against the run-manifest
declarations, and re-verifies each receipt's canonical hash — live, on every load.
The same evidence files are plain HTTP artifacts anyone can fetch and hash:

```bash
for f in run_manifest.json eval_receipt.json training_summary.json \
         thesis_formula_index.json science_source_ledger.json curriculum.json; do
  curl -sL https://huggingface.co/spaces/SZLHOLDINGS/szl-forge-lab/resolve/main/$f | sha256sum
done
```

## Local validation

```bash
python -m unittest discover -s tests -v
python -m py_compile forge_lab.py forge_runtime_contract.py app.py  # legacy Gradio sources, kept for provenance
```

---

<div align="center">

**[🛡️ SZLHOLDINGS on Hugging Face →](https://huggingface.co/SZLHOLDINGS)**   ·   **[a-11-oy.com →](https://a-11-oy.com)**   ·   **[Estate hub — live →](https://szlholdings-szl-estate-live.static.hf.space)**

### Governed AI you can prove.

<sub>SLSA: L1 honest · L2 attested · L3 roadmap. Λ = Conjecture 1 (advisory, never a theorem). Trust ceiling 0.97 — never 100%. Labels honest by default: MEASURED / REPORTED / MODELED / HEURISTIC / UNKNOWN / UNAVAILABLE. locked-proven = exactly 8 {F1,F4,F7,F11,F12,F18,F19,F22}.</sub>

</div>
