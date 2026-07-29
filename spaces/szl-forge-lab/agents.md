# SZL Forge Lab agent contract

## Purpose

Inspect the Forge's reproducibility, evaluation, formula, source-policy, and
curriculum evidence without implying that the snapshots are live model state.

## Read-only tools

- `GET /config` — Gradio interface metadata.
- `GET /gradio_api/info` — named endpoint descriptions.
- Named Gradio endpoints: `/status`, `/integrity`, `/evaluation`, `/receipt`,
  `/formulas`, `/formula`, `/sources`, `/source`, `/curriculum`, and
  `/curriculum-stage`.

## Evidence rules

The Space is a snapshot showcase. `RUNNING` and `REACHABLE` mean transport
availability only. The local training run is measured, but the weights are not
published. Raw-model policy compliance is 1/12 while the deterministic governed
runtime is 12/12; neither is a broad capability benchmark. A model is
not claimed releasable unless measured weights and a run receipt are present, and
a formula is not claimed proven unless an independent formal checker passes.

## Limits

The Space performs no external mutations, deployment, model promotion,
training, publication, or scientific-data ingestion. Source-policy records are
not legal advice and must be rechecked at artifact-acquisition time.
