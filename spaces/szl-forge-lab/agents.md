# SZL Forge Lab agent contract

## Purpose

Inspect the Forge's reproducibility, evaluation, formula, source-policy, and
curriculum evidence without implying that the snapshots are live model state.

## Read-only static surface

The deployed Space uses `sdk: static` with `app_file: index.html`. Retained
Python and Gradio source files do not define the deployed HTTP API.

- `GET /` - platform entry redirect to the static page.
- `GET /index.html` - evidence console.
- `GET /run_manifest.json` - packaged run manifest.
- `GET /eval_receipt.json` - packaged evaluation receipt.
- `GET /training_summary.json` - packaged training summary.
- `GET /thesis_formula_index.json` - formula metadata snapshot.
- `GET /science_source_ledger.json` - source-policy snapshot.
- `GET /curriculum.json` - curriculum blueprint.
- `GET /model_portfolio.json` - model and kernel evidence inventory.

Initial page load reads these seven JSON files, checks packaged evidence hashes
locally in the browser, and reads the public Hugging Face Space metadata API to
display its repository revision. That revision is not a running-process
attestation. The page's revision and byte-hash controls do not publish receipts.

Gradio metadata paths `/config` and `/gradio_api/info`, and named Gradio
prediction endpoints, are not available APIs of this static deployment. Do not
invoke prediction or queue endpoints to inspect its evidence.

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
