# SZL Evidence Lab operational package

Run local hybrid retrieval, inspect exact source spans, compare an optional identifier filter, and verify the running API against a frozen evaluation. The original four model/evaluation modules retain their original bytes. This release adds the optional filter to the web interface, a command-line operator, complete source checks, and a live verification command.

Use an isolated Python 3.11 environment without system-site-packages. The recorded model runtime includes torch 2.11.0+cu128, transformers 5.16.1, huggingface-hub 1.29.0 and tokenizers 0.23.2. `doctor` checks those bindings, all model/data hashes, and completed-run index/calibration links before loading models. Do not reuse a shared environment whose dependencies have drifted.

From this package directory, using the tested interpreter:

```text
python frontier.py doctor --lab-root <existing-lab-root> --run-id baseline-20260908-1
python frontier.py test
python frontier.py preview --lab-root <existing-lab-root> --run-id baseline-20260908-1 --port 8766
```

Open http://127.0.0.1:8766/. Leave the server process running. In another terminal:

```text
python frontier.py query --question "Which city hosted LongUnknownIdentifier012345?" --guarded
python frontier.py verify --lab-root <existing-lab-root> --run-id baseline-20260908-1 --output live-receipt.json
python frontier.py evaluate-risk --lab-root <existing-lab-root> --run-id baseline-20260908-1 --guarded --output risk-run-1
```

The `verify` command checks real local model responses and all runtime source hashes. Existing receipt filenames are never overwritten. The optional guard appears as a checkbox in corpus mode and has its own `/api/query-guarded` endpoint. Passage mode retains its original behavior. Unknown identifiers may trigger abstention even when a human could resolve them, so the feature is off by default. `evaluate-risk --guarded` compares actual guarded HTTP responses with the source-defined V3 filter; without `--guarded`, it runs the separate V2 global-margin experiment, which does not enable or modify the deployed guard.

For a new machine, create an isolated Python environment and install compatible PyTorch from the official selector, then `python -m pip install -r requirements-runtime.txt`. For the recorded Windows CUDA run, install `torch==2.11.0+cu128` from `https://download.pytorch.org/whl/cu128`. `python -m pip check` must succeed. `download` fetches only selected files at immutable public Hub revisions and verifies the pinned hashes. Then run `prepare --device cpu` (or verified `cuda`), `evaluate`, `doctor`, and `preview` in that order with one new `--run-id`. Expect roughly 1.73 GB of model and dataset files in addition to dependencies. Node.js is needed for the JavaScript behavior regression test; without it that one test reports an explicit skip.

```text
python frontier.py download --lab-root ./lab
python frontier.py prepare --lab-root ./lab --run-id baseline-1 --device cpu
python frontier.py evaluate --lab-root ./lab --run-id baseline-1
python frontier.py doctor --lab-root ./lab --run-id baseline-1
python frontier.py preview --lab-root ./lab --run-id baseline-1
```

Read `EXPERIMENT_REVIEW.md` before interpreting risk metrics. Synthetic identifiers make the filter's test family unusually easy to recognize. The experiment is useful for regression coverage and cannot establish general semantic answer safety. The weights are the upstream Qwen and deepset weights; this release creates no trained SZL weights.

The server binds only to loopback and uses local inference. Production deployment, external credentials, firewall rules and human witnessing require separate concrete provider integration. The source bundle contains neither credentials nor private estate inventories. Its SHA-256 manifest establishes integrity relative to the supplied bundle; it is not an independent signature.
