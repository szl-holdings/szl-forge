# SZL Evidence Lab: local web preview

This presentation layer calls the existing frozen Qwen retrieval and extractive
reader in-process. It does not change the four evaluation modules, model weights,
calibration, indexed corpus or evaluation result. It is a working local research
interface, not estate-wide production admission or a newly trained SZL model.

## Start the existing Windows run

Run from `C:/Users/steph/Documents/Codex/2026-08-09/v` in PowerShell:

```powershell
& './work/retrieval-integration-2026-09-07/preview-venv/Scripts/python.exe' -I -B './work/estate-audit-2026-09-05/kernels/szl-forge/operational/external_retrieval_preview.py' --lab-root './work/retrieval-integration-2026-09-07' --run-id baseline-20260908-1 --port 8766
```

Open http://127.0.0.1:8766 after `Preview ready` appears. Cold startup verifies
roughly 1.7 GB of local assets and loads the models; allow several minutes.
Stop the foreground process with Ctrl+C. The app deliberately has no public bind
option, background persistence, cloud connection or administrator requirement.

The local preview environment overlays transformers 5.16.1, huggingface-hub
1.29.0 and tokenizers 0.23.2 while inheriting the already-installed CUDA PyTorch
2.11.0+cu128 and web/data packages. It leaves shared packages unchanged, but is
not a fully hermetic container. Startup rejects drift in the model-runtime
versions. For portable installation, use a fully isolated virtual environment
and the one-payload setup guide, choosing the official PyTorch wheel for the host.

## Use it

- **Search corpus:** enter a question and retrieve an extractive candidate from
  the top three hybrid-ranked passages, or select **Find passages only**.
- **Check a passage:** paste a question and passage. The answer or abstention is
  scoped to that passage. Its rights and calibration transfer are not established.
- **Recorded examples:** replay previously evaluated positive and negative
  cases. The negative example always switches to passage mode; it is not a
  corpus-wide unsupported-query test.
- Inspect highlighted source spans, attribution, raw score margin, frozen-run
  metrics and JSON receipts. Character offsets count Unicode code points.

The original JSON API remains a separate command on port 8765. This preview's
same-origin endpoints are `/api/status`, `/api/query`, `/api/search`,
`/api/answer-context` and `/api/notices`. POST requires JSON and `X-SZL-Preview: 1`.
That header prevents simple cross-site browser submission; it is not a secret or
authentication. Local programs can call the API. Do not tunnel it to the internet.

## Protections and limits

Exact loopback Host/port and same-origin checks; cross-site Fetch Metadata
denial; 64 KiB request cap; duplicate/non-finite JSON rejection; field validation;
single-GPU concurrency; immutable static-asset hashes; no external scripts,
fonts or model requests; CSP, no framing, no-store and noindex headers. Artifact
hashes are checked before/after model startup. Changes require a restart.

None of this implements OS sandboxing, independent identity, two-person approval,
HSM signing, credential revocation, host firewall enforcement, or a public
production service. Mutable local files and receipts are not independent
attestation. The displayed evaluation is the original public development-set
result, not newly collected quality evidence. False-answer behavior after
retrieval selection and corpus-wide unsupported queries remain unmeasured.

No model training, cloud jobs, GitHub/HF publication, production deployment or
provider security mutations are performed by starting this preview. Read
EXTERNAL_RETRIEVAL_NOTICES.md for original model and dataset notices.

## Tests

```text
python -I -B -m unittest discover -s tests -p test_external_retrieval_*.py -v
python -I -B -m unittest discover -s operational/tests -p test_external_retrieval_preview.py -v
```

Wrapper tests use a stub service and verify contracts, not model quality. Actual
model HTTP/browser verification is recorded separately. Screenshot/layout checks
are not a full accessibility certification or a hidden benchmark.
