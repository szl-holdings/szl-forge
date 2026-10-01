# SZL local evidence retrieval

This is a separate **EXTERNAL_BENCHMARK_EVAL_ONLY** lane. It adds real local
inference, lexical/dense/hybrid retrieval, calibrated given-context abstention,
source-span citations, and a loopback JSON API. It does not replace Forge's
in-house evaluation or modify training/publication admission. Nothing is uploaded.

## Models, evidence and boundaries

- `Qwen/Qwen3-Embedding-0.6B` supplies native dense embeddings. BM25 is an
  independent lexical baseline; equal-weight reciprocal-rank fusion uses k=60.
- `deepset/roberta-base-squad2` extracts spans from individual passages. A raw
  span-minus-null logit margin is not a probability. Overflow windows are merged
  only within the same document. No valid span or invalid input fails closed.
- SQuAD 2 supplies human-authored questions, answer spans and passage-specific
  unanswerable labels. The fixed sample has 60+60 calibration and 120+120
  evaluation questions in disjoint article components. All 1,204 validation
  paragraphs remain in the shared retrieval index. Positive retrieval qrels are
  derived from original answer spans, not exhaustive relevance judgments.
- HotpotQA supplies 60 fixed evaluation cases with native supporting-fact
  positives and distractor candidate pools. This is not full-Wikipedia retrieval.
- Both datasets are public development sets. Base-model contamination is
  **UNKNOWN**. Article separation does not establish an unseen-model benchmark.

The threshold is selected on calibration questions only, before evaluation.
It maximizes calibration QA F1 subject to an empirical <=5% false-answer rate
among calibration negatives. Evaluation reports a Wilson interval; 5% is not a
guaranteed population limit. The threshold has only been calibrated on original
given contexts. Selecting a winner from retrieved passages changes the input
distribution: **global unsupported-query error remains unmeasured**. An answer
is a model-produced extractive candidate with a checkable source span, not a
human-certified assertion. An abstention does not prove no answer exists elsewhere.

See [EXTERNAL_RETRIEVAL_NOTICES.md](EXTERNAL_RETRIEVAL_NOTICES.md) for separate
dataset/model/runtime rights and attribution requirements. Publicly declared
licenses are not blanket legal clearance or automatic training admission.

## Dependencies and downloads

Python 3.11+ is required. The witnessed local environment uses torch
2.11.0+cu128, transformers 5.16.1, huggingface-hub 1.29.0, numpy 2.4.6,
pyarrow 25.0.0, fastapi 0.140.13, uvicorn 0.51.0, and httpx 0.28.1.
These are recorded environment versions, not a claim of dependency vulnerability
clearance. Use a separate environment and a PyTorch build appropriate to the host.
No dependency packages were installed by this build. CPU is supported but was
not the performance target; use `--device cpu` at preparation, not mid-run.

Use `hf download` for only the selected files at these exact revisions:

| Repository | Revision | Destination under LAB |
| --- | --- | --- |
| rajpurkar/squad_v2 | 3ffb306f725f7d2ce8394bc1873b24868140c412 | assets/squad-v2 |
| hotpotqa/hotpot_qa | 1908d6afbbead072334abe2965f91bd2709910ab | assets/hotpot-qa |
| deepset/roberta-base-squad2 | adc3b06f79f797d1c575d5479d6f5efe54a9e3b4 | assets/reader |
| Qwen/Qwen3-Embedding-0.6B | 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3 | assets/qwen or existing verified snapshot |

The data adapter lists the two required validation Parquets. `QWEN_FILES` and
`READER_FILES` in the lab module list every runtime asset and required SHA-256.
Keep README/license notices as well. No pickle weights or remote Python code
are loaded. Runtime uses local files only and disables Hub network access.

## Run from this repository

For example, set `LAB` to a writable evaluation directory and `ENCODER` to the
verified local Qwen snapshot. Commands below use environment-neutral placeholders:

```text
python operational/external_retrieval_lab.py prepare --lab-root LAB --run-id baseline-1 --encoder-path ENCODER --device cuda
python operational/external_retrieval_lab.py evaluate --lab-root LAB --run-id baseline-1
python operational/external_retrieval_lab.py query --lab-root LAB --run-id baseline-1 --question "What is the capital of France?"
python operational/external_retrieval_lab.py serve --lab-root LAB --run-id baseline-1 --port 8765
```

Preparation exclusively creates a run directory, freezes data, model hashes,
device, source-code hashes and settings. Evaluation exclusively creates its
attempt receipt and saves calibration before reading evaluation predictions.
Repeated evaluation of the same run is refused. Failed attempts remain visible;
fix infrastructure and create a new run ID, recording whether labels were already
viewed. Do not reuse an exposed evaluation slice as fresh evidence after tuning.

Successful runs contain `freeze.json`, `payload.json`, `calibration.json`,
`documents.npy` and `result.json`. The service refuses source/config/asset/index
changes or a missing completed result. These local hashes detect ordinary drift;
they are not independent attestation or protection from a hostile administrator
who can rewrite both files and receipts.

## Local API

The server binds only `127.0.0.1`, one worker. It has no cloud deployment or
production authentication. No broad CORS is enabled; browser Origin requests
are refused. Bodies are bounded to 64 KiB, questions to 512 characters,
reader questions to 128 tokens, supplied contexts to 16,000 characters and
eight windows. GPU requests are serialized; concurrent requests receive 429.

| Endpoint | JSON input | Output |
| --- | --- | --- |
| GET /health | none | ready state, corpus size, run and source fingerprint |
| POST /search | question, optional k=1..10 | ranked passages and citations |
| POST /query | question | top-three retrieval, extraction or abstention, citations and scope warnings |
| POST /answer-context | question, context | extraction or abstention within the supplied text only |

Use `Content-Type: application/json`. There is no public HTML interface. Stop
the foreground process with Ctrl+C. Do not expose the port via a reverse proxy
or change the loopback binding without adding a separate reviewed security layer.

## Verification

```text
python -I -B -m unittest discover -s tests -p "test_external_retrieval_*.py" -v
```

The tests cover math, split/label integrity, source tampering, source-bound
extraction, padding, and HTTP input handling. They are distinct from actual
model inference and the frozen evaluation results. Models and dataset bytes
are free to download under their stated licenses; local electricity, hardware,
storage and any future hosted operation are not guaranteed cost-free.
