# SZL Research Engine: First Vertical Slice

The objective is an SZL research engineer that can investigate the ecosystem,
form hypotheses, build experiments, and improve from externally verified results.
This change implements only its read-only evidence-to-proposal cycle. It does
not train weights, execute experiments, establish novelty, or grant autonomy.

## Architecture Decision

Use a hybrid system: a replaceable strong reasoning backend for difficult work,
plus an owned model trained on permissioned, independently checked task outcomes.
Do not attempt a foundation-model pretraining run on repository text. Retrieval
provides changing facts; fine-tuning should teach investigation and tool use.
Select the backend by measured task success, cost, latency, license, and privacy,
not by model name or model-card claims. No provider is invoked by tests or CI;
the hosted CLI is an explicit, potentially billable operator command.

Reuse the existing estate responsibilities:

| Responsibility | Existing owner | Next capability |
| --- | --- | --- |
| Immutable source knowledge | Second Brain and canonical repositories | Revision-aware retrieval and access filtering |
| Model serving | Router and governed inference | Compare owned and hosted backends on the same tasks |
| Experiments and model development | Forge | Independent execution results, evaluation, then training |
| Authorized software actions | Existing A11oy admission and owned-agent controls | Bounded patch/test jobs with explicit resource limits |
| Publication and runtime evidence | Existing canonical publishers | GitHub, then HF revision verification, then domain probes |

These are integration targets, not a claim that this module wires those services.
This module introduces no competing publisher or automatic deployment writer.

## Implemented Interface

`inference.research_cycle.run_cycle` accepts a question, an in-memory SQLite FTS5
corpus, and a trusted generator callback taking chat messages and returning JSON.
The model can search, read discovered records, finish a proposal, or abstain.
Unknown tools, unread citations, invented quotations, malformed responses, and
exceeded turn budgets stop the cycle. The callback is application code and must
be trusted; this is not a hostile-code sandbox or a general prompt-injection
defense. Quotation matching does not establish semantic support or source truth.

Each JSONL source row must have exactly `id`, `text`, `source`, `revision`,
`visibility` (`public` or `private`), and `content_sha256` (UTF-8 text SHA256).
Revision must be a full 40- or 64-character lowercase hexadecimal identifier.
The ingester must independently verify source authenticity, visibility and rights;
the cycle checks only supplied revisions, byte digests, and quotation presence.
Use short chunks, ideally around 1,500 characters, for the local model adapter.

Remote callbacks reject a corpus containing any private row before receiving any
messages. This relies on honest visibility labels and execution-place declarations;
it is not a network enforcement boundary. Questions and public source text can
still contain secrets: review inputs before wiring a hosted adapter.

The local adapter reuses `benchmark_ollama.LocalClient`, admits only already
installed local GGUF models, disables redirects and inherited proxy routing,
requires complete responses, and checks the model digest again after the run.
It refuses to disturb already loaded model work and never downloads a model.

```powershell
python -B local-compute/research_ollama.py `
  --model szl1:latest --corpus C:/work/public-source-corpus.jsonl `
  --question "What bounded experiment could improve cross-repository diagnosis?" `
  --max-turns 6 --output C:/work/research-run-001.json
```

Output is a local, unsigned development record, including the model digest,
runner and engine hashes, transcript, evidence references, and proposal. Keep it
private when inputs are private. It is not an A11oy execution receipt. A return
code of zero means a quotation-checked proposal exists, not that it is correct.

`local-compute/research_hf.py` supplies the hosted counterpart using the existing
HF router and local HF authentication. Specify `--model repository:provider`
and `--public-inputs-reviewed` explicitly, along with the same corpus, question,
output and turn bound. Review both the corpus and question before transmission.
There is no automatic provider fallback: at most six calls with 1,536 generated
tokens each by default and a 90-second timeout per call. The explicit
`--max-output-tokens` setting is capped at 4,096, allowing room for models whose
reasoning consumes the completion allowance. Incomplete answers are rejected;
the finish status and requested budget are retained. The adapter disables redirects and
inherited proxies. Reports retain token usage when supplied, but cost is not
measured and the provider's actual serving revision remains unverified.

## Capability Milestones

1. Freeze a permissioned, time-stamped task set: cross-repository fault diagnosis,
   missing source-to-model bindings, reproducible repairs, and experiment design.
   Keep development tasks and future locked evaluation tasks separate.
2. Compare existing SZL models and a strong reasoning backend on identical tasks.
   Measure patch test success, unsupported claims, citation correctness, cost,
   latency, abstention, and actual task completion. Protocol tests are not scores.
3. Connect a bounded worker through the existing admission path. Bind the proposed
   experiment to its input revisions, test commands, budget and result artifacts.
   Require real execution results before calling any experiment successful.
4. Curate licensed training examples from successful, independently verified
   rollouts. Preserve failure examples for analysis; never treat fluent proposals
   or evaluator answers as successful training labels. Split by task family and
   repository lineage to reduce contamination.
5. Train an owned adapter only after a baseline and explicit compute budget exist.
   Promote only after locked-task improvement, privacy checks, and reproducible
   serving. A failed candidate stays a failed candidate.

The research ambition is test-generating, cross-repository engineering: discover
an unmet invariant, propose an experiment, produce a patch and regression test,
then verify improvement outside the proposing model. It is an engineering
direction, not an established scientific first or a proven new model architecture.

## Verification

`python -m pytest -q tests/test_research_cycle.py tests/test_research_ollama.py tests/test_research_hf.py`

The public tests use scripted generators to check control flow and boundaries.
They are neither model intelligence evaluations nor hidden promotion gates.
