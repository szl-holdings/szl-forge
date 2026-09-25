# Bounded research cycle — development instrument

An executable first slice of the SZL research-and-engineering loop: propose a
typed improvement, compare it with a frozen baseline, preserve negative results,
and produce a readable result alongside machine-checkable evidence.

This extends Forge's existing `inference` package. It is **not another service,
new model, qualification lane, autonomous executor, or publisher**. It has no
production permissions. A11oy remains the consequential-action authority.

## What runs

1. Validate a digest-bound public synthetic study before calling a model.
2. Measure a fixed lexical retrieval baseline.
3. Give the proposer document text, development questions and previous
   development results. Never include validation questions in this context.
4. Accept only `title_weight`, `body_weight`, and `normalize_length`.
   No shell commands, Python, URLs, tool calls or dynamic imports are admitted
   from model output. This is typed configuration search, not generated-code
   execution or a hostile-code sandbox.
5. Evaluate at most three proposals with the same deterministic evaluator.
6. Choose only a strict development improvement with no per-query regression.
7. Assess that one selected candidate on validation once. Do not select a
   replacement after seeing validation.
8. Emit a JSON result and an escaped, script-free local HTML evidence page.

Ties are not improvements. Invalid JSON, duplicate keys, extra fields,
non-finite numbers, bool-as-integer weights, duplicate candidates and model
identity changes cannot qualify. An unavailable model stops the cycle without
automatic retry. Earlier measured attempts are retained, but the incomplete
cycle reports `MODEL_UNAVAILABLE`.

### Measurement limits

The included corpus is **four synthetic documents**, with three development
and three validation questions. Its distractor is deliberately simple. MRR is
computed over the complete document ranking with document-ID tie-breaking.
This is a software demonstration, not a search-quality benchmark.

The public validation slice is withheld from the proposal prompt, **not sealed**.
It is visible in this repository and can be overfit across repeated studies.
Its results cannot establish broad capability, independent verification,
novelty, model promotion, or training admission. Input labels are declarations;
the tool does not establish consent or license rights for an arbitrary file.
Use only approved public synthetic input here, never secrets or private data.

`frontier/harness/LANE_CONTRACT.md` and its signed hidden-set qualification
requirements remain unchanged. No L4+ charter or hidden benchmark is created.

## Run a repeatable software demonstration

Python 3.11+, standard library only, from the Forge repository root:

```powershell
$suite = 'experiments/research-cycle/suite.json'
$digest = (Get-FileHash $suite -Algorithm SHA256).Hash.ToLowerInvariant()
python -m inference.research_cycle --suite $suite --suite-sha256 $digest --replay experiments/research-cycle/replay.json --output reports/research-cycle-replay
```

Use a new output directory per run. Existing output is refused before model
inference. `SYNTHETIC_REPLAY` means **no model executed**; the checked-in recipe
is authored test input and must never be advertised as a model discovery.
The JSON result's `suite_sha256` is the canonical JSON digest, while the CLI
argument binds the exact input file bytes. The encoding is explicitly recorded.

## Optional local model proposal

Use only an already-installed, explicitly chosen local Ollama model. The
operator must read its current `/api/tags` digest and supply that exact value:

```powershell
python -m inference.research_cycle --suite $suite --suite-sha256 $digest --live-loopback --model EXACT_INSTALLED_NAME --model-digest EXACT_64_HEX_DIGEST --attempts 1 --output reports/research-cycle-local-model
```

The adapter uses only `http://127.0.0.1:11434`, disables proxy inheritance and
redirects, checks the named model digest before and after generation, requests
at most 128 output tokens, and applies a 20-second per-request socket timeout.
No retries, downloads, installs, daemon restarts, provider configuration changes
or public listeners. A socket timeout is not a hard process-runtime deadline.
At most three candidate generations are allowed; the default is one.

Failed local attempts retain a fixed `failure_phase`: `inventory_before`,
`generation`, or `inventory_after`. `elapsed_seconds` measures that failed
phase using a monotonic clock, not the whole cycle or server compute time.
Response validation is part of its phase, so a changed post-generation model
identity is still unavailable, even when generation returned text. The original
`error_type` is preserved; exception text and response bodies are not recorded.
Generic callbacks cannot supply these adapter-specific diagnostic fields.
An observation is reset at the start of every call. This distinguishes where a
failure was detected; it does **not** establish a server root cause, model
quality, runtime confinement, or successful recovery. No automatic retry or
timeout increase is introduced.

The local daemon is a trusted dependency, **not an isolated process**. Model
names containing `cloud` and declared remote models are refused; this does not
prove the daemon cannot contact another service. A local digest readback is
server-reported inventory, not cryptographic attestation of execution weights,
hardware or model quality. No cloud API option or credential handling is added.

Official adapter protocol: [chat](https://docs.ollama.com/api/chat) and
[model inventory](https://docs.ollama.com/api/tags).

## Use the existing governed inference coordinator

The Python integration consumes already-admitted Forge components:

```python
from inference.research_cycle import GovernedProposer, load_suite, run_cycle, write_result

proposer = GovernedProposer(
    {"request_id": "research-001", "tenant_id": tenant_id, "principal_id": principal_id},
    retriever=admitted_retriever,
    hydrator=admitted_hydrator,
    generator=admitted_generator,
    witness=admitted_nemo_witness,
)
result = run_cycle(load_suite(suite_path, suite_file_sha256), proposer,
                   mode="GOVERNED_PROPOSAL", attempts=1)
write_result(new_output_directory, result)
```

`GovernedProposer` invokes `governed_infer`; it does not bypass retrieval,
digest-verified hydration, formula binding, or pre/post witnesses. It requires
`PROPOSAL`, `NO_ACTION_AUTHORITY`, `executed=false`, and a matching output /
inference-receipt digest. REVIEW, ABSTAIN and BLOCKED remain unaccepted.
Each accepted proposal records the existing inference receipt's digest.
Callbacks are trusted integration code, not a confinement mechanism.

## Evidence, exit codes and presentation

- `result.json`: exact local evaluator-file digest, canonical study digest,
  mode, model identity boundary, baseline scores, every attempt, selected
  recipe, one-shot validation, decision and limitations.
- `index.html`: local evidence view of that result, escaped and without script
  or remote assets. This is not a deployment to either public domain.
- `receipt_sha256`: integrity over every result field except itself, using
  sorted compact UTF-8 JSON. It is **not a signature or external authority**.
- Result writer rejects changed digests or elevated authority fields.
- Exit `0`: a bounded development cycle completed, including ties or negative
  findings. It does **not** mean improvement, qualification, or production.
- Exit `2`: model unavailable, invalid inputs, or output failure. Inspect the
  decision; never treat process completion as research success.

Every result keeps `production_disposition=HOLD`,
`training_admission=false`, `publication_eligible=false`,
`model_quality_established=false` and `signature_status=UNSIGNED_LOCAL`.
Raw model output, credentials and private reasoning are not retained.

## Tests

```sh
python -m pytest -q tests/test_research_cycle.py tests/test_inference_control_plane.py
ruff check inference/research_cycle.py tests/test_research_cycle.py --select E9,F
```

The existing base Python workflow discovers the new tests. No additional
scheduled job, deployment writer, credential, GPU allocation or publisher is
introduced.

## Next integration gates

This slice operationalizes one bounded experiment, not the entire requested
research operator. The remaining sequence is:

1. Qualify a real proposer against estate tasks and the existing model gates;
   do not select model size or provider solely from marketing or train loss.
2. Admit a provenance- and license-reviewed estate corpus to the existing
   Second Brain, keeping changing facts in retrieval, not blindly in weights.
3. Use an independently approved execution environment for generated patches
   and experiments. The current typed evaluator grants no such authority.
4. Bind successful task evidence to existing protected PR and publication
   workflows; keep exact source, CI, Hub bytes and live runtime separate.
5. Curate successful and failed trajectories with consent, deduplication and
   evaluation exclusion before training any owned adapter. Declare compute and
   spending limits before a paid job.
6. Add the qualified operator to the existing a-11-oy.com product experience,
   retain public evidence pointers on a11oy.net, and publish only appropriately
   admitted models/datasets/evaluations to Hugging Face.

These are remaining gates, not accomplished features or automatic permissions.
