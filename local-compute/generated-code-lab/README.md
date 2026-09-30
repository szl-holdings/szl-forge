# Canonical Forge experimental worker candidate

This additive lab does not replace existing read-only local-compute tools, recovery admission, qualified models, or canonical publishers. It is not connected to the governed production execution path.

# SZL Local Model Lab — free, local GPU baseline

This is a local development baseline around **Qwen3-4B-Instruct-2507**, distributed as
`qwen3:4b-instruct` in Q4_K_M format. Its upstream license is Apache-2.0.
It is **not newly trained SZL weights**, a frontier-scale foundation model,
or a production estate-control agent. No cloud inference or paid GPU is used.

**Current verification:** see `RESULTS_2026-09-29.md`. Offline tests and live
sandbox probes have passed. A successful full model-generated smoke run after
the inference-grammar repair is **not yet verified**; disk/memory pressure on
the host prevented completion. Do not interpret a source bundle as that proof.

The initial goal is concrete: a local model generates Python implementations,
the lab tests them outside the model, and retains both failures and successes.
Those reviewed trajectories can later inform a real specialization experiment.

## What is installed

- Native Windows Ollama, already installed on this machine, serves the model.
- Task-specific endpoint: `http://127.0.0.1:11439`.
- Cloud features are disabled for this process; redirects and environment proxies
  are disabled in the Python client. No public listener or tunnel is created.
- One loaded model, one concurrent generation, 4,096-token working context.
- Downloaded model files live in the existing user Ollama model store. The older
  `szl1` files and existing Ollama services are preserved.
- `model.lock.json` binds the exact manifest and verified SHA256 of every blob.
- Model candidate code runs only in a local Docker Desktop Linux container.
  Inference itself uses the native Windows NVIDIA GPU, not GPU-in-Docker.

The process has no automatic startup registration. It can be stopped by closing
the task-specific Ollama process after checking the PID/listener. The launcher
sets a two-minute default keep-alive, but the lab's `ask`, `run`, and `smoke`
requests explicitly override it with `keep_alive: 0`. This requests model unloading
after each answer to reduce overlap with Docker testing; weights stay on disk.
Retries may take longer because the model must reload. This is a resource policy,
not a measurement or guarantee of reclaimed RAM/VRAM. See the
[Ollama chat request reference](https://docs.ollama.com/api/chat).
Other Ollama processes and their listeners are outside this lab's changes.

## Use the existing hosted model now

[Open the SZL Khipu model interface](https://szlholdings-szl-model-inference-lab.hf.space/#run-lab)
and select **Run bounded inference** after entering a short prompt.

This is a **separate hosted Khipu 1.5B demonstration**, not this local Qwen3-4B
lab or a deployment of this source package. No API token is required; do not
submit credentials or sensitive data. The public CPU service permits one request
at a time and at most 32 generated tokens, with best-effort availability and no
SLA. It has no tools, repository access or deployment authority.

Opening that link is an explicit choice. This installer and the local commands
do not contact the hosted service or fall back to remote inference. Its
[live contract](https://szlholdings-szl-model-inference-lab.hf.space/.well-known/szl-inference-contract.json)
and [build information](https://szlholdings-szl-model-inference-lab.hf.space/api/build-info)
describe its own source/model identity and limits. Check them independently of
this lab's local runtime qualification.

## Use the local model

Run these commands from PowerShell:

```powershell
Set-Location 'local-compute/generated-code-lab'

# Start only if the task-specific process is not already running.
.\start-local.ps1

# Verify availability and the pinned model identity.
python -B .\lab.py status

# Wait for the actual local API and model identity, not just process creation.
python -B .\lab.py ready --timeout 60

# Ask the local model a question. This mode has no repository or shell tools.
python -B .\lab.py ask 'Suggest three testable ways to reduce latency in a model-backed evidence-selection API. Separate hypotheses from measurements.'

# Generate code, execute it in a bounded container, and score it externally.
python -B .\lab.py run .\tasks\stable_provider_routing.json --attempts 2

# Run all three development smoke tasks.
python -B .\lab.py smoke

# Test the harness and task fixtures without model generation.
python -B -m unittest discover -s . -p 'test_*.py' -v

# Opt-in live restriction/timeout/output-limit/false-success probes.
python -B .\probe_sandbox.py

# Export an allowlisted source ZIP and one Markdown Python-installation payload.
python -B .\build_release.py
```

Use Python 3.11 or 3.12. The launcher expects Python and Ollama on PATH.

## Task format

A task is a local JSON file with an `id`, an `instruction`, and `cases`:

```json
{
  "id": "example",
  "instruction": "Implement solve(value): return the sum of the integer list value. An empty list returns 0.",
  "cases": [
    {"input": [], "expected": 0},
    {"input": [2, -1, 5], "expected": 6}
  ]
}
```

The model receives the instruction, not the case file. It must return a complete
`solution.py` defining `solve(value)`, encoded as a `code_lines` array with one
literal source line per item and a short `explanation`. The harness joins lines
without unescaping or repairing Python string literals. It cannot choose a host write path, edit the
test expectations, issue shell commands, or deploy anything through this harness.
There are at most three attempts per task. Feedback is a development pass count
or error label; these are not sealed held-out research results.

The pinned local Docker engine/image must pass a bounded preflight before a model
request is sent. An unavailable execution engine is an infrastructure failure,
not a reason to run generated code on the Windows host or to mark a task solved.
The smoke suite stops immediately after infrastructure or cleanup failure; it
continues to later tasks after ordinary candidate-answer failures. Optional GPU
telemetry errors are recorded separately and cannot rewrite external scoring.
The retained `gpu_runtime_readback` is only a post-run model snapshot. It can be
empty after an unload request and does not prove that generation used the GPU,
that unloading completed before sandbox startup, or that memory was reclaimed.

The inference wire schema constrains structure without expanding thousands of
bounded grammar repetitions. The host still enforces 400 lines, 2,000 characters
per line, 24,000 source bytes, and 4,000 explanation characters. Generation has
a separate 1,800-token budget and the client caps API responses at 2 MiB.

Runs are retained in `runs/<timestamp>-<random-id>/`:

- Full model generation and token/timing metadata.
- Each generated candidate and the fixed execution worker.
- Bounded container output and execution result.
- A receipt binding model identity, task, harness and worker hashes.
- A development trajectory explicitly marked **not automatically training eligible**.

Early Docker/model preflight failures also retain receipts. If disk space is below
512 MiB, the run refuses to start; it may not be possible to save a new receipt on
an exhausted volume. Do not remove this guard to force inference. Leave several
GiB of stable disk and memory headroom for Windows, the GPU runner, and Docker.
The launcher waits for readiness, retains timestamped logs, restores its caller's
environment, and never silently restarts an existing service.

The three bundled tasks contain 32 synthetic cases: canonical report hashing,
source/publication admission logic, and stable latency-constrained provider routing.
They test small components inspired by ecosystem work, not live credentials,
cryptographic provenance, every repository, or deployment readiness.

## Execution boundary

The Docker client is explicitly bound to the local Windows named pipe; a remote
`DOCKER_HOST` or selected context cannot silently rent or use remote compute.
The existing image is pinned by its content identity and pulled with `--pull=never`:

`sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea`

Containers have no network, a read-only root, a non-root user, dropped capabilities,
no-new-privileges, CPU/memory/PID limits, bounded output and a wall-clock timeout.
Only the fresh candidate directory is mounted, read-only. No credential stores,
repository checkouts, Docker socket, or model files are mounted into candidates.
Test expectations stay in the host evaluator; a candidate's printed claim of
success is not accepted as a pass.

This is defense in depth for bounded development experiments, not a formal
guarantee against container/runtime vulnerabilities or malicious adversarial code.
It is Windows/Docker-Desktop-specific and fails if the pinned local image is absent.
Generated answers are untrusted until the external checks actually pass.

## Source bundle and single-file payload

`build_release.py` exports only named source files, fixtures and documentation,
with SHA256 integrity metadata. It excludes runtime logs, experiment runs,
credentials, model weights, caches and unrelated repositories. The Markdown
contains one Python installer; it writes into a **new** directory, checks its
embedded content hashes, refuses overwrites, and does not run generated code,
install dependencies, download models or start services. Inspect the files first.
The source ZIP is deterministic for the same source inputs; hashes establish
byte integrity, not independent human approval or third-party attestation.

## Next model-development milestone

Collect permissioned, reviewed multi-file engineering trajectories from real SZL
tasks; split by task/repository family to reduce train/test leakage; establish an
unchanged-base baseline; then train an adapter and compare it under equal budgets.
Do not train on these smoke answers and then call success on the same cases
generalization. No automatic self-training or production-weight replacement is enabled.

## Licenses and sources

- Upstream weights: [official model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), Apache-2.0.
- Quantized distribution: [Ollama Qwen3 4B Instruct](https://ollama.com/library/qwen3:4b-instruct).
- Ollama software: [MIT license](https://github.com/ollama/ollama/blob/main/LICENSE).
- New lab source files use SPDX `Apache-2.0`; model and runtime licenses remain separate.

The software/model downloads do not require a paid inference account. Local
electricity, storage and hardware are still resources; no paid service was ordered.
