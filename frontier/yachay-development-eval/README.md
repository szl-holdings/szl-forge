# Yachay / Nemotron development evaluation candidate

This adds executable source for the missing base-versus-adapter comparison.
It does **not** establish A6000 runtime qualification, model quality, a completed
training run, publication eligibility, or exact training resumption.

The retained campaign used
`nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` at
`dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f`. Historical C94/C138 custom
checkpoints, the RTX 5050 contract, and later Node B PEFT campaigns are distinct.
This runner accepts only causal LoRA PEFT snapshots. It does not load C138's
custom adapter/optimizer format or reinterpret the 5050 contract for an A6000.

## Implemented commands

```bash
python evaluate.py check plan.template.json
# Expected exit 2: enumerates UNKNOWN bindings. Does not load models or probe GPU.

python evaluate.py verify /absolute/path/to/completed-plan.json
# Verifies complete local snapshot inventories, SHA-256, adapter/base binding,
# loader-review binding, data schema and exact row count. No model loading.

python evaluate.py run /absolute/path/to/completed-plan.json \
  --plan-sha256 EXTERNALLY_RETAINED_PLAN_SHA256 \
  --output /absolute/path/to/new-output-directory
```

`run` is an **explicit local inference action**, for an already qualified,
owner-coordinated Linux window. Nothing in this package acquires access,
connects over SSH, provisions compute, installs dependencies, starts a service,
dispatches CI to Node B, or restarts a trainer. Do not retry a denied route with
this tool. The private `_worker` command is for the supervisor only.

## Artifact and runtime requirements

Replace every UNKNOWN using independently observed evidence. Keep the plan's
SHA-256 outside the mutable plan. Bind `evaluator_sha256` to the reviewed source.
The template's original step-100 candidate remains unknown; a different campaign
or checkpoint must not silently replace it. To compare a different candidate,
freeze a separate plan with its own campaign ID and file hashes.

Each snapshot `files` map must exhaustively list relative filenames and SHA-256
values. Base files include weights, indexes, config, and local Python source;
tokenizer files include its config and template dependencies. The evaluator
rejects extra files, symlinks, path traversal and mismatched hashes. Materialize
snapshots without HF cache symlinks in the authorized environment, then mount
them read-only. Adapter directories contain only `adapter_config.json` and
`adapter_model.safetensors`; original checkpoints are not modified.

The upstream model uses custom code. Review its pinned local code and installed
dependencies before execution. A small loader-review JSON has these fields:

```json
{
  "binding_sha256": "SHA256_OF_CANONICAL_BINDING_OBJECT",
  "local_custom_code_reviewed": true,
  "reviewer": "THE_ACTUAL_REVIEWER"
}
```

The binding object is `{"base": BASE_FILES_MAP, "tokenizer": TOKENIZER_FILES_MAP,
"packages": EXACT_PACKAGES_MAP}`, serialized with sorted keys, no whitespace,
UTF-8 and no nonfinite numbers. Bind the review file's path/hash in the plan.
This record is a reviewer declaration, **not** a signature, independent witness,
package attestation or runtime-qualification receipt. Pin all five installed
package versions from the actual interpreter; historical version reports are
not proof of present compatibility.

Execution checks Python 3.12, hostname, current boot ID, the selected RTX A6000
GPU UUID, sm_86 capability and idle GPU state. It requires existing cgroup-v2
CPU, memory and PID limits and read-only artifact mounts. The owner must provide
the resource lease, qualified environment and execution window separately.
The parent terminates only its newly spawned worker process group on timeout
or exit. It never signals a saved trainer PID. The limits allow at most 900
seconds for the worker, 256 examples, 2048 tokens per example, four CPU cores
and a 64 GiB cgroup memory ceiling.

The CUDA allocator fraction is **not** a hard cap on all GPU/driver allocations.
The cooperative file lock does not exclude unrelated trainers. The idle GPU
probe does not prove a future lease. Offline HF flags prevent ordinary library
downloads; they are **not** an OS network sandbox for custom code. The source
is not a replacement for the missing independently qualified native
observer/watchdog, resource isolation or guarded launcher. No such qualification
is claimed by a schema check or this CPU suite.

## Data and metric protocol

Supply UTF-8 JSONL, one example per line, with exactly `id` and `messages`:

```json
{"id":"example-1","messages":[{"role":"user","content":"Question"},{"role":"assistant","content":"Answer"}]}
```

Any normalization of historical rows is a separate reviewed transform with its
own source/output hash; this evaluator never silently rewrites the source data.
The final user/assistant turn is required. Earlier turns are context. The
tokenized generation prompt must be an exact prefix of the tokenized full
conversation. Only final-answer tokens (including template end tokens) are
scored. Prefix mismatch, empty targets and oversized sequences stop the run;
there is no truncation, padding or dropped-row fallback.

Each candidate starts from a fresh base load, a fixed seed, `eval()` and
`torch.inference_mode()`, with all gradients disabled. There is no optimizer,
backward pass, generation or checkpoint write. Raw shifted next-token
cross-entropy sums and token counts are retained per example. The aggregate
is token-weighted NLL, with perplexity when representable. This is a new
explicit protocol; do not directly compare its values to historical losses
until the old encoder/reduction semantics are recovered and matched.

All results are `development_diagnostic`, `held_out_quality=NOT_MEASURED` and
`production_eligible=false`. The original six validation rows have already
been used repeatedly for selection. Neither a good validation score nor this
source test suite supplies the missing sealed, contamination-reviewed held-out
evaluation or refusal/abstention evidence.

## Output and failure evidence

An existing output directory is rejected. A new directory retains the frozen
plan, evaluator hash, planned candidates, worker log, append-only raw rows,
candidate events and final receipt with file hashes. Missing candidates remain
NOT_RUN; failed and timed-out attempts remain visible. Partial event lines are
retained and reported as errors. A parent crash may leave `started.json` without
`receipt.json`: that is an incomplete attempt, never a completed evaluation.
These local receipts are unsigned and do not authenticate the model provider.

## Validation performed

```bash
python -I -B -m unittest discover -s . -p 'test_*.py' -v
```

24 CPU software tests passed on Linux. Three POSIX-specific tests are skipped
on Windows. Fixture bytes are deliberately not model weights. The timeout test
starts and reaps a sleeping Python child; model/GPU probes are mocked in that
test. No torch/Transformers/PEFT packages were installed or imported, no model
weights were loaded, and no CUDA inference was executed during this validation.
The numerical test verifies token-weighted aggregation, not the GPU loader.

Sources for the loader API:

- [Pinned NVIDIA model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16/blob/dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f/README.md)
- [PEFT PeftModel reference](https://huggingface.co/docs/peft/en/package_reference/peft_model)

Next required measured step: provide the missing immutable bindings and a
qualified Node B window, then run the bounded diagnostic once. A separate
held-out protocol and explicit acceptance evidence are still required for any
training continuation or promotion decision. Historical counters stay consumed.
