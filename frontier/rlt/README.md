# SZL Forge recurrent continuity research

**Executable research software, not trained language weights or production routing.**
The canonical owner is this directory in `szl-holdings/szl-forge`. Do not copy an
engine into every vertical. The package is isolated from the dependency-free
production controller and does not change its model selection or authority.

## Study and original implementation

Concept source: Yifan Zhang, *Recurrent Looped Transformer*, September 12, 2026.
https://github.com/yifanzhang-pro/recurrent-looped-tranformer
https://yifanzhang-pro.github.io/recurrent-looped-tranformer/

Original SZL implementation: one shared encoder/decoder self-attention and FFN
block, separate cross-attention, previous decoder-output feedback, bounded local
decoder KV, learned absolute positions. This is not the report's 48+48 model or
the upstream experiment's approximately 79k-parameter trained implementation.
The default model has 7,105 random parameters. No upstream code or weights copied.

## Actual execution path

`model.py` -> controller-owned `Session` -> complete tensor checkpoint ->
independently digest-bound restore or fresh replay -> sanitized observation ->
JSON receipt and local HTML evidence view.

The inference session owns a model copy, validates actual parameter/state bytes,
requires tenant/policy/evidence/tokenizer/template digests, and rejects changed
history, changed execution, malformed state, nonfinite tensors and wrong pending
tokens. A token emitted by `emit_greedy` is pending until consumed by `append`.
The local attention window includes the current token: only W-1 prior positions
are retained. Failed appends do not partially commit session state.

Checkpoint bytes contain inference state and are **private**, not public proof.
Only the sanitized receipt is eligible for later review as a public projection.
Expected checkpoint digests must be independently held by the trusted controller.
Hash equality does not authenticate a producer or grant tool authority. This is
not protection against hostile code in the same Python process. The API does not
load pickle, connect providers, hydrate private graphs or execute tools.

`model.py` retains full gradients for training research. `runtime.py` checkpoints
are detached inference values and cannot stand in for full-gradient training
checkpoints. Model changes require replay. There is no arbitrary-context or
cross-hardware bitwise-replay guarantee; the reference is CPU/FP32, greedy,
little-endian, one session row, <=512 tokens, <=64 dimensions and <=512 vocabulary.
Digest checks have material overhead; no speedup is claimed.

## Install and run

Use a fresh, dedicated CPU environment from this repository checkout. Do not run
these commands in an existing laptop CUDA/training environment. No GPU, model
download or paid service is required. The project declaration is the single
Torch version authority; every install below retains the derived CPU constraint.

```bash
set -euo pipefail
CPU_CONSTRAINT=$(mktemp)
trap 'rm -f "$CPU_CONSTRAINT"' EXIT
python tools/rlt_cpu_environment.py requirement > "$CPU_CONSTRAINT"
python -m pip install -r "$CPU_CONSTRAINT" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -c "$CPU_CONSTRAINT" ./frontier/rlt
python -m pip check
python tools/rlt_cpu_environment.py verify
szl-rlt-research --output ./rlt-run-001
```

The Bash example applies to the isolated CPU research lane, not the deferred
owner-laptop training lane. CI uses the same helper, constrains subsequent
resolver invocations, and verifies distribution/module versions plus CUDA/HIP
build metadata both before tensor tests and after wheel installation. A CPU
runner alone does not prove that a CPU wheel was installed. Missing or mismatched
identity fails rather than downgrading a pin. This is local package identity,
not a complete hash-locked dependency closure or wheel-byte provenance. Existing
synthetic training and installed-package integration remain separate gates.

The package declares `safetensors[torch]==0.7.0` and `numpy==2.3.5`: NumPy is
required by the safetensors PyTorch serializer even though SZL's tensor-digest
helper does not use it. Install declared dependencies; do not use `--no-deps`
unless an independently verified compatible environment already exists.

`--output` must not exist; its parent must exist. It creates `receipt.json`,
`checkpoint.safetensors`, and a local `index.html`. Open the HTML locally to
inspect the actual measurements. The viewer has no JavaScript, CDN or networking.
The receipt records source hashes and environment. A local pass means only that
the disclosed synthetic continuity checks passed. It is unsigned, not a quality
certificate. Source checkout equivalent: `python -m frontier.rlt --output ...`.

```bash
python -m pytest frontier/rlt/tests -q
```

The dedicated workflow tests Python 3.11/3.12, builds a wheel, installs it, and
runs outside the checkout. Production's existing base workflow remains unchanged.

## Downstream integration boundary

GitHub reviewed source -> qualified HF runtime/artifact -> a-11-oy.com product
observation -> a11oy.net proof. Version 0.2 implements a trained synthetic experiment and an installed-package
research integration, not the entire production deployment chain.

Forge's `inference/governed_inference.py` remains the production coordinator.
Nemo remains the independent witness; A11oy retains action admission. The random
TinyRLT is deliberately not registered as a natural-language generator. Production
integration still requires an appropriate task-qualified model, broader quality
and abstention evaluation, a reviewed HF sole-writer publication contract and
live revision/byte readback. The v0.2 section below defines the narrower trained
register experiment and actual installed Nemo/Second Brain research integration. No such success
is inferred from a CPU test, a model card, a digest or a reachable website.

Ayllu observation producer/consumer repair is separately tracked by
https://github.com/szl-holdings/ayllu/pull/35 . No duplicated Ayllu patch lives here.
Clinical result interpretation/release, private graph training admission, financial
execution, branch-rule changes and autonomous promotion are outside this package.

## v0.2 — actual training and installed estate integration

The existing untrained continuity demonstration is retained. A separate bounded
experiment now trains this architecture on **synthetic register state labels**:
SET0, SET1, FLIP, KEEP, REVOKE. This is not natural-language pretraining or a
fine-tune of any existing SZL model. The model predicts one of three state labels
from output slots 8/9/10; those trained weights must not be registered as a general
text generator or interpreted using unrestricted `emit_greedy` vocabulary output.

```bash
python -m frontier.rlt.training --output ./register-run-001 --steps 120
python -m frontier.rlt.reporting --evaluation ./register-run-001/evaluation.json --output ./register-run-001/index.html
```

Nine runs: three disclosed seeds and three variants. Training uses 256 programs
of 16 operations, 120 optimizer steps and full BPTT. Validation uses 128 programs;
held-out tests use 128 programs each at lengths 16, 32 and 64. Programs with the
same first six operations always share a split. All datasets and weights are
hash-bound; code movement during an experiment prevents a terminal receipt.
Every seed/variant is reported, and the test split selects neither checkpoints
nor hyperparameters. This is shared-grammar generalization, not semantic
independence or proof of broad reasoning. Labels come from a separate oracle.

`no_output_feedback` disables only the previous-output merge; SWA and encoder
memory remain. The two-layer causal Transformer baseline has the same width,
examples and optimizer steps but different parameter and compute budgets.
**No parameter/FLOP/time-matched superiority or speedup claim is made.** Timings
are measured host observations. All weights are inference artifacts, not
optimizer-resume checkpoints, and all production/publication gates remain false.

`ecosystem.py` is not a replacement inference service. It is exposed as `python -m frontier.rlt.ecosystem`. It adapts the
installed exact-commit Second Brain public index and Nemo E1–E10 envelope witness.
Install `./frontier/rlt[estate]` and the existing root Forge controller package.
It rejects unpinned installed versions and corpus environment overrides. The
existing Forge control-plane contract supplies formula identities; no formula
application or authorization basis is invented for the synthetic task.

An enclosing trusted controller supplies authorization. PRE and POST Nemo
witnesses bind each exact envelope; refusal does not advance an accepted cache.
Compatible prefix continuation restores actual state; edited histories or changed
public evidence force reconstruction. Second Brain content never enters gradients
or the register task inputs. Its handles bind cache validity only: this does not
establish RAG answer quality. The synthetic oracle may verify a single register
answer or cause abstention; it does not turn Nemo into a factual oracle or grant
A11oy tool authority.

CI performs actual installed-package integration outside the checkout, retains
synthetic model weights/data and sanitized evidence, and never publishes private
inference state. The Python 3.12 lane repeats all nine full bounded experiments;
Python 3.11 runs an eight-step optimizer/export smoke, not the same quality run.
The main-push workflow is unfiltered so its check can run for each main revision.
This workflow is read-only toward repository contents and has no HF credential,
provider-write path, automatic promotion, or production deployment.
