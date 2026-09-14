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

Use a fresh environment. The CPU wheel index avoids unintentionally provisioning
a CUDA runtime. No GPU, model download or paid service is required.

```bash
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install safetensors==0.7.0
python -m pip install --no-deps ./frontier/rlt
szl-rlt-research --output ./rlt-run-001
```

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
observation -> a11oy.net proof. This PR implements the first executable research
package and its local observation, not the entire deployment chain.

Forge's `inference/governed_inference.py` remains the production coordinator.
Nemo remains the independent witness; A11oy retains action admission. The random
TinyRLT is deliberately not registered as a natural-language generator. Later
integration must supply trained checkpoint bytes, task-specific quality and
abstention evaluation, actual Nemo/Second Brain integration tests, a reviewed HF
sole-writer publication contract and live revision/byte readback. No such success
is inferred from a CPU test, a model card, a digest or a reachable website.

Ayllu observation producer/consumer repair is separately tracked by
https://github.com/szl-holdings/ayllu/pull/35 . No duplicated Ayllu patch lives here.
Clinical result interpretation/release, private graph training admission, financial
execution, branch-rule changes and autonomous promotion are outside this package.
