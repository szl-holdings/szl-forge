# SZL Model Lab

Python backend, Python-rendered operator interface, and two trainable CPU research
baselines inside `szl-holdings/szl-forge`. This is a source integration, not a
trained-model release or a qualification of owner hardware.

## Implemented tracks

| Track | Architecture | Target | Proposed HF model identity |
| --- | --- | --- | --- |
| router | 8 -> 16 -> 1 MLP; 161 parameters | labeled task success | SZLHOLDINGS/A11OY-Router |
| invariant | 8 -> 16 -> 1 MLP; 161 parameters | labeled violation | SZLHOLDINGS/A11OY-Invariant |

Both models are advisory. They cannot authorize execution, override hard denies,
or replace the existing router, invariant implementation, or signing authority.
YARQA encoder work remains DESIGN_ONLY; the kernel suite is not one trained model.
No trained weights are included. Test optimizer runs use synthetic fixtures only.

## Install and inspect locally

Use an isolated Python 3.11/3.12 environment. The existing CUDA environments and
owner workloads must not be modified to install this small CPU module.

```bash
python -m venv .venv-model-lab
# Activate the virtual environment using the command appropriate for your shell.
python -m pip install 'torch==2.10.0+cpu' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -c model-lab/constraints-test.txt -e 'model-lab[test]'
szl-model-lab plan
python -m pytest -q -c model-lab/pyproject.toml model-lab/tests
```

`constraints-test.txt` records exact test versions, not an artifact hash lock or a
production platform qualification. Hosted CI records the resolved environment.
The root Forge package and its required tests remain separate and unchanged.

## Operator interface

Set `SZL_LAB_ACCESS_TOKEN` to an owner-generated random value of at least 32
characters in the process environment, then run `szl-model-lab serve`.
Open `http://127.0.0.1:8765`; HTTP Basic username is `operator`, password is that
token. The default is loopback-only. Do not expose the development HTTP service
publicly or send this credential across an unencrypted network.

`/api/catalog` reports honest model states. `/api/nodes` and `/compute` inspect
only endpoints explicitly configured in `SZL_LAB_OLLAMA_ENDPOINTS_JSON`. Values
must be loopback or literal IPv4 tailnet addresses at port 11434; arbitrary URLs,
redirects and inherited proxies are rejected. No endpoint is guessed. The two
read-only Ollama requests are `/api/tags` and `/api/ps`. Inventory is not serving
qualification, GPU availability, pooled VRAM, a signature, or a quality benchmark.

`SZL_LAB_ROUTER_ARTIFACT` and `SZL_LAB_INVARIANT_ARTIFACT` optionally select trusted
owner-controlled local candidate directories. Missing, corrupt and fixture-only
weights are unavailable, never replaced by random-weight inference. `/api/score`
checks exact features and normalization and returns explicitly uncalibrated advice.
No web route trains, executes commands, publishes, merges or creates jobs.

## Training remains an explicit later owner operation

Read `docs/DATA_CONTRACT.md` and validate data first. The CLI requires explicit
acknowledgements, clean source revision identity, separate train/validation/test
splits and exclusive output directories. Held-out evaluation is a separate command.
Safetensors and local hashes are used; pickle/joblib are not loaded. Unsigned
candidate manifests do not replace existing Forge release receipts.

The current handoff authorizes source and code-only HF alignment, not training,
paid provider calls, remote teacher generation, model pulls or GPU disruption.
Existing Owned Agent Control / gpu-bridge own eventual execution. `szl-router`
retains admission and dispatch; no active learned routing hook is enabled here.

## GitHub -> Hugging Face source alignment

The `Model Lab source and blueprint contract` workflow tests pull requests without
secrets. Its explicitly dispatched main-only publication jobs reuse
`tools/acquire_hf_publisher_token.py`. They export an allowlisted source tree from
an exact Git revision to the two model repositories above, with no checkpoints,
fixtures, private node configuration or tokens. No kernel identity is moved.

The publication default is off. It is permitted only after normal protected-main
admission, a matching dispatch source SHA and passing tests. Each write uses a
conditional HF parent commit and is followed by immutable file-by-file readback.
Existing non-blueprint artifacts are preserved by refusal, not overwritten. Reports
record request-start stages so a timeout is not mistaken for verified publication.
Inspect these reports and the remote state before retrying an uncertain write.

A successful blueprint publication means CODE_ONLY_PUBLICATION_VERIFIED and
BLUEPRINT_NOT_TRAINED. It is not a trained model, an inference-ready HF endpoint,
or a production deployment. The existing trained-model publisher remains separate.
The static Forge Lab Space is not repurposed into a training UI.

## Remaining runtime gates

Owner-machine/tailnet observations, canonical pool attestation, an admitted router
hook, production dependency hash locks, owner-runner registration, genuine training,
held-out evaluation, qualified weights and public product/proof projections remain
separate. Keep `a-11-oy.com` product status and `a11oy.net` evidence downstream of
those actual observations; never infer readiness from a reachable page.
