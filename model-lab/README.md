# SZL Model Lab

Python backend, Python-rendered operator interface, and two trainable CPU research
baselines inside `szl-holdings/szl-forge`. This is a source integration, not a
trained-model release or a qualification of owner hardware.

## Implemented tracks

| Track | Architecture | Target | Proposed HF model identity (hub_state NOT_PUBLISHED) |
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
python -m venv ../szl-model-lab-venv
# Activate the virtual environment using the command appropriate for your shell.
python -m pip install 'torch==2.14.0+cpu' --index-url https://download.pytorch.org/whl/cpu
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
weights are unavailable, never replaced by random-weight inference. `/api/score/{track}`
checks exact features and normalization and returns explicitly uncalibrated advice.
No web route trains, executes commands, publishes, merges or creates jobs.

## Evidence lineage

`/evidence` and `/api/evidence-lineage` share one authenticated backend projection.
Supply the owner-controlled `SZL_LAB_EVIDENCE_SNAPSHOT` file and its external
`SZL_LAB_EVIDENCE_SNAPSHOT_SHA256` to inspect the existing public coverage contract.
The optional what-if query shows documentation projections that would need recheck
if an input changed; it changes no baseline, model, qualification or permission.
Missing or invalid evidence returns 503 with null inventory, not a passing score.
See [EVIDENCE_LINEAGE.md](docs/EVIDENCE_LINEAGE.md) for the trust boundary, tests,
research basis and code-only export closure. No additional service or dependency
is installed, and the existing runtime and publisher controls remain in force.

## Release identity decision (HOLD-only)

`/release-decision` and `/api/release-decision` share a single authenticated,
read-only comparison of the source, Space, gateway, and receipt identities.
Supply `SZL_LAB_RELEASE_SNAPSHOT` and `SZL_LAB_RELEASE_SNAPSHOT_SHA256` as an
owner-controlled file and an externally pinned digest. The viewer never fetches
provider state, verifies signatures, grants execution/publication authority, or
qualifies a release. Incomplete or conflicting inputs remain visible as HOLD;
what-if input invalidation is ephemeral. See
[RELEASE_DECISION.md](docs/RELEASE_DECISION.md) for the exact schema and tests.

## Training remains an explicit later owner operation

Read `docs/DATA_CONTRACT.md` and validate data first. The CLI requires explicit
acknowledgements, clean source revision identity, separate train/validation/test
splits and exclusive output directories. Held-out evaluation is a separate command.
Safetensors and local hashes are used; pickle/joblib are not loaded. Unsigned
candidate manifests do not replace existing Forge release receipts.

For an owner-controlled private archive, `docs/ARCHIVE_STAGING.md` describes
the local byte-pinned stage, training manifest gate, and local candidate copy.

The current handoff authorizes source and code-only HF alignment, not training,
paid provider calls, remote teacher generation, model pulls or GPU disruption.
Existing Owned Agent Control / gpu-bridge own eventual execution. `szl-router`
retains admission and dispatch; no active learned routing hook is enabled here.

## GitHub -> Hugging Face source alignment

Neither proposed model identity exists on the Hub. `/api/catalog` and
`szl-model-lab plan` report `hub_state: NOT_PUBLISHED` for every track, and no
committed workflow holds a write path to either id. The `Model Lab source and
blueprint contract` workflow runs tests and credential-free code-only plans only;
its former dispatch-only publication jobs, which could create both repositories,
were removed (HF upgrade plan P9: one committed writer per Hub asset, no latent
create paths to absent ids). `tests/test_workflow_contract.py` fails if any
workflow names an unpublished catalog id.

`blueprints.py` still builds the allowlisted code-only projection and keeps its
conditional-parent, file-by-file readback publisher for a future release. Creating
either repository is an owner decision; it then needs one deliberate writer
workflow and a `hub_state` change in the same pull request. A blueprint
publication would mean CODE_ONLY_PUBLICATION_VERIFIED and BLUEPRINT_NOT_TRAINED,
never a trained model, an inference-ready HF endpoint or a production deployment.
The static Forge Lab Space is not repurposed into a training UI.

## Remaining runtime gates

Owner-machine/tailnet observations, canonical pool attestation, an admitted router
hook, production dependency hash locks, owner-runner registration, genuine training,
held-out evaluation, qualified weights and public product/proof projections remain
separate. Keep `a-11-oy.com` product status and `a11oy.net` evidence downstream of
those actual observations; never infer readiness from a reachable page.
