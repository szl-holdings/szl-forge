# Payload 1 — Installed HF tooling integration, 2026-09-10

Owner: `szl-holdings/szl-forge`.
Order: GitHub → Hugging Face → a-11-oy.com → a11oy.net.

## Continue the existing work

Forge #212 already merged the TRL/Transformers/Accelerate exact-source guard.
Forge #215 already merged the Tau 0.4.2 contract. This change adds executable
adapters and real installed-library evaluation, not duplicate intake contracts.
It also adds the Hub 1.31.0 compatibility lane. No production dependency lock,
model route, Space, deployment, secret or branch protection is changed.

| Distribution | Evaluation version | Exact upstream Git commit |
|---|---|---|
| huggingface-hub | 1.31.0 | 495b17c8529614759ae0f1ccf1ebe9a61c148b7c |
| trl | 1.13.0 | 3d9261f1fec9f9a8140099c78a65c7da73dce79c |
| transformers | 5.17.0 | 856157a2f3e9594954310df18fdccc31ffddebe9 |
| accelerate | 1.15.0 | 6afc1e5ee217051fde702b23de2813344dc0fd33 |
| tau-ai | 0.4.2 | 55df51608b8b2d172c4bbac2cd11e8345e307476 |

Hub's annotated v1.31.0 tag resolves through tag object
`32ccc9ee57f3b3546165de105b4a0d8eba1b7446` to the commit above.
Tau's distribution is **tau-ai**, not an unrelated package named tau.
Tau requires Python 3.12 or newer; the installed-runtime matrix uses Python 3.12.

## Files and callable integration points

`inference/hf_tooling.py` is included in Forge's existing inference package.
It exposes exact PEP 610 Git-install verification, pinned requirements generation,
`job_labels`, `sft_config`, `tau_evidence_entries`, and `authorized_catalog`.

`job_labels` maps a non-secret run ID, source SHA and vertical to dedicated Sandbox
labels. It does not create a billable Job. The SDK tests verify both rejected
reserved/invalid labels and the valid-input boundary before any provider call.

`sft_config` constructs actual TRL SFTConfig with chunked_nll, no packing, no
remote model code, no Hub push and no experiment-tracker publication. It does
not load weights or train. Context length is bounded to 1,048,576; accepting that
configuration is emphatically not evidence of executing at that length.

`tau_evidence_entries` creates actual Tau custom_message and bookmark objects
containing source/run/receipt references, never credentials or raw private memory.
The host must authorize the tenant and own a single-writer append_batch transaction.
The entries cannot authorize a command. Catalog discovery intersects the host's
explicit allowlist; a newly discovered model is not a newly approved model.

`tools/evaluate_hf_tooling.py` runs real library probes after checking both the
installed version and the exact upstream Git source in direct_url.json. It feeds
the measured narrow CPU results back into the existing core-stack guard and binds
the existing Tau contract digest. Missing dependencies emit UNAVAILABLE, failed
probes emit SMOKE_FAIL; neither becomes a green runtime result.

## Actual probe scope

Hub: validate Sandbox labels before provider access; reject unsafe POSIX and
Windows-style remote filenames through actual HfFileSystem.get_file before local
writes; verify the shared httpx exception API. The optional live probe reads only
an immutable, bounded public README, compares three concurrent cached downloads,
and checks that a dry run does not copy cached bytes into local_dir. It cannot
download model weights or execute the README. Linux and Windows get separate jobs.
A dry run may create the inspected SDK cache bookkeeping (.gitignore, CACHEDIR.TAG
and lock files); the check explicitly records those and rejects any requested
README payload or unknown file. It does not claim zero filesystem writes.
This is not an injected timeout test or proof of every mutable-ref race scenario.

TRL: eight small CPU float32 chunked-NLL loss/gradient comparisons against ordinary
PyTorch cross entropy, including masking, all-masked input, causal versus explicit
shift, bias, scaling and softcap. Also construct a 1M-token config without allocating
that sequence; generate with a tiny random local Llama; save and restore a single-
process Accelerate CPU checkpoint after one synthetic optimizer step. No downloaded
model, real user corpus or paid GPU is used. These are correctness smoke probes,
not a throughput, model-quality, large-context or distributed-training benchmark.

Tau: append/reopen/replay first-class custom-message and bookmark entries, clear
a bookmark, replay the old compaction/leaf-pointer shape, reject malformed session
records, and verify SZL's catalog non-authority boundary. All records are synthetic.
Real provider streams, shell execution and private session corpora are not exercised.

## Execute

Offline controller tests need only the standard library:

```bash
python -m unittest discover -s tests -p 'test_hf_tooling_runtime.py' -v
python -m tools.evaluate_hf_tooling plan
python -m tools.evaluate_hf_tooling requirements --lane hub
```

Use the dedicated `HF tooling installed runtime` workflow. Each lane installs into
its own fresh hosted-runner venv. TRL uses an explicitly CPU-only Torch 2.10.0 install.
The job captures the pip install report and actual resolved dependency inventory.
The primary sources are exact Git pins; transitive dependencies are **not yet a
fully hash-locked closure**. Reproducible release promotion must add that closure,
not silently claim it from a list of top-level version pins.

For a separately prepared matching evaluation environment:

```bash
python -m tools.evaluate_hf_tooling run --lane trl --output artifacts/trl-runtime.json
python -m tools.evaluate_hf_tooling run --lane tau --output artifacts/tau-runtime.json
python -m tools.evaluate_hf_tooling run --lane hub --live-hub --output artifacts/hub-runtime.json
python -m tools.evaluate_hf_tooling verify --output artifacts/hub-runtime.json --expected-source "$(git rev-parse HEAD)"
```

Run only from a clean tracked-source checkout. Runtime commands record the actual
Git SHA and source-file SHA-256s. Local controller fixtures are not runtime evidence.
The local Python socket trap catches accidental egress; it is not an OS sandbox
for hostile extensions. No credentials are supplied to the evaluation jobs.

## One Python handoff block

```python
from pathlib import Path
import subprocess
import sys

required = ("inference/hf_tooling.py", "tools/evaluate_hf_tooling.py",
            "tests/test_hf_tooling_runtime.py", ".github/workflows/hf-tooling-runtime.yml")
missing = [name for name in required if not Path(name).is_file()]
if missing:
    raise SystemExit(f"Missing committed integration files: {missing}")
subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests",
                "-p", "test_hf_tooling_runtime.py", "-v"], check=True)
subprocess.run([sys.executable, "-m", "tools.evaluate_hf_tooling", "plan"], check=True)
print("Controller validated. Use the installed-runtime workflow for real library evidence.")
```

## Evidence, admission and downstream ownership

All generated receipts retain HOLD and false production/job/tool/publishing flags.
SMOKE_PASS describes only the listed executed checks. Unmeasured requirements stay
UNAVAILABLE and the full core-stack allChecksPass flag remains false. Hashes check
integrity, not publisher authenticity. Verify the exact GitHub workflow, source SHA,
job/OS/lane and artifact archive digest before accepting a report. CI artifacts have
30-day retention, not permanent immutable storage.

The corresponding Frontier intakes are #61 (core stack) and #62 (Tau successor).
Use their existing records to attach exact executed evidence. Do not create more
Spaces or duplicate source-only waves. Hub 1.31 is an infrastructure update, not a
new SZL foundation model. Normal publisher regression and a fully admitted dependency
closure must pass before selectively upgrading production publisher clients.

After the exact-head CI and review gates pass, merge normally, inspect the main-run
artifacts, and attach their precise hashes to the existing intakes. Only the existing
single-writer publication paths may update HF/runtime projections. Product status
on a-11-oy.com and proof on a11oy.net must distinguish installed compatibility from
production/model qualification; no live domain consumption is implied by this PR.
Inventory issue .github#728 and Lyte issue lyte-services#18 remain independent and
must be re-observed before any claim that estate drift is resolved.

Rollback: reviewed revert of these additive files. Existing production requirements
and serving behavior were never changed, so this slice requires no model rollback.

## Primary sources

- https://github.com/huggingface/trl/releases/tag/v1.13.0
- https://huggingface.co/docs/trl/long_context_training
- https://github.com/huggingface/huggingface_hub/releases/tag/v1.31.0
- https://github.com/huggingface/huggingface_hub/pull/4824
- https://github.com/huggingface/huggingface_hub/pull/4828
- https://github.com/huggingface/tau/releases/tag/v0.4.2
- https://github.com/szl-holdings/szl-forge/pull/212
- https://github.com/szl-holdings/szl-forge/pull/215

Upstream performance figures are not copied into SZL measurements.
