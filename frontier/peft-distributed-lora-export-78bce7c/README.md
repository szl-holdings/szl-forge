# PEFT export integrity: byte inspection and real CPU execution

Owner: Forge #257 / PR #258. Canonical intake: Frontier #110.
Upstream: `huggingface/peft@78bce7cb48f800a7ad0d352b68a46302e13e1687`.

The full P0-P5 plan remains EVALUATION/HOLD. Source admission, CPU fixture
execution, distributed gathering and production publication are separate gates.
No command in this lane publishes, trains, downloads pretrained weights, or
accepts a user-supplied model, URL or export directory.

## Dependency-free byte inspector

```sh
python -m tools.evaluate_peft_export
python -m unittest discover -s tests -p 'test_peft_*.py' -v
python -O -m unittest discover -s tests -p 'test_peft_*.py' -v
```

`inspect_adapter(artifact_bytes, config_bytes, binding)` checks a bounded linear
LoRA safetensors subset: exact declared inventory and shape, complete A/B pairs,
rank/dtype agreement, JSON, payload accounting, finite values and content/config
hashes. It accepts previously acquired bytes, not paths. Limits: 4 MiB artifact,
64 KiB header/config, 256 tensors, F16/BF16/F32/F64. This is not a replacement for
the official safetensors loader. Unsupported formats are unqualified, not corrupt.
DoRA/AdaLoRA auxiliary parameters are not malformed A/B findings; they remain
UNSUPPORTED by the linear-LoRA evaluator. Caller bindings remain
CALLER_DECLARED_NOT_AUTHENTICATED. Do not derive an expected tensor inventory from
the artifact under inspection in a qualification path.

## Real pinned-PEFT CPU lane

The `PEFT export runtime evaluation` workflow installs an isolated CPU environment
on Python 3.11 and 3.12. Existing project dependencies and production workflows
are unchanged. It executes the exact PR head (or admitted push SHA), verifies a
clean checkout, validates PEP 610 Git bindings for PEFT/Transformers/Accelerate/Hub,
and checks the installed critical PEFT save/load file against its upstream Git
blob. Torch 2.10.0+cpu, safetensors 0.7.0 and numpy 2.3.5 are fixed. Ancillary
resolved dependencies and pip download/installation reports are captured with
the observation; they are not represented as a pre-existing hash-locked closure.

The executable `python -m tools.evaluate_peft_runtime run` covers:

- Real PEFT save/load for a deterministic, nontrivial linear LoRA adapter.
- Exact synthetic base bytes, strict output equality before/after reload, a
  wrong-base negative, and unchanged adapter bytes after reload.
- Twelve real save_pretrained cases: A/B times flat/empty/scalar times
  inferred/explicit state dict. Inferred states must warn but still write;
  explicit states must bypass the upstream warning. All malformed A/B exports
  must remain rejected by the separate SZL structural evaluator.
- Legitimate 1-D DoRA/AdaLoRA helper inputs must not trigger the upstream A/B
  warning. A real DoRA export remains UNSUPPORTED by the linear-LoRA evaluator.
- Fail-closed observation validation: partial cases, wrong revisions, invalid
  reload evidence or authority flags cannot exit successfully.

The synthetic base is twelve fixed CPU parameters, identified as
SYNTHETIC_RECIPE_NOT_HUB_MODEL. Its recipe binds to the exact Forge execution
commit and its actual bytes have a SHA-256. It is not an upstream pretrained
model or proof of an external model's authenticated lineage. No Hub weights
are downloaded or rehosted. Files exist only in a private temporary directory
and are removed on exit. Python socket attempts fail during fixture execution;
this trap is not an OS sandbox. Hosted workflow permissions are contents:read,
with no secrets, no credential persistence, and a 15-minute job limit.

To reproduce in an isolated environment, install the CPU torch pin separately
from the official PyTorch CPU index, then generate and install the exact-source
requirements with `python -m tools.evaluate_peft_runtime requirements`. Run the
runtime command only from a clean checkout. All output evidence goes to stdout.
The workflow is the reference installation/command sequence; there is no new
production dependency default.

## Evidence and remaining acceptance

Each hosted attempt uploads source.txt, installation reports, the resolved
package environment, runtime.json and official safetensors interoperability
output, even on failure. A missing/failed runtime.json is not success. Verify the
outer Actions run head and artifact digest before relying on the contents.
CPU_EVALUATION_PASS is limited to these real-package CPU fixtures, never to
actual ZeRO-3/FSDP gathering or production qualification. PEP 610 is local
installation metadata, not an independent package attestation.

The original local-observation.json and local-observation-v2.json are preserved
as dated predecessor byte-inspector evidence, not new CPU-execution receipts.
A run result is recorded only after the actual job and artifact are inspected.

Real distributed gathering, distributed cancellation/failure/rollback,
authenticated external-base lineage, separately authorized immutable provider
readback, model quality, and independent publisher admission remain required.
No successful test substitutes for them. The original evaluation-plan.json and
all acceptance criteria remain unchanged. PEFT's warning is not write denial.

Rollback reverts these standalone evaluation files and their CPU-only workflow;
existing model artifacts, publishers, routes and permissions remain unchanged.
