# PEFT export integrity: executable CPU evidence

Owner: Forge #257 / PR #258. Canonical intake: Frontier #110.
Upstream: `huggingface/peft@78bce7cb48f800a7ad0d352b68a46302e13e1687`.
The original evaluation-plan.json and its entire P0-P5 acceptance remain intact.
Production qualification and source admission are separate gates.

## Run the bounded lanes

```sh
python -m tools.evaluate_peft_export
python -m unittest discover -s tests -p 'test_peft_*.py' -v
python -O -m unittest discover -s tests -p 'test_peft_*.py' -v
# In the exact isolated runtime environment from the workflow:
python -m tools.evaluate_peft_runtime run
python -m tools.evaluate_peft_fsdp
# In the separate original safetensors 0.7 environment:
python -m tools.evaluate_peft_export_interop
```

The reference installation and execution sequence is
`.github/workflows/peft-export-runtime.yml`. Run from a clean Git checkout.
Commands have no model, export-path or URL arguments. No pretrained weights are
downloaded, training is not performed, and there is no publisher in this lane.
All exports are small synthetic fixtures in private temporary directories,
removed when each command exits. Observations go to stdout.

## Byte inspector

`inspect_adapter(artifact_bytes, config_bytes, binding)` checks declared
artifact/config digests, source/base bindings, exact independently declared
tensor inventory and shapes, complete A/B pairs, ranks, dtypes, strict JSON,
finite payload values and contiguous byte accounting. Layout is checked before
values, bounding work even on forged overlapping descriptors.

Scope: 4 MiB artifact, 64 KiB header/config, 256 tensors, F16/BF16/F32/F64 linear
LoRA only. This is not a replacement for the official safetensors loader.
DoRA/AdaLoRA auxiliary vectors are not malformed A/B findings; the linear-LoRA
inspector leaves those methods UNSUPPORTED. Caller binding is
CALLER_DECLARED_NOT_AUTHENTICATED. A structural pass is never publication rights,
a complete distributed gather, model quality or authenticated external lineage.

## Real PEFT CPU exports

The runtime pins PEFT, Transformers, Accelerate and Hub to exact Git commits;
validates PEP 610 installation metadata; and checks the actual loaded critical
PEFT save/load file against its upstream Git blob. PEP 610 is local metadata,
not independent package attestation. Torch 2.10.0+cpu, safetensors 0.8.0 and
numpy 2.3.5 are fixed. Ancillary resolved dependencies and pip installation
reports are retained, not misrepresented as a pre-existing hash-locked closure.

The twelve-parameter synthetic base is explicitly SYNTHETIC_RECIPE_NOT_HUB_MODEL.
Its recipe is bound to the actual Forge execution commit and its serialized
bytes have a SHA-256. A valid nontrivial adapter must export and reload with exact
output equality, unchanged adapter bytes and a successful wrong-base negative.
Twelve real save_pretrained cases cover A/B x flat/empty/scalar x inferred/explicit
state dict. The inferred path must warn AND write; explicit state dict must
bypass the warning. The separate byte inspector still rejects malformed A/B.
Direct helper tests cover legitimate one-dimensional DoRA/AdaLoRA vectors, and
a real DoRA export remains UNSUPPORTED by the linear-LoRA inspector.

## Real FSDP CPU gathering and supervised failure

The FSDP command starts two owned CPU processes using Gloo on loopback, with
`use_orig_params=True`. It does not emulate the actual sharding or gather.
Each rank must observe its real one-dimensional parameter shards. Ungathered
PEFT export must warn, write malformed bytes and fail the independent inspector.
`FSDP.summon_full_params(..., writeback=False, rank0_only=False)` must produce the
expected complete A/B shapes and equal gathered artifact hashes on both ranks.
After leaving the gather context, every local shard must be restored exactly.
Both gathered exports must reload against the exact synthetic base with zero
absolute output error and unchanged adapter bytes.

Three separate real two-process failure fixtures run after both workers have
initialized FSDP and exposed shards: abrupt owned rank-0 exit with code17, a
stalled group with a bounded timeout, and parent-initiated cancellation. Every
fixture must reject the group, stop all workers and emit no success result.
The supervisor has a 120-second absolute bound and terminates/kills and joins
remaining workers. These are local CPU process/failure tests and gather-context
restoration, not GPU/ZeRO-3 qualification, a cancellation of the hosted Actions
run, or a production release rollback.

## Independent dependency scopes and evidence

Transformers 5.17 at the selected source requires safetensors>=0.8.0. Its real
PEFT environment uses exactly0.8.0. The original0.7.0 official-loader
interoperability gate remains unchanged in a separate clean job, with
packaging26.3 (required by that older Torch wrapper). No dependency resolver,
version check or test is bypassed. Historical0.7 results do not qualify0.8.
The standard project/production dependency set is unchanged.

The new workflow runs real PEFT and FSDP on Python3.11/3.12, with exact PR-head
checkout, contents:read, no secrets or credential persistence, offline Hub
flags, and15-minute job bounds. Single-process Python socket attempts fail;
that trap is not an operating-system sandbox. The distributed lane requires
loopback Gloo sockets. There are no external provider or model-download calls.

Each attempt retains exact source, installation reports, resolved package
versions and runtime/FSDP JSON, even on failure. Verify the outer Actions run's
head and artifact digest before accepting an inner observation. Missing or
failed execution is not a pass. All observations remain unsigned and require
that external evidence binding. Older local-observation files remain dated
byte-inspector observations; they are never rewritten as new runtime evidence.

## Production boundary and rollback

CPU fixture passes do not grant training, publication, routing or automatic
promotion. GPU and ZeRO-3 behavior, authenticated external-model lineage,
separately authorized immutable provider readback, model quality, independent
publisher admission and production rollback remain unqualified. The original
P0-P5 gate list stays authoritative; no missing predicate becomes PASS.

Rollback reverts the standalone evaluators and their CPU-only workflow. Existing
model artifacts, publishers, production dependency defaults and routes remain
unchanged. No weights are rehosted, no HF asset is mutated, and no protected-main
bypass or credential change is required.
