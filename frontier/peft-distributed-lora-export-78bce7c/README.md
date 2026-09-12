# PEFT export integrity: executable bounded byte evaluation

Owner: Forge #257 / PR #258. Canonical intake: Frontier #110.
Upstream binding: `huggingface/peft@78bce7cb48f800a7ad0d352b68a46302e13e1687`.

This advances a **subset of P4**, not completion of P0-P5. PEFT's upstream
inferred-state-dict heuristic warns; it does not refuse publication. Its
explicit-state-dict bypass is unchanged. This new SZL evaluator does not import
PEFT or modify any publisher, dependency lock, model route, or production default.

## Run without optional dependencies

From the repository root:

```sh
python -m tools.evaluate_peft_export
python -m unittest discover -s tests -p 'test_peft_export_bytes.py' -v
python -O -m unittest discover -s tests -p 'test_peft_export_bytes.py' -v
```

The first command inspects a fixed synthetic artifact and prints its bounded
report to stdout. It opens no paths, creates no model artifacts, and calls no
provider. Exit zero means only that this fixture passed the structural check.

`inspect_adapter(artifact_bytes, config_bytes, binding)` accepts bytes that a
separately authorized caller has already acquired. The binding must declare the
exact artifact/config SHA-256, base repository/revision, upstream PEFT revision,
state-dict provenance, and an independently specified complete tensor/shape map.
Never derive the expected inventory from the artifact being tested in a real
qualification path. This function compares the declarations but does not
authenticate them: `CALLER_DECLARED_NOT_AUTHENTICATED` is always explicit.

## What this evaluator actually checks

The parser is deliberately bounded to a 4 MiB artifact, 64 KiB header/config,
256 tensors, and F16/BF16/F32/F64 storage. It rejects duplicate JSON keys,
non-finite JSON numbers (including exponent overflow), malformed descriptors,
truncated bytes, forged lengths, gaps, overlap, trailing payload, empty/flat
A/B shards, missing A/B pairs, inventory/shape/rank/dtype mismatches, and actual
NaN/Inf values in the tensor payload. It does not allocate a model or interpret
pickle or custom code. Input-size or format limits are evaluation rejection,
not evidence that an otherwise valid larger adapter is corrupt.

Only standard serialized **linear LoRA** pairs receive STRUCTURE_ONLY PASS.
DoRA magnitude vectors and AdaLoRA lora_E are recognized as outside the A/B
shape heuristic; they lead to UNSUPPORTED/HOLD, never a false malformed-A/B
finding or a claim that those methods were evaluated. Convolutional tensors,
other dtypes, rank patterns, modules-to-save, biases and other formats need
separate scoped evaluation. Unsupported means unqualified, not defective.

## Optional official-loader interoperability

`python -m tools.evaluate_peft_export_interop` requires the exact recorded
CPU environment: torch `2.10.0+cpu`, safetensors `0.7.0`, numpy `2.3.5`.
It refuses a different version set rather than silently transferring evidence.
It does not install dependencies or change project dependencies.

This cross-check reads our fixtures with the official safetensors loader,
reads official serialized bytes with the bounded inspector, and compares a
small synthetic linear expression before/after load for all four storage
formats. Numerical comparison uses float64 arithmetic. These are eight
serialization directions and four synthetic roundtrips, **not a PEFT model
reload, distributed execution, performance benchmark, or adapter quality test**.
The dependency-free tests are collected by the existing base Python CI.
The optional interoperability command is separate and its local evidence must
not be described as a hosted CI result.

## Remaining gates / Codex continuation

Keep PR #258 draft until normal source checks and review pass. Preserve the
existing evaluation plan and its entire P0-P5 acceptance list. Execute pinned
PEFT warning/inferred/explicit-state-dict tests under the real package; produce
actual ZeRO-3/FSDP gathered/ungathered evidence on authorized hardware; verify
an exact base-model/config-bound save/reload and separately authorized immutable
provider readback. Review this subset parser against more official adversarial
fixtures before considering use outside evaluation. Do not wire it into a
publisher until that publisher's independent controls admit the change.

Every report keeps production/publication/automatic promotion false. Binding
authentication, distributed gathering, PEFT execution, exact-base reload and
provider readback remain NOT_RUN. No signed receipt or full source admission
is implied by the unsigned local observation JSON. Rollback removes only these
standalone evaluation files; existing publisher behavior and artifacts remain
unchanged.
