# Candidate adapter structural guard

This source slice covers newly generated estate payload kits and the explicitly
updated `frontier/brain-navigator-r2-candidate-20261001` copy. It does not silently
regenerate other candidates or alter their manifests, bases, recipes or gates.

Before `attach_adapter` returns a model, the packaged canonical Chaski guard
checks the strict checkpoint header, complete target-name coverage, and exactly
one enabled, active, unmerged `default` namespace. Missing, malformed, empty,
partially covered or inactive adapters fail before their scoring/conversion.
The existing model-only API remains available for export callers. Evaluators
request the loader/structural report for both candidate and comparator. Explicit
bare-base runs instead record `mode=bare_base` and `adapter_keys=null`.

The two vendored `adapter_guard.py` copies must equal `chaski/adapter_guard.py`
after CRLF-to-LF normalization. Their SHA256 pin in `candidate_lib.py` identifies
reviewed repository source, **not** an independent signature or authorization.
The loader reads at most 65,537 bytes, verifies the normalized source digest and
compiles that same buffer; it never falls back to an import-path shadow module.
The bootstrap fetch list includes the guard. The generator copies it without
candidate-template substitutions. Do not run the bootstrap/generator to test
these source contracts: those entry points have external mutation lanes.

Canonical provenance: `chaski/adapter_guard.py` at Forge commit
`89f5d6b5712327e677d3dab3d9b82011598c89c3`; normalized SHA256
`ad2cb8334cddf28782dc3d858d9ccfd5f0de7f892bc02b4447b7180fab525cd2`.

## Boundary

The inherited `applied`/`fully_applied` fields count parameter names, not tensor
values. Successful reports retain `coverage_basis=parameter_names_and_model_status`
and `numerical_application=UNKNOWN`. A source checksum is `REPO_DECLARED` trust.
Real tensor application, immutable loaded-artifact identity, qualified native
dependencies, runtime profile, model quality and publication remain separate
gates. Existing backend/class/precision fallbacks are unchanged by this slice;
they are not newly qualified. In particular, a misaligned loader can now fail
structural coverage instead of being scored as if its adapter were applied.

No trainer, inference, model evaluation, conversion, VM operation or publication
is run by these tests. No budget/ledger or eligibility state is modified.

## Base revision boundary

Both consumer copies reject a missing, mutable or malformed base revision before
the tokenizer/model loader imports its framework. Training, evaluation, export
and runtime qualification also validate immediately after reading candidate
metadata, before data/report reads, GPU/package probes, network access or output
creation. Only an exact lowercase 40-hex string is accepted, unchanged; no
trimming, normalization or default-branch resolution is performed. CPU and
offline flags do not bypass this boundary. Metadata-only `load_candidate` reads
still permit an unbound candidate for source/card/curriculum inspection.

The qualifier additionally compares the Hub response's SHA with the requested
pin, using exact string equality. Wrong, missing, malformed or unavailable
resolution fails qualification and cannot create or replace a runtime lock.
`--offline` records identity as `UNKNOWN`, produces a non-PASS verdict and never
locks, even with a matching existing lock or `--relock`. A matched response does
not erase another failed prerequisite. The remaining lock comparison/relock
behavior is preserved.

The loaders' syntax checks and the qualifier's Hub metadata equality are not
proof that loaded tensors/tokenizer bytes match the pin, nor compute/training
authorization. Those immutable-content and qualified-loader bindings remain
separate gaps. Existing fallback behavior is unchanged. No existing candidate
manifest, recipe, gate, signed artifact or authority is rewritten by this source
change; tests use mocked candidate/lock objects and write sinks only.

## Offline source regressions

```text
python -I -S -B -m unittest discover -s tests -p test_candidate_adapter_attachment.py -v
python -I -S -B -O -m unittest discover -s tests -p test_candidate_adapter_attachment.py -v
python -I -S -B -m unittest discover -s tests -p test_candidate_revision_boundary.py -v
python -I -S -B -O -m unittest discover -s tests -p test_candidate_revision_boundary.py -v
```

Tests use stdlib-only SIMULATED models, checkpoint headers, paths and report sinks
in memory. They do not load actual weights or held-out/curriculum rows. The normal
`base-python-ci` tests-directory suite also discovers them; no new workflow or
remote job dispatch is necessary.

## Explicit loader descriptor (source-only)

`tools/estate_payload/loader_profile.py` adds a pure comparison API,
`observe_loader_profile`. It accepts immutable UTF-8 JSON bytes up to 16 KiB,
checks an externally supplied raw SHA256 before parsing, and requires exact
candidate snapshot, recipe snapshot and purpose comparisons. It rejects
duplicate, missing, unknown or mistyped fields and unsupported choices.
Base and tokenizer identities, runtime-identity digest, one backend/factory,
expected class/type/view, quantization/dtypes and CPU/CUDA placement are explicit.
No-offload is the only supported descriptor policy in this first slice.

Acceptance returns immutable `MODELED` / `BLOCKED` comparison evidence with
`UNKNOWN` signing trust and permanently false execution, training and publication
properties. Boolean coercion raises. These properties are not dataclass fields:
future serialization must explicitly retain them; ordinary `asdict`/`repr` is
not a complete evidence export. No serializer is introduced here.

The module does not authenticate caller pins, read artifacts, import frameworks,
load a model, observe actual tensors/precision/placement or choose a fallback.
It is not integrated into existing loading entry points yet; actual loader
selection and returned-model inspection remain separate work. No real profile,
candidate ID, native dependency pin, signed SKU, runtime lock or grant is created.

```text
python -I -S -B -m unittest discover -s tests -p test_loader_profile_contract.py -v
python -I -S -B -O -m unittest discover -s tests -p test_loader_profile_contract.py -v
```

The 13 tests use only in-memory SIMULATED descriptors. The existing tests-directory
CI discovers them; test success is not runtime qualification or model evaluation.
