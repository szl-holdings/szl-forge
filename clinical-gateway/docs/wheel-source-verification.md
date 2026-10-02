# Source-bound wheel verification

`tools/verify_wheel_source.py` checks the bytes of this project's wheel before
installation, without extracting it, importing packaged Python, or executing the
build backend. It uses only Python 3.11+ standard-library modules and local Git.

```text
python -I -B clinical-gateway/tools/verify_wheel_source.py \
  --wheel clinical-gateway/dist/szl_oac_clinical_gateway-2.5.1-py3-none-any.whl \
  --git-revision <full-lowercase-40-character-commit> \
  --output new-wheel-source-receipt.json
```

The declared revision must exist locally as a commit. Mutable refs are rejected;
Git replacement objects and inherited Git redirection settings are disabled.
Expected bytes are read by immutable blob ID, not from the working tree. A dirty
source checkout therefore cannot redefine the expected package.

## Closed project profile

The verifier reads `pyproject.toml` and the literal `SOURCE_ASSETS` declaration in
`setup.py` from the declared commit. The supported profile is the existing
explicit `src` layout, five configured Python modules, the explicit
`oac_clinical_resources` package, and its source asset bundle. Python module names
are derived from the declaration rather than hardcoded in the verifier. Package
layout, discovery, data, license layout, or build-profile changes require an
explicit verifier update; unsupported declarations fail closed.

The static `dependencies` and `optional-dependencies` arrays in that same
immutable `pyproject.toml` define the complete `Requires-Dist` and
`Provides-Extra` sets. The current verifier accepts package names with simple
numeric version bounds and the wheel's `extra == "name"` marker for a declared
optional dependency. It compares every requirement and extra, including counts;
added, removed, duplicated, retargeted, or weakened dependencies fail even if
the wheel's RECORD is rehashed. Direct URLs, dependency extras, source markers,
and other requirement syntax need an explicit verifier update before packaging.
The same supported numeric bound syntax checks `Requires-Python`; bound order
may differ from `pyproject.toml`, but a changed bound fails.

For the current package, verification covers:

- 17 exact source-byte mappings: five modules, two package Python files, nine
  assets, and the source license.
- Two generated asset-integrity files, reconstructed independently from the
  declared source assets. The generated Python file may use consistent Windows
  line endings from `Path.write_text`; receipts hash the observed bytes. These
  are not labeled as direct Git-byte matches.
- Generated distribution metadata: identity, version, Python requirement,
  summary, license identity, all declared dependencies and extras, wheel
  format/tag, console entry points, and top-level names. Generated metadata is
  not labeled as exact Git-byte provenance; the descriptive metadata body is
  not independently reconstructed by this gate.
- The exact archive member set and all RECORD hashes/sizes, including the
  required unhashed RECORD self-entry.

A maliciously modified module with a recomputed RECORD still fails the immutable
source comparison. Unexpected code, metadata files, and deprecated embedded
signature files are refused by the closed profile.

The gate follows the [PyPA wheel specification](https://packaging.python.org/en/latest/specifications/binary-distribution-format/)
for ZIP packaging and SHA-256 RECORD entries, and the
[PyPA project metadata specification](https://packaging.python.org/en/latest/specifications/pyproject-toml/#dependencies)
for static dependency and extra declarations. It deliberately supports only the
project's pure-Python `py3-none-any` wheel, not every legal wheel or installed
package layout. [Installed package records](https://packaging.python.org/en/latest/specifications/recording-installed-packages/)
have a broader contract and are not accepted as a substitute for wheel records.

## Bounds and receipts

Wheel input is capped at 2 MiB, each member at 1 MiB, expanded members at 4 MiB
total, 256 members, and a 100:1 compression ratio. Only stored/deflated regular
files are accepted. Duplicate/case-aliased names, traversal, Windows-reserved
paths, directories, symbolic/special files, encryption, unsupported compression,
invalid records, and mismatched bytes are rejected before installation. Local
Git reads have byte/time limits, and 512 MiB free storage is required.

The CLI emits and saves either a complete verification receipt or a sanitized
failure receipt. Output paths are created exclusively: an existing receipt is
never overwritten. Receipt write failure returns a separate nonzero result.
The Linux build and Windows consumer retain success/failure receipts with
`always()` and missing-file errors. Linux uploads only the exact wheel filename
that passed the gate; Windows independently verifies it before installation.

This gate proves package/source integrity within the declared profile. It does
not authenticate a source signature, reproduce a build, grant clinical or PHI
authority, validate a site, publish to Hugging Face, or authorize production
promotion. Those fields remain explicitly false, including on failure.
