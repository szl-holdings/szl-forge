# vLLM-Omni binary prerequisite observation

Continues Forge #287 / #292 and canonical Frontier #129. It does not open a new
candidate, replace the byte inspector, or bypass the unresolved runtime gates.

The acquisition implementation on protected main proves a pinned wheel's bytes
and package-source correspondence. Installing an engine also needs a viable
resolved dependency environment. This successor reuses that exact byte inspector
and executes a real pip dry-run in an isolated virtual environment. It does not
install the upstream wheel or execute a model.

## Run

Use the two-checkout layout in the workflow: Forge and the exact upstream
`aff7d64948f6c0e81e9b3272234623044e215967` are sibling directories named `forge`
and `upstream-vllm-omni`. All executing source files must match their Git revision.
On Linux, create a fresh virtual environment and install the hash-locked resolver
from `resolver-requirements.txt`. From the Forge root:

```sh
/path/to/resolver/bin/python -m tools.evaluate_vllm_omni_prerequisites --observe
```

The fixed interpreter tool is pip 26.2.1. Its primary PyPI wheel SHA-256 is pinned;
this is not a runtime dependency lock or a production toolchain change. The child
receives no inherited credential, proxy, pip-index override or PYTHONPATH. Pip
configuration is disabled. The dedicated hosted workflow blocks egress outside its
GitHub/PyPI endpoint list and the hardener's platform-required endpoints. This is
not a network sandbox for manual runs of the Python command. Every resolver invocation uses the public PyPI index,
`--dry-run --ignore-installed --only-binary=:all:`. No arbitrary user requirement,
shell command, source build, dependency skipping or alternate index is accepted.

The actual pinned Omni METADATA requires `openai-whisper>=20250625`. Its binary
prerequisite is tested first, avoiding a large graph download when that necessary
condition already fails. Only after that passes does the full published Omni
wheel requirement resolution run. A nonzero resolver exit is
`BLOCKED_OR_UNAVAILABLE`, not automatically a proven dependency conflict; read
the retained log to distinguish missing wheels, connectivity and solver errors.
Timeout/size errors remain `INCOMPLETE`. Neither state is success. The workflow
stays red when prerequisites are unmet; it does not use continue-on-error.

## Evidence boundaries

Reports bind source revision/file hashes, Python/platform, installed resolver
version, fixed input, raw log hash, exit status, observed timestamps, and any
validated pip report. Reported package URLs/digests are metadata evidence,
**not** independent acquisition or signature verification for those packages.
The existing acquisition result is retained separately within the observation.
Each phase is bounded to 180 seconds, an 8 MiB report/log and an observed 1 GiB
workspace ceiling; polling is a guard, not an OS-enforced disk sandbox. The child
process group is terminated on timeout or bound failure. Source is rechecked
before returning. A fresh output directory prevents reuse of stale reports.

Success means only `OMNI_BINARY_PLAN_REPORTED_NOT_INSTALLED`; disposition remains
HOLD. Full engine closure, base-vLLM version/artifact/hardware selection, build
attestation, installed imports, licensed model execution, GPU correctness,
quality/performance, serving, rollback and production authorization are separate
unmet gates. The full plan covers only requirements declared by the published
Omni distribution; an externally supplied base-vLLM runtime is not inferred.
Transformers >=5.13.0,<5.15 remains unchanged, not forced into the separate 5.17 lane.

A necessary source-only dependency needs its own exact-source, reviewed,
reproducible isolated-build path and artifact provenance before later resolution.
Do not turn off binary-only checking for arbitrary dependencies to get a green
run. Do not convert the preliminary plan into a production lock automatically.
No upstream wheel/weights are rehosted; Actions retains reports and logs only.
A11oy #2010 / Frontier #96 source-projection repair remains independent.
