# vLLM-Omni RC: executed wheel acquisition, not runtime qualification

Owner: szl-forge#287. Canonical candidate: szl-frontier#129, admitted by #130
at `fa16050c930a2f5a258619101a98bfc2013cb783`. This is the measured successor
to Forge #288; it does not create another intake or replace that contract.

## Exact public artifact

- Upstream `vllm-project/vllm-omni@aff7d64948f6c0e81e9b3272234623044e215967`.
- Wheel `vllm_omni-0.29.0rc1-py3-none-any.whl`, exactly 7,275,029 bytes.
- Wheel SHA-256 `5c5ab1d7e66b2289e81a5f01511bda41f734ea7f02bae1150534e8df16207d79`.
- Core METADATA SHA-256 `d4a701d5851e05b1ab1594897a35d90cedb0dba6333bedeb8cdda1bb36519023`.
- Primary metadata: https://pypi.org/pypi/vllm-omni/0.29.0rc1/json
- Primary file record: https://pypi.org/project/vllm-omni/0.29.0rc1/#files

PyPI's release metadata constrains Transformers to `>=5.13.0,<5.15`. Do not
install this package into the independent Transformers 5.17 evaluation lane.
The Python/ABI wheel tag is not evidence of a complete platform dependency
closure. This acquisition workflow installs only the pytest test harness and imports no
upstream package. It records the resolved test environment separately; that is
not a runtime dependency lock. The file page reports Trusted Publishing was not used; that
is not a malicious-content finding or proof that no other attestation exists.

## Execute

Use a clean Forge checkout with the exact admitted upstream repository checked
out at sibling directory `upstream-vllm-omni`. No upstream setup/build/import
step is run. The workflow creates both checkouts with immutable revisions and
read-only, non-persisted repository credentials, then runs on Python 3.11/3.12:

```sh
python -m pip install --only-binary=:all: 'pytest==9.0.3'
python -m pip check
python -m pytest -q tests/test_vllm_omni_029rc1_contract.py tests/test_vllm_omni_wheel.py
python -O -m unittest discover -s tests -p 'test_vllm_omni_wheel.py' -v
python -m tools.evaluate_vllm_omni_wheel --acquire
```

The network opt-in fetches only the fixed release metadata and wheel URLs.
Redirects, changed size/digest, yanking, malformed metadata and excess bytes
are rejected. TLS validation is preserved; no credentials or ambient proxy
configuration are forwarded. A network-restricted environment must report a
failure rather than bypass transport validation.

The wheel is held in memory, never extracted, installed, imported or uploaded.
Bounded ZIP inspection rejects unsafe/nonregular/duplicate members and verifies
every RECORD hash and size. Packaged source is compared with immutable Git blob
identities, protected additionally by the independent outer SHA-256 pin.
Upstream `setup.py` generates only the specifically identified `_version.py`
exception; its bytes are hashed separately and never claimed Git-matched.
Completeness is required for tracked Python files. Unpackaged non-Python files
are explicitly listed; this is not a claim of model/resource completeness.

## Evidence and limits

The JSON report binds the actual Forge revision and four executing source-file
hashes, upstream revision, downloaded wheel hash/size, METADATA hash, declared
dependencies, all archive member hashes and source correspondence. Reports are
exclusive-created to avoid overwriting an earlier local observation. A failed
attempt exits nonzero and retains failure evidence when preflight permits.
Hosted artifacts contain only reports/logs, never upstream weights or wheels.

`ACQUISITION_VERIFIED_RUNTIME_NOT_RUN` means only that the measured acquisition
and file/source correspondence checks passed. The overall disposition stays
**HOLD**. Observations are unsigned. Source correspondence does not establish
an authenticated/reproducible build, resolved dependencies, an installed
engine, model quality, GPU execution, rollback, or production authorization.
The old qualification contract is not fed fabricated runtime PASS values.

Review must download the Actions evidence archive, verify its recorded digest,
match repository/event/run/attempt/source and inspect the inner observations.
Fixtures are synthetic and cannot replace this real download. A PR result
cannot be relabeled as a fresh protected-main result after merge.

Next qualification remains in Forge with GPU Bridge hardware evidence and
szl-serve serving/rollback ownership: isolated dependency locking compatible
with these exact package constraints, package/build provenance, explicitly
licensed model selection, bounded matched runtime measurements and failures.
Mooncake completion, VAD interruption constraints and migration are separate
unproven gates. No provider mutation, model rehosting, production route/default,
policy bypass or estate alignment claim is granted by this lane.
