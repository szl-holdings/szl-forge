# Actual Kernel 0.16.1 CPU observation

This is a separate evaluation lane, not a replacement for the canonical publisher.
The source-first chain and existing 0.16.0 production-release verifier are unchanged.

The workflow `kernel-0161-qualification.yml` runs two independent, disposable Linux
CPU containers. Both actually install `kernels==0.16.1`; tests do not substitute a
mock client for the hosted execution. The source candidate is the admitted
`szl-holdings/szl-kernels@782c04a5c6affffdb8f497e77ae2d0fd8d73ed70`. Four Git blob
identities are checked before source is staged; SHA-256 fingerprints are also
recorded and rechecked by the host. The fixed provider candidate is first-class
`SZLHOLDINGS/szl-kernels@09818b62d683c33d200fca32e2ebfd95c64c65c7`, previously
observed as v1. This pin is a dated candidate, not a claim about the latest branch.

## Two scopes, never an implicit fallback

The **source** lane uses `get_local_kernel` on a minimal source staging tree with
both supported loader layouts. It has no network at execution time. It evaluates
the actual public CPU API selfcheck, independently computed RMS normalization,
four invalid-threshold cases, and fixed cosine-retrieval values and receipt bytes.
The host reuses the existing canonical numerical/receipt validator, and validates
source fingerprints. Success means `CPU_SMOKE_VERIFIED` for this source-local
candidate. It does not mean the Hub can load it, a model was trained, all suite
operations were exhaustively verified, or a GPU kernel is qualified.

The **provider** lane first reads the fixed first-class repository revision and
compares the expected eight executable paths (four modules in two layouts) with
the admitted source. Missing or unexpected build files produce
`SOURCE_PROVIDER_DRIFT` without importing the provider package. Complete matching
bytes permit the client-policy check. Untrusted or unknown publisher status stays
blocked. A supported publisher is loaded using the real `get_kernel` with
`trust_remote_code=False`, and matching loader-origin metadata is required before
the same CPU smoke checks. There is no retry with broad trust, repository-list
trust, local overrides or offline bypass. The source lane is planned separately;
it is never used to turn a failed provider observation into provider success.

A completed provider drift/trust observation exits zero because the read-only
observation succeeded, but `runtime_qualified` is false and no numerical result is
present. This is explicitly an observation workflow, not a fleet readiness gate.
Runtime/network/malformed-data failures exit two and fail their job. Do not infer
fleet or production readiness from the workflow's color. Both lanes always keep
`production_authorization=false` and `gpu_qualified=false`. No publisher permission,
consumer pin update, provider ref creation or model promotion follows automatically.

## Isolation and evidence

The Docker base image is inherited from the existing digest-pinned runtime. Direct
Python dependencies are version-pinned and both pip install reports retain actual
wheel URL/hash provenance. Transitive packages are recorded, not advertised as an
exhaustive reproducible dependency lock. The container image ID and inspection,
executing Forge checkout, kernel source revision, stderr, numerical/blocked report,
and cleanup observation are retained as Actions artifacts for 30 days. These are
run evidence, not immutable long-term storage or independent signatures.

Runtime containers have no credential or workspace mounts, use a non-root user,
read-only root, dropped capabilities, no-new-privileges, bounded temporary storage,
128 PIDs, 3 GiB memory, two CPUs and a 240-second execution limit. Both checkouts
disable persisted Git credentials. The source runtime has network=none. The provider
runtime uses bridge networking to contact public HF; this is not an OS-level egress
allowlist. No GPU, owner device, HF paid job, cloud resource, or secret is requested.
The cleanup step removes only the exact run/attempt/lane container and verifies
that no such container remains. Ordinary runner loss remains outside this local
cleanup observation. Provider/library-reported origin is not independent authorship
or build-chain attestation.

Local regression command:

```bash
PYTHONPATH=tools python -m unittest -v tools.test_qualify_kernel_0161
```

A hosted run is needed for actual installation and execution. Re-observe and review
source/provider pins intentionally for successors. Never repoint a branch or weaken
trust just to make a report green. The canonical release workflow remains the only
publication path; website/Space deployment remains with the native estate operator.

Version-specific primary implementation:
https://github.com/huggingface/kernels/blob/1dc5c9d05683e8a594bb6d7a1ab18318b89e2caa/kernels/src/kernels/utils.py

Migration reference: https://huggingface.co/docs/kernels/migration
