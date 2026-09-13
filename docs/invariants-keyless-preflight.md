# Invariants keyless preflight — source-only successor

## Status and deliberately narrow scope

This change adds a non-publishing credential preflight, offline tests, and a
read-only CI lane. It does **not** modify `publish-szl-invariants.yml` or
`publish_szl_invariants.py`, configure Hugging Face, exchange real credentials,
upload artifacts to Hugging Face, deploy, train, or qualify a runtime.

The complete publisher integration remains separate. A write of that larger
publisher change was blocked by the tool safety check during preparation; it
was not rerouted or split to evade that check. The smaller preflight-only scope
leaves the live release behavior unchanged. Do not call this publication fixed.

## Observed blocker

Runs 34727235269 and 34727278848 passed final source authorization but failed at
the first model metadata read because their configured OAuth token was expired.
A publication-named evidence archive contained only authorization evidence.
Artifact upload success is not a Hugging Face publication receipt.

## Two identities for two targets

Official Hugging Face Trusted Publishers issue short-lived write grants scoped
to one repository. These two publication targets therefore need separate grants:

| Hub type | Exact `HF_OIDC_RESOURCE` |
|---|---|
| Model mirror | `SZLHOLDINGS/szl-invariants` |
| First-class kernel | `kernels/SZLHOLDINGS/szl-invariants` |

Both repository Trusted Publisher bindings should require:
- Provider: GitHub Actions; issuer `https://token.actions.githubusercontent.com`.
- Audience: `https://huggingface.co`.
- `repository`: `szl-holdings/szl-forge`.
- `branch`: `main`.
- `workflow`: `publish-szl-invariants.yml`.

Configure bindings in each typed target's Settings → Trusted Publishers. A
Hub maintainer with the required access must do this; this PR does not.
Do not substitute an account-level CI/CD identity, an unrelated Space binding,
or the separate trusted-kernel-loader badge.

Protocol: https://huggingface.co/docs/hub/trusted-publishers

The helper uses the pinned official `hf auth token` CLI in a temporary HF_HOME.
It removes ambient HF credentials, alternate endpoints, and supplied OIDC ID
tokens from the child environment. It has no PAT/cache fallback. Provider-side
OIDC signature and claim verification remains authoritative; local environment
checks are only an additional guard. Both grants must be acquired before a
pair is returned. Tokens are not printed, exported to GITHUB_ENV, or written to
reports. The helper cannot select another repository or grant itself access.

## SDK compatibility is explicit

In `huggingface_hub==1.26.0`, `HfApi.auth_check` excludes kernel from its accepted
repository types. No SDK constants are monkeypatched and no undocumented
kernel write-check endpoint is invented.

The model uses the SDK write auth-check. The kernel uses its successful
repo-scoped provider exchange as the write-grant basis plus a supported read
of immutable `main` and `v1` refs. That read is **not** independent proof of write
access: preflight records `kernel_access_check=READ_REFS_ONLY` and
`kernel_write_independently_verified=false`.

Pinned source examined:
https://github.com/huggingface/huggingface_hub/blob/v1.26.0/src/huggingface_hub/hf_api.py
https://github.com/huggingface/huggingface_hub/blob/v1.26.0/src/huggingface_hub/constants.py

## Evidence and acceptance

A preflight report starts at CHECKING, then becomes either
BOTH_REPO_SCOPED_CREDENTIALS_VALIDATED or KEYLESS_CREDENTIALS_UNAVAILABLE.
It always records repository_mutation=NOT_ATTEMPTED and
publication_verified=false. Reports are atomic replacements, not certificates.
SIGKILL or storage failures may leave an incomplete marker; never infer success.

Local: 12 focused tests passed under normal and optimized Python 3.13.5 with
installed huggingface_hub 1.16.1. The real-SDK regression tests argument handling,
authorization headers, and refs decoding using simulated HTTP. The 1.26.0 SDK
source was inspected but that version was not installed locally. DNS failed on
the local host. Hosted 1.26.0 CI, actual OIDC exchange and live provider writes
remain separate evidence requirements.

```bash
python -m unittest discover -s tests -p test_invariants_keyless_credentials.py -v
python -O -m unittest discover -s tests -p test_invariants_keyless_credentials.py -v
```

Before the future publisher integration is admitted, it must route each token
only to its typed target, create a current failure receipt before initial Hub
reads, preserve partial-mutation uncertainty, and pass regression tests for
both target readbacks. Existing closed file sets, exact-source authorization,
branch protections, signatures, quarantine, kernel main preservation, and
publication concurrency must remain. Configure both Hub bindings, then perform
one fresh protected-head publication and inspect the actual publication JSON.
Do not repeat the old dispatches or claim a verified release from this PR.
