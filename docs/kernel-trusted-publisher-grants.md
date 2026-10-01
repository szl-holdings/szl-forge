# Kernel Trusted Publisher grant boundary

## Decision

The kernel publisher treats a successful Hugging Face repository Trusted
Publisher exchange as the provider's write-authority decision for that exact
resource. It does not submit the resulting synthetic `hf_jwt_*` repository
grant to `HfApi.auth_check`.

This is not a permission bypass. The exchange remains fail closed and is more
specific than the removed probe:

1. The workflow must be `workflow_dispatch` from protected `main` at an exact
   Forge commit and with the exact workflow identity.
2. Ambient Hub credentials are rejected, and the official CLI runs in an empty
   temporary home with a small environment allowlist.
3. The provider must accept two separate exchanges: one for
   `SZLHOLDINGS/szl-kernels` and one for
   `kernels/SZLHOLDINGS/szl-kernels`.
4. Both returned values must be bounded, well-formed `hf_jwt_*` tokens and must
   be distinct before either is returned to the publisher.
5. GitHub OIDC request variables are removed before publication starts. There
   is no PAT, cached-token, or organization-token fallback.

The official Trusted Publishers contract says a repository exchange returns a
short-lived token with write access to exactly the requested repository. An
exchange fails when no publisher matches the issuer and exact configured
claims. Repository tokens are attributed to the synthetic `[OIDC]` system user,
not to the human account that configured the binding:

- https://huggingface.co/docs/hub/trusted-publishers
- https://github.com/huggingface/huggingface_hub/blob/v1.26.0/src/huggingface_hub/_oidc.py

## Why `auth_check` was removed

Run 36663468383 at Forge revision
`79105217be35a97ee417d323fd2ff1e8f4c25674` stopped with
`MODEL_WRITE_ACCESS_VALIDATION_FAILED` before any provider write. That fixed
code can only be reached after both exact-resource exchanges returned distinct,
well-formed grants. The failure therefore occurred at the additional model
probe, not at either Trusted Publisher exchange.

The pinned `huggingface_hub==1.26.0` implementation documents `auth_check` as a
check for a provided user token. With `write=True` it calls the separate route
`GET /api/models/{repo_id}/auth-check/write`. That route is not part of the
Trusted Publisher token-exchange protocol, and the SDK does not admit Kernel as
an `auth_check` repository type:

- https://github.com/huggingface/huggingface_hub/blob/v1.26.0/src/huggingface_hub/hf_api.py#L11929-L12002

Continuing to require that user-token endpoint would reject an exact
repository grant after the provider had already issued it. Inventing a Kernel
write-check endpoint or monkeypatching SDK repository types would be less
defensible. The publisher therefore uses only documented operations.

## Remaining provider checks

Before mutation, `publish_szl_kernels.run` uses the model grant to read the
current model revision and declared critical files. It uses the Kernel grant to
read repository metadata, exact `main` and `v1` parents, and the closed file
sets. These supported reads detect missing state and parent drift. Because the
repositories are public, they are structural predicates and are explicitly not
reported as independent proof of write permission.

Write authority is exercised only by the real typed operations:

- the pinned `kernel-builder` upload receives only the Kernel grant;
- the model `create_commit` receives only the model grant and the exact
  observed parent commit;
- failures are terminal and retained as partial-mutation uncertainty;
- success requires exact post-write branch discovery, byte-for-byte readback,
  signature verification, and the isolated credentialless runtime witness.

No source merge, green CI run, successful token exchange, or successful read
is a release. Only the final provider receipt
`PUBLISHED_AND_EXACT_READBACK_VERIFIED` supports that statement.

## Evidence boundary

This source change performs no Hub mutation and does not claim the kernel has
been published. The observed failed run's artifact report was
`PUBLICATION_FAILED_NO_PROVIDER_WRITE`. After protected admission, a fresh
exact-head run and its immutable provider readback remain required.
