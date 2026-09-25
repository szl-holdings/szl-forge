# Read-only canonical pool snapshot view

`/pool` and `/api/pool` extend the existing authenticated Model Lab. They display
an operator-supplied snapshot of `szl.compute-pool/v1`; they do not query a node,
verify DSSE signatures, launch a job, choose a route, or change pool membership.
The existing a11oy producer remains the pool qualification authority.

## Configure a snapshot, not a second authority

Use a snapshot obtained through the existing owner-controlled a11oy evidence path.
Record its SHA-256 independently when admitting those bytes for inspection. Set
both process environment variables before starting the existing workbench:

```text
SZL_LAB_POOL_SNAPSHOT=<owner-controlled local snapshot path>
SZL_LAB_POOL_SNAPSHOT_SHA256=<independently recorded 64-character lowercase SHA-256>
```

Keep the file and its ancestors owner-controlled. Neither value belongs in a
public model card or repository. Do not configure network-mounted paths. This
change does not provide snapshot acquisition, a remote filesystem adapter, or a
new secret store. It reuses the package's bounded regular-file reader and rejects
symlinks/reparse points. It is not a defense against a hostile same-user process.

Keep `SZL_LAB_ACCESS_TOKEN` and the existing loopback service configuration. The
same authentication protects both new routes and occurs before a snapshot read.
There is no request parameter for a path, digest, URL, credential or trust root.
A path without a digest, or a digest without a path, is rejected at startup.
The snapshot is read again on each authenticated view; a byte change without a
matching operator-supplied digest becomes unavailable rather than silently trusted.

## Evidence meanings

The adapter checks a maximum 512 KiB of JSON, duplicate keys, exact schema fields,
bounded node identities, strict booleans, digest syntax, aware timestamps and
cumulative count agreement with individual records. The schema reference is pinned
to the inspected a11oy source revision and Git blob in `pool_view.py`. That source
reference is not an attestation of which software produced a supplied snapshot.

`REPORTED_SNAPSHOT` means only that the snapshot's declared generation time is
inside the checked age window. A newer generation time does not refresh a receipt:
its reported observation time and reported freshness are carried separately.
`STALE_REPORTED_SNAPSHOT` preserves old or future-dated observations as stale.
An upstream `SERVING` state or `ready=true` stays explicitly reported, not verified.
Every local `ready`, signature verification and qualification flag stays false.

An unavailable or unconfigured inventory has `nodes=null` and
`counts_reported=null`. A valid snapshot containing zero nodes instead has `[]`
and zero counts. These are deliberately different states. Failures are redacted:
endpoint addresses, raw bodies, receipt reasons and local paths are not echoed.
The view exposes only bounded node/model names and reported model digests. It is
an authenticated operator view, not a public inventory or telemetry publication.

## GitHub to Hugging Face packaging

The existing `blueprints.py` explicit source allowlist includes this document and
`pool_view.py`, so its code-only projection does not ship an app with a missing
import. Snapshot files, node configuration and credentials are not added to that
allowlist. The current protected-main, explicit-dispatch publisher remains the
only blueprint write path; this view neither invokes nor relaxes its controls.
No learned model identity, target, parameter shape or existing weights are changed.

## Verification and remaining gates

Run the existing Model Lab workflow and its full package test command. Focused
regressions are in `tests/test_pool_view.py`, including authentication-before-read,
stale timestamps, mismatch/duplicates, unavailable versus empty, HTML escaping,
source-export closure, and preservation of strict JSON request rejection.
Synthetic fixtures do not establish owner-node liveness, GPU availability, pool
signature validity, trained-model quality, or GitHub/HF deployment success.
Real owner observations, active routing integration and training remain separate.
