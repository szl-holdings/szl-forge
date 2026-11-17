# Release-decision identity readback (research-only)

The existing authenticated **Model Lab** now exposes **`/release-decision`** and
**`/api/release-decision`** using the *same* Python projection and the existing
loopback, Basic-authentication, host, no-store and restrictive CSP controls.
There is **no new control plane, provider reader, writer, deployer or scheduler**.

## Input / authority boundary

Configure an owner-controlled regular local JSON file and its **separately
obtained** lowercase SHA-256 digest:

- `SZL_LAB_RELEASE_SNAPSHOT`
- `SZL_LAB_RELEASE_SNAPSHOT_SHA256`

Both are required together. Neither can be supplied by an HTTP parameter. Do
not serve the workbench outside its existing loopback boundary or put private
tokens, messages, weights, lab records, or unpublished dataset bytes into this
snapshot. Even matching externally supplied bytes do **not** prove the source's
issuer, a digital signature, execution, current provider state or authorization.

The schema is `szl.release-decision-input/v1`. Example **synthetic, untrusted**
payload (the fixed hashes below are illustrative, not observed source identities):

```json
{
  "schema": "szl.release-decision-input/v1",
  "scope": "owner_local_unattested",
  "release_id": "synthetic-rehearsal-001",
  "observed_at": "2026-11-16T22:00:00Z",
  "rails": {
    "source": {
      "expected": {"repository": "szl-holdings/a11oy", "revision": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},
      "observed": {"repository": "szl-holdings/a11oy", "revision": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}
    },
    "space": {
      "expected": {"space_id": "SZLHOLDINGS/a11oy", "revision": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", "boot_id": "synthetic-boot-1", "run_nonce": "synthetic-run-1"},
      "observed": null
    },
    "gateway": {
      "expected": {"repository": "szl-holdings/szl-router", "revision": "cccccccccccccccccccccccccccccccccccccccc", "route": "a11oy"},
      "observed": null
    },
    "receipt": {
      "expected": {"repository": "SZLHOLDINGS/frontier-54-runs", "repo_type": "dataset", "revision": "dddddddddddddddddddddddddddddddddddddddd", "path": "receipts/example/result.json", "sha256": "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"},
      "observed": null
    }
  }
}
```

The source, Space, gateway and receipt identities are exact; `observed` is
either the full same-kind identity or JSON `null`. Extra keys, duplicate keys,
nonfinite values, mutable revisions, noncanonical paths, unauthorized
repository namespaces, future timestamps and unsafe bytes are rejected. The
reader uses the existing bounded, symlink-resistant `safeio.read_regular` and
strict JSON parser. The snapshot age is calculated from the server clock
solely to flag old operator-supplied data; **not live provider freshness**.
When more than 900 seconds old it receives a stale-input hold reason.

## States and design constraints

- **503 / UNCONFIGURED** or **INVALID_OR_UNAVAILABLE_SNAPSHOT**: no configured
  usable evidence; null counts, no local path or secret disclosure.
- **200 / HOLD_UNVERIFIED**: a parseable input; matches mean equality *within
  supplied data*, not provider parity. Missing observations remain explicit.
- **200 / HOLD_CONFLICT_REPORTED**: at least one rail differs within that data.
- `?invalidate=source|space|gateway|receipt` performs an in-memory **WHAT-IF**
  recheck label without editing the input or claiming a real incident.
- Unknown rails return sanitized **404**. POST is **405**. No API accepts
  credentials, dispatches work, calls HF/GitHub/Cloudflare or changes provider state.

**Every state holds:** `execution_authority=false`,
`publication_authority=false`, `signature_verified=false`,
`independent_witness_verified=false`, `source_byte_parity_verified=false`,
`live_readback_performed=false`, `production_ready=false`, and
`action_performed=false`. These flags cannot be overridden by input.

## Engineering and production closure

This readback is a **view contract**, not the release gate. Actual admission
requires the existing owner-controlled workflow to fetch source and deployment
bytes at immutable revisions, compare the six protected A11oy paths, bind
the deployed boot/run nonce and gateway route, verify receipt bytes and
signatures where applicable, obtain named approval and required countersigns,
capture rollback state, and recheck at the moment of action.

The canonical hold and human-selection requirement live in
[szl-holdings/a11oy#2951](https://github.com/szl-holdings/a11oy/issues/2951).
A successful GET, matching value, signed-looking field or completed frontend
test must never close that issue by itself.

The core principle is four distinct rails: **evidence, validity, authority,
execution**. A release identity is a prerequisite for verified action but does
not alone prove model quality, scientific validity, permissions or side effects.
This separates the operational concept from Palantir-style object/action
composition, LangGraph checkpoints, Temporal histories and NVIDIA inference
observability while preserving SZL's own controller and Lutar invariants.

## Test commands

From `model-lab/`:

```sh
python -m pytest -q tests/test_release_decision.py tests/test_evidence_lineage.py
python -m pytest -q tests
```

All tests use synthetic snapshots and the real FastAPI/Jinja routes. No
provider call, GPU compute, secret, deployment or independent verification
is asserted. The code-only model blueprints include the new reader, router,
template and documentation in their exact-source allowlist; no weights or
publish eligibility is inferred.
