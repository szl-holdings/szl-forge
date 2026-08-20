# Owned Agent Control v2 Threat Model

## Security objective

OAC gives a local owner a verifiable, stop-only lifecycle boundary around an
explicit process tree that OAC launches on a qualified Windows machine. It aims
to make target registration, context qualification, authorization, isolation,
and local evidence fail closed under accidental misuse and ordinary file or
request tampering.

## Protected assets

- Integrity of the registered executable path, argv, working directory, hash,
  target generation, and run identity
- Confidentiality of operator private keys and supervisor-local credentials
- Integrity and one-shot state of signed isolation envelopes
- Integrity of context trace evidence and the local audit chain
- Availability of the owner-authorized stop operation

## Trusted computing base

- The Windows kernel, Job Object implementation, process and file APIs
- The fixed local NTFS volume and its ACL enforcement
- The operating Windows identity and Python/cryptography runtime
- The OAC package bytes and the exact registered executable bytes
- At least two honest independent operator-key custodians

## Adversaries and handled cases

| Case | Control |
|---|---|
| Edited or replayed isolation request | Canonical Ed25519 signatures, expiry/skew bounds, exact controller-root and immutable target-registration binding, and one-shot consumption |
| Replaced registered executable | SHA-256 is checked again before launch; drift fails closed and requires a new target ID |
| Child spawns descendants | Windows Job Object binds and terminates the supervised process tree |
| PID reuse | Stored process creation identity is checked with the PID |
| Partial state initialization | Database is staged, validated, and atomically published; failure cleanup avoids a trusted half-state |
| SQLite row mutation | STRICT tables, immutability/append-only triggers, cross-table verification, and a hash-chained audit ledger expose inconsistent local state |
| Empty or inconsistent context | At least one comparison is required; the consistency threshold and transition/policy gates fail closed before start |
| A11oy attempts process control | Export schema fixes `a11oy_process_control=false`; no remote mutation is performed by export |

## Explicitly out of scope

- Hostile containment from code running under the **same Windows identity**.
  That code can target the user's files, processes, Python runtime, or controller
  inputs. A Job Object is lifecycle control, not an OS security sandbox.
- Administrator, SYSTEM, kernel, firmware, hypervisor, boot-chain, or physical
  attackers
- Network egress, DNS, firewall policy, remote API/provider state, cloud IAM,
  SaaS credentials, model weights/state, or messages
- Processes not launched by this OAC state root
- Independent timestamping, transparency witnessing, or immutable evidence
- Availability against machine loss, disk failure, resource exhaustion, or a
  privileged denial-of-service attacker
- General AI correctness, alignment, or safety certification

For untrusted or hostile agent code, use a separate low-privilege identity plus
an OS sandbox, VM, container boundary with validated isolation, and network
policy. OAC can complement those controls but does not replace them.

## Context and isolation independence

Enriched context is diagnostic evidence and a **start gate**. It is not an
authorization mechanism and cannot enlarge target permissions. An authorized
isolation request must remain actionable when the current context is absent,
stale, or failed. This preserves stop availability without treating context as
security enforcement.

## Key lifecycle

Private keys are encrypted at rest by the cryptography runtime and should have
independent custody. Public-key registration becomes immutable when the trust
store is sealed. OAC v2 intentionally does not claim online key rotation. Loss
or compromise after sealing requires a newly qualified state root and a
documented migration; do not mutate the old database.

## Residual risks

- An attacker with the owner identity can tamper before a check, replace the
  controller/runtime, or race filesystem reads; OAC narrows but does not remove
  this class.
- SHA-256 binds bytes but not publisher identity, code safety, or all files a
  target may load after launch.
- The post-creation, pre-resume path/hash recheck does not cryptographically
  identify the already mapped image section against a hostile same-identity
  filesystem race; such an identity is outside the containment claim.
- A byte-for-byte state-root clone made before request consumption is not
  distinguishable without an external monotonic anchor.
- A local audit chain detects inconsistent edits when verified; an attacker who
  can rewrite the database and recompute every value can forge a new local
  history. External witnessing is required for stronger non-repudiation.
- CI evidence qualifies only its exact runner. Production hardware, identity,
  policy, and state-path behavior must be qualified independently.

## Required production controls

1. Dedicated least-privilege Windows identity and fixed local NTFS state path.
2. Separate operator-key custody and protected offline backups.
3. Exact wheel/hash pinning and Windows `doctor` plus behavioral `self-test`.
4. Central collection of exported evidence without granting the collector
   process authority.
5. Independent sandbox/network controls for any target treated as hostile.
6. Incident procedure that preserves the full state root and never edits audit
   or authorization rows in place.
