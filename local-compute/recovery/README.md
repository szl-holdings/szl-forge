# Laptop recovery: source-controlled diagnostic interface

This is the narrow GitHub successor of the separately supplied
`SZL_LAPTOP_RECOVERY_v1.zip` (SHA-256
`f710f03ed78dfeaca0c18aa745a75fd5d617a6018d893c728a32d9d48c3a0d44`).
Version 1.1 changes the execution default to **metadata inspection only**;
`--run-paired` explicitly opts into bounded inference. There is no installer,
package download, model mutation, training, remote dispatch or publication path.

## Why this exists

The owner-supplied September 12 five-model baseline completed the protocol.
ReceiptAgent scored 5/6: its draft marker had correct boolean values but an
extra `claim` property. Both `szl1` and `szl-sovereign-qwen` scored 0/6 with
128 repeated `@` characters per response and token-limit termination.
That is observed degeneration, not a proven weight-import root cause.
Both Nemotron names scored 6/6 on this small diagnostic set.

The helper preserves all original scores and evidence files. It records current
model metadata hashes and provider-reported FROM blob identities without copying
raw local Modelfile paths or system prompts. No raw machine report, owner archive,
credentials or environment dump is stored in this public repository.

The baseline protocol is `local-compute/benchmark_ollama.py` at Forge commit
`1686c0cfb607a33579413ca85bc113fa8d61919b`. The exact owner evidence hashes and
installed-model digests are fixed in this helper; they are not auto-rebased to
whatever model is currently present. This is a scoped recovery lane, not an
estate-wide benchmark or a generic trainer.

## Owner-laptop commands

Run from an inspected checkout using ordinary PowerShell and Python 3.12.
The entry point only admits the BETTERWITHAGE Windows host, preserves the old
workspace, and expects its existing completed baseline under the owner's home.
It checks explicit known cache/merge paths, not the whole drive.

```powershell
# Default: inspect local metadata and write a new report; no generation.
py -3.12 -I -B .\local-compute\recovery\szl_recovery.py

# Explicit experiment: only after other GPU work has finished normally.
py -3.12 -I -B .\local-compute\recovery\szl_recovery.py --run-paired
```

The live experiment sends exactly 12 requests on the **same pinned installed
1.5B ReceiptAgent**: six unconstrained and six schema-constrained. Field types
and exact allowed keys are specified; schemas do not encode correct answer
values with constants/enums. Independent strict grading still rejects extra
keys, duplicate keys, wrong values/types, Markdown, tool calls and token-limit
termination. Prompts/settings remain identical across modes, order alternates,
and each response and timing is receipted. The old 5/6 score is never revised.

Only literal loopback `127.0.0.1:11434` is used, with proxies and redirects
refused. Metadata endpoints and generation identities are fixed. The two
`@`-generating names are not executable targets. No existing application routes
are changed. Active foreign GPU work, loaded Ollama work, identity drift, missing
evidence, thermal/resource limits or errors stop the operation. No process is
terminated and no failed HTTP request is automatically retried.

Requests use a bounded 30-second keep-alive, with zero on the final request.
Timing is not a measured speedup versus the historical cold-load protocol.
Timeouts/locks/process accounting are best-effort; a disconnected request can
continue server-side. This is not a hostile-code sandbox or proof of GPU
exclusivity. Preserve unknowns and do not clear locks blindly.

Outputs go to a new home-directory `szl-recovery-*` folder, including
`RETURN_TO_CHAT.zip`. Inspect process IDs and hardware metadata before sharing.
A successful paired run remains
`SCHEMA_INTERFACE_PASSED_REUSED_SIX_CASES_ONLY`, not a fresh blind benchmark,
production promotion, trained model, or repair of the two degenerate weight sets.

The existing Qwen3.5 0.8B native trainer is a different lineage from the installed
Qwen2-family 1.5B ReceiptAgent. This helper never uses that recipe as an implicit
replacement. Missing training snapshots remain a separate prerequisite.

## Offline validation

```powershell
py -3.12 -I -B -m unittest discover -s local-compute/recovery -p 'test_*.py' -v
```

All model/API responses and GPU observations in these tests are fixtures.
Passing Linux/Windows CI verifies these contracts, not owner GPU execution,
live structured-output quality, model-weight repair or deployment.
`MANIFEST.sha256.json` is an unsigned file-integrity manifest, not a trust-root
signature. Trust requires the inspected source commit; changing both code and
an adjacent manifest cannot create independent approval.

Official interface documentation:
- https://docs.ollama.com/capabilities/structured-outputs
- https://docs.ollama.com/api/chat
- https://docs.ollama.com/api/ps
