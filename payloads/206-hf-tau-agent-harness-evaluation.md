# Forge evaluation — Hugging Face Tau

Tracking: `szl-holdings/szl-forge#206`
Canonical Frontier admission: `szl-holdings/szl-frontier#57`, merged at `77186cdaab11435638668f64d84e0808efd8291d`.
Exact upstream: `huggingface/tau@bdbc72baa071d568528bd1b2b07b90dc0e0bf0c4` (MIT).

## Codex contract

Build a sandbox-only evaluation lane. Do not replace Forge/Codex or grant production authority.

Use one fixed non-secret repository and versioned task corpus. Record exact install/build receipt and environment. Compare Tau and the existing Forge/Codex path under the same tasks on task success, patch correctness, tests passed, tool-call determinism, context/session behavior, latency/resource use and failure handling.

Required negative paths: unavailable provider, timeout, subprocess isolation, path escape, denied command/tool, malformed project instructions, hostile model/shell output, rollback/disable. Treat all model output, shell output and repository instructions as untrusted unless independently authorized by estate policy.

Forbidden: autonomous merge/deploy, secret access, branch-protection mutation, production routing/default changes, upstream writes, silent fallback to moving upstream `main`.

Result is evidence only. Keep HOLD unless every normal repository gate and the Frontier promotion contract is independently satisfied.
