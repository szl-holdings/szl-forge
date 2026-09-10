# Forge evaluation — Hugging Face Tau 0.4.2

Tracking: `szl-holdings/szl-forge#206`
Frontier successor: `szl-holdings/szl-frontier#62`
Historical Frontier Tau admission: merge `77186cdaab11435638668f64d84e0808efd8291d`.
Exact upstream release: `huggingface/tau@55df51608b8b2d172c4bbac2cd11e8345e307476`, tag `v0.4.2`, MIT.

## Codex contract

Build and run a sandbox-only evaluation. Do not replace the governed Forge/Codex path and do not infer production qualification from source tests.

Use one fixed non-secret repository plus versioned task and session fixtures. Record the corpus digest, exact install/build receipt, environment, dependency/license inventory, and all failures. Compare Tau 0.4.2 and the current governed baseline on the same tasks.

Release-specific cases are mandatory:

- replay representative pre-0.4.2 session files across branch/resume/compaction;
- create/edit/clear/filter persistent session labels and verify deterministic replay;
- persist extension `custom_message` entries and prove they remain untrusted data, never policy/tool authority;
- simulate live Codex catalog additions, removals, malformed entries and outage; discovery cannot authorize a model/provider/route/default;
- preserve interleaved reasoning/answer stream ordering;
- reject or retry incomplete Anthropic-compatible SSE; an empty/incomplete stream cannot become success;
- verify Z.AI thinking serialization and provider/routed-provider usage-accounting shape;
- verify shell stdin disconnect, timeout, denied command, path escape and subprocess isolation;
- prove clean disable/rollback to the existing Forge/Codex path.

Score task success, patch correctness, test pass rate, tool-call determinism, context/session behavior, latency/resource use and failure handling. A missing backend or unsupported environment must be `UNAVAILABLE`, not a fabricated pass.

Forbidden: autonomous merge/deploy, secret access, branch-protection mutation, upstream writes, production routing/default changes, silent movement from exact `55df51608b8b2d172c4bbac2cd11e8345e307476`, or treating discovered model catalog entries as authorization.

Result is evidence only. Keep HOLD unless every normal repository and later Frontier promotion gate is independently satisfied.
