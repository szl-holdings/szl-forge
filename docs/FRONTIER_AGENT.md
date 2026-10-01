# Frontier Agent

`tools/szl_frontier_agent.py` executes a bounded research or repair mission with
the locally installed, authenticated Codex CLI. It uses the configured default
model and does not store an API key. Each invocation terminates.

The repair cycle is reproduce, propose, apply, verify, and review. The model
runs read-only and returns a structured edit proposal. An authorized local
executor validates the complete proposal, applies only explicitly assigned
files in a fresh detached worktree, and runs operator-owned checks. Publication
remains a separate protected repository operation.

## Mission

The source checkout must be clean, have a `szl-holdings` GitHub origin, and
match the mission's exact revision. Python 3.11+, Git, and an authenticated
native Codex executable are required. Shell shims are rejected. On Windows,
`codex.exe` is resolved directly so an earlier npm `.CMD` wrapper cannot shadow
it. An explicit native executable can still be selected with `--codex-path`.

Provide a JSON object with schema `szl.frontier-agent-mission/v1`, `mode`
(`research` or `build`), `objective`, absolute local `repository`, exact
`source_revision`, `timeout_seconds` (30 to 1800), and `evidence` entries with
absolute `path` and `sha256`. Evidence files contain JSON. Build missions also
require 1..6 `checks`, trusted command argument arrays, never shell text or
model-generated commands. Include independent acceptance and new regression
tests. `allowed_paths` lists 1..12 unique explicit relative filenames;
`baseline_exit_codes` gives one expected 0/1 result per check with at least one
failure; `check_files` binds 1..12 absolute independent acceptance files and
their SHA-256 values. Every expected baseline result must reproduce.

Mission, evidence, edit-proposal, and model-event intake reject duplicate object keys at any
depth, including escaped-equivalent keys, and non-finite numbers (`NaN`,
infinities, and floating-point overflow). Valid JSON content is preserved;
evidence digests remain bound to original file bytes, not reserialized JSON.
Every model-event record must be a JSON object. Corrupt or non-object records
produce a sanitized `INCOMPLETE` receipt, even after an apparent completion,
before any proposal application or candidate check. The original event stream
is retained; unknown well-formed event types remain compatible.

Optional `source_paths` supplies at most twelve relative repository files as
revision-verified source context, each capped at 128 KiB. This lets research
continue when the local sandbox denies shell access, without weakening that
policy. Source IDs and hashes are retained separately from model statements.

```powershell
python examples/frontier_agent/prepare_powershell_repair.py --repository C:/path/to/szl-org-health --output-dir runs/powershell-mission
python tools/szl_frontier_agent.py --mission mission.json --run-dir runs/prepare-001
python tools/szl_frontier_agent.py --mission mission.json --run-dir runs/execute-001 --execute --codex-path C:/path/to/codex.exe
```

The example produces `runs/powershell-mission/mission.json`. It requires the
historical defect to be present: after that repair is merged, the expected
failing baseline no longer reproduces and execution stops before model use.

Use a new run directory each time. The source checkout must be clean and match
the mission revision. Evidence digests are checked before and after execution.
The launcher preserves the worktree, model events, output, proposal, baseline
and candidate check results, changed-file hashes, and receipt. The receipt also
binds the runner's own source bytes. Research proposals require a baseline,
metric, and falsification test.
The original checkout is not used as the model's working directory.

## Edit proposals

The model returns `summary` and 1..12 `edits`, each containing `path`, `before`,
and `after`. An existing UTF-8 file requires a unique exact `before` substring
and a replacement that changes it. A new file requires empty `before` and
nonempty full `after`. Duplicate, ambiguous, out-of-scope, symlinked, oversized,
no-op, and unsafe Windows paths are rejected before the first write. Direct
model writes are rejected. Checks must leave source and protected check bytes
unchanged. Check commands are trusted operator input and execute as the user.

## Evidence Limits

`PREPARED` means no model ran. `MODEL_RESPONSE_OBSERVED` requires a completed model turn
with no worktree changes. It does not certify the mission's substance, and
`tool_execution_observed` is reported separately. `REPAIR_VERIFIED` requires the
expected failing baseline, a completed model turn, a nonempty scoped edit, and
all independent candidate checks passing. `INCOMPLETE` retains any failed gate.
These states never authorize a release or training.
New files remain in the retained worktree; the tracked diff alone is not the
whole result. Model statements about test success are not accepted as checks.

Before publishing, review every changed file, run newly added tests and the
repository release checks, and refresh the protected branch's reviewed base.

The launcher sets a process deadline and bounds logs, not a token or dollar
budget. Codex usage follows the signed-in account. It retains local logs and
prompts, so supply only evidence appropriate for that account and protect the
run directory. Token-like environment variables are removed from child
environments; existing filesystem credentials are not an isolation boundary.
User configuration is ignored to avoid inheriting custom MCP servers or hooks.
On native Windows the launcher explicitly requests `windows.sandbox="elevated"`
because ignoring user configuration also discards that setting. Use an already
prepared native sandbox; no weaker or full-access fallback is selected when
setup or enforced policy prevents execution. The model remains read-only in
both mission modes. Sandbox configuration does not prove successful commands;
model proposals and local executor checks are observed separately. See the
official [Windows sandbox documentation](https://learn.chatgpt.com/docs/windows/windows-sandbox).
CLI sandbox enforcement still depends on the local runtime. This wrapper is
not a hostile-code sandbox, and operator check commands execute as the user.

Publishing, merging, DNS changes, GPU training, and scheduling are not automated
by this entry point. Use canonical repository release workflows after review.
Successful experiments can be curated into a training corpus later, with
licensing, provenance, held-out evaluation, and explicit training admission.
This is a working agent adapter, not a newly trained SZL foundation model.
