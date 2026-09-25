# Frontier Agent

`tools/szl_frontier_agent.py` connects an operator-defined mission to the locally
installed, authenticated Codex CLI. It uses the CLI's default model without
storing an API key. Research uses its read-only sandbox; engineering uses
workspace-write in a new detached worktree. Each invocation terminates.

The intended cycle is observe, propose, experiment, implement, verify, and review.
This entry point supplies the model execution and local verification steps.
`szl-org-health` supplies census evidence, `szl-frontier` supplies evaluation
contracts, and the existing estate operator retains publication ownership.

## Mission

Provide a JSON object with schema `szl.frontier-agent-mission/v1`, `mode`
(`research` or `build`), `objective`, absolute local `repository`, exact
`source_revision`, `timeout_seconds` (30 to 1800), and `evidence` entries with
absolute `path` and `sha256`. Evidence files contain JSON. Build missions also
require `checks`, a list of trusted command argument arrays. Commands are
operator input, never extracted from the model response.

Optional `source_paths` supplies at most twelve relative repository files as
revision-verified source context, each capped at 128 KiB. This lets research
continue when the local sandbox denies shell access, without weakening that
policy. Source IDs and hashes are retained separately from model statements.

```powershell
python tools/szl_frontier_agent.py --mission mission.json --run-dir runs/prepare-001
python tools/szl_frontier_agent.py --mission mission.json --run-dir runs/execute-001 --execute
```

Use a new run directory each time. The source checkout must be clean and match
the mission revision. Evidence digests are checked before and after execution.
The launcher preserves the worktree, model events, output, check results, and
receipt. Research proposals require a baseline, metric, and falsification test.
The original checkout is not used as the model's working directory.

## Evidence Limits

`PREPARED` means no model ran. `MODEL_RESPONSE_OBSERVED` requires a completed model turn
with no worktree changes. It does not certify the mission's substance, and
`tool_execution_observed` is reported separately. `LOCAL_CHECKS_PASSED` requires the model turn and all
operator-defined checks. These states never authorize a release or training.
New files remain in the retained worktree; the tracked diff alone is not the
whole result. Model statements about test success are not accepted as checks.

The launcher sets a process deadline and bounds logs, not a token or dollar
budget. Codex usage follows the signed-in account. It retains local logs and
prompts, so supply only evidence appropriate for that account and protect the
run directory. Token-like environment variables are removed from child
environments; existing filesystem credentials are not an isolation boundary.
User configuration is ignored to avoid inheriting custom MCP servers or hooks.
CLI sandbox enforcement still depends on the local runtime. This wrapper is
not a hostile-code sandbox, and operator check commands execute as the user.

Publishing, merging, DNS changes, GPU training, and scheduling are not automated
by this entry point. Use canonical repository release workflows after review.
Successful experiments can be curated into a training corpus later, with
licensing, provenance, held-out evaluation, and explicit training admission.
