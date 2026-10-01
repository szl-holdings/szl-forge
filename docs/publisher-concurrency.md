# Model-card publication concurrency

Both family writers bind an actual Hub mutation to the exact 40-character Hub
main revision inspected before deciding whether the three managed files differ.
They send `revision="main"`, `create_pr=False` and `parent_commit=expected_parent`.
The parent is not refreshed after a conflict or across bounded 429 retries.
HTTP 412 propagates. Missing or malformed repository heads prevent a write.

The reviewed client is exactly `huggingface-hub==1.23.0`, already pinned by both
publication workflows. Its source is
[`0c92853b8e07bc50ee0817e307e9fd88194dd4f3`](https://github.com/huggingface/huggingface_hub/tree/0c92853b8e07bc50ee0817e307e9fd88194dd4f3).
The upstream
[stale-parent regression](https://github.com/huggingface/huggingface_hub/blob/0c92853b8e07bc50ee0817e307e9fd88194dd4f3/tests/test_hf_api.py#L746)
expects 412.

That client can optimize away all additions and return another writer's current
head without sending a conditional commit. A public `CommitInfo` alone cannot
distinguish this shortcut from a successful mutation. The writers therefore
check a private invariant of this exact pinned source: every new addition has
`_is_committed is False` initially, and the client marks all original additions
`True` only after `_send_commit` succeeds. An unavailable capability or an
unconfirmed return fails closed; another client version is rejected before Hub
initialization. This is a pinned client control-flow check, not an independent
provider receipt. Any client upgrade needs a reviewed replacement contract and
the race fixtures. See the
[initial marker](https://github.com/huggingface/huggingface_hub/blob/0c92853b8e07bc50ee0817e307e9fd88194dd4f3/src/huggingface_hub/_commit_api.py#L191)
and
[commit-success marker](https://github.com/huggingface/huggingface_hub/blob/0c92853b8e07bc50ee0817e307e9fd88194dd4f3/src/huggingface_hub/hf_api.py#L5214).

Already matching assets use immutable readback and a second Hub-head check.
They create no commit. That observation does not lock a branch against subsequent
changes. Each accepted conditional mutation still requires exact-byte readback
at its returned immutable revision. Readback or confirmation failure can follow
an accepted commit; the writer records failure and does not blindly repeat the
mutation. An owner must reconcile the target before a retry.

Khipu additionally checks local Git HEAD against the supplied immutable source
revision and freshly queries the fixed canonical GitHub main ref. It checks
before credential acquisition in the workflow, before initializing the Hub
client, before each write attempt, and before accepting a no-change result.
Lookup failure, missing/ambiguous main, checkout mismatch or stale source stops
publication. Dry runs remain offline and require no token or fresh-main query.
The GitHub observation and Hub commit remain separate services; this guard does
not create an atomic transaction across them or independently attest a worktree.

## Release integration

This repair's writer and workflow paths themselves match both automatic
main-push publication triggers. Do not merge it alone while main still contains
the stale family cards. Forge #424 supplies the stronger dated source facts and
the original-Khipu correction. The release owner must review an integration
revision containing both this repair and #424's corrected source inputs before
the first publishing push to protected main. The two existing card branches are
unchanged by preparation of this repair; the queue owns any reviewed branch
integration, required-check refresh, owner disposition and merge.

Forge #426 has its separate eleven-target writer and is not changed here. Its
publication scope and release prerequisites remain as reviewed. Neither card
publication grants model promotion, runtime or autonomous execution authority.
