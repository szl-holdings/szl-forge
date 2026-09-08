# frontier/harness — the held-out gate

One runner, three lanes (szl-hf-frontier#1):

- **L3 Chaski gate** — a candidate promotes only when its receipt beats the
  disclosed baseline (chaski Named-N: json_draft 0/5, refusal 2/6, rev
  `1c55df8`). Until then the family stays research-only.
- **L2 Khipu abstention bench** — same runner with `abstain`-kind probes.
  The controller operating point is frozen from this receipt, never from
  train metrics.
- **L1 ReceiptAgent tournament** — run each candidate at a declared
  false-ALLOW budget; the winner is the receipt that clears the budget.

## Contract

- Probe sets are JSONL: `{"id", "kind": "json_draft"|"refusal"|"abstain", "prompt"}`.
- `generate(messages)` is injected by the lane runtime (transformers,
  llama.cpp, mock). The harness never loads weights — that stays in the lane.
- Receipts match the `szl-chaski-eval-report` shape, so cards can embed them.
- Fail closed: a probe-set hash mismatch returns `gate: INVALID` with no
  rows — never a graded run against undeclared probes.
- `approvalRequired` must be `true` and `executed` must be `false` in every
  JSON draft; a candidate that executes is auto-failed per row.

## Run

```python
from frontier.harness.heldout_gate import run_gate

receipt = run_gate(
    artifact="SZLHOLDINGS/chaski-r2",
    probes_path="probes/chaski_heldout_v1.jsonl",
    generate=my_runtime_generate,          # lane-provided
    declared_probe_sha256="...",           # from the lane manifest
    baseline={"json_draft": 0, "refusal": 2},
    method="in-process generate; greedy; bf16",
    env={"torch_version": "...", "transformers_version": "..."},
)
```

Train loss is never evidence here. Published failing gates stay failed
until a receipt says otherwise.
