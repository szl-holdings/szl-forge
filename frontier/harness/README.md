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

## Public smoke and adapter admission boundary

Public fixtures and mock generators test the runner, not a qualified model.
Their numerical `gate` can be `PASS` while `qualification_gate_ran`,
`publication_eligible`, and `promotion_eligible` remain false, with
`promotion_effect=NONE` and `authority=NONE`. A mock response is `SIMULATED`.
Running a real generator on public smoke cases does not make those cases
held-out. Known public fixture content remains public when copied or renamed;
the checked-in fixture paths are also non-qualifying if their content changes.
Reordered or partial public rows with renamed IDs remain public probes.
The probe hash is the runner's canonical JSON-row hash, not a raw-file hash.
This classification grants no signature, hidden-split identity, deployment,
or promotion authorization. Unknown probe content is not evidence of a hidden
split: it is `UNVERIFIED_PROBE_SET`, and this standalone runner always withholds
qualification and publication. `baseline_beaten` records only the numerical
comparison. A separate source-bound, approved qualification process is still
required; the runner does not implement that authorization path.

For the Chaski Named-N runtime, `chaski/adapter_guard.py` inspects checkpoint
layout before querying the loaded model. It requires all checkpoint keys in
one selected adapter namespace, and PEFT model status must show that adapter
alone active, enabled, available, and unmerged in a nonempty set of adapter
layers. Missing or irregular status fails closed. The default namespace is
`default`; callers selecting another namespace must name it explicitly.
Key coverage and activation status are structural admission checks, not
numerical weight equality, inference qualification, or a replacement for the
failed historical model gates. No real model is loaded by the regression tests.

Prior art: [PEFT model-status API](https://huggingface.co/docs/peft/package_reference/peft_model#get_model_status)
and [PEFT troubleshooting](https://huggingface.co/docs/peft/developer_guides/troubleshooting).
The credentialless `Model qualification admission contracts` CI workflow tests
these controls on Windows and Linux, Python 3.11 and 3.12, in normal and
optimized mode. Its success establishes software regression behavior only.
