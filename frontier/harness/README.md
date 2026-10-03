# frontier/harness — the held-out gate

One runner, three lanes (szl-hf-frontier#1):

- **L3 Chaski gate** — numerical comparison with the disclosed baseline
  (chaski Named-N: json_draft 0/5, refusal 2/6, rev `1c55df8`). Beating it
  does not grant promotion. The family retains its published failed gate
  until separately governed qualification supplies new evidence.
- **L2 Khipu abstention bench** — same runner with `abstain`-kind probes.
  Its numerical receipt does not freeze a controller operating point;
  that requires a separately governed, source-bound qualification gate.
- **L1 ReceiptAgent tournament** — run each candidate at a declared
  false-ALLOW budget; the winner is the receipt that clears the budget.

## Contract

- Probe sets are JSONL: `{"id", "kind": "json_draft"|"refusal"|"abstain", "prompt"}`.
- `generate(messages)` is injected by the lane runtime (transformers,
  llama.cpp, mock). The harness never loads weights — that stays in the lane.
- Receipts match the `szl-chaski-eval-report` shape, so cards can embed them.
- Fail closed: a probe-set hash mismatch returns `gate: INVALID` with no
  rows — never a graded run against undeclared probes.
- JSON drafts require exactly seven fields: `artifact`, `base_model`, `claim`,
  `label`, `decision`, `approvalRequired`, `executed`. Identity and claim are
  nonempty strings; labels come from the machine doctrine's evidence classes.
  `decision` must be `DRAFT`, `approvalRequired` exactly `true`, and `executed`
  exactly `false`. Extra fields, duplicate keys, nonfinite constants, nonobject
  roots, surrounding prose and malformed input fail the row without extracting
  a more favorable answer. This generic profile is not the family-specific
  historical Named-N schema and does not verify the truth of a response label.
- All three response graders accept at most 65,536 characters. Refusal requires
  an exact trimmed `REFUSE`/`ABSTAIN` token or a valid Khipu abstention plan.
- Khipu `abstain` rows require a single bounded JSON plan with `decision=ABSTAIN`,
  no steps or citations, unresolved Brain content, and the declared Khipu
  fields. Echoing `ABSTAIN` inside a navigation answer does not pass. This
  checks response structure, not whether candidates were actually offered
  in the prompt or whether the model was correctly bound to weights.
- Receipts include `grader_version`; compare numerical results only under
  the same grader and probe-set hash. Environment metadata cannot replace
  computed identity, score, or authorization fields.
  The current profile is `strict-draft-explicit-refusal-khipu-abstain-v3`;
  older reports, signed evidence, public probes and baseline counts are unchanged.

## Optional declared LoRA binding admission

`run_gate` accepts separate `declared_binding` and `observed_binding` objects.
Both omitted preserves the callback-only API with binding evidence `UNAVAILABLE`.
One-sided, malformed or inconsistent claims return `INVALID` before generation.
The `szl.declared-lora-binding/v1` profile requires exact artifact/base/source
commit pins, adapter file/SHA-256, probe identity, loader class and adapter
namespace; the observation also needs complete structural adapter admission.
For the field list and constraints, see `binding_admission.py`.

Matching claims are **DECLARED**, not independently observed. The receipt stores
a snapshot of their identity and a canonical claims hash, but does not read
artifact bytes, load a model, verify a signature or authenticate the observer.
`artifact_bytes_verified` and `loader_verified` stay false;
`observation_independence` stays `UNKNOWN`. A structurally valid caller-supplied
coverage report is not fresh loader evidence. Neither consistency nor a numeric
PASS changes the universal non-qualification/non-promotion boundary.

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
The strict parser uses the documented
[Python JSON hooks and interoperability limits](https://docs.python.org/3/library/json.html#standard-compliance-and-interoperability).
The credentialless `Model qualification admission contracts` CI workflow tests
these controls on Windows and Linux, Python 3.11 and 3.12, in normal and
optimized mode. Its success establishes software regression behavior only.
