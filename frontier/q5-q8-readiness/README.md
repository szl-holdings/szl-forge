# Q5–Q8 training-lane readiness (2026-11-27)

These records prepare four approved lanes. They do not launch a job, push an
image, or write to the Hub. Q4 (Chaski 4B) is not approved and has no recipe.

The train image is the transformers 5.22.2 / peft 0.21.2 / trl 1.17.1 pin
shared with `kallpa/nemotron-h-trainer`. That branch owns the Nemotron-H
trainer. This directory does not copy it. The forge trainer's TRL 0.23.1
Unsloth path is rejected for these lanes.

| Lane | Status | Why it is not a launch |
|---|---|---|
| Q5 ReceiptAgent 0.8B | `READY_NOT_LAUNCHED` | Pins, eval-only heldout, and a tiny CPU dry-run are recorded. Operator admission, heldout independence, and a native Qwen3.5 step are UNKNOWN. |
| Q6 a11oy-code | `BLOCKED` | Apache-2.0 coder base is pinned. Eval-only heldout is ABSENT. |
| Q7 embedder | `BLOCKED` | Embedding base is pinned. Pair dataset and eval-only heldout are UNKNOWN. |
| Q8 brain-navigator | `BLOCKED` | Causal base is pinned. Dataset and eval-only heldout are UNKNOWN. |

Spend on every approved lane is capped at 100 USD per run, 500 USD per day,
and 5000 USD per month. The only allowed providers are `salad` and `massed`.
Which of those two would run a future job is UNKNOWN. GPU-hour figures are
planning estimates inside the approval; measured GPU-hours are UNKNOWN.

Checked JSON is `*.manifest.json`. `training/q5_q8_lanes.py` rejects a record
that raises the caps, reuses `chaski` or `chaski-r4`, or sets
`launch_authorized`. `training/q5_q8_dry_run.py` takes two CPU steps on a
tiny random config. Q7's step is contrastive.
