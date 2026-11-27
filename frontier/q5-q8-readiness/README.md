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
| Q6 a11oy-code | `READY_NOT_LAUNCHED` | Private split is pinned. Data status is PASS_WITH_CONDITIONS. Embedding-similarity was UNAVAILABLE. GPQA, the szl-chaski/rotation heldouts, AbstentionBench, SimpleQA, LiveCodeBench, and MTEB were not checked. The datasets API commit sha is UNKNOWN. |
| Q7 embedder | `READY_NOT_LAUNCHED` | Private pair file is pinned, including CC BY 4.0 doctrine attribution declared on the card. Same conditions as Q6. Per-row attribution bytes were not rechecked. |
| Q8 brain-navigator | `READY_NOT_LAUNCHED` | Private navigator file is pinned, including heldout_abstain. The base stays Qwen/Qwen3-1.7B. Same conditions as Q6. |

Spend on every approved lane is capped at 100 USD per run, 500 USD per day,
and 5000 USD per month. The only allowed providers are `salad` and `massed`.
Which of those two would run a future job is UNKNOWN. GPU-hour figures are
planning estimates inside the approval; measured GPU-hours are UNKNOWN.

Checked JSON is `*.manifest.json`. `training/q5_q8_lanes.py` rejects a record
that raises the caps, reuses `chaski` or `chaski-r4`, or sets
`launch_authorized`. `training/q5_q8_dry_run.py` takes two CPU steps on a
tiny random config. Q7's step is contrastive.
