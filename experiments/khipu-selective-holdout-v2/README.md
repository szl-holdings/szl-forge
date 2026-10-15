# Khipu selective-abstention v2 diagnostic

**State: DECLARED pre-run design. Model inference is NOT RUN in this source anchor.**
The public owner-reported result for `SZLHOLDINGS/SZL-Khipu-1.5B` remains 2/6 abstentions, a release blocker. The separate v1 local diagnostic recorded 0/6 on a different constructed set; its frozen runner and receipt remain on draft PR #610. Neither denominator is pooled with this experiment.

This is a paired **exploratory** test of the same pinned original merged checkpoint at Hub revision `8a5c666c4fc1ec75a51b8bd9784b4504e807757a`. The baseline arm uses the canonical training system prompt. The policy arm appends [a fixed abstention instruction](policy_prompt.txt). The 16 fictional cases comprise eight metadata-supported handle lookups and eight unsupported requests. Case and arm order are fixed; arm order alternates within each pair. The private case file is outside the repository and is committed only by SHA-256 in [the preregistration](preregister.json), signed and published before inference. Cases are not used for training and are never added to the curriculum.

The policy arm tests whether an explicit decision rule changes proposals on this narrow fixture. It does not change weights, establish learned model quality, or substitute for a private future release gate. Once cases or outputs are published, they become public development material and cannot serve as an unseen promotion set. The source anchor binds the dataset hash, immutable model revision, declared base lineage, source-base blobs, runner, both prompts, seed, decode settings, and expected public artifact bytes. The original owner training receipt names a base model but does not pin the actual immutable training implementation revision; that field remains **UNKNOWN**. The separately published adapter is inspected for structural coverage and is not reapplied to the merged checkpoint. Structural coverage does not prove numerical equivalence.

The runner rejects duplicate JSON keys and non-JSON constants, validates the plan schema, checks the exact query and offered candidates, requires cited and traversed handles to be offered, and scores NAVIGATE and ABSTAIN separately. It does not grade the semantic truth of free-text rationales. Khipu only proposes JSON; A11oy/Alloy controls resolution and effects. The experiment loads no node content, exposes no tools to the model, and performs no controller action.

Preflight and inference both require the working-tree grading schema to match the preregistered schema hash. The same checked bytes are parsed and retained for grading before any output directory or model work begins. The source-review repair updated the preregistered runner hash before inference; it does not create a model result or alter the sealed case commitment.

From a **clean, signed** Forge checkout containing the preregistration, with the sealed fixture outside the checkout:

```powershell
$repo = 'C:\Users\steph\szl-work\khipu-selective-abstain-v2-20261015'
$cases = 'C:\Users\steph\szl-work\khipu-selective-v2-private-20261015\cases.json'
$assets = 'C:\Users\steph\szl-work\khipu-original-checkpoint-20261014'
$output = 'C:\Users\steph\szl-work\khipu-selective-v2-output-20261015'
$python = 'C:\Users\steph\szl-work\khipu-mixed-runtime-20261014\Scripts\python.exe'
& $python "$repo\experiments\khipu-selective-holdout-v2\run.py" --cases $cases --preflight-only
& $python "$repo\experiments\khipu-selective-holdout-v2\run.py" --cases $cases --model-dir $assets --output-dir $output
```

The output directory must not already exist. The runner requires at least 8 GiB free disk after artifact preparation and 5 GiB free VRAM before loading. It checks the public model, adapter, and support-file bytes; uses greedy decoding and no remote code; records every raw response; and writes an unsigned chained local receipt only for a complete 32-output run. An interrupted run leaves partial output and a `BLOCKED_NO_SCORE` receipt. The receipt records dataset, source anchor, model and declared base revisions, seed, hardware, loss `NOT_APPLICABLE_INFERENCE_ONLY`, harness, output hash, and arm-wise metrics. Joules remain **UNAVAILABLE** without a real NVML delta; carbon remains **ROADMAP**. Independent model replay is **NOT RUN** until a separate party actually performs it.

Source checks:

```powershell
ruff check experiments/khipu-selective-holdout-v2 --select E9,F
python -m pytest -q tests/test_khipu_selective_v2.py
git diff --check
```

Run these checks from the Forge repository root. The preregistration's timestamp is self-reported; the verified signed Git commit and remote readback are the pre-run anchor. No score, deployment, production readiness, authorization, or promotion claim follows from passing source tests or preflight.
