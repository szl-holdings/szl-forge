# One-GPU cost envelope — source-only production admission aid

**Status: HOLD.** This module is an additive, offline cost-envelope gate. It is
not a GPU launcher, a paid-job authorization, a replacement for
`training/launch_contract.py`, a receipt signer, or a Salad controller.

## Why this exists

A GPU group marked `running` need not have an allocated worker or verified
trainer step. A trainer that reaches its requested step count can still lack a
durable checkpoint, independent holdout, or owner-approved serving release.
Budget and retry limits must be bound to **one immutable candidate**, not to
an unscoped fleet-scale request.

The existing Forge launch contract remains the single training-data admission
owner; the existing GPU Bridge Salad worker/terminal-evidence path remains the
runner/receipt authority. This tool only models the cost of one proposed worker
once a trusted operator has supplied fresh provider pricing and a spending cap.

## Bounded candidate

Invoke `python -m training.gpu_canary_cost_envelope --plan path/to/plan.json`
from a non-production analysis environment. The module returns **exit 2 even
for a valid cost envelope**: no source-only estimate can become an automated
launch-success predicate. It reads one local JSON document, computes an
upper-bound model for GPU running seconds, and makes zero provider calls.

Required schema: `szl.forge.gpu-canary-cost-envelope/v1`.

Inputs include `provider=salad`, `organization=alloyscape`,
`project=default`, canonical container-group name, pinned student/dataset
40-hex revisions, distinct 64-hex train/holdout file digests, an immutable
`worker_image` using `@sha256:<64-hex>`, exactly one GPU,
`max_attempts=1`, `max_running_seconds` from 1 to 7200,
`observed_gpu_rate_usd_hour` and `operator_cost_cap_usd` as decimal
strings, and a UTC `price_observed_utc` no older than 30 minutes.

The CLI rejects duplicate JSON members at **any** nesting depth, nonfinite
constants, floating-point JSON numbers, non-object plans and inputs larger than
64 KiB before calling its cost model. Monetary values stay exact decimal strings;
all running-time and GPU/retry counts stay integers. Invalid input exits with
status 2 and never calls Salad or a training launcher.

The estimate assumes one continuously running, billable GPU, ignores startup,
data egress and other unspecified charges, and rounds **up** to cents. A
self-supplied price, cap, path, hash or approval is **not authenticated**.
The output always has `launch_authorized=false`.

## Actual GPU launch: separate acceptance obligations

1. Independently obtain immutable student/base model and training-dataset
   revisions, actual licensed rights and consent, clean/disjoint holdout and
   exact corpus bytes. Never train on benchmark holdouts or unreviewed papers.
2. Validate an owner-authenticated Forge operator-admission receipt, intended
   training family and adapter method, and the canonical
   `training/launch_contract.py` result.
3. Identify the **one** existing Salad/Massed controller and current leases,
   resolve active/stopped status conflicts, then independently verify the
   provider price, exact digest-pinned worker image, replica, capability and
   budget authorization. No second watcher or blind `start`/replica change.
4. Run only through the existing owned worker with a short, enforced runtime
   and retry cap. Capture GPU identity, real optimizer/step counters,
   loss/grad-norm, and a durable checkpoint with a re-read of exact bytes.
5. Evaluate a separately held-out, provenance-reviewed set and maintain
   independent receipts. Only then consider separately governed promotion,
   serving and rollback.

## External research translated into SZL requirements

- Hugging Face TRL: use pinned `SFTTrainer` or `GRPOTrainer` APIs as appropriate
  to *approved* source labels; resume from a real checkpoint, not by rerunning
  a stale recipe: https://huggingface.co/docs/trl/sft_trainer
- Ray Train: recover model, optimizer and epoch/step state from persistently
  stored checkpoints rather than trusting worker restarts:
  https://docs.ray.io/en/latest/train/user-guides/checkpoints.html
- NVIDIA NeMo: multi-GPU checkpoint shards need coordinated, resumable state
  with integrity: https://docs.nvidia.com/nemo-framework/user-guide/26.02/nemotoolkit/checkpoints/dist_ckpt.html
- Salad: rates depend on GPU class, priority and actual running state; a
  pricing-page example is not a verified project-specific quote:
  https://salad.com/pricing

All adaptations here are original standard-library Python. No source, training
data, model weights or licenses are copied from external projects.

**This PR alone does not close rights, signer, budget, capacity, quality,
checkpoint or serving gates.**
