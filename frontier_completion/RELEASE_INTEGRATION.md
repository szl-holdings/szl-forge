# Frontier adoption: existing owners, bounded changes

Source-only integration, 2026-09-13. This module is an offline helper, not a new
model factory, public UI, scheduler, training admission or publisher.

## Implemented in this slice

The existing `registry.CANDIDATES` recapture list now includes Nex-N2.5-mini,
Nex-N2.5-Pro, Nex-N2.5-Max, OUI-1 and Agnes-3.0-Flash. Correct Mini identity:
`nex-agi/Nex-N2.5-mini`. Pro/Max and Agnes are WATCH/reference tracks. These entries
are inputs to the existing metadata scanner, not immutable artifact admission,
new-release evidence, installed models or provider-routing configuration. The
first metadata observation remains baseline capture, not an alert.

`data_review.py` adds bounded UltraData pilot-format inspection to the existing
CLI as `review-data`. It uses the unchanged strict JSON/hash primitives in
`frontier_completion.core`. Input is stdin, report is stdout; there is no file
writer, network acquisition, schema-reference resolution, tool execution or model
load in this action. All authority fields stay false.

At most 4 MiB, 256 JSONL records and 1 MiB per record are inspected. Required
source fields, supported message shapes, frozen function definitions, strict JSON
arguments, call/result pairing and supported RL stdin/stdout test-pair shapes are
checked. Exact duplicate content (ignoring uuid only) and duplicate UUIDs cannot
inflate the number of format-checked rows. Other harness/ground-truth/content
shapes return REVIEW_REQUIRED, not an invented coercion or success result.

This is not full JSON Schema argument validation, semantic/cross-harness dedup,
held-out decontamination, privacy review, license clearance, replay or reward
verification. Extra source fields are flagged and are never used as authority.
No normalized training corpus is exported. Original pilot rows, failure traces,
source-specific masks and sampling context must be retained under the owner's
data policy. The report contains hashes and fixed diagnostics, not record text.
Hashes are not anonymization and reports are not automatically public.

## Local use after source admission

Only supply a legitimately held, separately approved public-data pilot. No
provider download is implicit. Review the actual executable and pilot identity;
the CLI's declared revision fields do not independently attest either source.
The following is Bash/WSL, not PowerShell text-pipe syntax:

```bash
python -m frontier_completion review-data \
  --source "$REVIEWED_EXECUTOR_SHA" \
  --dataset ultradata-agent \
  --dataset-revision "$REVIEWED_DATASET_SHA" \
  --input-sha256 "$EXPECTED_PILOT_SHA256" < "$PILOT_JSONL"
```

Use `ultradata-rl` for the RL adapter. Pilot byte hash must match exactly. Nothing
is overwritten by the command; the operator controls any stdout redirection.
Exit 0 means format checks passed on the supplied pilot, exit 2 means at least
one record needs review, exit 1 means bounded input/binding validation failed.
Every result remains HOLD and NOT_ADMITTED regardless of process exit status.

Local validation for this slice: 35 synthetic reviewer/actual-CLI tests passed
with the existing core.py matched to Git blob
`5749806773259b7b52d2f98843390eeff5e6dffb`. Existing package initialization was
matched to `78d731def40dd026a1f2aa4b7776e39758d3877c`.

```bash
python -m unittest discover -s tests -p 'test_ultradata_pilot_review.py' -v
python -O -m unittest discover -s tests -p 'test_ultradata_pilot_review.py' -v
```

Those are scoped offline tests, not a full-checkout, Windows, GPU, real-dataset
or model-performance result. Required hosted repository checks own admission.

## Existing source ownership and upstream references

**Canonical release intake remains szl-frontier.** Its existing
`python/szl_frontier/edge_lane.py` already records both UltraData releases. Do not
create another UltraData admission or overwrite its historical revision pins.

- Nex: https://huggingface.co/nex-agi/Nex-N2.5-mini ;
  https://huggingface.co/nex-agi/Nex-N2.5-Pro ;
  https://huggingface.co/nex-agi/Nex-N2.5-Max . Start with sandboxed proposal-only
  evaluation. Capture observation identity, action schema, post-action evidence,
  failures and budget. No automatic computer control or provider substitution.
- UltraData SFT: https://huggingface.co/datasets/openbmb/UltraData-SFT-Agent-2609 .
  Its current card describes static traces without released replay environments;
  failure traces are not success labels, and frozen virtual tools are not our
  runtime APIs. Counts are trajectories, not necessarily unique tasks.
- UltraData RL: https://huggingface.co/datasets/openbmb/UltraData-RL-2609 .
  Supplied query/ground-truth records remain inert. Any verifier execution needs
  a separately admitted sandbox and environment, not Python eval of dataset text.
- OUI: https://huggingface.co/thesysdev/OUI-1 . Model-proposed declarative layouts
  must pass the product's deterministic component/schema validation. Python binds
  values, evidence and permissions. No arbitrary JS, URLs, backend queries or
  model-supplied verified-status badges. No generator is installed by this slice.
- Agnes: https://huggingface.co/Agnes-AI/Agnes-3.0-Flash . Research reference for
  hybrid-state behavior; qualify exact custom code and state isolation separately.
- Lyte/Granite: https://huggingface.co/ibm-granite/granite-timeseries-patchtst-fm-r2 .
  Extend existing Lyte forecast evaluation, with rolling holdouts and matched
  seasonal/statistical baselines. Forecasting is not causal explanation.
- TRL: https://huggingface.co/docs/trl/main/long_context_training . Keep the TRL
  optimization experiment separate from current production CUDA dependencies.
  Forge #282 and merged #288 own the separate vLLM 0.29/Omni qualification work;
  don't widen the older AsyncGRPO compatibility range merely to clear a gate.
- GGUF: https://huggingface.co/blog/bartowski/per-tensor-layout-maps-for-gguf-quantization .
  Reuse the existing quantization canary and add task/abstention regression evidence.
- DeepSeek: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash . Inspect actual
  consumer aliases and upstream mapping before repinning. A hosted model name
  does not independently establish immutable served weight identity.

A dataset's top-level license label does not resolve constituent-source rights.
The UltraData cards state additional source and redistribution restrictions.
This change does not mirror, rehost, ingest for gradients or claim rights to any
upstream dataset, checkpoint or code. Retain attribution for all adaptations.

## Source-to-surface completion, not achieved by this PR alone

GitHub -> HF SZLHOLDINGS -> a-11-oy.com product -> a11oy.net proof.

Reuse admitted Forge Model Lab rather than the older whole-workbench ZIPs.
Reconcile #271's alternative architectures/task identities without publishing
incompatible models under the same undifferentiated name. Preserve existing
ReceiptAgent, Khipu, Chaski, kernel identities and quarantine.

A11oy #2151 repairs recognition of the existing publisher's code-only blueprint
tags in the product preparation API/UI. It grants no trained/runtime status.
Only the existing source-bound workflow publishes supported model blueprints,
with exact-source and immutable HF byte readback. Check unresolved publisher
credentials/holds before dispatch; chat authentication is not CI authentication.

`szl-router` retains route/policy authority. A11oy retains compute-pool evidence;
Ollama/Tailscale/Docker configuration does not prove available hardware or pooled
VRAM. The existing owner supervisor/bridge owns any later explicit training run.
No model API, training, remote teacher call or paid job is launched by this slice.

The product and proof must project the same typed source/artifact/evaluation
records through existing collectors and publishers. Do not add proposed IDs to a
live inventory or turn a static snapshot into current runtime evidence. Normal
source checks/review, valid publication identity, immutable readback, deployment
and live contract verification remain separate acceptance gates.
