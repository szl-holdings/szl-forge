# 2026-09-13 — kernel-model source slice and compute map

## Scope and observations

Inspected Forge source at `b6cc0cc635459fc8740db85ff0c6e98eeac84007`
(tree `53c103c6ca014adab9dffd67ff3d8fa0678b4e3a`), organization repository
search, focused file searches, existing open Forge PRs and selected HF metadata.
This is NOT a complete read of every organization file, private asset census,
weight-byte audit, live device audit or deployment certification. A Git clone in
the isolated test environment failed DNS; tests were run on the NEW source slice,
not a reconstructed/full Forge checkout. GitHub connector reads remained usable.

Remote Desktop returned no connected device. No owner host commands, Tailscale
changes, container changes, provider calls, purchases, training, HF writes,
website writes, branch-protection edits or production promotions were performed.
No secret values are needed in this source slice.

## Source movement readback

Before writing, main advanced to `7789e6051de75365730e0bca6fb5f9ef8b306683`
(tree `49e560b9ae424f4be1548132411ff95ab8608b01`). The compare from the inspected
base contains only `audit/HF_MODEL_TRAINING_2026-09-12.md` and three
`local-compute/recovery/*gguf_forensics*`/documentation additions. Those changes
are preserved by basing the new source commit on the newer main. No new
candidate-package or pyproject overlap was returned by that comparison.

## Reuse map

| Concern | Existing source inspected or located | Disposition |
| --- | --- | --- |
| Training, recovery, release | `szl-forge/local-compute/`; recovery merge #268, remaining #264 | Keep current supervised work and receipts. |
| Two-box GPU inference pool | `a11oy/box-scripts/litellm_config.yaml` at `4845411b81f6aa32c9802dae0b851928fc4f36c6` | Declares `sovereign-llm` across `omen` and `betterwithage`; not live observation. |
| Ollama/Tailscale/Docker topology | `a11oy/box-scripts/SOVEREIGN_MESH_RUNBOOK.md` at the same revision | Native Ollama, private tailnet, stateless Docker sidecars. No public port 11434. |
| Docker declaration | `a11oy/box-scripts/docker-compose.yml` referenced by the inspected runbook | Located; current running containers and complete Compose semantics not verified. |
| Remote teacher/fallback | `a11oy/docs/LAPTOP_ONLY_WIRING.md` located in source search | OpenRouter is remote inference, not owned compute or pooled training VRAM. |
| Policy gateway | `szl-router/README.md` and `router_control.app` | Existing mandatory-auth/provider-allowlist control; do not replace with learned policy. |
| Coordination mesh | `szl-mesh/README.md` | UDS/CRDT coordination, not the two-GPU inference pool; design claims are not runtime evidence. |
| Portfolio authority | `szl-forge/portfolio/model_portfolio.json` | Existing 16-entry declared scope, not a fresh HF census; left unchanged. |
| Public Forge frontend | `szl-forge/spaces/szl-forge-lab/README.md` | STATIC / SNAPSHOT / READ-ONLY; explicitly not a training console. |
| New source-only models | `model_candidates/` in Forge | Three prototypes, isolated Python UI; no new repo, scheduler, publisher or Space. |

Primary source coordinates:
- https://github.com/szl-holdings/szl-forge/tree/b6cc0cc635459fc8740db85ff0c6e98eeac84007/local-compute
- https://github.com/szl-holdings/a11oy/blob/4845411b81f6aa32c9802dae0b851928fc4f36c6/box-scripts/litellm_config.yaml
- https://github.com/szl-holdings/a11oy/blob/4845411b81f6aa32c9802dae0b851928fc4f36c6/box-scripts/SOVEREIGN_MESH_RUNBOOK.md
- https://github.com/szl-holdings/szl-router/blob/2399f28c796a859049cd0642a6dd66d6ea64f533/README.md
- https://github.com/szl-holdings/YARQA-ATTN/blob/main/README.md
- https://github.com/szl-holdings/szl-lambda-gate/blob/main/README.md

## Material findings and successor order

**P0: preserve working source and existing candidates.** Forge #262 already
contains a focused 46-known-ID audit; it is not a complete author enumeration.
Do not create a competing census. #243 remains a separate GPU-evaluation lane,
not a dependency to bypass. The inspected local-compute README records a prior
continuation whose held-out loss worsened and whose reload smoke did not improve.
A completed training command is not a passing candidate. Current #264 recovery
and current supervised candidates must remain separate from these new prototypes.

**P1: repair ownership ambiguity before publication.** The current portfolio's
`szl-kernels` source points at `governed-receipt-spec`, while a dedicated kernels
repo also exists. The Lambda README describes HF as canonical despite the desired
GitHub-first chain. Invariants involve both a dedicated source and the formal
registry. Reconcile exact publisher inputs and artifact identity per asset; do
not silently overwrite source pointers because repository names look similar.

**P1: verify the existing pool before adding compute orchestration.** Collect
current node, installed-model digest, loaded-work, Tailscale policy, gateway auth,
actual serving model and capability observations through the owner-host path.
The inspected LiteLLM YAML treats two deployments under one logical model as
interchangeable. Verify actual same-weight identity where claiming identical
replicas; otherwise retain explicit heterogeneous route identities in receipts.
Audit environment-variable resolution and retry/timeout behavior in the installed
LiteLLM version. Source configuration alone proves neither readiness nor VRAM fit.
Do not launch Ray/Kubernetes or assume inference load balancing creates a
multi-host training-memory pool.

**P1: tighten the historical network recipe before reuse.** The old runbook's
`OLLAMA_HOST=0.0.0.0` and wildcard origins need a current firewall/ACL review.
Prefer loopback behind a restricted private transport, or explicit private bind
and least-privilege firewall policy. Tailscale Serve is private tailnet sharing;
Funnel is public. Do not substitute Funnel or publicly expose Ollama. The new
local-only candidate workbench is not a service to expose through either.

**P1: learned counterparts, not renamed kernels.** Router/risk prototypes produce
advisory scores; hard authorization remains in existing deterministic services.
YARQA's original noncausal CPU kernel cannot be treated as a causal LM unchanged.
The new fixed-width causal reference is explicitly separate and tiny. Check
causality, gradient isolation, baseline quality and runtime support before any
scaling or export claim. `szl-kernels` remains software/family, not one neural net.

**Correction to the earlier migration alert.** Model-type metadata for the four
kernel IDs does NOT establish absence of first-class kernel repositories. The
YARQA source README already records a kernel-type publication. The HF connector's
model/dataset/Space lookup cannot by itself reconcile the kernel-type inventory.
Do not delete, recreate or reclassify assets on the earlier inference alone.
The three proposed new model IDs were not returned by model lookup; this is not
proof about inaccessible/private repositories. No HF repository was created.

**P2: publication and evidence projection.** After reviewed source, admitted data,
owner-supervisor integration, measured training, matched evaluation and current
release approvals, use the existing Forge publisher. GitHub -> HF -> a-11-oy.com
-> a11oy.net remains the order. Public cards must distinguish scaffold, random
initialization, trained candidate, evaluated candidate and qualified runtime.

Technical references (primary documentation; no upstream performance claims):
- https://huggingface.co/docs/kernels/migration
- https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
- https://huggingface.co/docs/safetensors/en/api/torch
- https://tailscale.com/docs/features/tailscale-serve
- https://openrouter.ai/docs/guides/routing/provider-selection
