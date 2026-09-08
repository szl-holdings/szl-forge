# MiniCPM5: executable qualification, not production promotion

Copyright 2026 SZL Holdings. SPDX-License-Identifier: Apache-2.0

Tracks `szl-holdings/szl-frontier#25`, after the source intake and live metadata
witness in frontier PRs #27/#28. This Forge lane moves from source metadata to
real model-execution evidence without replacing an existing model or public lab.

## Canonical source and boundary

- Model: https://huggingface.co/openbmb/MiniCPM5-2B
- Fixed revision: `3497c460c89e00520c3cfa2e73f49ab7647f1177`.
- Source: `inference/minicpm5_qualification.py` in this repository.
- Authority: GitHub -> Hugging Face -> a-11-oy.com -> a11oy.net.

The model card describes a standard Llama architecture, a 131072-token context
and SGLang's `minicpm5` tool parser. This runner qualifies none of those broader
claims. It deliberately tests a 4096-token-bounded direct Transformers path,
not SGLang tool parsing, production tenant isolation, or the whole context range.
The upstream Apache-2.0 model is evaluated only; the separate UltraData training
corpora are not downloaded, executed, admitted to training, or republished.

## Actual implementation

The dependency-free default command prints a deterministic plan:

```bash
python inference/minicpm5_qualification.py
python -m unittest discover -s tests -p 'test_minicpm5_qualification.py' -v
```

An explicit `--run` on separately provisioned CUDA hardware verifies the exact
Hub model revision, selects only named configuration/tokenizer files and
safetensors, checks a 6 GiB aggregate download limit, hashes every file, checks
weight SHA-256 against upstream metadata, rejects remote-code mappings, and
loads locally with `trust_remote_code=False` / `use_safetensors=True`.
It never starts a server, trains a model, executes a tool, uploads weights or
mutates the model-inference-lab Space.

The suite contains 12 **public synthetic** probes: lookup, missing evidence,
cross-tenant evidence, revoked access, instruction-like untrusted notes and
Unicode grounding, twice each. They are smoke/contract probes, not a hidden
held-out benchmark. The actual existing controller still owns access decisions;
correct answers to these prompts do not demonstrate security isolation.

Each generation is bounded to 96 new tokens and at most 30 seconds of generation
stopping-time allowance. Overall generation allowance is 300 seconds. The
external job must also have a hard timeout; backend stopping criteria are not
an operating-system isolation boundary. No automatic retry loop is configured.

Outputs must match the strict JSON schema and exact expected grounded values.
Duplicate keys, NaN/Infinity/overflow numbers, code fences, extra fields,
truncation and private-reasoning prefixes fail. There is no repair-to-green.
Only output hashes, result codes, generation token counts, latency and CUDA
allocation/reservation counters are persisted; no raw output/reasoning or
private Second Brain content enters the report.

## Reproducible execution contract

Use an isolated official PyTorch CUDA image and pin the top-level evaluation
libraries, for example `transformers==5.6.0` and `huggingface_hub==1.23.0`.
The full installed package inventory is recorded by the runner. An image tag
plus that inventory is NOT an immutable deployment closure: the report keeps
`imageDigestVerified=false` and `runtimeQualified=false`. Resolve an immutable
image and hash-locked dependency closure before a deployment qualification.

Fetch the script from a full GitHub commit, verify its SHA-256, then invoke:

```bash
python minicpm5_qualification.py --run \
  --source-revision <exact-github-commit> \
  --image <declared-runtime-image> \
  --output /tmp/minicpm5-qualification.json
```

Do not pass `main`, an invented image digest, a Hugging Face write token, or
production data. The report binds the declared source revision and executed
file hash; verify both against GitHub. Running a reviewed feature-branch source
for an evaluation does not admit that branch into production.

Exit 0 means all 12 smoke probes passed; exit 2 means completed with at least one
quality failure; exit 1 means incomplete. Atomic checkpointing replaces stale
PASS data, and the final report is emitted with `SZL_QUALIFICATION_JSON=` in job
logs. A hard-killed job without a complete final report is incomplete, not PASS.

## Interpretation and downstream owners

Generation latency covers the complete generation call, with CUDA synchronization;
it is not TTFT, streaming inter-token latency, or production throughput. p50/p95
use nearest-rank over this small 12-case sequence, including cold first-generation
effects. CUDA allocated/reserved peaks are process counters, not total host/GPU
usage. Energy, billed cost and TTFT stay null unless separately measured.

Every result remains unsigned and `publicationEligible=false`,
`productionDisposition=HOLD`, `runtimeQualified=false`, with no tool authority.
Even 12/12 is only a bounded smoke result, never production readiness.

Forge owns this evaluation and future model/runtime selection. Nemo remains the
independent envelope witness; szl-serve remains recipe/validator/receipt owner.
Neither is replaced by this runner. Next: matched-baseline held-out evaluation,
SGLang parser/fallback qualification, real controller/Nemo/Serve integration,
immutable runtime closure, separate data-rights/decontamination review, rollback,
and truthful A11oy product/proof projections. Existing signed model artifacts,
formula semantics, Second Brain storage and public serving defaults are unchanged.
