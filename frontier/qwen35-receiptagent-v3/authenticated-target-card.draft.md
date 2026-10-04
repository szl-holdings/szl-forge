---
tags:
- receipt-agent
- research
- unreleased-target
---

<p><a href="https://huggingface.co/spaces/SZLHOLDINGS/szl-command-lab"><img src="https://raw.githubusercontent.com/szl-holdings/.github/main/profile/assets/szl/logos/szl_mark_holographic.svg" alt="SZL Holdings" width="112"></a></p>

# ReceiptAgent v3 authenticated target

A reserved repository for a receipt-oriented adapter whose future release must carry an authenticated training, evaluation, and comparison chain.

**Artifact:** Release destination; no published adapter at the audited revision  
**Stage:** `UNRELEASED_TARGET`

**[Open Command Lab](https://huggingface.co/spaces/SZLHOLDINGS/szl-command-lab)** · [Build source](https://github.com/szl-holdings/szl-forge/tree/b18f2604c309710847158443a907f9a20be9e95a/frontier/qwen35-receiptagent-v3) · [Audited repository revision](https://huggingface.co/SZLHOLDINGS/szl-receiptagent-qwen35-0.8b-v3-authenticated/tree/af86a187f507a5423f5236c7cf77cf48ca67d077)

- **Inference is unavailable from this target.** The audited tree has no adapter, model configuration, tokenizer, or release manifest.
- **Qualification is pending.** Source tests and native Windows observer checks do not establish completed training or model quality.
- **Authority stays external.** The planned model can propose structured receipt responses; it cannot approve, execute, sign, or promote its own output.

## What is being built

The [source candidate](https://github.com/szl-holdings/szl-forge/blob/b18f2604c309710847158443a907f9a20be9e95a/frontier/qwen35-receiptagent-v3/candidate.json) declares a future LoRA/PEFT adapter for `DRAFT`, evidence `RECOVERY`, and `REFUSE` responses. Its implementation base is `unsloth/Qwen3.5-0.8B` pinned to `23c69c53358a07516b5827588b3fdb12ae78fd65`. That is a source configuration, not evidence of weights in this repository.

The curriculum is project-authored synthetic policy/schema material. The committed final evaluation set is public and preregistered; it does not constitute blind or independent certification.

## Release requirements

The [owner Windows/WSL proof](https://github.com/szl-holdings/szl-forge/issues/546) and the [host-memory guard](https://github.com/szl-holdings/szl-forge/blob/b18f2604c309710847158443a907f9a20be9e95a/frontier/qwen35-receiptagent-v3/HOST_MEMORY_GUARD.md) remain part of the run requirements. A complete candidate also needs measured training and evaluation, the frozen comparison, and authenticated evidence.

The [release builder](https://github.com/szl-holdings/szl-forge/blob/b18f2604c309710847158443a907f9a20be9e95a/frontier/qwen35-receiptagent-v3/prepare_release.py) generates the measured model card from that evidence. The [publisher](https://github.com/szl-holdings/szl-forge/blob/b18f2604c309710847158443a907f9a20be9e95a/tools/publish_receiptagent_v3.py) then requires an exact parent revision and verifies immutable repository bytes. A separate runtime witness is still required for a runtime claim.

This status page is documentation. It carries no authenticated release receipt or model-performance result.

## Dated evidence

At `2026-10-04T14:42:54.991300Z`, the Hub file roster for revision `af86a187f507a5423f5236c7cf77cf48ca67d077` contained only `.gitattributes`. An authenticated read of `README.md` at that same immutable revision returned HTTP 404 at `2026-10-04T17:28:27.566919Z`.

The word “authenticated” in the repository name identifies the intended release lane; the name itself supplies no evidence.
