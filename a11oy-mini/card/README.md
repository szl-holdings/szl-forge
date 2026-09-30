---
thumbnail: https://huggingface.co/SZLHOLDINGS/A11OY-MINI/resolve/7ea56236ea7988b3915b5ef07548cb2c3930a3cf/og-card.png
license: apache-2.0
language:
  - en
pipeline_tag: text-generation
library_name: llama.cpp
tags:
  - szl-holdings
  - series-a
  - doctrine-v11
  - governed-ai
  - gguf
szl:
  doctrine: v11-LOCKED
  lean: "749/14/163"
  lambda: "Conjecture 1 — advisory, never a theorem"
  artifact_class: GGUF
  originality: QUANT_OF_SZL_ORIGINAL
  legacy_parent: SZLHOLDINGS/chaski
  r2_parent: SZLHOLDINGS/chaski-r2
  legacy_parent_hub_sha: 1c55df8652e9d0f7b84356b1e2d54849165ae884
  forbidden_parent: SZLHOLDINGS/chaski-5050
  base: Qwen/Qwen3.5-0.8B
  seed: 11
  evals: R2_NAMED_N_MEASURED_LEGACY_NOT_EVALUATED
  publication_eligible: false
  autonomy_eligible: false
  lab_load: forbidden
  gpu: LOCAL_CUDA_TEXT_OBSERVED_2026_09_24
  status: BYTES_AVAILABLE
  weights: AVAILABLE
  khipu_lab_pin: false
  tok_s_claim: false
  third_llm: false
  new_train: false
  r2_rebuild: 2026-09-17
  r2_gate: "5/5 + 6/6 MEASURED ollama"
  native_text_observation: "2026-09-24 envelope 5/5 + 6/6; semantic failures; HOLD"
---

> **DERIVED GGUF RUNTIME · R2 NAMED-N RECORD · LEGACY FILES DEPRECATED · NOT PROMOTABLE · NOT AUTONOMOUS**

# A11OY-MINI

## What this is

A derived GGUF runtime family. The R2 files are a quantized rebuild from a local Chaski-R2 adapter; the retained legacy F16 and Q4_K_M files derive from Chaski. It is not a new training run, not Chaski-5050, not a flagship release, and not an authorized A11oy production deployment.

## Evidence and artifact identity

Card evidence reviewed on 2026-09-30 UTC at Hub revision [`7ea56236ea7988b3915b5ef07548cb2c3930a3cf`](https://huggingface.co/SZLHOLDINGS/A11OY-MINI/tree/7ea56236ea7988b3915b5ef07548cb2c3930a3cf). The historical artifact/evaluation snapshot below is `c936dc749743c94586706345a0142c79094a581c`. The values below are the Hub's LFS SHA-256 identities and sizes at that revision. R2 values match the canonical [R2 conversion receipt](https://github.com/szl-holdings/szl-forge/blob/a6b58f623185820eddfb0b279d21b7d81b87137c/chaski_r2/a11oy_mini_r2_gguf/gguf_receipt.json); legacy values match [conversion_receipt.json](https://huggingface.co/SZLHOLDINGS/A11OY-MINI/blob/c936dc749743c94586706345a0142c79094a581c/conversion_receipt.json). Listing and receipt comparison do not constitute a new download-and-rehash of the GGUFs.

| Runtime artifact | Published file | Bytes | SHA-256 |
|---|---|---:|---|
| R2 text Q4_K_M | [a11oy-mini-r2-Q4_K_M.gguf](https://huggingface.co/SZLHOLDINGS/A11OY-MINI/blob/c936dc749743c94586706345a0142c79094a581c/a11oy-mini-r2-Q4_K_M.gguf) | 541903328 | `6d42341c932a76e91b2c04a859a4248e7d2c77308f998a32771f43802f097b62` |
| R2 BF16 projector | [a11oy-mini-r2-BF16-mmproj.gguf](https://huggingface.co/SZLHOLDINGS/A11OY-MINI/blob/c936dc749743c94586706345a0142c79094a581c/a11oy-mini-r2-BF16-mmproj.gguf) | 207346048 | `4855efe034435b9b3b289b2c07d09b263c088749d1f85c43c1cf4672bc7fcbf2` |
| Legacy F16 — deprecated | [a11oy-mini-f16.gguf](https://huggingface.co/SZLHOLDINGS/A11OY-MINI/blob/c936dc749743c94586706345a0142c79094a581c/a11oy-mini-f16.gguf) | 1557662240 | `a5df00e4e3ca07f65a4b43aad4ef1505625952a3105e6dcc0dba87f2fa35fc57` |
| Legacy Q4_K_M — deprecated | [a11oy-mini-q4_k_m.gguf](https://huggingface.co/SZLHOLDINGS/A11OY-MINI/blob/c936dc749743c94586706345a0142c79094a581c/a11oy-mini-q4_k_m.gguf) | 541903392 | `06136ba385b2e052cf4cdb3dc8d333948e0b612bd15a541b314e170399c2faa6` |

The [gguf_gate_receipt.json](https://huggingface.co/SZLHOLDINGS/A11OY-MINI/blob/c936dc749743c94586706345a0142c79094a581c/gguf_gate_receipt.json) records **JSON drafts 5/5** and **adversarial refusals 6/6** on 2026-09-17, using **Ollama, temperature 0, context 4096**, with the named Chaski gate files `chaski/gate/json_drafts.n5.jsonl` and `chaski/gate/adversarial_refusals.n6.jsonl`. This is a small owner-run GGUF-level record for the named **R2 Q4_K_M text artifact**.

The gate receipt names the R2 artifact but does not itself contain the artifact SHA-256 or an immutable Hub revision. The separate canonical conversion receipt records the R2 file hashes and initially says `gguf_scores=UNMEASURED`; the later gate receipt records the measured run. The [hub_put_receipt.json](https://huggingface.co/SZLHOLDINGS/A11OY-MINI/blob/c936dc749743c94586706345a0142c79094a581c/hub_put_receipt.json) records the later authorized R2 upload decision. These documents provide scoped lineage and measured-record evidence; they do not make the evaluation receipt an independently verified cryptographic binding to public bytes.

## New exact-byte text observation — 2026-09-24

**HOLD remains in force.** A new owner-local CUDA observation binds the R2 text
GGUF at Hub revision `0619dd65b92a135501af35b3c4e3b4e762be1d7d` to local
SHA-256 `6d42341c932a76e91b2c04a859a4248e7d2c77308f998a32771f43802f097b62`,
verified before and after inference. The [unaltered receipt](https://huggingface.co/SZLHOLDINGS/A11OY-MINI/blob/7ea56236ea7988b3915b5ef07548cb2c3930a3cf/evidence/2026-09-24-native-cuda/receipt.json)
records the installed llama.cpp native CUDA runtime, RTX 5050 Laptop GPU,
temperature 0, seed 0, context 4096, immutable fixture identities, and all outputs.
The historical envelope checker reports **5/5 drafts and 6/6 refusal prefixes**;
that is a format result, not semantic correctness.

[2026-09-25 semantic review](https://huggingface.co/SZLHOLDINGS/A11OY-MINI/blob/7ea56236ea7988b3915b5ef07548cb2c3930a3cf/evidence/2026-09-24-native-cuda/semantic_review.json) found four
unsupported claims: treating adapter files as an evaluation/gate closure,
presenting training loss as evaluation, inventing job state, and declaring
COMPLETED inside a refusal. The receipt's `MEASURED_BOUNDED_PASS` is preserved
with its original envelope-only meaning. Production disposition stays **HOLD** with review status `SEMANTIC_FAILURES_OBSERVED`,
with `publication_eligible=false`, `autonomy_eligible=false`,
`promotion_effect=NONE`, `PROPOSAL_ONLY`, and `UNSIGNED_HONEST`.

No semantic pass rate is reported (`semantic_pass_rate=null`).

The [full evidence and exact reproduction sources](https://huggingface.co/SZLHOLDINGS/A11OY-MINI/blob/7ea56236ea7988b3915b5ef07548cb2c3930a3cf/evidence/2026-09-24-native-cuda/README.md)
are also retained in [canonical GitHub source](https://github.com/szl-holdings/szl-forge/tree/7cf86ae27e0c68b6fc694560038d04ac31065ce6/chaski_r2/a11oy_mini_r2_gguf/evidence/2026-09-24-native-cuda).
This text-only observation uses known named fixtures. It does not retroactively
bind the 2026-09-17 run, qualify the projector/vision path, establish held-out
generalization, authorize a house-lab retarget, or establish a deployed service
or autonomy.

## What the evidence does and does not establish

The **R2 5/5 + 6/6 result applies only to the named R2 Q4_K_M artifact**. It does not transfer to the legacy GGUFs, the projector's visual performance, the Chaski baseline, Chaski-5050, all adapters, another quantization, or a new training run. It is not a public leaderboard result, a broad benchmark, a speed claim, or independent third-party qualification.

The legacy [conversion_receipt.json](https://huggingface.co/SZLHOLDINGS/A11OY-MINI/blob/c936dc749743c94586706345a0142c79094a581c/conversion_receipt.json) retains `evals=none-this-run` and identifies Chaski parent revision `1c55df8652e9d0f7b84356b1e2d54849165ae884`. Its byte measurements are not behavioral evaluations. Both legacy files remain published for provenance and are **deprecated**. The R2 parent has a real [Chaski-R2 Hub page](https://huggingface.co/SZLHOLDINGS/chaski-r2), but the canonical R2 conversion receipt identifies a local source-adapter path rather than an immutable parent Hub revision.

## Release, autonomy, and deployment boundary

`publication_eligible=false` and `autonomy_eligible=false` remain unchanged. **House-lab load is forbidden.** This is not a lab retarget, a release qualification, deployment authorization, approval authority, or autonomous tool executor. Public artifact availability does not change those gates. No tokens-per-second or vision benchmark is claimed. Lambda uniqueness remains **Conjecture 1: open and advisory, never a theorem**.

## Relationship and source of truth

Legacy parent: [SZLHOLDINGS/chaski](https://huggingface.co/SZLHOLDINGS/chaski). R2 source family: [SZLHOLDINGS/chaski-r2](https://huggingface.co/SZLHOLDINGS/chaski-r2). Neither is [SZLHOLDINGS/chaski-5050](https://huggingface.co/SZLHOLDINGS/chaski-5050). This derived runtime does not replace the source adapters or the [Khipu GGUF house-lab artifact](https://huggingface.co/SZLHOLDINGS/SZL-Khipu-1.5B-GGUF).

Use [gguf_gate_receipt.json](https://huggingface.co/SZLHOLDINGS/A11OY-MINI/blob/c936dc749743c94586706345a0142c79094a581c/gguf_gate_receipt.json), [hub_put_receipt.json](https://huggingface.co/SZLHOLDINGS/A11OY-MINI/blob/c936dc749743c94586706345a0142c79094a581c/hub_put_receipt.json), the legacy [conversion_receipt.json](https://huggingface.co/SZLHOLDINGS/A11OY-MINI/blob/c936dc749743c94586706345a0142c79094a581c/conversion_receipt.json), and the canonical [R2 conversion receipt](https://github.com/szl-holdings/szl-forge/blob/a6b58f623185820eddfb0b279d21b7d81b87137c/chaski_r2/a11oy_mini_r2_gguf/gguf_receipt.json) as the evidence sources. The [historical source documentation](https://github.com/szl-holdings/szl-forge/blob/a6b58f623185820eddfb0b279d21b7d81b87137c/a11oy_mini/README.md) describes the original runtime family; later R2 evidence is scoped separately above.

Canonical card authoring path: `a11oy-mini/card/README.md` in `szl-holdings/szl-forge`. This source correction preserves the stronger dated published evidence; it changes no receipt, model file, release gate or deployed service.
