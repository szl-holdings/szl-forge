---
thumbnail: https://huggingface.co/SZLHOLDINGS/A11OY-MINI/resolve/main/og-card.png
license: apache-2.0
language:
  - en
pipeline_tag: text-generation
library_name: llama.cpp
base_model: Qwen/Qwen3.5-0.8B
# base_model is the silhouette (Qwen3.5 instruct); the cut is SZL Chaski / Chaski-R2.
# No base_model_relation: quantized — the Chaski parents are not publication-eligible (szl-forge rule).
tags:
  - szl-holdings
  - doctrine-v11
  - governed-ai
  - proposal-only
  - gguf
szl:
  doctrine: v11-LOCKED
  lean: "749/14/163"
  lambda: "Conjecture 1 — advisory, never a theorem"
  artifact_class: GGUF
  originality: QUANT_OF_SZL_ORIGINAL
  legacy_parent: SZLHOLDINGS/chaski
  legacy_parent_artifact: merged-shard
  legacy_parent_hub_sha: 1c55df8652e9d0f7b84356b1e2d54849165ae884
  r2_parent: SZLHOLDINGS/chaski-r2
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
  hub_put: R2_UPLOAD_2026-09-17  # hub_put_receipt.json decision UPLOAD; GGUF bytes OBSERVED on the Hub tree
  khipu_lab_pin: false
  tok_s_claim: false
  third_llm: false
  new_train: false
  r2_rebuild: 2026-09-17
  r2_gate: "5/5 + 6/6 MEASURED ollama"
  native_text_observation: "2026-09-24 envelope 5/5 + 6/6; semantic failures; HOLD"
  card_resync: 2026-09-25
---

> **DERIVED GGUF RUNTIME · R2 NAMED-N RECORD · LEGACY FILES DEPRECATED · NOT PROMOTABLE · NOT AUTONOMOUS**

# A11OY-MINI

> Card re-synced from `szl-holdings/szl-forge` on 2026-09-25: frontmatter now carries `base_model: Qwen/Qwen3.5-0.8B` from the source card ([`a11oy_mini/README.md`](https://github.com/szl-holdings/szl-forge/blob/main/a11oy_mini/README.md)); the `series-a` tag was removed. The source card's "no GGUF bytes / ROADMAP until a `.gguf` exists" statements describe an earlier checkout and are superseded by the Hub tree below; the R2 evidence sections come from the newer [`chaski_r2/a11oy_mini_r2_gguf`](https://github.com/szl-holdings/szl-forge/tree/main/chaski_r2/a11oy_mini_r2_gguf) kit, whose receipts and evidence directory are byte-identical copies of the files on this Hub ID.

## What this is

A derived GGUF runtime family. The R2 files are a quantized rebuild from a local Chaski-R2 adapter; the retained legacy F16 and Q4_K_M files derive from Chaski. It is not a new training run, not Chaski-5050, not a flagship release, and not an authorized A11oy production deployment.

## Hub tree (OBSERVED 2026-09-25)

The source card's condition ("not a quantized Hub child until a `.gguf` file exists on `SZLHOLDINGS/A11OY-MINI`") is met on the bytes: four `.gguf` files are present. The card still declares no `base_model_relation: quantized`, because the Chaski parents remain `publication_eligible=false`.

| Path | Bytes | Role |
|---|---:|---|
| `a11oy-mini-r2-Q4_K_M.gguf` (LFS) | 541,903,328 | R2 text Q4_K_M — the only artifact with a measured gate |
| `a11oy-mini-r2-BF16-mmproj.gguf` (LFS) | 207,346,048 | R2 BF16 projector — no vision/projector evaluation on record |
| `a11oy-mini-f16.gguf` (LFS) | 1,557,662,240 | legacy F16 — deprecated, `evals=none-this-run` |
| `a11oy-mini-q4_k_m.gguf` (LFS) | 541,903,392 | legacy Q4_K_M — deprecated, `evals=none-this-run` |
| `Modelfile.a11oy-r2.gguf` | 142 | Ollama Modelfile for the R2 Q4_K_M (temperature 0, num_ctx 4096) |
| `conversion_receipt.json` | 2,224 | legacy conversion receipt (2026-08-28, local convert) |
| `gguf_gate_receipt.json` | 636 | R2 gate record (2026-09-17, Ollama) |
| `hub_put_receipt.json` | 794 | R2 upload decision (2026-09-17) |
| `evidence/2026-09-24-native-cuda/` | receipt.json 59,040 + README, manifest, semantic review, logs, `repro/` | native CUDA text observation package |
| `bom/model-bom.cdx.json` | 3,382 | CycloneDX BOM |
| `og-card.png` (LFS) | 2,066,185 | thumbnail |

Sizes were read from the Hub listing; no GGUF was downloaded or re-hashed by this pass. The revision SHA was not captured by this pass.

## Evidence and artifact identity

Reviewed Hub revision: `c936dc749743c94586706345a0142c79094a581c`. The values below are the Hub's LFS SHA-256 identities and sizes at that revision. R2 values match the canonical [R2 conversion receipt](https://github.com/szl-holdings/szl-forge/blob/a6b58f623185820eddfb0b279d21b7d81b87137c/chaski_r2/a11oy_mini_r2_gguf/gguf_receipt.json); legacy values match [conversion_receipt.json](https://huggingface.co/SZLHOLDINGS/A11OY-MINI/blob/c936dc749743c94586706345a0142c79094a581c/conversion_receipt.json). Listing and receipt comparison do not constitute a new download-and-rehash of the GGUFs. The 2026-09-25 re-sync re-listed the tree (sizes above unchanged) and re-read the receipts; the hashes below were not recomputed.

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
verified before and after inference. The [unaltered receipt](./evidence/2026-09-24-native-cuda/receipt.json)
records the installed llama.cpp native CUDA runtime, RTX 5050 Laptop GPU,
temperature 0, seed 0, context 4096, immutable fixture identities, and all outputs.
The historical envelope checker reports **5/5 drafts and 6/6 refusal prefixes**;
that is a format result, not semantic correctness.

[Semantic review](./evidence/2026-09-24-native-cuda/semantic_review.json) found four
unsupported claims: treating adapter files as an evaluation/gate closure,
presenting training loss as evaluation, inventing job state, and declaring
COMPLETED inside a refusal. The receipt's `MEASURED_BOUNDED_PASS` is preserved
with its original envelope-only meaning. Production disposition stays **HOLD**,
with `publication_eligible=false`, `autonomy_eligible=false`,
`promotion_effect=NONE`, `PROPOSAL_ONLY`, and `UNSIGNED_HONEST`.

The [full evidence and exact reproduction sources](./evidence/2026-09-24-native-cuda/README.md)
are also retained in [canonical GitHub source](https://github.com/szl-holdings/szl-forge/tree/7cf86ae27e0c68b6fc694560038d04ac31065ce6/chaski_r2/a11oy_mini_r2_gguf/evidence/2026-09-24-native-cuda).
This text-only observation uses known named fixtures. It does not retroactively
bind the 2026-09-17 run, qualify the projector/vision path, establish held-out
generalization, authorize a house-lab retarget, or establish a deployed service
or autonomy.

## What the evidence does and does not establish

The **R2 5/5 + 6/6 result applies only to the named R2 Q4_K_M artifact**. It does not transfer to the legacy GGUFs, the projector's visual performance, the Chaski baseline, Chaski-5050, all adapters, another quantization, or a new training run. It is not a public leaderboard result, a broad benchmark, a speed claim, or independent third-party qualification.

The legacy [conversion_receipt.json](https://huggingface.co/SZLHOLDINGS/A11OY-MINI/blob/c936dc749743c94586706345a0142c79094a581c/conversion_receipt.json) retains `evals=none-this-run` and identifies Chaski parent revision `1c55df8652e9d0f7b84356b1e2d54849165ae884`. Its byte measurements are not behavioral evaluations. Both legacy files remain published for provenance and are **deprecated**. The R2 parent has a real [Chaski-R2 Hub page](https://huggingface.co/SZLHOLDINGS/chaski-r2), but the canonical R2 conversion receipt identifies a local source-adapter path rather than an immutable parent Hub revision.

## Source recipe (szl-forge `a11oy_mini/`, re-synced 2026-09-25)

> **CHAWPI silhouette.** Silhouette from Qwen3.5 instruct. Cut is original live Chaski. This folder exports that cut to GGUF. We do not republish someone else's tensors and we do not train a third messenger.

| | |
|---|---|
| **SKU** | `SZLHOLDINGS/A11OY-MINI` |
| **Legacy parent** | `SZLHOLDINGS/chaski` **merged shard** (`model.safetensors-00001-of-00001.safetensors`, parent revision `1c55df86…`) — deprecated files |
| **R2 parent** | `SZLHOLDINGS/chaski-r2` (local adapter path per the R2 conversion receipt) |
| **Not parent** | `SZLHOLDINGS/chaski-5050` |
| **Base / silhouette** | `Qwen/Qwen3.5-0.8B` (Apache-2.0, Qwen3.5 instruct). Cut is live SZL Chaski / Chaski-R2. |
| **Convert** | llama.cpp F16 (`convert_hf_to_gguf.py --outtype f16`), then `Q4_K_M` via `llama-quantize` (or `ollama create --quantize q4_K_M` FROM the F16 GGUF) |
| **Banned** | Direct safetensors→Ollama (MEASURED 2026-07-12 `@` spam) |
| **Evals** | legacy files: none-this-run (inherited from live Chaski). R2 Q4_K_M: envelope-only 5/5 + 6/6 MEASURED, semantic failures on review; HOLD |
| **Publication** | `publication_eligible: false` |
| **Lab** | House CPU lab stays **Khipu GGUF**. Do not retarget inference-lab. |
| **License** | Apache-2.0 |
| **Doctrine** | v11 LOCKED. Seed 11. Λ = Conjecture 1 (advisory, never a theorem). |

Scripts in the source kit:

```bash
python a11oy_mini/convert_a11oy_mini_gguf.py
python a11oy_mini/eval_a11oy_mini.py
python a11oy_mini/serve_a11oy_mini.py --check
```

Convert, when a Chaski merge is on disk at `a11oy_mini/chaski-merged/`: `python a11oy_mini/convert_a11oy_mini_gguf.py --convert --fetch-llama-cpp`. `ollama create` from a safetensors directory is refused. Default `--status` writes `conversion_receipt.json` with `publication_eligible: false`. Serve pins `SZLHOLDINGS/A11OY-MINI` only; it refuses live Chaski overwrite and refuses a 5050 parent. It does not pin `SZLHOLDINGS/SZL-Khipu-1.5B-GGUF` or the inference lab.

Superseded statements from the source card, kept for the record: "Scripts only this week. No Hub PUT. No GGUF bytes." and "ROADMAP until a `.gguf` file exists on `SZLHOLDINGS/A11OY-MINI`" described the checkout before the 2026-08-28 legacy conversion and the 2026-09-17 R2 upload; the Hub tree above now holds the bytes, and `hub_put_receipt.json` records the upload decision.

## What this is NOT

- Not a new train and not a third LLM
- Not `chaski-5050`
- Not a `base_model_relation: quantized` Hub child (parents are not publication-eligible)
- Not the Khipu lab pin
- Not a tokens/s claim
- Not a semantically qualified release (2026-09-24 review recorded four unsupported claims; HOLD)

## Release, autonomy, and deployment boundary

`publication_eligible=false` and `autonomy_eligible=false` remain unchanged. **House-lab load is forbidden.** This is not a lab retarget, a release qualification, deployment authorization, approval authority, or autonomous tool executor. Public artifact availability does not change those gates. No tokens-per-second or vision benchmark is claimed. Lambda uniqueness remains **Conjecture 1: open and advisory, never a theorem**.

## Relationship and source of truth

Legacy parent: [SZLHOLDINGS/chaski](https://huggingface.co/SZLHOLDINGS/chaski). R2 source family: [SZLHOLDINGS/chaski-r2](https://huggingface.co/SZLHOLDINGS/chaski-r2). Neither is [SZLHOLDINGS/chaski-5050](https://huggingface.co/SZLHOLDINGS/chaski-5050). This derived runtime does not replace the source adapters or the [Khipu GGUF house-lab artifact](https://huggingface.co/SZLHOLDINGS/SZL-Khipu-1.5B-GGUF).

Use [gguf_gate_receipt.json](https://huggingface.co/SZLHOLDINGS/A11OY-MINI/blob/c936dc749743c94586706345a0142c79094a581c/gguf_gate_receipt.json), [hub_put_receipt.json](https://huggingface.co/SZLHOLDINGS/A11OY-MINI/blob/c936dc749743c94586706345a0142c79094a581c/hub_put_receipt.json), the legacy [conversion_receipt.json](https://huggingface.co/SZLHOLDINGS/A11OY-MINI/blob/c936dc749743c94586706345a0142c79094a581c/conversion_receipt.json), and the canonical [R2 conversion receipt](https://github.com/szl-holdings/szl-forge/blob/a6b58f623185820eddfb0b279d21b7d81b87137c/chaski_r2/a11oy_mini_r2_gguf/gguf_receipt.json) as the evidence sources. The [source card](https://github.com/szl-holdings/szl-forge/blob/main/a11oy_mini/README.md) describes the original runtime family and conversion recipe; later R2 evidence is scoped separately above. GitHub (`szl-holdings/szl-forge`) is canonical; this Hub card is the mirror.
