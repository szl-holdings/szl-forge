# chaski-r4 — qualification state (after receipt C)

Last updated 2026-10-01 after [szl-forge #475](https://github.com/szl-holdings/szl-forge/pull/475) merged receipt C.
States follow the model-qualification-gates vocabulary: training, evaluation, publication and promotion are
assigned independently and never collapsed.

## Receipt C

`chaski_r4/evidence/canonical_rerun_20261001_140615.receipt.json` — sha256 `5ae3de970014726f190dfe8e4d5b35e667fd737b1be29e27205c25d143bdf403`
(report_sha256 `f166a31b74b9da6fd9d1652578957ec382aafb1f98bad42d947cfad8a135a93b`). Evaluator `chaski/bakeoff_named_n.py` at main 55c3027b (LF sha256
`251d966beb4ff9601de25191f12293a0d67ee2e732c9b6618a073201163998e1`), label `MEASURED`, `gate_ran=True`,
`publication_eligible=False` (fixed by the runner), `held_out_in_gradients=False`,
`hub_put=False`. Owner metal: NVIDIA GeForce RTX 5050 Laptop GPU, torch 2.11.0+cu128, transformers 5.16.1, peft 0.20.0,
Windows-10-10.0.26200-SP0, computed 2026-10-01T18:11:50Z. Base `huggingface:Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17`.
Gate files: `json_drafts.n5.jsonl` `dd093403e4aeeb90…`,
`adversarial_refusals.n6.jsonl` `1e58dda9cf07e224…`.

| Candidate | JSON drafts | Adversarial refusals | adapter_sha256 | adapter tensors applied | loader class / prompt renderer |
|---|---|---|---|---|---|
| `base-qwen35-0.8b` | 0/5 | 6/6 | `—` | base (no adapter) | `Qwen3_5ForConditionalGeneration` / AutoProcessor |
| `chaski-5050` | 5/5 | 6/6 | `fc7da61d9e30d9bc` | 192/192 | `Qwen3_5ForConditionalGeneration` / AutoProcessor |
| `chaski-r2` | 5/5 | 6/6 | `e35df3bea9b9d260` | 192/192 | `Qwen3_5ForConditionalGeneration` / AutoProcessor |
| `chaski-r4-local` | 5/5 | 6/6 | `e1abc37a5c41a82b` | 192/192 | `Qwen3_5ForConditionalGeneration` / AutoProcessor |

Every adapter tensor landed (192/192) under the evaluator's loader class; the coverage preflight that enforces this
is `tools/chaski_margin_probe.py --coverage-only` and it fails closed (`ADAPTER_NOT_APPLIED`). The prompt renderer was
`AutoProcessor`, the same path as the 2026-09-17 canonical receipt; each case carries `prompt_sha256`.

## How receipt C relates to receipts A and B

| | Receipt A (2026-09-16) | Receipt B (2026-09-17) | Canonical 2026-09-17 | Receipt C (2026-10-01) |
|---|---|---|---|---|
| Evaluator | `chaski_r4/bakeoff_canonical_four_way_r4.py` (`f4ca282a…`, `AutoModelForCausalLM`) | same | `chaski/bakeoff_named_n.py` (`AutoModelForImageTextToText`) | `chaski/bakeoff_named_n.py` + adapter guard (`251d966b…`) |
| Torch | 2.11.0+cu128 | 2.10.0+cu130 | 2.10.0+cu130 | 2.11.0+cu128 |
| base | 0/5, 6/6 | 0/5, 6/6 | 0/5, 6/6 | 0/5, 6/6 |
| chaski-5050 | 0/5, 6/6 (`b077cafe…`) | 5/5, 6/6 (`fc7da61d…`) | 5/5, 6/6 (`fc7da61d…`) | 5/5, 6/6 (`fc7da61d…`) |
| chaski-r2 (control, `e35df3be…`) | 5/5, 6/6 | 5/5, 6/6 | 5/5, 6/6 | 5/5, 6/6 |
| chaski-r4 | 0/5, 3/6 (original, `b116832a…`) | 5/5, 6/6 (retrained, `e1abc37a…`) | not evaluated | 5/5, 6/6 (retrained, `e1abc37a…`) |

Receipt C reproduces receipt B's integer counts for the retrained r4 beside the r2 control, in receipt A's torch build,
with a loader that demonstrably applies the adapters. Output bytes differ case by case across torch/CUDA builds
(base 10/11 identical to B, r2 10/11, r4 8/11, 5050 5/11); the gate counts do not.

**What C does not settle.** Receipts A and B show adapter effects that the `f4ca282a…` evaluator copy cannot produce
with the current adapter files under any transformers/peft pairing probed (transformers 5.4.0–5.18.0, peft
0.18.1–0.21.1 map `AutoModelForCausalLM` to `Qwen3_5ForCausalLM` and apply 0/192 tensors of a `language_model`-layout
adapter). Their provenance is recorded as unresolved in `tools/geh_v8/THREAD_AUDIT.md`; receipt C neither explains nor
depends on them. The local r2 control (`a16be6dd…` weights, dir-hash `e35df3be…`) is not byte-identical to the
published `SZLHOLDINGS/chaski-r2` adapter (`6f12981e…`); both score 5/5 + 6/6 under this evaluator.

## State block

```text
Artifact: chaski-r4 local adapter, retrained (weights f1a2cdc313795775966280bc8648367005700327dd28010da8d2f88d2a5e2a02, dir-hash e1abc37a5c41a82b0fc2cd98ccd6edbb2a08fceb8ca9e883bcfb76b861c221cf; matches chaski_r4/training_receipt.json)
Canonical source: szl-holdings/szl-forge chaski/bakeoff_named_n.py @ 55c3027b; receipt C committed in #475 (06804ce4)
Training: TRAINED_CHALLENGER (training_receipt.json 2026-09-17T02:36Z; dataset 0fea0d85…; training_plan hash mismatch recorded)
Evaluation: MEASURED — receipt C: r4 5/5 + 6/6 beside r2 control 5/5 + 6/6; 5050 5/5 + 6/6; base 0/5 + 6/6; adapters 192/192 applied
Publication: UNPUBLISHED (chaski-r4 is absent from the Hub; evidence is published in this repository only)
Promotion: NOT_PROMOTABLE — publication_eligible=false is fixed by the evaluator; promotion is an owner decision taken outside the gate
Blocking gate: none from the gate itself. Open items: (1) owner decision on publishing r4 as a PUBLIC_EXPERIMENTAL_ARTIFACT with receipts A, B, C and the provenance caveat above the fold; (2) receipts A/B provenance stays unresolved
Next bounded action: owner decision. If publish: card first (status line, receipt C hash, limits), bytes second, never the reverse
Protected state: receipts A and B, chaski_r4/training_receipt.json, adapters, gate files, pristine evaluator copy; none modified
Completion evidence: this file + receipt C sha256 5ae3de970014726f… + GEH v8 chain (12 receipts, head 344df7b4…) in chaski_r4/evidence/geh/
```

## Claim boundary

Receipt C's own boundary applies verbatim: Owner-metal named-N bake-off. Integer counts only. Small synthetic gate (n=5 JSON drafts, n=6 adversarial refusals). Not a broad quality or safety benchmark. Not SOTA. Gate files stay held-out and keep gate_ran=false. Parent eval_chaski.py remains a kit stamp. publication_eligible stays false. No Hub PUT. Live SZLHOLDINGS/chaski is not overwritten. House CPU lab stays signed Khipu GGUF.
