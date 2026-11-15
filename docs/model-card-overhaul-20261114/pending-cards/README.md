# Pending wave-4 honest minimal cards (2026-11-14 audit remediation)

The 2026-11-14 audit of the Hugging Face estate found that 8 of the 40 wave-4
`-v2-trained` repositories still carried the default auto-generated model card
while their 32 siblings carry the honest minimal `eval-battery-bound` card.
These 8 are exactly the wave-4 entries the overhaul manifest lists as
`reads: 0, pub_map: UNMAPPED`.

Their evaluation battery receipts DO exist in `SZLHOLDINGS/frontier-54-runs`
(`benchmark/mass-battery-*.json`), and each repository mirrors its receipt at
`receipts/eval_battery.json` with a matching byte size. The cards in this
directory were written from those receipts and each repository's
`adapter_config.json` on 2026-11-14, following the overhaul rules: no metric
without a verified receipt binding; no invented license; failures disclosed.

## Staged cards and verified facts

| Repository | Base | Battery | Answerable | Abstained | Receipt sha256 (dataset copy) |
|---|---|---|---|---|---|
| willay-v2-trained | Qwen/Qwen3-1.7B | 3/6 | 3/3 | 0/3 | 046364235ff96edae6780213211784ef5f41605b3ec6c5d4143dc1051159801c |
| ouroboros-v2-trained | Qwen/Qwen3-1.7B | 3/6 | 3/3 | 0/3 | 2cf97ba5ac08ea178d920e345660798388a7e2e905a4f13f691ae1eeaab543a7 |
| khipu-r2-v2-trained | Qwen/Qwen2.5-1.5B | 3/6 | 3/3 | 0/3 | 762d07b2065ae2fa85cb82a88c0cf05e9aaf8907e56e241b3824623e06f4f9f7 |
| freeenergy-v2-trained | Qwen/Qwen3-1.7B | 3/6 | 3/3 | 0/3 | 8574d95c5caa1d01e67dfaa0376c7a0d8391a5ad595b3bf0b8d0ca4950c70f77 |
| chaski-v2-trained | Qwen/Qwen3-1.7B | 3/6 | 3/3 | 0/3 | 9136fae82354f09cddd126e8a85d198ac5a4f444dd3e7ef9fba8c46497894a38 |
| doctrine-v2-trained | Qwen/Qwen3-1.7B | 4/6 | 3/3 | 1/3 | 6d03d1049b3563731ab482f8b0b28f15bc3309bfbc4ad6b81660e120fedd3bb8 |
| brain-navigator-v2-trained | Qwen/Qwen3-1.7B | 3/6 | 3/3 | 0/3 | 3ae2dadc72b192cac0e4ec4a380edb2a7a634ca405b0dc6c0904139a865bda80 |
| chakana-v2-trained | Qwen/Qwen3-1.7B | 4/6 | 3/3 | 1/3 | 6b410c8f9f5375a5e9b4c7b2bd6a2e2d59818d6a99b60c0d32d0d52091a484de |

All receipts: kind `szl.estate-battery/mass-training`, node
`0056-kci3-sma6-prxmx200114` (NVIDIA RTX A6000), state `MEASURED`, measured
2026-11-13T22:20Z-23:01Z. Mirror byte sizes match the dataset receipts exactly
(3513/3663/3303/3680/3660/3671/3641/3365 bytes in table order). Adapters verified
per repository: rank 16, alpha 32, dropout 0.05, targets q/k/v/o_proj.

## Why these cards are staged here and not yet on the Hub

- The org requires PR-gated changes to repository main branches; the audit
  session's connector credential is read-only for the org (preupload:
  Authorization error), and the operator laptop could not reach
  huggingface.co directly at audit time (all direct calls hang).
- Direct-to-main writes are off-route by doctrine regardless.

## Recommended publication route

Extend the closed-profile card publisher (tools/publish_chaski_card.py /
publish-khipu-card.yml pattern) with these eight profiles, or publish through
the hf-sync window, then verify Hub bytes exactly as the existing publication
workflows do. Card text for each repository is in `<repo-name>.md` beside
this note, byte-ready for `README.md`.
