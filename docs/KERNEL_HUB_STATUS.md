# Kernel Hub fleet — 2026-09-11 observation

State: `HOLD` (kernel-type live; leftover model-type mirrors not retired).

Live unauthenticated `GET https://huggingface.co/api/kernels?author=SZLHOLDINGS` at 2026-09-11T23:07Z:

- kernels API list cardinality: **14**
- objects carrying tag `kernel`: **9**
- objects on that list without tag `kernel`: **5** (`szl-governed-norm`, `governed-inference-meter`, `szl-maskmod`, `szl-block-kv`, `szl-receipt-attn`)

Those two numbers are different predicates. Do not add them to membership 46/35/21 or inventory 44/30/48.

Prior same-day observation still records missing kernel-type `v1` branches on `szl-maskmod`, `szl-block-kv`, `szl-receipt-attn`, `YARQA-ATTN`. Branch presence was not re-verified in the 23:07Z tag/list sweep; that field stays HOLD.

- Hugging Face model-type compiled-kernel takedown starts **2026-09-13**
- Consumer contract: `kernels==0.16.1`, repo type `kernel`, exact revision (`v1` SHA preferred). Floating `main` and omitted revision are HOLD.
- `trust_remote_code=True` is a review flag, not a compatibility repair
- This change does **not** delete Hub mirrors, load weights, or mint production authorization

Run:

```bash
python tools/hf_kernel_hub_ops.py --output docs/KERNEL_HUB_FLEET_LIVE.json
python -m unittest tools/test_hf_kernel_hub_ops.py
```

Existing `tools/hf_kernel_migration_guard.py` remains the single-repo 0.16.1 / v1 load guard. This fleet operator covers the org-wide mirror clock the single-repo guard does not.
