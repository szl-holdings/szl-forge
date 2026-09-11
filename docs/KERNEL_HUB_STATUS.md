# Kernel Hub fleet — 2026-09-11 observation

State: `HOLD` (kernel-type live; leftover model-type mirrors not retired).

- 14 first-class kernel-type repositories on `SZLHOLDINGS`
- 14 leftover model-type mirrors, all SHA-divergent from the kernel-type head
- Hugging Face model-type compiled-kernel takedown starts **2026-09-13**
- Missing kernel-type `v1` branch: `szl-maskmod`, `szl-block-kv`, `szl-receipt-attn`, `YARQA-ATTN`
- Consumer contract: `kernels==0.16.1`, repo type `kernel`, exact revision (`v1` SHA preferred). Floating `main` and omitted revision are HOLD.
- `trust_remote_code=True` is a review flag, not a compatibility repair
- This change does **not** delete Hub mirrors, load weights, or mint production authorization

Run:

```bash
python tools/hf_kernel_hub_ops.py --output docs/KERNEL_HUB_FLEET_LIVE.json
python -m unittest tools/test_hf_kernel_hub_ops.py
```

Existing `tools/hf_kernel_migration_guard.py` remains the single-repo 0.16.1 / v1 load guard. This fleet operator covers the org-wide mirror clock the single-repo guard does not.
