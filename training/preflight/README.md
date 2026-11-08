# Training Preflight (non-executing)

A preflight that cannot train, push, or write. It only READS and REJECTS.
A wave that has not passed preflight must not start.

Checks (all fail-closed, typed error codes):
1. PREFLIGHT_NON_LM_TARGET - the destination must not carry .npz fixture/kernel
   artifacts; a merged causal-LM checkpoint must never overwrite them
2. PREFLIGHT_TARGET_UNREADABLE / TARGET_MISMATCH - destination must resolve
   to exactly the requested identity
3. PREFLIGHT_MODEL_UNREADABLE / PIN_MISMATCH - student and teacher reads use
   the configured commit; the resolved sha must be that commit
4. PREFLIGHT_DATASET_UNREADABLE / NO_DATA / PIN_MISMATCH - the dataset read
   uses the configured commit, and that commit must contain data files
5. PREFLIGHT_METHOD_NOT_TRAINABLE - only qlora/lora16 run
6. PREFLIGHT_CONFIG_MISSING_FIELD - the wave config must be complete

The shared launch contract in `training/launch_contract.py` runs before any Hub
read. A rejected config does not need `HF_TOKEN` and does not open a Hub client.
Student, teacher, and dataset Hub reads pass the configured commit as
`revision`. A moved default branch is not accepted in place of that commit.
Run: python preflight.py --config path/to/wave.json
`training/w30/distill-r2.blocked.json` is rejected by that contract.
Unit tests use fake Hub clients - no network, no GPU, no writes.
