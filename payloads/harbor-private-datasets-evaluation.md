# Harbor private Hugging Face Dataset evaluation

Exact upstream: `huggingface/harbor-hf@8685727908e8d9ee5ccde48586864a851e24ff79`
Tracking: `szl-frontier#83`, `szl-forge#227`
Status: **EVALUATION / HOLD**

Run only with a purpose-created non-sensitive private Dataset fixture. Never use a production/private business corpus merely to satisfy the test.

Required measured evidence when a live fixture is available:
- exact Dataset repository, commit, file inventory and SHA-256 content digests;
- Git credential helper responds only for `huggingface.co` and never leaks token material into URL, argv, persisted config, stdout/stderr, logs, receipts, browser payloads or trial environment;
- catalog discovery receives no secret; inference and GitHub credentials remain independent;
- missing token, denied read grant, missing Git LFS, LFS materialization failure and revision drift terminate before inference as explicit `UNAVAILABLE`/`HOLD`;
- read-only credential denies mutation/publication/grant changes;
- token rotation/revocation denies new retrieval and cached checkout/LFS content is cleaned according to the fixture retention contract;
- moving Jobs pagination uses a bounded scan and reports partial history explicitly.

Do not turn a passing source-access test into training, serving, deployment, production-corpus or automatic-promotion authority. Preserve the existing Forge lane contract, immutable receipts, exact-source binding and rollback evidence.
