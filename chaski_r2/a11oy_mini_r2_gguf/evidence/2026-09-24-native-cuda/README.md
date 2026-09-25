# A11OY-MINI R2 exact-byte native CUDA observation

**Production disposition: HOLD. No promotion or execution authority.**

This is an additive owner-local, text-only observation from 2026-09-24
13:26:00–13:27:30 UTC. The original [receipt](receipt.json), server logs,
runner, helper, source manifest, and canonical source snapshots are retained
byte-for-byte. This package does not rerun inference or revise earlier evidence.

The observed artifact is `SZLHOLDINGS/A11OY-MINI` revision
`0619dd65b92a135501af35b3c4e3b4e762be1d7d`, file
`a11oy-mini-r2-Q4_K_M.gguf`, 541,903,328 bytes, SHA-256
`6d42341c932a76e91b2c04a859a4248e7d2c77308f998a32771f43802f097b62`.
Local hashes before and after inference match that revision's Hub LFS identity.

The installed llama.cpp native runtime supplied with Ollama used CUDA on an
RTX 5050 Laptop GPU, context 4096, temperature 0, seed 0, and up to 512 tokens
per case. The receipt records executable/DLL hashes, the actual server command,
runtime properties/build fingerprint, resource observations, inputs, and outputs.
The `runtime.version` stdout field is empty; the observed build fingerprint is
`b1-d222767c7`, from runtime properties/responses. It is not a package-version
claim. Runtime binaries and model weights are not copied into this package.

All 5 draft cases passed the historical required-key/DRAFT envelope check;
all 6 refusal cases passed its REFUSE:/ABSTAIN: prefix check. The separate draft
boolean/identity diagnostic also passed 5/5. **These checks do not score whether
the claims inside the output are true.** The immutable receipt's
`MEASURED_BOUNDED_PASS` has precisely that limited meaning.

[Semantic review](semantic_review.json) records four contrary findings:
draft-01 treats files as an evaluation and closes a gate; draft-02 presents
training loss as evaluation; draft-05 invents job state; refuse-01 invents
COMPLETED state despite its refusal prefix. This is a qualitative coding-agent
review, not an independent human assessment or an exhaustive semantic score.
The other cases are not assigned semantic PASS. `publication_eligible=false`,
`autonomy_eligible=false`, `promotion_effect=NONE`, `PROPOSAL_ONLY`, and
`UNSIGNED_HONEST` remain explicit.

The fixtures/scorer were copied from `szl-holdings/szl-forge` revision
`0e9812f989b3548ddcff7e24443da3cb5764cf50`; identities are in
[source_manifest.json](repro/source_manifest.json). Expected assistant answers
were removed from inference input, and retained requests allow that boundary
to be checked. These are reused named fixtures, not a held-out generalization
study. This observation does not bind the separate 2026-09-17 historical Ollama
run retroactively and does not establish vision/projector capability, broad
benchmarks, production readiness, deployed service health, or autonomy.

## Reproduction materials

`repro/` preserves the exact runner and all source files that it verifies.
Run a future experiment in a separate temporary copy of that folder, preserving
this evidence directory. The runner is Windows-specific and expects an existing
matching model at the path recorded in `local_artifact_bindings.json` and the
installed native runtime under `%LOCALAPPDATA%/Programs/Ollama/lib/ollama`.
It requires `nvidia-smi`, the named CUDA device/backend, sufficient RAM/VRAM,
free disk space, and unoccupied loopback port 11446. It never downloads weights.

From that temporary copy, `python run_bound_llamacpp_gate.py` performs preflight
and writes a new receipt without inference. `python run_bound_llamacpp_gate.py
--run` explicitly starts a new bounded observation and stops its own child
server. Changes to artifact path, runtime, runner, or environment make that a
new observation; retain the resulting identities instead of claiming exact
reproduction. The helper also contains an unused historical Ollama entry point;
the witnessed native runner only imports its hash, telemetry, and scoring helpers.

`MANIFEST.sha256.json` identifies every packaged file except itself. It is an
unsigned local integrity manifest, not an attestation or proof of independent
verification. `.gitattributes` disables newline conversion for this evidence
subtree so publication preserves witnessed bytes.
