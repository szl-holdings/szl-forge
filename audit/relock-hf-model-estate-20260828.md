# Hugging Face model and kernel estate current-main relock

- Requested: 2026-08-28
- Source authority: the exact protected `main` produced by protected squash merge of this PR
- Canonical targets: SZL Forge model, adapter, GGUF, kernel, and inference-lab publication lanes

This audit-only receipt creates one protected-main promotion edge after the current GitHub convergence. It changes no model weights, training data, kernel source, workflow, secret, ruleset, provider configuration, or runtime claim.

Completion requires the repository-native publication workflows to bind every published artifact to the exact merged source, validate cards/lineage/digests and declared autonomy boundaries, and independently read back the immutable Hugging Face revisions. A merge alone is not a publication-success claim.
