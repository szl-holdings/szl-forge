# SZL ReceiptAgent Qwen3.5 0.8B v2 candidate

This directory is a **qualification and evidence lane**, not by itself a model
release.

It pins the Apache-2.0 Qwen/Qwen3.5-0.8B lineage and the exact Unsloth
implementation revision used on the RTX 5050 laptop. The first gate performs a
real CUDA tensor operation, loads the 0.8B checkpoint in 4-bit mode, and runs a
bounded local generation. Its report remains outside Git because it is
host-specific execution evidence.

Run from Linux or WSL with a CUDA-enabled Unsloth environment:

```bash
python qualify_runtime.py \
  --report /path/outside/repository/qwen35-runtime-qualification.json
```

Passing this gate proves only that the pinned model loaded and generated on the
observed GPU. It does not prove that an SZL fine-tune exists. No repository,
model card, adapter, or public portfolio entry may be created until training,
5/5 held-out schema conformance, 6/6 adversarial refusal, and the signed
training/evaluation receipt chain all pass.

After the runtime gate passes, run a single real optimizer step as the training
smoke:

```bash
python train_candidate.py \
  --max-steps 1 \
  --output-dir /path/outside/repository/qwen35-training-smoke
```

Only after that succeeds may the bounded full run use the declared 64-step
configuration (four curriculum passes at effective batch size two). The older
700-step proposal is rejected because it would overfit this 31-row admitted
curriculum. Both runs save the adapter and a content-addressed training report
outside Git; neither creates a public model automatically.

Evaluate the exact saved adapter against the committed 5-case contract set and
6-case adversarial refusal set:

```bash
python evaluate_candidate.py \
  --adapter-dir /path/outside/repository/qwen35-full/adapter \
  --training-report /path/outside/repository/qwen35-full/training-report.json \
  --report /path/outside/repository/qwen35-full/eval-report.json
```

The evaluator reloads the pinned base revision, verifies the adapter and
training-report digests, and records raw integer counts plus per-case output
hashes. A passing evaluation is still not publication approval: owner-signed
training/eval receipts and a post-publication Hub readback remain mandatory.

The bounded run completed 64 optimizer steps over 31 admitted training rows on
an NVIDIA RTX 5050 Laptop GPU. The exact adapter then passed 5/5 held-out JSON
contract cases and 6/6 adversarial refusal cases. These are **MEASURED** raw
counts, not a general capability benchmark and not evidence of autonomy.

After committing the qualification source, mint the two-receipt chain against
that exact source commit:

```bash
python evidence_chain.py mint \
  --training-report /path/outside/repository/training-report.json \
  --evaluation-report /path/outside/repository/eval-report.json \
  --source-commit 0123456789abcdef0123456789abcdef01234567
python evidence_chain.py verify
```

`mint` refuses mismatched report self-digests, adapter hashes, dataset hashes,
acceptance counts, candidate identity, or source commits. It signs the
training evidence first and binds the evaluation receipt to the training
receipt's canonical SHA-256. The private Ed25519 key remains outside Git.
