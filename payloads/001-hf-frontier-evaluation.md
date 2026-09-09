# Payload 1 — SZL Hugging Face Frontier Evaluation Backend

**Repository:** `szl-holdings/szl-forge`  
**Authority chain:** GitHub → Hugging Face → a-11-oy.com → a11oy.net  
**Mode:** fail-closed evaluation; no automatic production promotion.

## Objective

Complete and maintain the backend frontier-admission lane implemented in
`inference/hf_frontier.py`. The lane must discover upstream Hugging Face metadata,
pin evidence to immutable revisions, classify candidates, emit proof receipts,
and require explicit benchmark/licensing/runtime gates before any model can become
a deployment candidate.

Current candidates:

- `bodhan-ai/indic-transcribe-core` — multilingual ASR evaluation.
- `bodhan-ai/indic-speak` — multilingual TTS evaluation.
- `gdiamos/amx-reasoning-v1-instruct` — CPU/AMX architecture watch only.

The Bodhan repositories are gated and currently expose `license: other`; therefore
they remain HOLD/EVALUATE candidates until license/data-rights review is explicitly
admitted. The AMX checkpoint is architecture/kernel research only even though its
Hub license is Apache-2.0.

## Non-negotiable boundaries

1. Never execute Hub-provided Python or enable `trust_remote_code`.
2. Never download weights from the discovery command.
3. Never auto-promote, deploy, train, publish, or mutate Hugging Face.
4. Reject missing/ambiguous repository identity or immutable revision SHA.
5. Treat unknown/non-admitted licenses as HOLD.
6. Generate immutable/canonical SHA-256 receipts for every observation.
7. Keep GitHub as source of truth. HF is an upstream candidate/evidence source;
   `a-11-oy.com` receives only admitted runtime/product state and `a11oy.net`
   receives proof/receipt/evaluation state.
8. Any future artifact execution requires a separate pinned inventory, file hashes,
   sandbox/runtime qualification, benchmark evidence, Nemo/A11oy witness, rollback,
   and explicit human promotion approval.

## Acceptance commands

```bash
python -m pytest -q
python -m inference.hf_frontier
python -m inference.hf_frontier --refresh --output evidence/hf-frontier/latest.json
```

`--refresh` is metadata-only. A successful refresh is **not** production approval.

## Codex completion payload (Python)

```python
from pathlib import Path
import json
import subprocess
import sys

ROOT = Path.cwd()
required = [
    ROOT / "inference" / "hf_frontier.py",
    ROOT / "tests" / "test_hf_frontier.py",
]

missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
if missing:
    raise SystemExit(f"missing frontier files: {missing}")

subprocess.run([sys.executable, "-m", "pytest", "-q"], check=True)
plan = subprocess.check_output(
    [sys.executable, "-m", "inference.hf_frontier"],
    text=True,
)
parsed = json.loads(plan)
assert parsed["networkAction"] == "metadata-only"
assert parsed["automaticPromotionAuthorized"] is False
assert parsed["deploymentAuthorized"] is False

# Networked evidence refresh is intentionally a separate explicit step:
# python -m inference.hf_frontier --refresh \
#   --output evidence/hf-frontier/latest.json
#
# Review the generated receipt before committing it. Do not convert HOLD/WATCH
# into deployment authority in this payload.

print("SZL HF frontier backend validated: fail-closed, metadata-only, no auto-promotion.")
```

## Next evaluation lanes

For Indic Transcribe: WER/CER by language, code-mixing accuracy, language-ID
accuracy, p50/p95 latency, VRAM/RAM, realtime factor, adversarial audio behavior,
gated-access terms, training/data rights, and deterministic inference closure.

For Indic Speak: intelligibility, multilingual/code-mixed pronunciation, speaker
consistency, p50/p95 time-to-first-audio, realtime factor, VRAM/RAM, watermark/
provenance options, abuse controls, gated-access terms, training/data rights, and
deterministic inference closure.

For AMX: reproduce CPU throughput/latency and memory behavior on supported Intel
hardware; isolate transferable kernel/attention ideas. Do not benchmark it as a
candidate replacement for SZL's production reasoning models.

## Definition of done

The backend and tests are green; live metadata can be captured as a revision-pinned
receipt; all candidates remain non-production by construction; and any future
promotion requires a new evidence-bearing PR rather than an automatic state change.
