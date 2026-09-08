# NeoMME retrieval smoke lane

Hcompany owns NeoMME; SZL owns this original evaluation harness. This is the
first executable probe for the NeoMME candidate already registered in Frontier
and Second Brain. It is NOT a new flagship, weight mirror, or production route.

## Scope

`neomme_probe.py` defaults to a dependency-free, no-network plan. Only explicit
`--execute` downloads the exact public checkpoint, verifies every allowed
metadata Git blob plus the safetensors SHA-256, loads it without remote Python
code, and runs inference on CPU. No dataset is downloaded. Six original synthetic
documents are evaluated as text and as 384x512 RGB page images against eight
queries. Dense cosine and masked MeanMaxSim are measured separately.

The direct Transformers class is required: the upstream integration intentionally
does not map `AutoModel` to the retrieval head. The runtime source pin includes
the upstream NeoMME kernel-decorator import repair. It is a smoke-test runtime
pin, NOT an admitted production image or a full supply-chain audit.

Primary references:
- https://huggingface.co/Hcompany/NeoMME-260M-Retriever
- https://github.com/huggingface/transformers/pull/47992
- https://github.com/huggingface/transformers/pull/48457

## Reproduce in an isolated CPU environment

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install 'transformers @ git+https://github.com/huggingface/transformers.git@cdfdcad31314fe4f23b40ab374e860a62403f72a' Pillow==11.3.0
python tests/test_neomme_probe.py
python experiments/neomme_probe.py
HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_TELEMETRY=1 \
  python experiments/neomme_probe.py --execute --output neomme-smoke.json
```

Use a 15-minute or shorter externally enforced job timeout. Artifact budget is
650,000,000 bytes and each encoded input is bounded to 1,536 tokens. No token,
write-scoped credential, or production endpoint is needed. Dependency versions
and the exact harness hash are included in a completed report; transitive
packages are recorded, not yet distributed as a hash-locked runtime image.

## Evidence and interpretation

`SMOKE_EXECUTED` means real inference completed. It does not mean the model won a
benchmark. Inspect all four metric groups and per-case rankings; do not drop
misses. A single expected document per query defines MRR, Recall@3 and nDCG@3.
The fixtures are easy authored smoke cases, not an independent held-out split.
Image pixel hashes bind the rendered inputs, while a separate fixture hash binds
the text and qrels. Item timings include processing plus forward execution. Peak
RSS is Linux process high-water memory, including imports and dependencies.

The report digest covers canonical JSON before its own `receipt_sha256` field is
added. `UNSIGNED_HONEST` is an integrity record, not a signature. A failed run
atomically replaces an old successful report when `--output` is given.

## Existing ecosystem boundaries

Forge owns model loading and qualification. Second Brain can consume the results
as review-required retrieval evidence; it does not gain a new authorized index
from these smoke results. Frontier publishes the observed status through its
existing Space and covenant dataset only after source review. Nemo and A11oy
retain their existing envelope and action authorization boundaries.

Production remains HOLD. Before a live retrieval rollout: add a frozen independent
SZL qrels split, baseline comparison, tenant/rights admission, content-bound
retrieval handles, fallback/rollback tests, an admitted runtime image, and the
existing explicit release authorization. No existing model default or public
inference Space is modified by this experiment.
