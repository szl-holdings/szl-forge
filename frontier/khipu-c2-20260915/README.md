# Khipu C2 - Owner-Metal Abstention Candidate

**Status:** Historical owner-metal measurement retained; current qualification
**BLOCKED**. No production promotion or controller operating-point freeze.

## 2026-10-11 strict offline regrade

**MEASURED (offline regrade of preserved outputs, not fresh model inference):**
the historical L2 receipt reported **12/12 PASS**, but all **12/12** preserved
responses fail the current `frontier/harness/heldout_gate.py` `grade_abstain`
contract, so the current-contract count is **0/12**. Every response omits the
required `groundedOnly` field and has `candidates: []`, while the committed
`khipu/khipu.schema.json` requires that field and `candidates.minItems: 1`.
This check read only the committed response rows; it did not rerun the private
probes, load weights, or call the model. It does not establish a new held-out
model score or independent replay.

Exact inputs at Forge source `3b00912d3e10e45ce043da1dc2ab9376be48cb61`:

| Input | SHA-256 |
|---|---|
| `lane_l2_hidden_c2.json` (preserved historical receipt) | `0664e2ed8d512723dcb520bb64dc14603ea7f41862b8f6ec7b8b2c06ff62f03c` |
| `../harness/heldout_gate.py` (strict grader) | `74872aebabfc91c2265d9308355b66fac3480761594d4a1673969f678b0d5eb2` |
| `../../khipu/khipu.schema.json` (output schema) | `b95f9927366dae7c5d36cfb7de6e229eb605524318ab642a6aa2292a212170d0` |

The historical receipt remains unchanged, including its old
`publication_eligible: true` field. That field is a dated runner output, not
publication authority. Current qualification and publication remain **BLOCKED**
until a Khipu-specific mixed NAVIGATE + ABSTAIN gate produces a valid,
artifact-bound result. The q4_K_M derivative still needs its own post-quant
evaluation.

## Historical result (preserved, not current qualification)

- Hardware: betterwithage, NVIDIA RTX 5050 Laptop GPU
- Base family: Qwen2.5-1.5B / SZL-Khipu BrainNavigator
- Candidate: khipu-abstain-c2-20260912
- Hidden-set probe SHA-256: `6de93da59ba8574174044716f798ede56d7676b82100c0e59f2b695161337cf8`
- L2 result reported by the historical runner: 12/12 abstain cases passed
- Declared L2 baseline: 3 correct abstentions
- Outcome reported at the time: a numerical strict L2 beat under that runner;
  the strict current-contract regrade above rejects every preserved response

All 12 observed outputs emitted proposal-only Khipu JSON with decision ABSTAIN,
zero citations, empty candidate lists, and an abstainReason. The empty candidate
lists and omitted `groundedOnly` field mean these outputs do not satisfy the
current Khipu output contract.

## Boundary

This does not freeze a controller operating point and does not promote the candidate. The attempted L3 receipt is retained as fail-closed evidence: lane_l3.py is a Chaski JSON-draft/refusal runner and correctly rejects an abstain-only Khipu probe suite.

Remaining work:
1. Khipu-specific mixed NAVIGATE + ABSTAIN operating-point gate.
2. Explicit invented-identifier metric.
3. Separate q4_K_M post-quant evaluation against that Khipu gate.
4. Controller-contract freeze only after the correct lane accepts the complete evidence.
