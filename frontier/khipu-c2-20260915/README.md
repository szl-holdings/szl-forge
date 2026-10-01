# Khipu C2 - Owner-Metal Abstention Candidate

**Status:** MEASURED candidate; not production-promoted.

## Result

- Hardware: betterwithage, NVIDIA RTX 5050 Laptop GPU
- Base family: Qwen2.5-1.5B / SZL-Khipu BrainNavigator
- Candidate: khipu-abstain-c2-20260912
- Hidden-set probe SHA-256: `6de93da59ba8574174044716f798ede56d7676b82100c0e59f2b695161337cf8`
- L2 result: 12/12 abstain cases passed
- Declared L2 baseline: 3 correct abstentions
- Outcome: strict L2 beat; evidence is owner-metal measured

All 12 observed outputs emitted proposal-only Khipu JSON with decision ABSTAIN, zero citations, empty candidate lists, and an abstainReason. The candidate avoided invented node identifiers on the evaluated abstain cases.

## Boundary

This does not freeze a controller operating point and does not promote the candidate. The attempted L3 receipt is retained as fail-closed evidence: lane_l3.py is a Chaski JSON-draft/refusal runner and correctly rejects an abstain-only Khipu probe suite.

Remaining work:
1. Khipu-specific mixed NAVIGATE + ABSTAIN operating-point gate.
2. Explicit invented-identifier metric.
3. Separate q4_K_M post-quant evaluation against that Khipu gate.
4. Controller-contract freeze only after the correct lane accepts the complete evidence.
