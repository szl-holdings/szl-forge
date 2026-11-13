# governed-distill-0.8b-w30-20261107

**Status: RESEARCH_CANDIDATE — NOT_PROMOTED — NOT_PUBLICATION_ELIGIBLE**

SZL-GovernedDistill-0.8B-v1: the first model in the estate trained only on
gate-accepted, copy-constrained teacher output. Lineage, training receipt,
and the measured eval FAIL (1/6 abstained) are recorded in `candidate.json`
and published verbatim:
- Hub model: SZLHOLDINGS/SZL-GovernedDistill-0.8B-v1 (honest card with the FAIL)
- Eval receipt: SZLHOLDINGS/model-eval-receipts/governed-distill-w30-student-eval-20261107.json

The scientific finding: training on grounded-by-construction output does not
transfer grounding to free generation. The confabulation floor is a decoding
property. The teacher-side **21.9% figure is corpus acceptance (46/210) through
the copy-constrained teacher gate, not a probe-response or abstention score**.
The earlier **1.3% prompt-assisted reference is unavailable for replay from an
accessible raw record and is excluded from quantitative claims under the
protocol**. Student evaluation is measured separately: 1/6 abstained; the
candidate remains **RESEARCH_CANDIDATE**, **NOT_PROMOTED**,
**NOT_PUBLICATION_ELIGIBLE**, and **NOT for biological validation**.

The corrected measurement class and permanent hold are recorded in the pinned
trace receipt:
- [VGL trace receipt](https://huggingface.co/datasets/SZLHOLDINGS/frontier-54-runs/resolve/3e5587156b1296d86f48c2fdb5facc24228747cb/collaboration/vgl-protocol-1.3-percent-trace.json)
- Embedded receipt SHA-256: `d636b4dcbd178eb6b9aeb1411846c3be64f9346416b9a5b64d4b01dce660f5f9`
