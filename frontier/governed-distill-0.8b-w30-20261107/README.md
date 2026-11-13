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
property. The teacher-side 21.9% figure is corpus acceptance (46/210), not a
probe-response score; the earlier 1.3% prompt-assisted reference is unavailable
for replay and is excluded under the protocol. The student result is 1/6, the
same as khipu-r3's 1/6.
