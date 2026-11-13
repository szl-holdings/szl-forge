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
property, confirmed from both directions (teacher: 1.3% prompt-assisted ->
21.9% copy-constrained; student: 1/6, same as khipu-r3's 1/6).

Corrections (2026-11-13): the teacher figures are corpus acceptance rates
(single runs) on the first 150 prompts of doctrine-corpus-v2: 4/150 with no
instruction, 2/150 (1.3%) with the rule stated in the prompt, 31/150 (20.7%)
copy-constrained. The 46/210 (21.9%) run is an extension of that pilot under
the same seed, not an independent replication (its first 150 verdicts are
identical; prompts 151-210 accepted 15/60). All rounds used the original
grounding gate at szl-forge ab0f1dd09adaf2196aedb550f8c2822cb42d39f6, not the
later revisions on main. The condition scripts are in
`training/distill/conditions/` (PR #724).

Student evaluation is measured separately: 1/6 abstained; the candidate
remains **RESEARCH_CANDIDATE**, **NOT_PROMOTED**,
**NOT_PUBLICATION_ELIGIBLE**, and **NOT for biological validation**.

The corrected measurement class and permanent hold are recorded in the pinned
trace receipt:
- [VGL trace receipt](https://huggingface.co/datasets/SZLHOLDINGS/frontier-54-runs/resolve/3e5587156b1296d86f48c2fdb5facc24228747cb/collaboration/vgl-protocol-1.3-percent-trace.json)
- Embedded receipt SHA-256: `d636b4dcbd178eb6b9aeb1411846c3be64f9346416b9a5b64d4b01dce660f5f9`
