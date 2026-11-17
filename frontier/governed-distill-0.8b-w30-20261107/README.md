# governed-distill-0.8b-w30-20261107

**Status: RESEARCH_CANDIDATE — NOT_PROMOTED — NOT_PUBLICATION_ELIGIBLE**

SZL-GovernedDistill-0.8B-v1: the first model in the estate trained only on
gate-accepted, copy-constrained teacher output. Lineage, training receipt,
and the measured eval FAIL (1/6 abstained) are recorded in `candidate.json`
and published verbatim:
- Hub model: SZLHOLDINGS/SZL-GovernedDistill-0.8B-v1 (honest card with the FAIL)
- Eval receipt: SZLHOLDINGS/model-eval-receipts/governed-distill-w30-student-eval-20261107.json

## Student result and claim boundary

On the six recorded free-generation probes, this student abstained on one
and answered five: the recorded verdict remains **FAIL**. That observation
does not establish a universal explanation of confabulation or prove that
grounding transfers to free generation. The candidate remains
**RESEARCH_CANDIDATE**, **NOT_PROMOTED**, **NOT_PUBLICATION_ELIGIBLE**, and
**NOT for biological validation**.

## Recovered teacher corpus acceptance

The earlier description of the 1.3% and 20.7% records as unavailable was
superseded by the public recovery record below. The recovered counts are
historical **corpus acceptance by the original teacher gate**, not
probe-response or abstention scores, student capability, or independent
replication. They do not change the student result or release status.

| Historical condition | Accepted / offered | Corpus acceptance | Pinned raw receipt |
|---|---:|---:|---|
| No-instruction baseline | 4 / 150 | 2.67% | [Baseline](https://huggingface.co/datasets/SZLHOLDINGS/frontier-54-runs/resolve/d5163816a9ad6dc58dfac462be11a4d203204347/receipts/distill-gate-receipt-2026-11-06T183445Z.json) |
| Prompt-rule condition, unconstrained decoding | 2 / 150 | 1.33% | [Prompt rule](https://huggingface.co/datasets/SZLHOLDINGS/frontier-54-runs/resolve/d5163816a9ad6dc58dfac462be11a4d203204347/receipts/distill-gate-receipt-2026-11-07T115228Z.json) |
| Copy-constrained pilot | 31 / 150 | 20.67% | [Pilot](https://huggingface.co/datasets/SZLHOLDINGS/frontier-54-runs/resolve/d5163816a9ad6dc58dfac462be11a4d203204347/receipts/distill-gate-receipt-2026-11-07T203231Z.json) |
| Copy-constrained extension | 46 / 210 | 21.90% | [Extension](https://huggingface.co/datasets/SZLHOLDINGS/frontier-54-runs/resolve/d5163816a9ad6dc58dfac462be11a4d203204347/receipts/distill-gate-receipt-2026-11-07T212651Z.json) |

The prompt-rule minus baseline difference is **-1.33 percentage points** of
corpus acceptance. The historical shorthand "1.3%" must not be presented as
a relative improvement or a capability gain. These are descriptive counts;
this correction makes no statistical-superiority claim.

The pilot and extension have the same prompt-index/verdict sequence for the
first 150 entries. The additional 60 entries contribute 15 acceptances. The
pilot and extension must not be pooled or counted as independent runs.

The [pinned condition documentation](https://github.com/szl-holdings/szl-forge/blob/3062802c9e7d24c0cb05b77652585d3c4154ae09/training/distill/conditions/README.md)
records the original `szl.distill.gate/v1` at
`ab0f1dd09adaf2196aedb550f8c2822cb42d39f6`. The raw receipts do not contain
condition labels; the mapping above comes from that documentation and the
receipt timestamps. In the archived prompt-rule and copy-constrained
scripts, the source reference answer was already in the teacher context;
under copy-constrained decoding it was in the copy-allowed token set.
Invocation arguments were not
logged, so the archived scripts' defaults are not proven run arguments.

## Correction provenance

- [Superseding recovery record](https://huggingface.co/datasets/SZLHOLDINGS/frontier-54-runs/resolve/d5163816a9ad6dc58dfac462be11a4d203204347/collaboration/vgl-protocol-1.3-percent-trace-recovery-20261114.json), dated 2026-11-14. Embedded receipt SHA-256: `18192b33d8ef63419b7a18c7812658154cd81ac841d8049add079446f545b985`.
- [Original trace receipt, retained as historical evidence](https://huggingface.co/datasets/SZLHOLDINGS/frontier-54-runs/resolve/3e5587156b1296d86f48c2fdb5facc24228747cb/collaboration/vgl-protocol-1.3-percent-trace.json). Embedded receipt SHA-256: `d636b4dcbd178eb6b9aeb1411846c3be64f9346416b9a5b64d4b01dce660f5f9`. Its claim that the underlying records were unavailable is superseded; the original receipt is unchanged.

Record recovery corrects provenance. It does not itself establish a fresh
replay, independent replication, generalization, biological validity,
protocol acceptance, or a frozen result.
