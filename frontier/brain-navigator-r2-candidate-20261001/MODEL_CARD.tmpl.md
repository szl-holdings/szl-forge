---
license: {{candidate:actual_training_base.license}}
base_model: {{candidate:actual_training_base.repo_id}}
tags:
- szl-holdings
- frontier-candidate
- experimental
---

# {{candidate:candidate_id}}

**Status: EXPERIMENTAL CANDIDATE - {{evaluation_report:promotion}}.** Release gate {{gate_results:release_gate}} ({{gate_results:gate_verdict}}).
Failing or unavailable gates: {{FAILING_GATES}}.

Every number below is read from a report in `out/`; none is typed by hand (`render_card.py`).

## Lineage

- Base: `{{candidate:actual_training_base.repo_id}}` @ `{{candidate:actual_training_base.revision}}` (license {{candidate:actual_training_base.license}})
- Predecessor (frozen comparator, not weight initialization): `{{candidate:predecessor.repo_id}}`
- Target repository: `{{candidate:target_repo_id}}` - the predecessor is never overwritten

## Training (report state: {{training_report:state}}, unsigned until the owner signs)

- Optimizer steps: {{training_report:optimizer_steps}} | seed {{training_report:seed}} | backend {{training_report:backend}}
- Final train loss (reported verbatim as a string): {{training_report:finalTrainLoss}}
- Adapter SHA-256: `{{training_report:adapterSha256}}`
- Curriculum: {{training_report:train_rows}} train rows; leakage verdict {{leakage_receipt:verdict}}
- Rights basis: {{candidate:training_data.rights}} ({{candidate:training_data.binding_status}})

## Evaluation (split {{evaluation_report:split}}, {{evaluation_report:rows}} rows, greedy, text-only tokenizer path)

| Metric | Value |
|---|---|
| exact match | {{evaluation_report:metrics.exact_match}} |
| schema validity | {{evaluation_report:metrics.schema_validity}} |
| true abstain | {{evaluation_report:metrics.true_abstain}} |
| false abstain | {{evaluation_report:metrics.false_abstain}} |
| evidence-handle validity | {{evaluation_report:metrics.evidence_handle_validity}} |
| ECE | {{evaluation_report:metrics.ece}} |

Claim scope: {{candidate:evaluation_protocol.claim_scope}}

## Quantization

Manifest status: {{QUANT_MANIFEST:status}} | parity verdict: {{parity_receipt:verdict}}

## Limitations

- Proposal-only artifact. Approval, execution and authoritative receipt minting stay outside the weights.
- Author-run metrics on a project-authored suite; not a blind benchmark or independent certification.
- Promotion requires owner-signed training and evaluation envelopes, immutable Hub byte readback and an independent inference check.
