# RBCG v1 — receipt-bound conformal guarantee, version 1 (prototype)

IN-SILICO RESEARCH ONLY · SYNTHETIC DATA ONLY · NOT FOR CLINICAL USE · Python 3.10+ standard library only.

RBCG v1 certifies a selective-risk threshold with Learn-then-Test using **scene-level betting e-values**
(the exchangeable unit is the scene, not the row), signs the guarantee as a DSSE (ECDSA P-256 / RFC 6979)
receipt, monitors deployment with **two conformal test martingales** (pre-registered primary: per-scene
OOD count; secondary: the v0 score), audits labels with an **anytime-valid betting confidence sequence**,
anchors receipts in an **RFC 9162-style Merkle transparency log**, and verifies everything offline with a
fail-closed verifier. See METHOD_v1.md for assumptions, validity sketches and verified references.

| file | role |
|---|---|
| `rbcg1.py` | scene-level loss, betting and HB p-values, fixed-sequence LTT, two-CTM monitor, betting-CS audit, canonical JSON, DSSE PAE, ECDSA P-256 + RFC 6979, receipts |
| `rbcg1_log.py` | Merkle log (RFC 9162 §2.1): MTH, inclusion and consistency proofs and their verification; signed tree heads; anchor bundles |
| `rbcg1_verify.py` | fail-closed verifier: VALID / VIOLATED / REFUSED:<CODE> incl. NOT_INCLUDED and INCONSISTENT_STH (also a CLI) |
| `rbcg1_protocol.py` | sign / verify the pre-registration protocol receipt (binds code hashes) |
| `rbcg1_triagex.py` | adapter to the TRIAGE-X stack (frozen advisory; refuses to touch the spent frozen splits) |
| `rbcg1_experiments.py` | design pilot, E1/E2 shards (v1 and v0 side by side on the same fresh data), E3 re-verification, merge |
| `v0_ref/` | the exact v0 `rbcg.py` and `rbcg_verify.py` (sha pinned) used as the comparator |
| `run_tests.py`, `tests/` | unittest suite; skips count as failures; fixtures include the RFC 6979/DSSE KATs and the pinned transparency-dev Merkle vectors |
| `mutate_rbcg1.py` | 50 hand-written mutants; refuses to score unless the clean suite is green and a no-op mutant survives |
| `tools/build_merkle_vectors.py` | rebuilds the Merkle fixture from the pinned upstream tarball (data only) |
| `protocol/` | the signed protocol receipt and its public key |
| `examples/log_sth/` | a real v1 guarantee, monitor chains, rows, labels and log anchors (STHs + proofs), verifiable offline |
| `examples/e3/` | the stored v0 E3 bundle that v1 re-verifies |
| `reports/` | VM logs, suite logs for four Python versions, mutation report |

Quick check (from this directory):

    python3 -B run_tests.py
    python3 -B mutate_rbcg1.py mutation_report.json
    python3 -B rbcg1_protocol.py verify --protocol protocol/protocol.dsse.json
    python3 -B rbcg1_verify.py --guarantee examples/log_sth/guarantee.dsse.json --pubkey examples/log_sth/public_key.pem \
        --cal-rows examples/log_sth/cal_rows.jsonl --monitor-chain examples/log_sth/in_dist/monitor_chain.jsonl \
        --deploy-rows examples/log_sth/in_dist/deploy_rows.jsonl --labels examples/log_sth/in_dist/labels.json \
        --anchor examples/log_sth/in_dist/anchor.json --log-pubkey examples/log_sth/log_public_key.pem \
        --policy examples/log_sth/policy.json

Pure-Python ECDSA is not constant-time: throwaway test keys only. Licence: Apache-2.0 (RBCG-LIC-1, 2026-11-12). See LICENSE and NOTICE.
