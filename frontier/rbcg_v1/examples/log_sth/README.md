# Log/STH example (rep 0 of the pre-registered run, synthetic only)

A real v1 guarantee, its calibration rows, two monitor chains (in-distribution and shifted), labels,
and for each chain an anchor bundle: the issuance STH with the guarantee's inclusion proof, the final
STH with the chain head's inclusion proof, and the issuance-to-final consistency proof. STHs are DSSE
envelopes signed with the log key (log_public_key.pem); receipts are signed with public_key.pem.
Throwaway keys; private keys were never written.

    python3 -B ../../rbcg1_verify.py --guarantee guarantee.dsse.json --pubkey public_key.pem \
        --cal-rows cal_rows.jsonl --monitor-chain in_dist/monitor_chain.jsonl \
        --deploy-rows in_dist/deploy_rows.jsonl --labels in_dist/labels.json \
        --anchor in_dist/anchor.json --log-pubkey log_public_key.pem --policy policy.json

Replace in_dist/ with shifted/ for the shifted stream. Each directory holds the verifier output that
the run recorded (verify_output.json).
