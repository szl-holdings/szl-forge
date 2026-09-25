import json, datetime
put = {
    "schema": "szl.a11oy-mini-hub-put/v1",
    "decision": "UPLOAD",
    "authorized_by": "owner (Steph) - execution of this block is the authorization record",
    "destination": "SZLHOLDINGS/A11OY-MINI",
    "files": ["a11oy-mini-r2-Q4_K_M.gguf", "a11oy-mini-r2-BF16-mmproj.gguf", "README.md"],
    "legacy_files": "a11oy-mini-f16.gguf, a11oy-mini-q4_k_m.gguf marked DEPRECATED in card; NOT deleted",
    "basis": "Rebuilt from chaski-r2 (named-N winner, adapter MEASURED 5/5 + 6/6). GGUF gate MEASURED on this artifact (gguf_gate_receipt.json).",
    "claim_boundary": "This PUT covers the A11OY-MINI GGUF SKU only. The chaski/r2/5050/r4 adapters, their receipts, and bakeoff publication_eligible=false remain unchanged and sealed.",
    "decided_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}
json.dump(put, open("chaski_r2/a11oy_mini_r2_gguf/hub_put_receipt.json", "w", encoding="utf-8"), indent=2)
print("hub-put receipt written")
