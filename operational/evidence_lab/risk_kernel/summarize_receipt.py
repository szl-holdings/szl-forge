"""Print compact failure records from an SZL nonce-kernel receipt."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("receipt", type=Path)
args = parser.parse_args()
data = json.loads(args.receipt.read_text(encoding="utf-8"))
threshold = float(data["base_threshold"])
key = "corpus_anchored_nonce" if "corpus_anchored_nonce" in data["records"] else "synthetic_nonce"
rows = []
for record in data["records"][key]:
    response = record["response"]
    if record["split"] != "holdout" or response["status"] != "ANSWER" or float(response["margin"]) <= threshold:
        continue
    anchor = record.get("anchor")
    passage_ids = [passage["id"] for passage in response["passages"]]
    rows.append(
        {
            "id": record["id"],
            "family": record["family"],
            "margin": response["margin"],
            "answer": response["answer"],
            "anchor_document_id": anchor["document_id"] if anchor else None,
            "anchor_retrieved": anchor["document_id"] in passage_ids if anchor else None,
            "evidence_document_id": response["evidence"]["document_id"],
        }
    )
print(json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False))
