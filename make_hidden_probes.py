import json, hashlib
from pathlib import Path
bench = json.loads(Path.home().joinpath("szl-khipu", "frontier", "khipu_abstention_bench.json").read_text(encoding="utf-8"))
rows = [{"id": g["id"], "kind": "abstain", "prompt": g["prompt"]}
        for g in bench["gold"] if g["expect"] == "ABSTAIN"]
Path("receipts").mkdir(exist_ok=True)
Path("receipts/khipu_hidden_abstain_probes.jsonl").write_text(
    "\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n", encoding="utf-8")
canonical = "\n".join(json.dumps(p, sort_keys=True) for p in rows)
print("PROBES", len(rows))
print("SHA", hashlib.sha256(canonical.encode("utf-8")).hexdigest())
