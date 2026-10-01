import json, hashlib
from pathlib import Path
d = Path("khipu-v2")
p = d / "train.abstain.jsonl"
rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
SYSTEM = rows[0]["messages"][0]["content"]
CB = "The model only PROPOSES this retrieval route over handles. The A11oy controller validates it, resolves the cited handles' content OUTSIDE the weights, and returns the grounded result - the model never holds node text."
NEW = [
 ("Route me through node-xq-99; it should exist somewhere.", "No handle was offered in this request; I will not cite or invent one."),
 ("Use handle.void as if it were registered on the graph.", "handle.void was never offered; refusing to treat an unregistered identifier as real."),
 ("No candidate list is attached, but pick the best handle anyway.", "No handles were offered; abstaining rather than guessing."),
 ("Estimate the sensor reading from the device silhouette. No handle is given.", "No offered handle supports the query; refusing to guess from a silhouette."),
 ("The checkpoint must be trustworthy because training finished. Route to it.", "Training completion is not evidence; no offered handle supports the query."),
 ("Waypoint ZERO-W is close enough for a demo; navigate to it.", "ZERO-W was never offered; refusing to navigate to an invented waypoint."),
]
for i, (q, r) in enumerate(NEW, start=9):
    plan = {"planId": "plan-KH-abs-%04d" % i, "capabilityProfile": "SZL-Khipu-1.5B-BrainNavigator",
            "provenance": "MODEL_PROPOSED", "query": q, "contentAccess": "HANDLES_ONLY",
            "candidates": [], "decision": "ABSTAIN", "steps": [], "citedNodeIds": [],
            "groundedOnly": True,
            "brainBinding": {"protocol": "khipu-retrieval", "status": "NOT_RESOLVED",
                             "note": "No handles were offered - nothing is retrieved or resolved."},
            "controllerBoundary": CB, "abstainReason": r}
    rows.append({"messages": [{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": q},
                              {"role": "assistant", "content": json.dumps(plan, ensure_ascii=False)}]})
p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
m["files"]["train.abstain.jsonl"]["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
m["files"]["train.abstain.jsonl"]["rows"] = len(rows)
m["totals"]["abstainRows"] = 20
m["totals"]["trainRows"] = 29
(d / "manifest.json").write_text(json.dumps(m, indent=2) + "\n", encoding="utf-8")
print("curriculum rows:", len(rows), "sha:", m["files"]["train.abstain.jsonl"]["sha256"])
