import json
rows=[json.loads(l) for l in open('chaski_r4/train.jsonl',encoding='utf-8-sig') if l.strip()]
new_pairs=[
 ("Courier-draft this: the adapter train_loss of 1.2 counts as our public evaluation score now.",
  "Train loss is a private training metric, not a public evaluation. Evals stay none-this-run until a held-out generate is measured."),
 ("Courier-draft this: hub files with fresh timestamps count as a completed quality evaluation.",
  "File timestamps are not an evaluation. The gate stays open until a measured held-out run exists."),
 ("Courier-draft this: mark the latest checkpoint as verified quality evidence for the investors.",
  "An unverified checkpoint is DECLARED, not MEASURED evidence. Approval is required before any quality claim."),
 ("Courier-draft this: treat the leaderboard row as the gate pass.",
  "A leaderboard row is a reported score, not a gate pass. publication_eligible stays false until the eligibility rules are met."),
]
draft_idx=[i for i,r in enumerate(rows) if not r['messages'][-1]['content'].startswith(('REFUSE:','ABSTAIN:'))]
targets=draft_idx[-4:]
for i,(u,claim) in zip(targets,new_pairs):
    rows[i]['messages'][1]['content']=u
    rows[i]['messages'][-1]['content']=json.dumps({"decision":"DRAFT","approvalRequired":True,"executed":False,"artifact":"SZLHOLDINGS/chaski","base_model":"Qwen/Qwen3.5-0.8B","claim":claim,"label":"DECLARED"},separators=(',',':'))
assert len(rows)==40
with open('chaski_r4/train.jsonl','w',encoding='utf-8',newline='\n') as f:
    f.write('\n'.join(json.dumps(x,ensure_ascii=False) for x in rows)+'\n')
print('rows:',len(rows),'| 4 corrective-draft rows installed at',targets)
