import json
rows=[json.loads(l) for l in open('chaski_r4/train.pre-fix-backup.jsonl',encoding='utf-8-sig') if l.strip()]
SYS='You are Chaski, the SZL Holdings courier. Emit one JSON object. decision must be DRAFT. approvalRequired must be true. executed must be false. You draft. You never execute.'
out=[]
for r in rows:
    msgs=[dict(m) for m in r['messages']]
    msgs[0]['content']=SYS
    for m in msgs:
        if m['role']=='assistant' and not m['content'].startswith(('REFUSE:','ABSTAIN:')):
            d=json.loads(m['content'])
            if d.get('label')!='DECLARED':
                d['label']='DECLARED'
                m['content']=json.dumps(d,separators=(',',':'))
    out.append({**r,'messages':msgs})
assert len(out)==40
with open('chaski_r4/train.jsonl','w',encoding='utf-8',newline='\n') as f:
    f.write('\n'.join(json.dumps(x,ensure_ascii=False) for x in out)+'\n')
print('rows:',len(out),'| uniform non-enumerated system applied')
