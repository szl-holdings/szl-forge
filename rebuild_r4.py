import json, re
rows=[json.loads(l) for l in open('chaski_r4/train.pre-fix-backup.jsonl',encoding='utf-8-sig') if l.strip()]
for r in rows:
    for m in r['messages']:
        if m['role']=='assistant' and not m['content'].startswith(('REFUSE:','ABSTAIN:')):
            d=json.loads(m['content'])
            if d.get('label')!='DECLARED':
                d['label']='DECLARED'
                m['content']=json.dumps(d,separators=(',',':'))
out=[]; swapped=0
for r in rows:
    out.append(r)
    for _ in range(2):
        msgs=[dict(m) for m in r['messages']]
        c0=msgs[0]['content']
        c1=re.sub(r'^You are [^.]*\.', 'You are Chaski, a proposal-only messenger of SZL Holdings. You draft. You refuse. You never execute.', c0, count=1)
        if c1!=c0: swapped+=1
        msgs[0]['content']=c1
        out.append({**r,'messages':msgs})
with open('chaski_r4/train.jsonl','w',encoding='utf-8',newline='\n') as f:
    f.write('\n'.join(json.dumps(x,ensure_ascii=False) for x in out)+'\n')
print('base rows:',len(rows),'| chaski-style swapped:',swapped,'| total:',len(out))
print('sample system:', out[1]['messages'][0]['content'][:220])
