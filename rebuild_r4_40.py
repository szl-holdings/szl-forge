import json, re
rows=[json.loads(l) for l in open('chaski_r4/train.pre-fix-backup.jsonl',encoding='utf-8-sig') if l.strip()]
out=[]; swapped=0
for i, r in enumerate(rows):
    for m in r['messages']:
        if m['role']=='assistant' and not m['content'].startswith(('REFUSE:','ABSTAIN:')):
            d=json.loads(m['content'])
            if d.get('label')!='DECLARED':
                d['label']='DECLARED'
                m['content']=json.dumps(d,separators=(',',':'))
    if i % 2 == 1:
        msgs=[dict(m) for m in r['messages']]
        c0=msgs[0]['content']
        c1=re.sub(r'^You are [^.]*\.', 'You are Chaski, a proposal-only messenger of SZL Holdings. You draft. You refuse. You never execute.', c0, count=1)
        if c1!=c0: swapped+=1
        msgs[0]['content']=c1
        r={**r,'messages':msgs}
    out.append(r)
assert len(out)==40, 'row count drifted: %d' % len(out)
with open('chaski_r4/train.jsonl','w',encoding='utf-8',newline='\n') as f:
    f.write('\n'.join(json.dumps(x,ensure_ascii=False) for x in out)+'\n')
print('rows:',len(out),'| chaski-style swapped:',swapped)
print('sample original-style:', out[0]['messages'][0]['content'][:150])
print('sample chaski-style:  ', out[1]['messages'][0]['content'][:150])
