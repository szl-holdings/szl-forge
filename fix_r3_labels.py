import json, collections
rows=[json.loads(l) for l in open('chaski_r3/train.jsonl',encoding='utf-8-sig') if l.strip()]
c=collections.Counter()
for r in rows:
    for m in r['messages']:
        if m['role']=='assistant' and not m['content'].startswith(('REFUSE:','ABSTAIN:')):
            c[json.loads(m['content']).get('label')]+=1
print('BEFORE census:', dict(c), '| rows:', len(rows))
fixed=0
for r in rows:
    for m in r['messages']:
        if m['role']=='assistant' and not m['content'].startswith(('REFUSE:','ABSTAIN:')):
            d=json.loads(m['content'])
            if d.get('label')!='DECLARED':
                d['label']='DECLARED'
                m['content']=json.dumps(d,separators=(',',':'))
                fixed+=1
with open('chaski_r3/train.jsonl','w',encoding='utf-8',newline='\n') as f:
    f.write('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows)+'\n')
print('labels normalized to DECLARED:', fixed)
