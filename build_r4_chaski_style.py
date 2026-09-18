import json, re
rows=[json.loads(l) for l in open('chaski_r4/train.jsonl',encoding='utf-8-sig') if l.strip()]
out=[]; swapped=0
for r in rows:
    out.append(r)
    for _ in range(2):
        msgs=[dict(m) for m in r['messages']]
        c0=msgs[0]['content']
        c1=re.sub(r'You are Chaski-R\d+, the SZL Holdings courier for SZLHOLDINGS/chaski-r\d+\.','You are Chaski, a proposal-only messenger of SZL Holdings. You draft. You refuse. You never execute.',c0)
        if c1!=c0: swapped+=1
        msgs[0]['content']=c1
        out.append({**r,'messages':msgs})
with open('chaski_r4/train.jsonl','w',encoding='utf-8',newline='\n') as f:
    f.write('\n'.join(json.dumps(x,ensure_ascii=False) for x in out)+'\n')
print('r4 rows:',len(rows),'| chaski-style added:',swapped,'| total:',len(out))
