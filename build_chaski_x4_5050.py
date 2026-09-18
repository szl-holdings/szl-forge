import json
rows=[json.loads(l) for l in open('chaski_r3/train.jsonl',encoding='utf-8-sig') if l.strip()]
out=[]
swapped=0
for r in rows:
    out.append(r)
    for _ in range(4):
        msgs=[dict(m) for m in r['messages']]
        c0=msgs[0]['content']
        c1=c0.replace('You are Chaski-R2, the SZL Holdings courier for SZLHOLDINGS/chaski-r2.','You are Chaski, a proposal-only messenger of SZL Holdings. You draft. You refuse. You never execute.')
        if c1!=c0:
            swapped+=1
        msgs[0]['content']=c1
        out.append({**r,'messages':msgs})
with open('szl_dataset_5050_chaski_x4.jsonl','w',encoding='utf-8',newline='\n') as f:
    f.write('\n'.join(json.dumps(x,ensure_ascii=False) for x in out)+'\n')
print('courier rows:',len(rows),'chaski-style added:',swapped,'total:',len(out))
