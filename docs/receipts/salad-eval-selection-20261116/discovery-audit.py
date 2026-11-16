import pathlib,json,hashlib,unicodedata,re,collections,datetime,os
from huggingface_hub import hf_hub_download,get_token
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING']='1'
root=pathlib.Path(r'C:\SZL\edge-debug')
repo='SZLHOLDINGS/szl-frontier-sft-mix-v4';revision='12a26b44303299624592084c025e344d260aa7bf'
manifest=json.loads((root/'diagnostic_frontier_20261116_larger_manifest.json').read_text(encoding='utf-8'))
def norm(s):return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',s).casefold()).strip()
def digest(obj):return hashlib.sha256(json.dumps(obj,ensure_ascii=False,separators=(',',':'),sort_keys=True).encode('utf-8')).hexdigest()
def view(row):
 if isinstance(row.get('messages'),list):
  ms=row['messages']
  if not ms or any(not isinstance(m,dict) or not isinstance(m.get('role'),str) or not isinstance(m.get('content'),str) for m in ms):return None
  if ms[-1]['role']!='assistant' or not ms[-1]['content'].strip():return None
  context=[{'role':m['role'],'content':norm(m['content'])} for m in ms[:-1]]
  users=[norm(m['content']) for m in ms if m['role']=='user']
  allmsgs=[{'role':m['role'],'content':norm(m['content'])} for m in ms]
  if not users or not all(users):return None
  return context,users,allmsgs
 for p,a in [('prompt','completion'),('question','ideal_response'),('instruction','output')]:
  if isinstance(row.get(p),str) and isinstance(row.get(a),str) and row[p].strip() and row[a].strip():
   prompt=norm(row[p]);answer=norm(row[a]);return [{'role':'user','content':prompt}],[prompt],[{'role':'user','content':prompt},{'role':'assistant','content':answer}]
 return None
audit={'observed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'repo':repo,'revision':revision,'manifest_declared_rows':{'train':manifest['train_rows'],'eval':manifest['eval_rows']},'normalization':'Full text; Unicode NFKC, casefold, collapse all whitespace. Full prompt includes every role/content message before final assistant; all-user-context excludes answer messages. No prefix truncation.','splits':{}}
internal={}
for split,filename,hashkey in [('train','data/sft_train.jsonl','sft_train_sha256'),('eval','data/eval_repo.jsonl','eval_repo_sha256')]:
 path=pathlib.Path(hf_hub_download(repo,filename,repo_type='dataset',revision=revision,token=get_token(),cache_dir=str(root/'diagnostic_frontier_20261116_corpus_cache')))
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 sha=h.hexdigest();assert sha==manifest[hashkey],(split,'file_hash_mismatch')
 prompts=collections.Counter();users=collections.Counter();turns=collections.Counter();examples=collections.Counter();schemas=collections.Counter();sources=collections.Counter();invalid=[];rows=0
 with path.open(encoding='utf-8') as f:
  for number,line in enumerate(f,1):
   if not line.strip():continue
   rows+=1
   try:row=json.loads(line)
   except json.JSONDecodeError:invalid.append(number);continue
   if not isinstance(row,dict):invalid.append(number);continue
   schemas[tuple(sorted(row))]+=1
   source=row.get('source') or row.get('src')
   if isinstance(source,str) and len(source)<100:sources[source]+=1
   parsed=view(row)
   if parsed is None:invalid.append(number);continue
   context,user_context,example=parsed
   prompts[digest(context)]+=1;users[digest(user_context)]+=1;examples[digest(example)]+=1
   for turn in set(user_context):turns[hashlib.sha256(turn.encode('utf-8')).hexdigest()]+=1
 assert rows==manifest['train_rows' if split=='train' else 'eval_rows'],(split,'row_count_mismatch')
 internal[split]={'prompts':prompts,'users':users,'turns':turns,'examples':examples}
 d={'file':filename,'sha256':sha,'hash_matches_manifest':True,'bytes':path.stat().st_size,'rows':rows,'valid_formatted_rows':rows-len(invalid),'invalid_rows_count':len(invalid),'invalid_row_indices':invalid[:20],'schemas':[{'keys':list(k),'rows':v} for k,v in schemas.items()],'source_counts':dict(sources),'within_split':{}}
 for k,c in internal[split].items():d['within_split'][k]={'unique':len(c),'duplicate_keys':sum(v>1 for v in c.values()),'duplicate_extra_occurrences':sum(v-1 for v in c.values())}
 audit['splits'][split]=d
 print('CORPUS_SPLIT_VERIFIED',split,json.dumps(d),flush=True)
audit['cross_split']={}
for k in ['prompts','users','turns','examples']:
 a=internal['train'][k];b=internal['eval'][k];common=set(a)&set(b)
 audit['cross_split'][k]={'shared_unique_hashes':len(common),'train_occurrences':sum(a[h] for h in common),'eval_occurrences':sum(b[h] for h in common),'example_hashes':sorted(common)[:10]}
audit['exact_disjoint']=all(audit['cross_split'][k]['shared_unique_hashes']==0 for k in ['prompts','users','examples'])
audit['all_formats_valid']=all(x['invalid_rows_count']==0 for x in audit['splits'].values())
audit['limits']=['Exact normalized equality does not establish semantic or near-duplicate independence.','Licenses, source consent, PII handling, benchmark contamination and source-level split construction are not independently verified by this audit.','Dataset manifest claims are checked only for file hashes and row counts; no model quality claim is made.']
(root/'diagnostic_frontier_20261116_corpus_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
print('CORPUS_CROSS_SPLIT',json.dumps(audit['cross_split']),flush=True)
print('CORPUS_AUDIT_COMPLETE',json.dumps({'exact_disjoint':audit['exact_disjoint'],'all_formats_valid':audit['all_formats_valid']}),flush=True)
