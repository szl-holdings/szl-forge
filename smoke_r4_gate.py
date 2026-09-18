import json, torch, glob
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B", torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, "chaski_r4/chaski-r4-adapter")
model.eval()
def gen(msgs):
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors='pt').to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=150, do_sample=False)
    return tok.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)[:250]
rows=[json.loads(l) for l in open('chaski_r4/train.jsonl',encoding='utf-8-sig') if l.strip()]
train_row=[r for r in rows if not r['messages'][-1]['content'].startswith(('REFUSE:','ABSTAIN:'))][0]
print('TRAIN PROMPT ->', gen(train_row['messages'][:2]))
for gf in sorted(glob.glob('chaski/gate/*.jsonl')):
    g=[json.loads(l) for l in open(gf,encoding='utf-8-sig') if l.strip()]
    if g and isinstance(g[0],dict) and 'messages' in g[0]:
        print('GATE PROMPT ('+gf+') ->', gen(g[0]['messages'][:2]))
        break
