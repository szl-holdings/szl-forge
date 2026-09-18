import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B", torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, "chaski_r4/chaski-r4-adapter")
model.eval()
rows=[json.loads(l) for l in open('chaski_r4/train.jsonl',encoding='utf-8-sig') if l.strip()]
row=[r for r in rows if not r['messages'][-1]['content'].startswith(('REFUSE:','ABSTAIN:'))][0]
prompt = tok.apply_chat_template(row['messages'][:2], tokenize=False, add_generation_prompt=True)
inputs = tok(prompt, return_tensors='pt').to(model.device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=150, do_sample=False)
print('TRAINING TAIL:', row['messages'][-1]['content'][:250])
print('MODEL OUTPUT :', tok.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)[:250])
