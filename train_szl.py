# SZL Forge — train SZL-1 on your own metal.
# Fine-tunes Qwen2.5-3B-Instruct with QLoRA via Unsloth, then merges to a
# 16-bit safetensors folder that Ollama can import directly (no llama.cpp
# build needed on Windows).
#
# Honest expectations:
# - First run downloads the ~2 GB 4-bit base model from Hugging Face.
# - Training: minutes-scale on an RTX 5050 (small dataset, 3 epochs).
# - Merge step needs ~8 GB free RAM and ~7 GB free disk for ./szl-model.

import json

from unsloth import FastLanguageModel

MAX_SEQ_LEN = 1024
BASE = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"

print(f"[szl-forge] loading base model: {BASE}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    use_gradient_checkpointing="unsloth",
    random_state=11,
)

print("[szl-forge] loading dataset: szl_dataset.jsonl")
with open("szl_dataset.jsonl", "r", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]
print(f"[szl-forge] {len(rows)} training examples")

texts = [
    tokenizer.apply_chat_template(
        r["messages"], tokenize=False, add_generation_prompt=False
    )
    for r in rows
]

from datasets import Dataset

dataset = Dataset.from_dict({"text": texts})

from trl import SFTConfig, SFTTrainer

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LEN,
    args=SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2e-4,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=11,
        output_dir="outputs",
        report_to="none",
    ),
)

print("[szl-forge] training...")
stats = trainer.train()
print(f"[szl-forge] training done: {stats.training_loss:.4f} final loss")

print("[szl-forge] merging to 16-bit safetensors at ./szl-model ...")
model.save_pretrained_merged("szl-model", tokenizer, save_method="merged_16bit")
print("[szl-forge] DONE. Next: ollama create szl1 -f Modelfile")
