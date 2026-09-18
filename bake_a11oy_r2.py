from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="chaski_r2/chaski-r2-adapter",
    max_seq_length=2048,
    load_in_4bit=True,
)
print("[a11oy] PEFT attached:", hasattr(model, "peft_config"))
model.save_pretrained_gguf("chaski_r2/a11oy_mini_r2", tokenizer, quantization_method="q4_k_m")
print("[a11oy] GGUF bake complete")
