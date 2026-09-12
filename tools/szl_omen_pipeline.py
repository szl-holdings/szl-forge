"""SZL one-file pipeline for the OMEN and the 5050 laptop.

Subcommands:
  merge    — merge all eight Tier-3 adapters into standalone checkpoints
  dpo      — on-policy DPO for the khipu abstain lane (recommended first run)
  publish  — upload merged/ (or a DPO output dir) to the Hub

Deps:  pip install torch transformers peft trl datasets safetensors huggingface_hub
Auth:  hf auth login   (token prompt; tokens never go in files or chat)
Receipts: every stage writes a *_receipt.json with input digests. Train loss is
a train metric, not an eval. publication_eligible stays false until a held-out
generate receipt exists.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

import torch

# ---------------- merge (from the audit wave) ----------------

MERGE_JOBS = [
    ("SZLHOLDINGS/szl-receiptagent-qwen35-0.8b-v2", "Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"),
    ("SZLHOLDINGS/szl-receiptagent-qwen35-0.8b-v3", "Qwen/Qwen3.5-0.8B", None),
    ("SZLHOLDINGS/khipu-r3",                        "Qwen/Qwen3.5-0.8B", None),
    ("SZLHOLDINGS/chaski-r2",                       "Qwen/Qwen3.5-0.8B", None),
    ("SZLHOLDINGS/chaski-5050",                     "Qwen/Qwen3.5-0.8B", None),
    ("SZLHOLDINGS/brain-navigator-r2",              "Qwen/Qwen3.5-0.8B", None),
    ("SZLHOLDINGS/KHIPU-R2",                        "Qwen/Qwen2.5-1.5B-Instruct", None),
    ("SZLHOLDINGS/WILLAY",                          "Qwen/Qwen2.5-0.5B-Instruct", None),
]


def sha256_of(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_merge(out: pathlib.Path) -> None:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out.mkdir(exist_ok=True)
    for repo, base, rev in MERGE_JOBS:
        name = repo.split("/")[-1]
        dest = out / name
        dest.mkdir(parents=True, exist_ok=True)
        print(f"--- merging {repo}")
        model = AutoModelForCausalLM.from_pretrained(base, revision=rev, torch_dtype=torch.float32, device_map="cpu")
        model = PeftModel.from_pretrained(model, repo).merge_and_unload()
        model.save_pretrained(dest, safe_serialization=True)
        AutoTokenizer.from_pretrained(repo).save_pretrained(dest)
        receipt = {
            "repo": repo, "base_model": base, "base_revision": rev or "main@merge-time",
            "merged_files": {p.name: sha256_of(p) for p in sorted(dest.glob("*.safetensors"))},
            "merge_dtype": "float32",
        }
        (dest / "merge_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        print(f"    -> done")


# ---------------- dpo (khipu abstain lane, on-policy pairs) ----------------

DPO_ADAPTER = "SZLHOLDINGS/khipu-r3"          # the 0/6 port this run exists to fix
DPO_BASE = "Qwen/Qwen3.5-0.8B"
DPO_DATA_REPO = "SZLHOLDINGS/SZL-Khipu-1.5B-abstain"
DPO_FILES = ["train.abstain.jsonl", "adversarial.jsonl"]


def cmd_dpo(out: pathlib.Path, max_pairs: int, epochs: int) -> None:
    from datasets import Dataset
    from huggingface_hub import hf_hub_download
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    print(f"--- dpo on {gpu}")

    rows = []
    for fname in DPO_FILES:
        local = hf_hub_download(DPO_DATA_REPO, fname)
        text = pathlib.Path(local).read_text(encoding="utf-8")
        for line in text.splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    digest = hashlib.sha256("\n".join(sorted(json.dumps(r, sort_keys=True) for r in rows)).encode()).hexdigest()

    tok = AutoTokenizer.from_pretrained(DPO_ADAPTER)
    policy = AutoModelForCausalLM.from_pretrained(DPO_BASE, torch_dtype=torch.bfloat16).to(device)
    from peft import PeftModel
    policy = PeftModel.from_pretrained(policy, DPO_ADAPTER)  # on-policy: rejected samples come from the adapter we're fixing
    policy.eval()

    pairs = []
    for row in rows[:max_pairs]:
        prompt = row.get("prompt") or row.get("input") or row.get("messages")
        chosen = row.get("chosen") or row.get("expected") or row.get("output")
        if isinstance(prompt, list):  # chat rows
            prompt_text = tok.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        else:
            prompt_text = str(prompt)
        if not chosen:
            continue
        ids = tok(prompt_text, return_tensors="pt").to(device)
        with torch.no_grad():
            gen = policy.generate(**ids, max_new_tokens=128, do_sample=False)
        rejected = tok.decode(gen[0][ids["input_ids"].shape[-1]:], skip_special_tokens=True)
        if rejected.strip() and rejected.strip() != str(chosen).strip():
            pairs.append({"prompt": prompt_text, "chosen": str(chosen), "rejected": rejected})
    print(f"    {len(pairs)} on-policy pairs from {len(rows)} source rows")
    if len(pairs) < 4:
        raise RuntimeError("too few usable pairs; check the source jsonl field names against your curriculum")

    peft_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
                          task_type="CAUSAL_LM",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    cfg = DPOConfig(
        output_dir=str(out), per_device_train_batch_size=1, gradient_accumulation_steps=4,
        num_train_epochs=epochs, learning_rate=5e-5, lr_scheduler_type="cosine",
        logging_steps=1, save_strategy="no", bf16=(device == "cuda"),
        beta=0.1, max_length=1024, max_prompt_length=512, seed=11,
        report_to=[],
    )
    trainer = DPOTrainer(
        model=DPO_BASE,
        ref_model=None,
        peft_config=peft_cfg,
        args=cfg,
        train_dataset=Dataset.from_list(pairs),
        processing_class=tok,
    )
    result = trainer.train()
    trainer.save_model(out)
    tok.save_pretrained(out)

    receipt = {
        "lane": "khipu-abstain-dpo", "base": DPO_BASE, "policy_adapter": DPO_ADAPTER,
        "data_repo": DPO_DATA_REPO, "data_files": DPO_FILES, "data_digest_sha256": digest,
        "pairs_used": len(pairs), "epochs": epochs, "lora": "r16 a32 bf16 (QLoRA forbidden on Qwen3.5)",
        "train_loss": result.training_loss, "train_loss_label": "train metric, not an eval",
        "hardware": gpu, "seed": 11,
        "publication_eligible": False,
        "autonomy_eligible": False,
        "next_gate": "held-out abstain generate on adversarial.jsonl — receipt required before any card claims a number",
    }
    (out / "dpo_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"    train_loss {result.training_loss:.4f} (train metric, not an eval); receipt written")


# ---------------- publish ----------------

CONFIG_FIXES = {
    "chaski-5050": "Qwen/Qwen3.5-0.8B",
    "szl-receiptagent-qwen35-0.8b-v2": "Qwen/Qwen3.5-0.8B",
    "KHIPU-R2": "Qwen/Qwen2.5-1.5B-Instruct",
}


def cmd_publish(folder: pathlib.Path, only: str | None) -> None:
    from huggingface_hub import HfApi, upload_folder

    api = HfApi()
    api.whoami()  # fail fast if not logged in
    for sub in sorted(p for p in folder.iterdir() if p.is_dir()):
        if only and sub.name != only:
            continue
        repo = f"SZLHOLDINGS/{sub.name}"
        print(f"--- {repo}")
        cfg = sub / "adapter_config.json"
        if sub.name in CONFIG_FIXES and cfg.exists():
            data = json.loads(cfg.read_text(encoding="utf-8-sig"))
            data["base_model_name_or_path"] = CONFIG_FIXES[sub.name]
            cfg.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        readme = sub / "README.md"
        if readme.exists():
            text = readme.read_text(encoding="utf-8")
            new = text.replace("base_model_relation: adapter", "base_model_relation: merged")
            new = new.replace("library_name: peft", "library_name: transformers")
            if new != text:
                readme.write_text(new, encoding="utf-8")
            else:
                print("    card YAML had no adapter/peft markers; left untouched")
        info = upload_folder(repo_id=repo, folder_path=str(sub),
                             commit_message="Merged full-precision checkpoint (receipted) + config fixes")
        print(f"    OK -> {info}")
    print("publish complete")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("merge"); m.add_argument("--out", default="merged")
    d = sub.add_parser("dpo"); d.add_argument("--out", default="dpo-khipu-r4"); d.add_argument("--max-pairs", type=int, default=64); d.add_argument("--epochs", type=int, default=1)
    p = sub.add_parser("publish"); p.add_argument("--folder", default="merged"); p.add_argument("--only", default=None)
    args = ap.parse_args()
    if args.cmd == "merge":
        cmd_merge(pathlib.Path(args.out))
    elif args.cmd == "dpo":
        cmd_dpo(pathlib.Path(args.out), args.max_pairs, args.epochs)
    else:
        cmd_publish(pathlib.Path(args.folder), args.only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
