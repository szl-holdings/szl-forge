#!/usr/bin/env python3
# SZL-Khipu-1.5B-BrainNavigator — QLoRA training on the owner's own metal.
# SPDX-License-Identifier: Apache-2.0
# (c) 2026 Lutar, Stephen P. - SZL Holdings
#
# Fine-tunes Qwen2.5-1.5B-Instruct with QLoRA (Unsloth) on the committed Khipu
# curriculum, merges to a 16-bit safetensors folder, then builds and SIGNS an
# owner training receipt that the Alloy backbone verifies.
#
# What this model IS: a governed retrieval NAVIGATOR. Given a query + candidate
# Brain node HANDLES (ids + synthetic metadata only), it emits a retrieval PLAN
# (route over handles, cite only the supporting ones, ABSTAIN when none support).
# It never resolves node content and never answers from memory. The curriculum is
# fully SYNTHETIC, so this measures routing-POLICY conformance over synthetic
# scenarios -- NEVER real-Brain navigation skill.
#
# BINDING honesty doctrine:
# - The receipt's baseModel is the CANONICAL Hugging Face id and MUST equal
#   EXPECTED_BASE_MODEL in the backbone (Qwen/Qwen2.5-1.5B-Instruct). We train
#   from the bnb-4bit variant of the SAME weights; the fine-tune still derives
#   from that base.
# - datasets is pinned by recomputing sha256 over the ACTUAL committed files and
#   cross-checking manifest.json -- a drifted checkout fails LOUD here.
# - finalTrainLoss is REPORTED verbatim as a STRING (never a re-derived float).
# - This NEVER touches SZL-1 (the 3B sovereign model) or the ReceiptAgent.
#   Different output dir, different Ollama name ('khipu').
# - Producing + signing a receipt does not "measure" anything server-side: the
#   backbone REPORTS the owner's attestation and DERIVES its verification.
import glob
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sign_receipt as sr  # noqa: E402

# The bnb-4bit is the SAME Qwen2.5-1.5B-Instruct weights, quantized for QLoRA.
BASE_TRAIN = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
BASE_CANONICAL = "Qwen/Qwen2.5-1.5B-Instruct"  # MUST equal EXPECTED_BASE_MODEL
SERVED_MODEL = "khipu"  # Ollama name (NEVER 'szl1' or 'receiptagent')
MAX_SEQ_LEN = 2048

MERGED_DIR = os.path.join(HERE, "khipu-model")
ADAPTER_DIR = os.path.join(HERE, "khipu-adapter")
TRAINING_RECEIPT = os.path.join(HERE, "training_receipt.signed.json")

# The 5 files the backbone regenerates + pins (buildKhipuCurriculum).
CURRICULUM_FILES = [
    "train.jsonl",
    "eval.jsonl",
    "train.abstain.jsonl",
    "adversarial.jsonl",
    "khipu.schema.json",
]
# Only these two are used for TRAINING (navigate plans + abstain plans).
# eval.jsonl and adversarial.jsonl are held out for eval_khipu.py.
TRAIN_FILES = ["train.jsonl", "train.abstain.jsonl"]
# Abstain parity (IN-MEMORY oversample). The committed abstain set (8) is smaller
# than the navigate set (15), which biases the model into ALWAYS navigating (it
# never learns to refuse when no handle supports the query). We repeat the
# abstain rows in memory to reach ~1:1 with navigate rows. This NEVER edits a
# committed file, so the backbone verifier -- which hashes the files ON DISK, not
# the training multiset -- still sees the exact pinned curriculum. Do NOT push
# past parity: over-abstention tanks routing conformance.
ABSTAIN_OVERSAMPLE = 2


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest() -> dict:
    with open(os.path.join(HERE, "manifest.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def build_datasets_map(manifest: dict) -> dict:
    """sha256 of every committed curriculum file, cross-checked against the
    manifest. This is exactly what the backbone regenerates and compares."""
    datasets = {}
    for name in CURRICULUM_FILES:
        path = os.path.join(HERE, name)
        if not os.path.exists(path):
            raise SystemExit(f"[train] missing curriculum file: {name}")
        digest = sha256_file(path)
        declared = manifest.get("files", {}).get(name, {}).get("sha256")
        if declared != digest:
            raise SystemExit(
                f"[train] {name} sha256 {digest} != manifest {declared}. "
                "The committed curriculum is inconsistent -- regenerate it "
                "(gen:khipu-curriculum) before training."
            )
        datasets[name] = digest
    return datasets


def sha256_safetensors_dir(directory: str) -> str:
    """Deterministic digest over every *.safetensors in a dir (name + bytes,
    sorted). Pins the exact produced artifact bytes."""
    files = sorted(glob.glob(os.path.join(directory, "*.safetensors")))
    if not files:
        raise SystemExit(f"[train] no *.safetensors found in {directory}")
    h = hashlib.sha256()
    for path in files:
        h.update(os.path.basename(path).encode("utf-8"))
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def load_train_rows(tokenizer):
    rows = []
    for name in TRAIN_FILES:
        reps = ABSTAIN_OVERSAMPLE if name == "train.abstain.jsonl" else 1
        file_rows = []
        with open(os.path.join(HERE, name), "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    file_rows.append(json.loads(line))
        for _ in range(reps):
            rows.extend(file_rows)
        print(f"[train]   {name}: {len(file_rows)} rows x{reps}")
    print(
        f"[train] {len(rows)} training rows total "
        f"(abstain oversampled x{ABSTAIN_OVERSAMPLE} to ~1:1 with navigate)"
    )
    return [
        tokenizer.apply_chat_template(
            r["messages"], tokenize=False, add_generation_prompt=False
        )
        for r in rows
    ]


def main() -> None:
    manifest = load_manifest()
    datasets = build_datasets_map(manifest)
    contract = manifest["contract"]
    print(f"[train] curriculum verified against manifest ({len(datasets)} files)")

    from unsloth import FastLanguageModel

    print(f"[train] loading base: {BASE_TRAIN}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_TRAIN,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        lora_alpha=64,
        lora_dropout=0,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
        random_state=11,
    )

    texts = load_train_rows(tokenizer)

    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=Dataset.from_dict({"text": texts}),
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        args=SFTConfig(
            per_device_train_batch_size=1,
            # Small curriculum -> pure memorization objective. Keep the same
            # recipe that got the ReceiptAgent over the quantization cliff:
            # ga=2 x 45 epochs is ~700 optimizer steps, and a
            # constant_with_warmup schedule (NOT cosine -- cosine decays LR to
            # ~0 by the end and starves the late consolidation the hardest
            # tokens need). r32/alpha64 buys the LoRA capacity to survive the
            # q4_K_M rebirth (the 1.5B base prior can flip small string tokens
            # like "1.5B" -> "15B" under 4-bit if the margin is thin).
            gradient_accumulation_steps=2,
            num_train_epochs=45,
            learning_rate=2e-4,
            warmup_steps=10,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="constant_with_warmup",
            seed=11,
            output_dir=os.path.join(HERE, "outputs"),
            report_to="none",
        ),
    )

    # Response-only loss. Every row shares the SAME ~1.0k-char system prompt plus
    # a candidate-handle block in the user turn, so training on the full
    # conversation spends most of the gradient re-memorizing the prompt while the
    # short target (a routing plan) barely moves. Mask everything up to the
    # assistant turn so loss lands ONLY on the plan.
    from unsloth.chat_templates import train_on_responses_only

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    print("[train] training...")
    stats = trainer.train()
    final_loss = f"{stats.training_loss:.4f}"
    print(f"[train] final loss (REPORTED verbatim): {final_loss}")

    print(f"[train] saving LoRA adapter -> {ADAPTER_DIR}")
    model.save_pretrained(ADAPTER_DIR)
    print(f"[train] merging to 16-bit safetensors -> {MERGED_DIR}")
    model.save_pretrained_merged(MERGED_DIR, tokenizer, save_method="merged_16bit")

    adapter_sha = sha256_safetensors_dir(ADAPTER_DIR)
    weights_sha = sha256_safetensors_dir(MERGED_DIR)

    payload = {
        "kind": "szl-khipu-training-receipt",
        "v": 1,
        "capabilityProfile": "SZL-Khipu-1.5B-BrainNavigator",
        "baseModel": BASE_CANONICAL,
        "datasets": datasets,
        "schemaFingerprintSha256": contract["schemaFingerprintSha256"],
        "outputSchemaSha256": contract["outputSchemaSha256"],
        "adapterSha256": adapter_sha,
        "weightsArtifactSha256": weights_sha,
        "servedModel": SERVED_MODEL,
        "trainedAt": datetime.now(timezone.utc).isoformat(),
        "host": platform.node() or "unknown-host",
        "finalTrainLoss": final_loss,
        # keyId is stamped by sign_receipt from the signing key.
        "keyId": "",
    }
    sr.sign_payload(payload, TRAINING_RECEIPT)
    print("[train] DONE.")
    print("[train] Next: rebirth-khipu.ps1 (GGUF import), then sanity_gate.py,")
    print("        then eval_khipu.py to produce the signed eval receipt.")


if __name__ == "__main__":
    main()
