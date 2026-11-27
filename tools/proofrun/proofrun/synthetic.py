"""Synthetic proof-carrying run for tests and the demo. Simulates a LoRA trainer: no GPU, no network.
The image digest is a synthetic sha256 (clearly labelled), the key is ephemeral and discarded."""
from __future__ import annotations

import hashlib
import json
import shutil
import struct
from pathlib import Path

from .canonical import sha256_bytes, write_json
from .safetensors_lite import write_safetensors
from .signing import Signer
from .writer import ProofRun

SYNTH_IMAGE = "ghcr.io/szl-holdings/salad-train@sha256:" + hashlib.sha256(b"proofrun-synthetic-image").hexdigest()


def _adapter(path: Path, step: int, zero_b: bool = False) -> None:
    a = struct.pack("<8f", *[0.01 * (i + 1) for i in range(8)])
    b = bytes(32) if zero_b else struct.pack("<8f", *[0.001 * step * (i + 1) for i in range(8)])
    write_safetensors(path, {
        "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": ("F32", [2, 4], a),
        "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": ("F32", [4, 2], b)})


def _save_ckpt(out: Path, step: int, max_steps: int, history: list, zero_b=False) -> Path:
    d = out / f"checkpoint-{step}"
    d.mkdir(parents=True, exist_ok=True)
    _adapter(d / "adapter_model.safetensors", step, zero_b)
    (d / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "r": 2, "lora_alpha": 4}))
    (d / "trainer_state.json").write_text(json.dumps({"global_step": step, "max_steps": max_steps,
                                                      "log_history": list(history)}))
    (d / "optimizer.pt").write_bytes(hashlib.sha256(f"opt{step}".encode()).digest() * 4)
    return d


def make_run(run_dir, steps: int = 20, save_every: int = 10, with_eval: bool = True,
             zero_b: bool = False, signer: Signer | None = None) -> dict:
    run = Path(run_dir)
    if run.exists():
        shutil.rmtree(run)
    inputs = run / "inputs"
    inputs.mkdir(parents=True)
    (inputs / "train.jsonl").write_text("\n".join(json.dumps({"q": i, "a": i * 2}) for i in range(50)))
    (inputs / "requirements.lock").write_text("transformers==5.22.2 --hash=sha256:" + "0" * 64 + "\n")
    (inputs / "config.json").write_text(json.dumps({"lr": 1e-4, "max_steps": steps, "lora_r": 2}))
    signer = signer or Signer()
    pr = ProofRun(run, signer, run_id="synthetic-lora-001")
    pr.start(image_ref=SYNTH_IMAGE, lockfile=inputs / "requirements.lock", dataset=inputs / "train.jsonl",
             config=inputs / "config.json", base_model="synthetic/base")
    out, hist = run / "artifacts", []
    for step in range(1, steps + 1):
        loss = round(2.0 / step, 6)
        hist.append({"step": step, "loss": loss})
        pr.beat(step, {"loss": loss})
        if step % save_every == 0:
            pr.checkpoint(_save_ckpt(out, step, steps, hist, zero_b))
    fin = out / "final"
    fin.mkdir()
    for n in ("adapter_model.safetensors", "adapter_config.json"):
        shutil.copy2(out / f"checkpoint-{steps}" / n, fin / n)
    pr.final(fin)
    quality = []
    if with_eval:
        pr.eval({"heldout_acc": 0.91}, eval_set_sha256=sha256_bytes(b"heldout-set"),
                harness_sha256=sha256_bytes(b"harness-v1"))
        quality = [{"metric": "heldout_acc", "op": ">=", "value": 0.9, "eval_receipt": "evals/heldout.json"}]
    pr.claim(steps=steps, quality=quality)
    return {"run_dir": str(run), "signer": signer.fingerprint, "steps": steps,
            "note": "synthetic image digest and ephemeral key; for tests only"}
