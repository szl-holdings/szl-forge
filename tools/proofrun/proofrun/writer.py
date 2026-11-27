"""ProofRun: the write side a trainer calls. Produces the directory `proofrun verify` checks.

    pr = ProofRun(run_dir, Signer(), run_id="f54-v3")
    pr.start(image_ref="ghcr.io/szl-holdings/salad-train@sha256:...", lockfile=..., dataset=..., config=...)
    pr.beat(step, {"loss": ...})                 # every optimizer step or every N steps
    pr.checkpoint(ckpt_dir)                      # after Trainer saves checkpoint-N
    pr.final(final_dir)                          # after save_model(final)
    pr.eval(metrics, eval_set_sha256=..., harness_sha256=...)  # by the eval harness
    pr.claim(steps=N, quality=[...])

`HFCallback(pr)` wires beat/checkpoint into a transformers Trainer without importing transformers here.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from .canonical import ProofError, load_json, tree_entries, write_json
from .records import (HeartbeatChain, build_checkpoint_receipt, build_claim, build_eval_receipt,
                      build_final_receipt, build_manifest)
from .signing import Signer


class ProofRun:
    def __init__(self, run_dir: Path | str, signer: Signer, run_id: str) -> None:
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.signer, self.run_id = signer, run_id
        self.manifest: dict | None = None
        self.chain: HeartbeatChain | None = None

    def start(self, **kw) -> dict:
        self.manifest = build_manifest(self.signer, run_id=self.run_id, **kw)
        write_json(self.dir / "run_manifest.json", self.manifest)
        self.chain = HeartbeatChain(self.signer, self.run_id, self.manifest)
        (self.dir / "heartbeats.jsonl").write_text("")
        return self.manifest

    def beat(self, step: int, metrics: dict | None = None) -> dict:
        if self.chain is None:
            raise ProofError("start() first")
        env = self.chain.beat(step, metrics)
        with open(self.dir / "heartbeats.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(env, sort_keys=True, separators=(",", ":")) + "\n")
        return env

    def _rel(self, p: Path) -> str | None:
        try:
            return Path(p).resolve().relative_to(self.dir.resolve()).as_posix()
        except ValueError:
            return None

    def checkpoint(self, ckpt_dir: Path | str) -> dict:
        env = build_checkpoint_receipt(self.signer, self.run_id, ckpt_dir,
                                       heartbeat_head=self.chain.head if self.chain else None,
                                       artifact_dir=self._rel(Path(ckpt_dir)))
        write_json(self.dir / "checkpoints" / f"{Path(ckpt_dir).name}.json", env)
        return env

    def final(self, final_dir: Path | str) -> dict:
        env = build_final_receipt(self.signer, self.run_id, tree_entries(final_dir), self._rel(Path(final_dir)))
        write_json(self.dir / "final.json", env)
        return env

    def eval(self, metrics: dict, *, eval_set_sha256: str, harness_sha256: str, name: str = "heldout",
             checkpoint: str | None = None) -> dict:
        ck = sorted((load_json(p) for p in (self.dir / "checkpoints").glob("*.json")),
                    key=lambda e: e["payload"]["global_step"])
        term = next(e for e in ck if e["payload"]["checkpoint"] == checkpoint) if checkpoint else ck[-1]
        fin = load_json(self.dir / "final.json")["payload"]
        adapter = next(f["sha256"] for f in fin["files"] if f["path"] == "adapter_model.safetensors")
        env = build_eval_receipt(self.signer, run_id=self.run_id,
                                 checkpoint_tree_sha256=term["payload"]["tree_sha256"], adapter_sha256=adapter,
                                 image_digest=self.manifest["payload"]["image"]["digest"],
                                 eval_set_sha256=eval_set_sha256, harness_sha256=harness_sha256,
                                 metrics=metrics, name=name)
        write_json(self.dir / "evals" / f"{name}.json", env)
        return env

    def claim(self, **kw) -> dict:
        env = build_claim(self.signer, run_id=self.run_id, **kw)
        write_json(self.dir / "claim.json", env)
        return env


class HFCallback:
    """Duck-typed transformers.TrainerCallback: pass to Trainer(callbacks=[HFCallback(pr)])."""

    def __init__(self, pr: ProofRun, every: int = 1) -> None:
        self.pr, self.every = pr, every

    def __getattr__(self, name):  # all other TrainerCallback hooks are no-ops
        if name.startswith("on_"):
            return lambda *a, **k: None
        raise AttributeError(name)

    def on_step_end(self, args, state, control, **kw):
        if state.global_step % self.every == 0:
            last = (state.log_history or [{}])[-1]
            self.pr.beat(state.global_step, {k: v for k, v in last.items() if isinstance(v, (int, float)) and math.isfinite(v)})

    def on_save(self, args, state, control, **kw):
        self.pr.checkpoint(Path(args.output_dir) / f"checkpoint-{state.global_step}")
