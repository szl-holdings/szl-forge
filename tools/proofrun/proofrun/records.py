"""Builders for the five receipt kinds. Each returns a signed envelope (see signing.py).

  run_manifest       image digest, lockfile hash, dataset hash, config hash (the environment)
  heartbeat          per-step, hash-linked to the previous heartbeat; genesis links to the manifest
  checkpoint_receipt sha256 per file, tree hash, trainer_state global_step, bound to a heartbeat
  final_adapter      sha256 per file of the published adapter
  eval_receipt       metrics bound to checkpoint tree hash, adapter hash and image digest
  claim              what the run asserts (steps, final adapter, quality)
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

from . import SCHEMA
from .canonical import (ProofError, is_hex64, load_json, sha256_file, sha256_obj, tree_entries,
                        tree_sha256)
from .signing import Signer

DIGEST_RE = re.compile(r"^(?:[a-z0-9][a-z0-9._/:-]*@)?sha256:[0-9a-f]{64}$")


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def image_digest_of(ref: str) -> str:
    if not isinstance(ref, str) or not DIGEST_RE.match(ref):
        raise ProofError(f"image must be pinned by @sha256 digest, got {ref!r}")
    return "sha256:" + ref.rsplit("sha256:", 1)[1]


def dataset_hash(path: Path | str) -> dict:
    p = Path(path)
    if p.is_file():
        return {"kind": "file", "sha256": sha256_file(p), "files": 1, "bytes": p.stat().st_size}
    if p.is_dir():
        ents = tree_entries(p)
        if not ents:
            raise ProofError(f"dataset dir {p} is empty")
        return {"kind": "tree", "sha256": tree_sha256(ents), "files": len(ents),
                "bytes": sum(e["size"] for e in ents)}
    raise ProofError(f"dataset not found: {p}")


def build_manifest(signer: Signer, *, run_id: str, image_ref: str, lockfile: Path | str,
                   dataset: Path | str, config: Path | str | dict, base_model: str = "",
                   extra: dict | None = None) -> dict:
    cfg = config if isinstance(config, dict) else load_json(config)
    payload = {"schema": SCHEMA, "run_id": run_id, "created": now(), "retroactive": False,
               "image": {"ref": image_ref, "digest": image_digest_of(image_ref)},
               "lockfile_sha256": sha256_file(lockfile), "dataset": dataset_hash(dataset),
               "config_sha256": sha256_obj(cfg), "config": cfg, "base_model": base_model,
               "extra": extra or {}}
    return signer.envelope("run_manifest", payload)


def build_retroactive_manifest(signer: Signer, *, run_id: str, source: dict, note: str) -> dict:
    """For runs that predate proofrun: no image digest, lockfile or heartbeat chain is known.
    The verifier only accepts this with --accept-retroactive and labels every result."""
    payload = {"schema": SCHEMA, "run_id": run_id, "created": now(), "retroactive": True,
               "image": {"ref": None, "digest": None}, "lockfile_sha256": None, "dataset": None,
               "config_sha256": None, "source": source, "note": note}
    return signer.envelope("run_manifest", payload)


class HeartbeatChain:
    def __init__(self, signer: Signer, run_id: str, manifest_env: dict) -> None:
        self.signer, self.run_id = signer, run_id
        self.manifest_sha = manifest_env["payload_sha256"]
        self.records: list[dict] = []

    @property
    def head(self) -> str:
        return self.records[-1]["payload_sha256"] if self.records else self.manifest_sha

    def beat(self, step: int, metrics: dict | None = None, ts: str | None = None) -> dict:
        payload = {"run_id": self.run_id, "seq": len(self.records), "step": int(step),
                   "ts": ts or now(), "prev": self.head, "manifest_sha256": self.manifest_sha,
                   "metrics": metrics or {}}
        env = self.signer.envelope("heartbeat", payload)
        self.records.append(env)
        return env


def read_trainer_state(path: Path | str) -> dict:
    ts = load_json(path)
    for k in ("global_step", "max_steps"):
        if not isinstance(ts.get(k), int):
            raise ProofError(f"trainer_state {path}: {k} missing or not an int")
    return ts


def checkpoint_payload_from_files(run_id: str, name: str, files: list[dict], global_step: int,
                                  max_steps: int, trainer_state_sha256: str, heartbeat_head: str | None,
                                  artifact_dir: str | None, source: dict | None = None) -> dict:
    for e in files:
        if not is_hex64(e.get("sha256")):
            raise ProofError(f"checkpoint {name}: bad sha256 for {e.get('path')}")
    return {"run_id": run_id, "checkpoint": name, "global_step": global_step, "max_steps": max_steps,
            "trainer_state_sha256": trainer_state_sha256, "files": sorted(files, key=lambda r: r["path"]),
            "tree_sha256": tree_sha256(files), "heartbeat_head": heartbeat_head,
            "artifact_dir": artifact_dir, "source": source, "created": now()}


def build_checkpoint_receipt(signer: Signer, run_id: str, ckpt_dir: Path | str, *,
                             heartbeat_head: str | None, artifact_dir: str | None = None) -> dict:
    d = Path(ckpt_dir)
    ts = read_trainer_state(d / "trainer_state.json")
    files = tree_entries(d)
    payload = checkpoint_payload_from_files(run_id, d.name, files, ts["global_step"], ts["max_steps"],
                                            sha256_file(d / "trainer_state.json"), heartbeat_head,
                                            artifact_dir)
    return signer.envelope("checkpoint_receipt", payload)


def build_final_receipt(signer: Signer, run_id: str, files: list[dict], artifact_dir: str | None,
                        source: dict | None = None) -> dict:
    return signer.envelope("final_adapter", {"run_id": run_id, "files": sorted(files, key=lambda r: r["path"]),
                                             "tree_sha256": tree_sha256(files), "artifact_dir": artifact_dir,
                                             "source": source, "created": now()})


def build_eval_receipt(signer: Signer, *, run_id: str, checkpoint_tree_sha256: str, adapter_sha256: str,
                       image_digest: str, eval_set_sha256: str, harness_sha256: str,
                       metrics: dict[str, Any], name: str = "heldout") -> dict:
    for k, v in (("checkpoint_tree_sha256", checkpoint_tree_sha256), ("adapter_sha256", adapter_sha256),
                 ("eval_set_sha256", eval_set_sha256), ("harness_sha256", harness_sha256)):
        if not is_hex64(v):
            raise ProofError(f"eval receipt: {k} must be sha256 hex")
    return signer.envelope("eval_receipt", {"run_id": run_id, "name": name,
                                            "checkpoint_tree_sha256": checkpoint_tree_sha256,
                                            "adapter_sha256": adapter_sha256,
                                            "image_digest": image_digest_of(image_digest),
                                            "eval_set_sha256": eval_set_sha256,
                                            "harness_sha256": harness_sha256, "metrics": metrics,
                                            "created": now()})


def build_claim(signer: Signer, *, run_id: str, steps: int, final_adapter: bool = True,
                quality: list[dict] | None = None, evidence_level: str = "proof-carrying",
                note: str = "") -> dict:
    """quality: [{"metric": "heldout_acc", "op": ">=", "value": 0.8, "eval_receipt": "evals/heldout.json"}]"""
    if evidence_level not in ("proof-carrying", "retroactive"):
        raise ProofError("evidence_level must be proof-carrying or retroactive")
    return signer.envelope("claim", {"run_id": run_id, "steps": int(steps), "final_adapter": final_adapter,
                                     "quality": quality or [], "evidence_level": evidence_level,
                                     "note": note, "created": now()})
