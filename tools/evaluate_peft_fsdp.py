"""Two-rank CPU FSDP export evidence; no GPU, pretrained model or publisher.

Copyright 2026 SZL Holdings. SPDX-License-Identifier: Apache-2.0
Uses private temporary directories and loopback-only Gloo. Worker identities,
paths, models and modes are internal fixed fixtures, never CLI-controlled.
"""
from __future__ import annotations

from datetime import timedelta
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from typing import Any
from unittest.mock import patch
import warnings

from tools.evaluate_peft_runtime import (
    BASE_ID, PEFT_REVISION, SAVE_AND_LOAD_BLOB, SHAPES,
    EvaluationError, bounded_read, check_installations, require, sha256, source_revision,
)


def supervise(processes, *, timeout: float, ready, cancel: bool = False, stall: bool = False) -> None:
    """Bound worker lifetime; any nonzero, cancellation or timeout rejects the group."""
    deadline = time.monotonic() + timeout
    ready_deadline = None
    try:
        while True:
            codes = [p.exitcode for p in processes]
            require(not any(c is not None and c != 0 for c in codes), "WORKER_NONZERO")
            if all(c == 0 for c in codes):
                return
            if ready():
                require(not cancel, "GROUP_CANCELLED")
                if stall and ready_deadline is None:
                    ready_deadline = time.monotonic() + 2.0
            now = time.monotonic()
            require(now < deadline and (ready_deadline is None or now < ready_deadline), "GROUP_TIMEOUT")
            time.sleep(0.05)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join(timeout=2)
        require(not any(p.is_alive() for p in processes), "WORKER_CLEANUP_FAILED")


def _worker(rank: int, temporary: str, revision: str, mode: str) -> None:
    # Gloo's C++ connection messages belong to logs, never the parent JSON.
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    require(type(rank) is int and rank in (0, 1), "INVALID_RANK")
    require(mode in {"normal", "loss", "stall", "cancel"}, "INVALID_FIXTURE_MODE")
    require(re.fullmatch(r"[0-9a-f]{40}", revision) is not None, "INVALID_SOURCE")
    check_installations()
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    import peft
    from peft import LoraConfig, PeftModel, PeftWarning, get_peft_model
    from safetensors.torch import load, save
    from tools.evaluate_peft_export import MAX_BYTES, MAX_HEADER, inspect_adapter

    content = Path(peft.utils.save_and_load.__file__).read_bytes()
    blob = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
    require(blob == SAVE_AND_LOAD_BLOB, "UPSTREAM_CRITICAL_FILE_MISMATCH")
    torch.set_num_threads(1)
    torch.manual_seed(1742)
    torch.use_deterministic_algorithms(True)
    root = Path(temporary)
    rank_root = root / f"rank-{rank}"
    rank_root.mkdir()
    dist.init_process_group("gloo", rank=rank, world_size=2,
                            init_method=(root / "rendezvous").as_uri(), timeout=timedelta(seconds=20))
    try:
        class TinyLinear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 4, bias=False)
                with torch.no_grad():
                    self.linear.weight.copy_(torch.arange(12, dtype=torch.float32).reshape(4, 3) / 16)
                self.name_or_path = BASE_ID

            def forward(self, x):
                return self.linear(x)

        base_bytes = save(TinyLinear().state_dict())
        config = LoraConfig(target_modules=["linear"], r=2, lora_alpha=4, lora_dropout=0.0,
                            bias="none", revision=revision)
        model = get_peft_model(TinyLinear(), config).eval()
        model.peft_config["default"].base_model_name_or_path = BASE_ID
        with torch.no_grad():
            model.base_model.model.linear.lora_A["default"].weight.fill_(0.25)
            model.base_model.model.linear.lora_B["default"].weight.fill_(0.5)
        sample = torch.tensor([[0.25, -0.5, 1.0], [1.0, 0.0, -0.25]])
        with torch.no_grad():
            reference = model(sample).clone()
        wrapped = FSDP(model, device_id=torch.device("cpu"), use_orig_params=True)
        local_shards = {n: p.detach().clone() for n, p in wrapped.module.named_parameters()}
        ab = {n: list(p.shape) for n, p in wrapped.module.named_parameters()
              if ".lora_A." in n or ".lora_B." in n}
        require(len(ab) == 2 and all(len(shape) == 1 for shape in ab.values()), "FSDP_NOT_SHARDED")
        (root / f"ready-{rank}").write_text("FSDP_CPU_SHARDED", encoding="utf-8")
        dist.barrier()
        if mode == "loss":
            if rank == 0:
                os._exit(17)  # controlled abrupt death of this owned fixture worker only
            dist.barrier()
            raise EvaluationError("RANK_LOSS_DID_NOT_PROPAGATE")
        if mode in {"stall", "cancel"}:
            while True:
                time.sleep(1)

        def export(directory: Path):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                wrapped.module.save_pretrained(str(directory), safe_serialization=True,
                                               save_embedding_layers=False)
            notices = [w for w in caught if issubclass(w.category, PeftWarning) and
                       "DeepSpeed ZeRO-3 / FSDP shards" in str(w.message)]
            raw = bounded_read(directory / "adapter_model.safetensors", MAX_BYTES)
            cfg = bounded_read(directory / "adapter_config.json", MAX_HEADER)
            binding = {"artifactSha256": sha256(raw), "configSha256": sha256(cfg),
                       "baseModelId": BASE_ID, "baseModelRevision": revision,
                       "peftRevision": PEFT_REVISION, "expectedTensorShapes": SHAPES,
                       "stateDictProvenance": "INFERRED"}
            return raw, inspect_adapter(raw, cfg, binding), notices

        ungathered, bad_verdict, notices = export(rank_root / "ungathered")
        require(bool(notices) and bad_verdict["state"] == "REJECTED" and
                bad_verdict["reason"] == "UNGATHERED_AB_SHAPE", "UNGATHERED_NOT_REJECTED")
        require(any(t.ndim < 2 or t.numel() == 0 for t in load(ungathered).values()), "NO_REAL_SHARD_BYTES")
        with FSDP.summon_full_params(wrapped, writeback=False, rank0_only=False):
            gathered, good_verdict, notices = export(rank_root / "gathered")
            require(not notices and good_verdict["state"] == "PASS", "GATHERED_EXPORT_FAILED")
            observed_shapes = {k: list(v.shape) for k, v in load(gathered).items()}
            require(observed_shapes == SHAPES, "GATHERED_SHAPE_MISMATCH")
        current_shards = dict(wrapped.module.named_parameters())
        require(set(current_shards) == set(local_shards), "SHARD_INVENTORY_CHANGED")
        for name, value in local_shards.items():
            require(torch.equal(value, current_shards[name].detach()), "SHARD_RESTORATION_FAILED")
        base = TinyLinear()
        base.load_state_dict(load(base_bytes), strict=True)
        reloaded = PeftModel.from_pretrained(base, str(rank_root / "gathered"),
                                             local_files_only=True, is_trainable=False).eval()
        with torch.no_grad():
            actual = reloaded(sample)
        torch.testing.assert_close(reference, actual, rtol=0, atol=0)
        require(torch.isfinite(actual).all().item(), "NONFINITE_RELOAD")
        require(bounded_read(rank_root / "gathered" / "adapter_model.safetensors", MAX_BYTES) == gathered,
                "RELOAD_MUTATED_BYTES")
        report = {"rank": rank, "worldSize": 2, "sourceRevision": revision,
                  "upstreamRevision": PEFT_REVISION, "baseSha256": sha256(base_bytes),
                  "shardedShapes": ab, "ungatheredSha256": sha256(ungathered),
                  "ungatheredWarnedAndWritten": True, "ungatheredRejected": True,
                  "gatheredSha256": sha256(gathered), "gatheredShapes": observed_shapes,
                  "gatheredStructurePass": True, "peftReloadPass": True,
                  "maxAbsoluteError": float((reference - actual).abs().max()),
                  "shardsRestoredExactly": True, "adapterBytesUnchanged": True}
        (root / f"result-{rank}.json").write_text(json.dumps(report, sort_keys=True, allow_nan=False), encoding="utf-8")
        dist.barrier()
    finally:
        dist.destroy_process_group()


def run_group(revision: str, mode: str) -> list[dict[str, Any]] | dict[str, Any]:
    context = multiprocessing.get_context("spawn")
    with tempfile.TemporaryDirectory(prefix="szl-fsdp-cpu-") as temporary:
        root = Path(temporary)
        processes = [context.Process(target=_worker, args=(rank, temporary, revision, mode)) for rank in (0, 1)]
        started = []
        try:
            for process in processes:
                process.start()
                started.append(process)
            def ready():
                return all((root / f"ready-{rank}").is_file() for rank in (0, 1))
            try:
                supervise(processes, timeout=120, ready=ready, cancel=mode == "cancel", stall=mode == "stall")
            except EvaluationError as exc:
                wanted = {"loss": "WORKER_NONZERO", "stall": "GROUP_TIMEOUT", "cancel": "GROUP_CANCELLED"}
                require(mode in wanted and str(exc) == wanted[mode] and ready(), "UNEXPECTED_GROUP_FAILURE")
                if mode == "loss":
                    require(processes[0].exitcode == 17, "INJECTED_LOSS_NOT_OBSERVED")
                require(not any((root / f"result-{rank}.json").exists() for rank in (0, 1)),
                        "FAILED_GROUP_PRODUCED_SUCCESS")
                return {"mode": mode, "rejectedAs": wanted[mode], "fsdpReadyBeforeInjection": True,
                        "allWorkersStopped": all(not p.is_alive() for p in processes), "successReports": 0}
            require(mode == "normal", "INJECTED_GROUP_DID_NOT_FAIL")
            return [json.loads(bounded_read(root / f"result-{rank}.json", 65536)) for rank in (0, 1)]
        finally:
            for process in started:
                if process.is_alive():
                    process.kill()
                process.join(timeout=2)


def validate_ranks(rows: Any, revision: str) -> None:
    require(type(rows) is list and len(rows) == 2, "RANK_EVIDENCE_MISSING")
    require(all(type(row) is dict for row in rows), "RANK_RECORD_INVALID")
    require({row.get("rank") for row in rows} == {0, 1}, "RANK_IDENTITY_MISMATCH")
    for row in rows:
        require(type(row.get("rank")) is int and type(row.get("worldSize")) is int and row["worldSize"] == 2 and
                row.get("sourceRevision") == revision and row.get("upstreamRevision") == PEFT_REVISION,
                "RANK_SOURCE_MISMATCH")
        for field in ("ungatheredWarnedAndWritten", "ungatheredRejected", "gatheredStructurePass",
                      "peftReloadPass", "shardsRestoredExactly", "adapterBytesUnchanged"):
            require(row.get(field) is True, "RANK_GATE_FAILED")
        require(type(row.get("maxAbsoluteError")) is float and row["maxAbsoluteError"] == 0.0,
                "RANK_PARITY_FAILED")
        require(row.get("gatheredShapes") == SHAPES, "RANK_GATHERED_SHAPES")
        shards = row.get("shardedShapes")
        require(type(shards) is dict and set(shards) == {
            "base_model.model.linear.lora_A.default.weight", "base_model.model.linear.lora_B.default.weight"
        }, "RANK_SHARDED_INVENTORY")
        require(all(type(shape) is list and len(shape) == 1 and type(shape[0]) is int and
                    0 <= shape[0] <= 26 for shape in shards.values()), "RANK_NOT_SHARDED")
        for field in ("baseSha256", "gatheredSha256", "ungatheredSha256"):
            require(type(row.get(field)) is str and re.fullmatch(r"[0-9a-f]{64}", row[field]) is not None,
                    "RANK_DIGEST_INVALID")
        require(row["ungatheredSha256"] != row["gatheredSha256"], "SHARDED_BYTES_NOT_DISTINCT")
    require(rows[0]["gatheredSha256"] == rows[1]["gatheredSha256"] and
            rows[0]["baseSha256"] == rows[1]["baseSha256"], "CROSS_RANK_BYTES_DISAGREE")


def main() -> int:
    try:
        revision = source_revision()
        dependencies = check_installations()
        with patch.dict(os.environ, {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                                     "HF_HUB_DISABLE_TELEMETRY": "1", "GLOO_SOCKET_IFNAME": "lo"}):
            rows = run_group(revision, "normal")
            validate_ranks(rows, revision)
            failures = [run_group(revision, mode) for mode in ("loss", "stall", "cancel")]
        report = {"schema": "szl.forge.peft-fsdp-cpu.v1", "state": "FSDP_CPU_EVALUATION_PASS",
                  "sourceRevision": revision, "upstreamRevision": PEFT_REVISION,
                  "dependencies": dependencies, "worldSize": 2, "device": "cpu", "backend": "gloo",
                  "evidenceClass": "UNSIGNED_EXECUTION_REQUIRES_OUTER_CI_BINDING",
                  "baseKind": "SYNTHETIC_RECIPE_NOT_HUB_MODEL", "ranks": rows, "failureTests": failures,
                  "productionAuthorized": False, "publicationAuthorized": False, "automaticPromotion": False,
                  "providerWrites": 0, "pretrainedWeightsDownloaded": False, "trainingExecuted": False,
                  "productionDisposition": "HOLD",
                  "remaining": ["GPU_AND_ZERO3_NOT_MEASURED", "AUTHENTICATED_EXTERNAL_MODEL_LINEAGE",
                                "IMMUTABLE_PROVIDER_READBACK", "MODEL_QUALITY_QUALIFICATION",
                                "INDEPENDENT_PUBLISHER_ADMISSION", "PRODUCTION_ROLLBACK_NOT_RUN"]}
        print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))
        return 0
    except Exception as exc:
        reason = str(exc) if isinstance(exc, EvaluationError) else type(exc).__name__
        print(json.dumps({"state": "BLOCKED", "reason": reason, "productionDisposition": "HOLD",
                          "productionAuthorized": False, "publicationAuthorized": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
