"""Execute the installed HF tooling lanes; never promote a model or production lock.

Copyright 2026 SZL Holdings. SPDX-License-Identifier: Apache-2.0
Use `python -m tools.evaluate_hf_tooling plan` without optional dependencies.
Only `run` imports the explicitly installed, exact-source evaluation libraries.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import importlib.metadata as metadata
import inspect
import json
import platform
import socket
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from inference.hf_frontier import AUTHORITY_CHAIN, sha256, strict_json, write_manifest
from inference.hf_tooling import (
    AUTHORITY, LANES, RELEASES, ToolingError, authorized_catalog, hex_digest,
    job_labels, require_install, requirements, sft_config, tau_evidence_entries,
)

ROOT = Path(__file__).resolve().parents[1]
CHECKS = {
    "hub": ("sandbox_label_validation", "filesystem_path_boundary", "httpx_exception_contract"),
    "trl": ("chunked_nll_loss_gradient_parity", "long_context_config_only",
            "synthetic_generation_offline", "accelerate_cpu_checkpoint_roundtrip"),
    "tau": ("custom_message_label_roundtrip", "legacy_compaction_replay",
            "malformed_session_rejected", "catalog_does_not_grant_authority"),
}
REMAINING = {
    "hub": ("billable_sandbox_job_roundtrip", "injected_header_timeout_resume",
            "mutable_ref_atomic_race", "production_publisher_regression"),
    "trl": ("million_token_training", "gpu_tensor_core_throughput", "fused_dpo_kto_grpo_parity",
            "distributed_vllm_weight_sync", "fsdp2_offload", "receipt_agent_quality_bakeoff"),
    "tau": ("provider_streaming_negative_paths", "zai_serialization", "shell_lifecycle_isolation",
            "real_agent_task_baseline", "tenant_authorized_host_integration"),
}
PUBLIC_REPO = "gdiamos/amx-reasoning-v1-instruct"
PUBLIC_REVISION = "b144ee0138929f0181b9219177f98fc7c8d259c9"
PUBLIC_FILE = "README.md"


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ToolingError(message)


@contextmanager
def offline_network():
    """Accidental Python socket egress trap, NOT an OS security sandbox."""
    def denied(*args, **kwargs):
        raise ToolingError("unexpected_network_attempt")
    with ExitStack() as stack:
        for owner, name in ((socket.socket, "connect"), (socket.socket, "connect_ex"),
                            (socket, "create_connection"), (socket, "getaddrinfo")):
            stack.enter_context(patch.object(owner, name, denied))
        yield


def source_revision() -> str:
    # Do not label dirty tracked source as an exact-source run. Output artifacts
    # are untracked; only the CI checkout and outer artifact authenticate a run.
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True, timeout=10).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                    cwd=ROOT, text=True, timeout=10)
    check(not dirty.strip(), "tracked_source_is_dirty")
    return hex_digest(revision)


def hub_labels() -> dict[str, Any]:
    from huggingface_hub import Sandbox
    from huggingface_hub._sandbox import RESERVED_SANDBOX_LABELS

    class ReachedApi(Exception):
        pass

    check("labels" in inspect.signature(Sandbox.create).parameters, "missing_labels_argument")
    labels = job_labels("fixture-run-42", "a" * 40)
    invalid = [[], {"test": 7}] + [{key: "override"} for key in RESERVED_SANDBOX_LABELS]
    with patch("huggingface_hub._sandbox.HfApi", side_effect=ReachedApi) as api:
        for value in invalid:
            try:
                Sandbox.create(labels=value)
            except ValueError:
                pass
            else:
                raise ToolingError("invalid_labels_accepted")
        check(api.call_count == 0, "invalid_label_reached_provider")
        try:
            Sandbox.create(labels=labels)
        except ReachedApi:
            pass
        else:
            raise ToolingError("positive_labels_did_not_reach_api_boundary")
        check(api.call_count == 1, "incorrect_api_boundary_count")
    return {"invalidCases": len(invalid), "validLabelsAcceptedBeforeApi": True,
            "providerCallsExecuted": 0, "labels": labels}


def hub_paths() -> dict[str, Any]:
    from huggingface_hub import HfFileSystem

    cases = ("../outside.txt", "folder/../../outside.txt", "folder/..\\..\\outside.txt")
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        fs = HfFileSystem(token=False, skip_instance_cache=True)
        for name in cases:
            destination = root / "must-not-exist" / "file.txt"
            with patch.object(fs, "resolve_path", return_value=SimpleNamespace(path=name)):
                try:
                    fs.get_file("fixture/model/path", str(destination))
                except ValueError:
                    pass
                else:
                    raise ToolingError("unsafe_filename_was_accepted")
            check(not destination.parent.exists(), "local_write_before_filename_validation")
    return {"unsafeCasesRejectedBeforeWrite": len(cases), "platform": platform.system(),
            "windowsStylePathCovered": True}


def hub_httpx() -> dict[str, Any]:
    from huggingface_hub.utils import get_session, httpx
    import httpx as underlying

    check(httpx.HTTPError is underlying.HTTPError, "httpx_exception_reexport_mismatch")
    check(callable(get_session), "hub_session_factory_missing")
    return {"sharedExceptionIdentity": True, "networkCalls": 0}


def verify_dry_run_payload(destination: Path) -> list[str]:
    """A dry run may create SDK bookkeeping, but must never copy payload bytes.

    Hub 1.31's local-folder helper creates CACHEDIR.TAG, .gitignore and lock
    files while resolving paths. The regression fixed by upstream #4817 is
    copying the cached requested file before the dry-run return, not those
    bookkeeping writes. Only the inspected SDK bookkeeping is admitted here.
    """
    allowed_files = {
        ".cache/huggingface/.gitignore", ".cache/huggingface/.gitignore.lock",
        ".cache/huggingface/CACHEDIR.TAG", ".cache/huggingface/download/README.md.lock",
    }
    allowed_directories = {".cache", ".cache/huggingface", ".cache/huggingface/download"}
    check(not destination.is_symlink(), "dry_run_root_is_symlink")
    if not destination.exists():
        return []
    files = []
    for index, path in enumerate(destination.rglob("*")):
        check(index < 16 and not path.is_symlink(), "dry_run_inventory_not_admitted")
        name = path.relative_to(destination).as_posix()
        if path.is_dir():
            check(name in allowed_directories, "dry_run_unexpected_directory")
        else:
            check(path.is_file() and name in allowed_files, "dry_run_copied_payload_or_unknown_file")
            check(path.stat().st_size <= 4096, "dry_run_bookkeeping_oversized")
            files.append(name)
    return sorted(files)


def hub_public_snapshot() -> dict[str, Any]:
    """Explicit, small README-only live probe. Never download weights or Python."""
    from huggingface_hub import snapshot_download

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        kwargs = {"repo_id": PUBLIC_REPO, "revision": PUBLIC_REVISION,
                  "allow_patterns": [PUBLIC_FILE], "token": False,
                  "cache_dir": str(root / "cache"), "max_workers": 1}
        planned = snapshot_download(**kwargs, dry_run=True)
        check(len(planned) == 1, "unexpected_download_inventory")
        file = planned[0]
        check(file.filename == PUBLIC_FILE and file.commit_hash == PUBLIC_REVISION,
              "download_identity_mismatch")
        check(type(file.file_size) is int and 0 < file.file_size <= 262144,
              "readme_download_size_not_admitted")

        def download(_: int) -> str:
            directory = Path(snapshot_download(**kwargs))
            path = directory / PUBLIC_FILE
            check(path.resolve().is_relative_to(root.resolve()), "cache_escaped_temporary_root")
            with path.open("rb") as stream:
                body = stream.read(262145)
            check(len(body) == file.file_size, "download_size_mismatch")
            return hashlib.sha256(body).hexdigest()

        with ThreadPoolExecutor(max_workers=3) as pool:
            hashes = list(pool.map(download, range(3)))
        check(len(set(hashes)) == 1, "concurrent_cache_bytes_disagree")
        destination = root / "dry-run-only"
        snapshot_download(**kwargs, dry_run=True, local_dir=destination)
        bookkeeping = verify_dry_run_payload(destination)
        return {"repoId": PUBLIC_REPO, "revision": PUBLIC_REVISION, "file": PUBLIC_FILE,
                "fileBytes": file.file_size, "fileSha256": hashes[0], "concurrentReaders": 3,
                "dryRunCopiedNoPayload": True, "dryRunBookkeepingFiles": bookkeeping,
                "weightDownload": False}


def trl_loss_parity() -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional
    from trl.trainer.sft_trainer import _chunked_cross_entropy_loss

    torch.manual_seed(1742)
    torch.set_num_threads(1)
    loss_errors, gradient_errors = [], []
    for mode in ("shifted", "causal", "all_masked", "scaled_softcap"):
        for chunk_size in (2, 16):
            hidden = torch.randn(2, 9, 7, requires_grad=True)
            weight = torch.randn(19, 7, requires_grad=True)
            bias = torch.randn(19, requires_grad=True)
            labels = torch.randint(0, 19, (2, 9))
            labels[:, 2:4] = -100
            if mode == "all_masked":
                labels.fill_(-100)
            scale, cap = (1.25, 2.0) if mode == "scaled_softcap" else (1.0, None)
            kwargs = {"labels" if mode == "causal" else "shift_labels": labels}
            loss, _, _, count = _chunked_cross_entropy_loss(
                hidden, weight, chunk_size, lm_head_bias=bias,
                logit_scale=scale, final_logit_softcapping=cap, **kwargs)
            h, y = (hidden[:, :-1], labels[:, 1:]) if mode == "causal" else (hidden, labels)
            logits = (h @ weight.t() + bias) * scale
            if cap is not None:
                logits = cap * torch.tanh(logits / cap)
            denominator = (y != -100).sum().clamp(min=1)
            reference = functional.cross_entropy(logits.reshape(-1, 19), y.reshape(-1),
                                                  ignore_index=-100, reduction="sum") / denominator
            torch.testing.assert_close(loss, reference, rtol=2e-5, atol=2e-6)
            check(int(count) == int((y != -100).sum()), "incorrect_valid_token_count")
            actual_grad = torch.autograd.grad(loss, (hidden, weight, bias), retain_graph=True)
            reference_grad = torch.autograd.grad(reference, (hidden, weight, bias))
            for actual, expected in zip(actual_grad, reference_grad, strict=True):
                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
                gradient_errors.append(float((actual - expected).abs().max().detach()))
            loss_errors.append(float((loss - reference).abs().detach()))
    return {"cases": len(loss_errors), "device": "cpu", "dtype": "float32",
            "maxLossAbsoluteError": max(loss_errors),
            "maxGradientAbsoluteError": max(gradient_errors),
            "performanceBenchmark": False}


def trl_context() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as temporary:
        config = sft_config(temporary, max_length=1_048_576, cpu=True)
        check(config.max_length == 1_048_576 and config.loss_type == "chunked_nll",
              "sft_config_contract_mismatch")
        check(config.trust_remote_code is False and config.packing is False and
              config.push_to_hub is False, "unsafe_sft_configuration")
    return {"configuredMaxLength": 1_048_576, "allocatedSequenceTokens": 0,
            "trainingExecuted": False, "capacityQualification": False}


def trl_generation() -> dict[str, Any]:
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(1742)
    config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                         max_position_embeddings=64, pad_token_id=0, bos_token_id=1, eos_token_id=2)
    model = LlamaForCausalLM(config).eval()
    prompt = torch.tensor([[1, 4, 6, 8]])
    with torch.no_grad():
        result = model.generate(prompt, max_new_tokens=2, do_sample=False,
                                attention_mask=torch.ones_like(prompt))
    check(result.shape[0] == 1 and 4 < result.shape[1] <= 6, "synthetic_generation_failed")
    return {"model": "random-local-one-layer-llama", "downloadedWeights": False,
            "promptTokens": 4, "generatedTokens": int(result.shape[1] - 4)}


def accelerate_checkpoint() -> dict[str, Any]:
    import torch
    from accelerate import Accelerator

    torch.manual_seed(1742)
    accelerator = Accelerator(cpu=True)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    model, optimizer = accelerator.prepare(model, optimizer)
    accelerator.backward(model(torch.ones(2, 3)).square().mean())
    optimizer.step()
    expected = {key: value.detach().clone() for key, value in model.state_dict().items()}
    with tempfile.TemporaryDirectory() as temporary:
        accelerator.save_state(temporary, safe_serialization=True)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(50)
        accelerator.load_state(temporary)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
    return {"singleProcessCpu": True, "optimizerSteps": 1, "weightsRestoredExactly": True,
            "fsdp2Qualification": False}


def tau_roundtrip() -> dict[str, Any]:
    from tau_agent.session import JsonlSessionStorage, LabelEntry, SessionState

    async def run(path: Path):
        entries = tau_evidence_entries(run_id="fixture-run-42", source_revision="a" * 40,
                                       receipt_sha256="b" * 64, timestamp=1742.0)
        await JsonlSessionStorage(path).append_batch(entries)
        loaded = await JsonlSessionStorage(path).read_all()
        state = SessionState.from_entries(loaded)
        check(state.labels_by_id.get(entries[0].id) == "SZL evidence", "bookmark_not_persisted")
        check(len(state.messages) == 1 and state.messages[0].details["evidenceIsAuthority"] is False,
              "custom_message_not_replayed")
        clear = LabelEntry(id="clear", parent_id=entries[-1].id, timestamp=1743.0,
                           target_id=entries[0].id, label=None)
        await JsonlSessionStorage(path).append(clear)
        state = SessionState.from_entries(await JsonlSessionStorage(path).read_all())
        check(entries[0].id not in state.labels_by_id, "bookmark_clear_not_persisted")
        return {"recordsPersisted": 3, "customMessageReplayed": True, "bookmarkClearReplayed": True,
                "sessionSha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    with tempfile.TemporaryDirectory() as temporary:
        return asyncio.run(run(Path(temporary) / "synthetic.jsonl"))


def tau_legacy() -> dict[str, Any]:
    from tau_agent.messages import UserMessage
    from tau_agent.session import (CompactionEntry, LeafEntry, MessageEntry, SessionState,
                                   entries_from_json_lines, entry_to_json_line)

    entries = [MessageEntry(id="m1", timestamp=1, message=UserMessage(content="first fixture")),
               MessageEntry(id="m2", parent_id="m1", timestamp=2,
                            message=UserMessage(content="keep this fixture")),
               CompactionEntry(id="c1", parent_id="m2", timestamp=3,
                               summary="fixture summary", replaces_entry_ids=["m1"]),
               LeafEntry(id="old-tip", timestamp=4, entry_id="m1")]
    restored = entries_from_json_lines([entry_to_json_line(entry) for entry in entries])
    state = SessionState.from_entries(restored)
    check(state.active_leaf_id == "c1" and len(state.messages) == 2,
          "legacy_compaction_or_leaf_replay_failed")
    check(state.messages[-1].content == "keep this fixture", "legacy_kept_message_missing")
    return {"legacyReplacementIdReplay": True, "obsoleteLeafCannotSelectTip": True,
            "fixtureSha256": sha256([entry.model_dump(mode="json") for entry in entries])}


def tau_malformed() -> dict[str, Any]:
    from tau_agent.session import SessionJsonlError, entry_from_json_line

    rejected = 0
    for text in ('{"type":"not-a-session-entry"}', '{"type":', '[]'):
        try:
            entry_from_json_line(text)
        except (SessionJsonlError, ValueError):
            rejected += 1
    check(rejected == 3, "malformed_session_was_accepted")
    return {"malformedEntriesRejected": rejected}


def tau_catalog() -> dict[str, Any]:
    result = authorized_catalog(["approved/model", "new/unapproved-model"], ["approved/model"])
    check(result == ["approved/model"], "discovery_created_permission")
    check(authorized_catalog(["new/model"], []) == [], "empty_allowlist_granted_permission")
    return {"boundary": "SZL host adapter", "liveCatalogContacted": False,
            "newModelsAutomaticallyAdmitted": False}


PROBES = {
    "hub": (hub_labels, hub_paths, hub_httpx),
    "trl": (trl_loss_parity, trl_context, trl_generation, accelerate_checkpoint),
    "tau": (tau_roundtrip, tau_legacy, tau_malformed, tau_catalog),
}


def evaluate(lane: str, revision: str, *, live_hub: bool = False) -> dict[str, Any]:
    check(lane in LANES and type(live_hub) is bool, "unknown_lane_or_mode")
    check(not live_hub or lane == "hub", "live_read_only_belongs_to_hub_lane")
    hex_digest(revision)
    sources, results = [], {}
    installation_error = None
    try:
        sources = [require_install(package) for package in LANES[lane]]
    except ToolingError as exc:
        installation_error = type(exc).__name__
    names = list(CHECKS[lane]) + (["public_readme_cache_roundtrip"] if live_hub else [])
    if installation_error:
        results = {name: {"status": "UNAVAILABLE", "reasonCode": "exact_install_missing_or_mismatched"}
                   for name in names}
    else:
        for name, probe in zip(CHECKS[lane], PROBES[lane], strict=True):
            try:
                with offline_network():
                    details = probe()
                results[name] = {"status": "PASS", "evidence": details}
            except Exception as exc:
                # Fixed synthetic probes only; exception classes are safe to publish.
                results[name] = {"status": "FAIL", "reasonCode": type(exc).__name__}
        if live_hub:
            try:
                results["public_readme_cache_roundtrip"] = {"status": "PASS", "evidence": hub_public_snapshot()}
            except Exception as exc:
                results["public_readme_cache_roundtrip"] = {"status": "FAIL", "reasonCode": type(exc).__name__}
    dependencies = sorted({(dist.metadata["Name"], dist.version) for dist in metadata.distributions()})
    report = {
        "schema": "szl.forge.hf-tooling-runtime.v1", "sourceRepository": "szl-holdings/szl-forge",
        "sourceRevision": revision, "authorityChain": list(AUTHORITY_CHAIN), "lane": lane,
        "observedAt": dt.datetime.now(dt.timezone.utc).isoformat(), "liveHubRequested": live_hub,
        "installationStatus": "UNAVAILABLE" if installation_error else "EXACT_SOURCE_VERIFIED",
        "sources": sources, "checks": results,
        "runtimeStatus": "UNAVAILABLE" if installation_error else
                         "SMOKE_PASS" if all(row["status"] == "PASS" for row in results.values()) else "SMOKE_FAIL",
        "environment": {"python": platform.python_version(), "platform": platform.system(),
                        "machine": platform.machine()},
        "dependencyInventory": [{"name": name, "version": version} for name, version in dependencies],
        "dependencyClosureFullyHashLocked": False,
        "remainingEvaluation": {name: "UNAVAILABLE" for name in REMAINING[lane]},
        "authority": dict(AUTHORITY), "productionDisposition": "HOLD",
        "sourceFilesSha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                              for name in ("inference/hf_tooling.py", "tools/evaluate_hf_tooling.py")},
    }
    if lane == "trl":
        from tools.hf_core_stack_guard import EXPECTED, REQUIRED_CHECKS, validate_evidence
        check(all(EXPECTED[key] == {"version": RELEASES[key]["version"],
                                    "revision": RELEASES[key]["revision"]} for key in EXPECTED),
              "existing_core_stack_pin_drift")
        # Do not certify the whole existing contract from narrower CPU smoke.
        report["existingCoreStackGate"] = validate_evidence({
            "releases": [{"package": key, **value} for key, value in EXPECTED.items()],
            "checks": {name: results[{
                "trl_chunked_nll": "chunked_nll_loss_gradient_parity",
                "trl_long_context_config": "long_context_config_only",
                "transformers_import_api": "synthetic_generation_offline",
                "accelerate_checkpoint_save_load": "accelerate_cpu_checkpoint_roundtrip",
            }[name]]["status"] if name in {"trl_chunked_nll", "trl_long_context_config",
                "transformers_import_api", "accelerate_checkpoint_save_load"} else "UNAVAILABLE"
                for name in REQUIRED_CHECKS},
        })
    if lane == "tau":
        contract = json.loads((ROOT / "frontier/tau_042_evaluation.json").read_text())
        check(contract["upstream_revision"] == RELEASES["tau-ai"]["revision"] and
              contract["production_eligible"] is False, "existing_tau_contract_drift")
        report["existingTauContractSha256"] = sha256(contract)
    report["reportSha256"] = sha256(report)
    return report


def verify_report(report: dict[str, Any], expected_source: str) -> None:
    """Check receipt integrity and bounded interpretation; not a signature verifier."""
    try:
        lane = report["lane"]
        check(lane in LANES and report["schema"] == "szl.forge.hf-tooling-runtime.v1", "wrong_schema")
        check(report["sourceRepository"] == "szl-holdings/szl-forge" and
              report["sourceRevision"] == hex_digest(expected_source), "wrong_source")
        check(report["reportSha256"] == sha256({k: v for k, v in report.items() if k != "reportSha256"}),
              "report_hash_mismatch")
        check(report["authorityChain"] == list(AUTHORITY_CHAIN) and
              report["productionDisposition"] == "HOLD" and
              sha256(report["authority"]) == sha256(AUTHORITY), "authority_changed")
        keys = {"schema", "sourceRepository", "sourceRevision", "authorityChain", "lane", "observedAt",
                "liveHubRequested", "installationStatus", "sources", "checks", "runtimeStatus",
                "environment", "dependencyInventory", "dependencyClosureFullyHashLocked",
                "remainingEvaluation", "authority", "productionDisposition", "sourceFilesSha256",
                "reportSha256"}
        keys |= {"existingCoreStackGate"} if lane == "trl" else {"existingTauContractSha256"} if lane == "tau" else set()
        check(set(report) == keys, "unexpected_report_fields")
        check(type(report["liveHubRequested"]) is bool and
              (lane == "hub" or report["liveHubRequested"] is False), "invalid_live_mode")
        names = set(report["checks"])
        required = set(CHECKS[lane]) | ({"public_readme_cache_roundtrip"} if report["liveHubRequested"] else set())
        check(names == required, "check_coverage_mismatch")
        check(report["dependencyClosureFullyHashLocked"] is False, "unearned_dependency_lock_claim")
        observed = dt.datetime.fromisoformat(report["observedAt"])
        check(observed.tzinfo is not None, "timestamp_is_naive")
        for key in ("inference/hf_tooling.py", "tools/evaluate_hf_tooling.py"):
            hex_digest(report["sourceFilesSha256"][key], 64)
        if lane == "tau":
            hex_digest(report["existingTauContractSha256"], 64)
        if lane == "trl":
            from tools.hf_core_stack_guard import DENIED_AUTHORITY, REQUIRED_CHECKS
            bound = report["existingCoreStackGate"]
            check(bound["disposition"] == "HOLD" and bound["allChecksPass"] is False and
                  sha256(bound["authority"]) == sha256(DENIED_AUTHORITY) and
                  set(bound["checks"]) == set(REQUIRED_CHECKS), "core_contract_changed")
        check(all(row["status"] in {"PASS", "FAIL", "UNAVAILABLE"} for row in report["checks"].values()),
              "invalid_status")
        if report["installationStatus"] == "EXACT_SOURCE_VERIFIED":
            expected = [{"package": name, **RELEASES[name]} for name in LANES[lane]]
            check(report["sources"] == expected, "installed_source_mismatch")
            status = "SMOKE_PASS" if all(row["status"] == "PASS" for row in report["checks"].values()) else "SMOKE_FAIL"
        else:
            check(report["installationStatus"] == "UNAVAILABLE" and not report["sources"] and
                  all(row["status"] == "UNAVAILABLE" for row in report["checks"].values()), "invalid_unavailable_report")
            status = "UNAVAILABLE"
        check(report["runtimeStatus"] == status, "false_runtime_status")
        check(report["remainingEvaluation"] == {name: "UNAVAILABLE" for name in REMAINING[lane]},
              "unmeasured_capability_claim")
    except (KeyError, TypeError, ValueError) as exc:
        raise ToolingError("runtime_receipt_failed_verification") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "requirements", "run", "verify"))
    parser.add_argument("--lane", choices=tuple(LANES), default="hub")
    parser.add_argument("--output", type=Path, default=Path("artifacts/hf-tooling/report.json"))
    parser.add_argument("--live-hub", action="store_true")
    parser.add_argument("--expected-source")
    args = parser.parse_args(argv)
    if args.action == "plan":
        print(json.dumps({"sources": RELEASES, "lanes": CHECKS,
                          "productionDisposition": "HOLD", "authority": AUTHORITY}, indent=2))
        return 0
    if args.action == "requirements":
        print(requirements(args.lane), end="")
        return 0
    if args.action == "verify":
        if args.expected_source is None:
            parser.error("verify requires --expected-source")
        with args.output.open("rb") as stream:
            report = strict_json(stream.read(2 * 1024 * 1024 + 1))
        verify_report(report, args.expected_source)
    else:
        report = evaluate(args.lane, source_revision(), live_hub=args.live_hub)
        verify_report(report, report["sourceRevision"])
        write_manifest(args.output, report)
    print(json.dumps({"runtimeStatus": report["runtimeStatus"], "checks": report["checks"],
                      "productionDisposition": "HOLD"}, indent=2))
    return 0 if report["runtimeStatus"] == "SMOKE_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
