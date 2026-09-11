"""Executable AsyncGRPO LoRA / vLLM evaluation lane.

Runs the admitted serving protocol against a job-local directory: adapter-only
versus merged transfer, load-before-evict, and the required negative paths.
Optionally prepares a job-local venv with the exact TRL pin and the recorded
PEFT/vLLM closure.

This module does not train, write the Hub, start a production server, or invent
CUDA numbers. GPU generation throughput stays UNAVAILABLE unless CUDA, vLLM,
PEFT, and the exact TRL pin are actually present on the host. Until that host
exists, overall disposition is UNAVAILABLE — a valid plan remains EVALUATION
only inside plan[].
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping

from frontier.asyncgrpo_lora_closure import (
    ADAPTER_ONLY_API_VERIFIED_AT_PIN,
    ADAPTER_ONLY_UPSTREAM_COMMIT,
    ADAPTER_ONLY_UPSTREAM_ISSUE,
    STABLE_BASELINE_REVISION,
)
from frontier.asyncgrpo_lora_contract import (
    DENIED_AUTHORITY,
    TRL_BASELINE_VERSION,
    TRL_REPOSITORY,
    TRL_REVISION,
    evaluate_plan,
)
from frontier.asyncgrpo_lora_gpu import (
    GPU_FIELDS,
    admit_gpu_measurements,
    gpu_blocker,
    try_gpu_negative_paths,
    try_gpu_sync_paths,
    unavailable_measurements,
)
from frontier.asyncgrpo_lora_install import prepare_job_local, probe_interpreter, read_direct_url
from frontier.asyncgrpo_lora_protocol import ProtocolServer, measure_sync_paths, run_negative_paths

SCHEMA = "szl.asyncgrpo-lora-vllm-evaluation/v1"
ALLOWED_DISPOSITIONS = frozenset({"UNAVAILABLE", "HOLD", "EVALUATION"})
REQUIRED_EVIDENCE = (
    "exact-trl-commit-install-receipt",
    "pep610-direct-url-provenance",
    "compatible-exact-peft-vllm-closure",
    "job-local-env-not-production-runtime",
    "transferred-sync-bytes",
    "sync-pause",
    "generation-throughput",
    "policy-version-correctness",
    "max-staleness",
    "adapter-rank-and-capacity",
    "loaded-evicted-adapter-identities",
    "checkpoint-presence",
    "resource-use",
    "adapter-only-vs-merged-equivalent-conditions",
    "vllm-without-lora-support",
    "runtime-adapter-update-unavailable",
    "unsupported-or-insufficient-max-rank",
    "insufficient-max-loras",
    "shared-path-unavailable",
    "path-escape-symlink",
    "partially-written-adapter",
    "load-failure",
    "load-before-evict-ordering",
    "server-restart",
    "serving-cache-eviction",
    "checkpoint-missing",
    "hybrid-recurrent-state-logprob-mismatch",
    "serving-cache-ephemeral-not-durable-artifact",
    "adapter-only-sync-api-at-pinned-trl-revision",
)
NEGATIVE_EVIDENCE = (
    "vllm-without-lora-support",
    "runtime-adapter-update-unavailable",
    "unsupported-or-insufficient-max-rank",
    "insufficient-max-loras",
    "shared-path-unavailable",
    "path-escape-symlink",
    "partially-written-adapter",
    "load-failure",
    "load-before-evict-ordering",
    "server-restart",
    "serving-cache-eviction",
    "checkpoint-missing",
    "hybrid-recurrent-state-logprob-mismatch",
    "serving-cache-ephemeral-not-durable-artifact",
)
DEFAULT_PLAN = {
    "lora_rank": 32,
    "max_lora_rank": 32,
    "max_loras": 6,
    "max_staleness": 4,
    "lora_server_enabled": True,
    "shared_storage_verified": True,
    "checkpoints_enabled": True,
    "adapter_servable": True,
}
MEASURED_FIELDS = GPU_FIELDS
SIBLING_HOLDS = (
    {"id": "ghlore", "owner": "szl-holdings/szl-forge#207", "disposition": "HOLD"},
    {"id": "harbor", "owner": "szl-holdings/szl-forge#217", "disposition": "HOLD"},
    {"id": "tau", "owner": "szl-holdings/szl-forge#206", "disposition": "HOLD"},
    {"id": "lyte", "owner": "szl-holdings/lyte-services#18", "disposition": "HOLD"},
    {"id": "browser-acceptance", "owner": "WP23", "disposition": "HOLD"},
)


class EvaluationRunnerError(ValueError):
    """Host probe or measurement admission failed closed."""


def package_present(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def cuda_available() -> bool:
    if not package_present("torch"):
        return False
    try:
        import torch
    except Exception:
        return False
    available = getattr(getattr(torch, "cuda", None), "is_available", None)
    if not callable(available):
        return False
    try:
        value = available()
    except Exception:
        return False
    return value is True


def probe_host(python: str | None = None) -> dict[str, object]:
    if python:
        probed = probe_interpreter(python)
        trl_rev = None
        trl_info = probed.get("trl")
        if isinstance(trl_info, dict):
            rev = trl_info.get("directUrlRevision")
            trl_rev = rev if type(rev) is str else None
        return {
            "trlPresent": probed.get("trlPresent") is True,
            "peftPresent": probed.get("peftPresent") is True,
            "vllmPresent": probed.get("vllmPresent") is True,
            "torchPresent": probed.get("torchPresent") is True,
            "cudaAvailable": probed.get("cudaAvailable") is True,
            "trlInstalledRevision": trl_rev,
            "exactTrlRevisionMatched": trl_rev == TRL_REVISION,
            "adapterOnlySyncApiAssumed": False,
            "adapterOnlyApiAtPinnedRevision": ADAPTER_ONLY_API_VERIFIED_AT_PIN,
            "adapterOnlyHardwareExecuted": False,
            "probedPython": python,
        }
    trl = read_direct_url("trl")
    return {
        "trlPresent": package_present("trl"),
        "peftPresent": package_present("peft"),
        "vllmPresent": package_present("vllm"),
        "torchPresent": package_present("torch"),
        "cudaAvailable": cuda_available(),
        "trlInstalledRevision": trl.get("directUrlRevision"),
        "exactTrlRevisionMatched": trl.get("directUrlRevision") == TRL_REVISION,
        "adapterOnlySyncApiAssumed": False,
        "adapterOnlyApiAtPinnedRevision": ADAPTER_ONLY_API_VERIFIED_AT_PIN,
        "adapterOnlyHardwareExecuted": False,
        "probedPython": None,
    }


def hardware_ready(probe: Mapping[str, object]) -> bool:
    return gpu_blocker(probe) is None


def admit_measurements(probe: Mapping[str, object], measurements: object) -> dict[str, object]:
    try:
        return admit_gpu_measurements(probe, measurements)
    except Exception as exc:
        raise EvaluationRunnerError(str(exc)) from exc


def _evidence_map(present: set[str]) -> list[dict[str, object]]:
    return [{"id": item, "present": item in present} for item in REQUIRED_EVIDENCE]


def evaluate_host(
    *,
    plan: Mapping[str, Any] | None = None,
    measurements: Mapping[str, object] | None = None,
    output_dir: Path | None = None,
    run_protocol: bool = True,
    install: bool = False,
    execute_gpu: bool = True,
) -> dict[str, object]:
    """Probe the host, run the job-local protocol, and emit a fail-closed receipt."""
    if type(run_protocol) is not bool or type(install) is not bool or type(execute_gpu) is not bool:
        raise EvaluationRunnerError("runner flags must be explicit booleans")
    probe = probe_host()
    plan_result = evaluate_plan(**dict(DEFAULT_PLAN if plan is None else plan))
    reasons: list[str] = list(plan_result["reasons"])
    if probe["trlPresent"] is not True:
        reasons.append("trl_uninstalled")
    elif probe["exactTrlRevisionMatched"] is not True:
        reasons.append("trl_revision_unmatched")
    if probe["peftPresent"] is not True:
        reasons.append("peft_uninstalled")
    if probe["vllmPresent"] is not True:
        reasons.append("vllm_uninstalled")
    if probe["cudaAvailable"] is not True:
        reasons.append("cuda_unavailable")
    if ADAPTER_ONLY_API_VERIFIED_AT_PIN is not True:
        reasons.append("adapter_only_sync_api_unverified_at_pin")
    if measurements is None:
        reasons.append("measured_comparison_absent")

    gpu_measured = admit_measurements(probe, measurements)

    owned_tmp: tempfile.TemporaryDirectory[str] | None = None
    if output_dir is None:
        owned_tmp = tempfile.TemporaryDirectory(prefix="asyncgrpo-job-local-")
        workspace = Path(owned_tmp.name)
    else:
        if not isinstance(output_dir, Path):
            raise EvaluationRunnerError("output_dir must be a pathlib.Path")
        workspace = output_dir

    present: set[str] = set()
    if ADAPTER_ONLY_API_VERIFIED_AT_PIN:
        present.add("adapter-only-sync-api-at-pinned-trl-revision")
    job_local: dict[str, object] | None = None
    protocol_measured: dict[str, object] | None = None
    negatives: dict[str, bool] = {}
    protocol_status = "SKIPPED"
    gpu_status = "MEASURED" if hardware_ready(probe) else "UNAVAILABLE"
    gpu_negatives = try_gpu_negative_paths(probe, workspace / "gpu-negatives", execute=False)

    try:
        if run_protocol:
            job_local = prepare_job_local(workspace, install=install)
            present.add("job-local-env-not-production-runtime")
            if job_local.get("venvPython"):
                probe = probe_host(str(job_local["venvPython"]))
                for code, flag in (
                    ("trl_uninstalled", "trlPresent"),
                    ("peft_uninstalled", "peftPresent"),
                    ("vllm_uninstalled", "vllmPresent"),
                    ("cuda_unavailable", "cudaAvailable"),
                    ("trl_revision_unmatched", "exactTrlRevisionMatched"),
                ):
                    if probe.get(flag) is True and code in reasons:
                        reasons.remove(code)
            closure = job_local["closure"]
            if isinstance(closure, dict):
                trl_info = closure.get("trl")
                if isinstance(trl_info, dict) and trl_info.get("exact"):
                    present.add("exact-trl-commit-install-receipt")
                if isinstance(trl_info, dict) and trl_info.get("pep610"):
                    present.add("pep610-direct-url-provenance")
                if closure.get("compatibleExactPeftVllmClosure"):
                    present.add("compatible-exact-peft-vllm-closure")

            server = ProtocolServer(
                workspace / "protocol",
                enable_lora=True,
                max_lora_rank=int(DEFAULT_PLAN["max_lora_rank"]),
                max_loras=int(DEFAULT_PLAN["max_loras"]),
                max_staleness=int(DEFAULT_PLAN["max_staleness"]),
                runtime_updates=True,
                checkpoints_enabled=True,
            )
            protocol_measured = measure_sync_paths(
                server,
                name="policy",
                rank=int(DEFAULT_PLAN["lora_rank"]),
                version=3,
            )
            present.update(
                {
                    "transferred-sync-bytes",
                    "sync-pause",
                    "policy-version-correctness",
                    "max-staleness",
                    "adapter-rank-and-capacity",
                    "loaded-evicted-adapter-identities",
                    "checkpoint-presence",
                    "resource-use",
                    "adapter-only-vs-merged-equivalent-conditions",
                }
            )
            negatives = run_negative_paths(workspace / "negatives")
            for code, ok in negatives.items():
                evidence_id = code.replace("_", "-")
                if ok and evidence_id in NEGATIVE_EVIDENCE:
                    present.add(evidence_id)
            protocol_status = "MEASURED"

        gpu_python = str(job_local.get("venvPython")) if isinstance(job_local, dict) and job_local.get("venvPython") else None
        gpu_attempt = try_gpu_sync_paths(
            probe,
            workspace / "gpu",
            execute=execute_gpu,
            venv_python=gpu_python,
        )
        gpu_negatives = try_gpu_negative_paths(
            probe,
            workspace / "gpu-negatives",
            execute=execute_gpu,
            venv_python=gpu_python,
        )
        if gpu_attempt.get("executed") is True:
            gpu_measured = admit_measurements(probe, gpu_attempt)
            gpu_status = "MEASURED"
            present.add("generation-throughput")
            probe = dict(probe)
            probe["adapterOnlyHardwareExecuted"] = True
            if "measured_comparison_absent" in reasons:
                reasons.remove("measured_comparison_absent")
        else:
            gpu_status = "UNAVAILABLE"
            if "gpu_generation_throughput_unavailable" not in reasons:
                reasons.append("gpu_generation_throughput_unavailable")

        if hardware_ready(probe) and gpu_measured.get("executed") is True and not plan_result["reasons"]:
            disposition = "EVALUATION"
        elif hardware_ready(probe):
            disposition = "HOLD"
        else:
            disposition = "UNAVAILABLE"

        if disposition not in ALLOWED_DISPOSITIONS:
            raise EvaluationRunnerError("disposition escaped the allowed set")

        missing = [item for item in REQUIRED_EVIDENCE if item not in present]
        return {
            "schema": SCHEMA,
            "sourceRepository": TRL_REPOSITORY,
            "sourceRevision": TRL_REVISION,
            "stableBaselineVersion": TRL_BASELINE_VERSION,
            "stableBaselineRevision": STABLE_BASELINE_REVISION,
            "upstreamAdapterOnlyCommit": ADAPTER_ONLY_UPSTREAM_COMMIT,
            "upstreamAdapterOnlyIssue": ADAPTER_ONLY_UPSTREAM_ISSUE,
            "probe": probe,
            "plan": plan_result,
            "jobLocal": job_local,
            "protocolMeasured": protocol_measured,
            "measured": {field: gpu_measured.get(field, "UNAVAILABLE") for field in GPU_FIELDS},
            "gpuAttempt": gpu_attempt,
            "negativePathsObserved": negatives,
            "gpuNegativePaths": gpu_negatives,
            "protocolStatus": protocol_status,
            "gpuStatus": gpu_status,
            "requiredEvidence": list(REQUIRED_EVIDENCE),
            "requiredEvidenceDetail": _evidence_map(present),
            "satisfiedEvidence": sorted(present),
            "missingEvidence": missing,
            "siblingHolds": list(SIBLING_HOLDS),
            "reasons": reasons,
            "disposition": disposition,
            "evaluationOnly": True,
            "productionEligible": False,
            "ciGreenClaimed": False,
            "allDone": False,
            "ato": False,
            "live": False,
            **DENIED_AUTHORITY,
        }
    finally:
        if owned_tmp is not None:
            owned_tmp.cleanup()


def write_receipt(path: Path, receipt: Mapping[str, object]) -> Path:
    if not isinstance(path, Path):
        raise EvaluationRunnerError("receipt path must be a pathlib.Path")
    if receipt.get("disposition") not in ALLOWED_DISPOSITIONS:
        raise EvaluationRunnerError("refusing to persist an illegal disposition")
    if receipt.get("productionTrainingAuthorized") or receipt.get("productionServingAuthorized"):
        raise EvaluationRunnerError("refusing to persist a production-authority claim")
    if receipt.get("allDone") or receipt.get("ato") or receipt.get("live"):
        raise EvaluationRunnerError("refusing to persist an ALL_DONE/ATO/LIVE stamp")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--skip-protocol", action="store_true")
    parser.add_argument("--receipt", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    receipt = evaluate_host(
        output_dir=args.output_dir,
        install=args.install,
        run_protocol=not args.skip_protocol,
    )
    if args.receipt is not None:
        write_receipt(args.receipt, receipt)
    payload = receipt if args.json else {
        "disposition": receipt["disposition"],
        "planDisposition": receipt["plan"]["disposition"] if isinstance(receipt["plan"], dict) else None,
        "protocolStatus": receipt["protocolStatus"],
        "gpuStatus": receipt["gpuStatus"],
        "reasons": receipt["reasons"],
        "sourceRevision": receipt["sourceRevision"],
        "productionServingAuthorized": receipt["productionServingAuthorized"],
        "satisfiedEvidence": receipt["satisfiedEvidence"],
        "missingEvidence": receipt["missingEvidence"],
        "ciGreenClaimed": receipt["ciGreenClaimed"],
        "allDone": receipt["allDone"],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
