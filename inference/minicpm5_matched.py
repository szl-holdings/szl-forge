"""Same-worker F16/Q4 synthetic comparison; never production authorization.

Copyright 2026 SZL Holdings. SPDX-License-Identifier: Apache-2.0
Reuse the admitted GGUF evaluator and unchanged original suite. Default is plan;
only explicit run starts two bounded, sequential, ephemeral CPU evaluations.
"""
from __future__ import annotations

import argparse
import base64
import os
import platform
import secrets
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from inference import minicpm5_gguf as g

ROOT = Path(__file__).resolve().parents[1]
GGUF_RUNNER_SHA = "e94f91230c25e712ae77b076a3bfabe60704ae70b244c8c8b12fb781dfa9a573"
ORDER = ("F16", "Q4_K_M")
MAX_SECONDS = 780
BOUNDARIES = {"productionDisposition": "HOLD", "modelOperational": False,
    "runtimeQualified": False, "publicationEligible": False, "sealed": False,
    "trainingAuthorized": False, "toolExecuted": False}


class MatchedError(ValueError):
    """Incomplete or mismatched evidence does not establish a comparison."""


def plan() -> dict[str, Any]:
    return {"schema": "szl.forge.minicpm5-matched-plan.v1", "variants": list(ORDER),
        "modelRevision": g.REVISION, "runtimeArchiveSha256": g.RUNTIME["sha256"],
        "ggufRunnerSha256": GGUF_RUNNER_SHA, "suiteSha256": g.SUITE_SHA,
        "caseCountPerVariant": 12, "maximumEvaluations": 2,
        "maxWallSeconds": MAX_SECONDS, "requestedJobHardLimitSeconds": 900,
        "automaticExecution": False, "automaticRetry": False,
        "comparisonClass": "SAME_WORKER_RUNTIME_PUBLIC_SYNTHETIC",
        "authorityChain": ["GitHub", "Hugging Face", "a-11-oy.com", "a11oy.net"],
        "remainingLimits": ["single sequential pass; order and cold-start effects",
            "GGUF conversion and embedded tokenizer/template lineage not proven",
            "not a held-out benchmark or native tool-parser test",
            "no causal quantization or production speedup claim"], **BOUNDARIES}


def checked_digest(value: dict[str, Any]) -> str:
    if not isinstance(value, dict):
        raise MatchedError("evidence must be an object")
    unsigned = dict(value)
    claimed = unsigned.pop("recordSha256", None)
    if not isinstance(claimed, str) or not g.base.SHA64.fullmatch(claimed) or g.base.digest(unsigned) != claimed:
        raise MatchedError("record digest mismatch")
    return claimed


def validate_variant(report: dict[str, Any], variant: str, revision: str) -> None:
    checked_digest(report)
    if variant not in ORDER or report.get("variant") != variant:
        raise MatchedError("unexpected variant or order")
    for key, expected in {
        "schema": "szl.forge.minicpm5-gguf-result.v1", "plan": g.plan(),
        "sourceRepository": "szl-holdings/szl-forge", "sourceRevision": revision,
        "runnerSha256": GGUF_RUNNER_SHA, "modelSha256": g.FILES[variant][2],
        "runtimeArchiveSha256": g.RUNTIME["sha256"], "phase": "complete",
        "artifactBytesVerified": True, **BOUNDARIES,
    }.items():
        if report.get(key) != expected or type(report.get(key)) is not type(expected):
            raise MatchedError("variant binding mismatch: " + key)
    if not g.base.SHA64.fullmatch(str(report.get("serverBinarySha256", ""))):
        raise MatchedError("native executable identity missing")
    if report.get("errorType") or report.get("completedCases") != 12:
        raise MatchedError("incomplete variant cannot be compared")
    if g.summarize(report, g.baseline()) != report:
        raise MatchedError("variant summary differs from recorded observations")
    observed = report.get("sourceObservation")
    checked_digest(observed)
    for key, expected in {"modelId": g.MODEL, "revision": g.REVISION,
        "files": g.plan()["files"], "license": "apache-2.0",
        "evidenceClass": "METADATA_ONLY", "weightsDownloaded": False,
        "productionDisposition": "HOLD"}.items():
        if observed.get(key) != expected or type(observed.get(key)) is not type(expected):
            raise MatchedError("source observation mismatch: " + key)
    if type(report.get("peakResidentBytes")) is not int or report["peakResidentBytes"] <= 0:
        raise MatchedError("resident memory measurement missing")
    for case in report["cases"]:
        for key, upper in (("generatedTokens", 96), ("inputTokens", 4000)):
            if type(case.get(key)) is not int or not 1 <= case[key] <= upper:
                raise MatchedError("case token observation outside bounds")
        if type(case.get("reasoningProduced")) is not bool:
            raise MatchedError("missing reasoning-output observation")


def worker_identity() -> dict[str, Any]:
    """Hash host identifiers rather than exporting hostnames or machine IDs."""
    job = os.environ.get("JOB_ID", "")
    if not job or len(job) > 100:
        raise MatchedError("provider JOB_ID required for same-worker execution")
    cpu_info = Path("/proc/cpuinfo").read_text(encoding="utf-8")
    models = sorted({line.split(":", 1)[1].strip() for line in cpu_info.splitlines() if line.startswith("model name")})
    return {"jobId": job, "system": platform.system(), "machine": platform.machine(),
        "python": platform.python_version(), "cpuCount": os.cpu_count(), "cpuModels": models,
        "workerHash": g.base.digest({"boot": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                                     "hostname": platform.node()})}


def summarize_pair(reports: list[dict[str, Any]], context: dict[str, Any], revision: str) -> dict[str, Any]:
    if not g.base.SHA40.fullmatch(revision) or len(reports) != 2:
        raise MatchedError("exact source and two variants required")
    if context.get("evidenceClass") not in {"LIVE_SAME_WORKER", "OFFLINE_FIXTURE"}:
        raise MatchedError("comparison observation class required")
    workers = context.get("workerObservations")
    if not isinstance(workers, list) or len(workers) != 2 or not all(isinstance(w, dict) for w in workers) or workers[0] != workers[1]:
        raise MatchedError("worker identities changed or incomplete")
    worker = workers[0]
    if not worker.get("jobId") or not g.base.SHA64.fullmatch(str(worker.get("workerHash", ""))):
        raise MatchedError("provider job and hashed worker identity required")
    if not g.base.SHA64.fullmatch(str(context.get("protocolSha256", ""))):
        raise MatchedError("request protocol identity required")
    expected_protocol = g.base.digest([g.request_body(c) for c in g.base.suite()])
    if context["protocolSha256"] != expected_protocol:
        raise MatchedError("request protocol changed")
    if not g.base.SHA64.fullmatch(str(context.get("comparisonId", ""))):
        raise MatchedError("comparison identity missing")
    for report, variant in zip(reports, ORDER, strict=True):
        validate_variant(report, variant, revision)
        if report.get("jobId") != worker["jobId"]:
            raise MatchedError("variant belongs to a different provider job")
    for key in ("host", "serverBinarySha256", "runtimeArchiveSha256", "runnerSha256"):
        if reports[0].get(key) != reports[1].get(key):
            raise MatchedError("execution mismatch: " + key)
    actual_host = reports[0]["host"]
    for key in ("system", "machine", "python", "cpuCount"):
        if actual_host.get(key) != worker.get(key):
            raise MatchedError("variant host differs from worker observation")
    left, right = reports
    changes = [{"id": a["id"], "f16Passed": a["passed"], "q4Passed": b["passed"],
        "sameOutputHash": a["outputSha256"] == b["outputSha256"],
        "sameInputTokenCount": a["inputTokens"] == b["inputTokens"]}
        for a, b in zip(left["cases"], right["cases"], strict=True)]
    live = context["evidenceClass"] == "LIVE_SAME_WORKER"
    result = {"schema": "szl.forge.minicpm5-matched-result.v1", "plan": plan(),
        "status": "EXECUTED" if live else "OFFLINE_FIXTURE", "matchedExecutionObserved": live,
        "context": context, "sourceRepository": "szl-holdings/szl-forge", "sourceRevision": revision,
        "variantReports": reports, "caseComparison": changes,
        "f16PassedCases": left["passedCases"], "q4PassedCases": right["passedCases"],
        "bothSmokePassed": left["passedCases"] == right["passedCases"] == 12,
        "outputHashParity": all(c["sameOutputHash"] for c in changes),
        "verdictParity": all(c["f16Passed"] == c["q4Passed"] for c in changes),
        "q4RegressedCaseIds": [c["id"] for c in changes if c["f16Passed"] and not c["q4Passed"]],
        "q4RecoveredCaseIds": [c["id"] for c in changes if not c["f16Passed"] and c["q4Passed"]],
        "quantizationIsolationVerified": False, "conversionLineageVerified": False,
        "speedupClaim": None, "latencyComparisonClaim": "single sequential observation only",
        "observationIntegrity": "unsigned content-addressed; not independent certification", **BOUNDARIES}
    result["recordSha256"] = g.base.digest(result)
    return result


def validate_pair(record: dict[str, Any]) -> None:
    checked_digest(record)
    if summarize_pair(record["variantReports"], record["context"], record["sourceRevision"]) != record:
        raise MatchedError("pair verdict or binding was altered")


def execute_child(command: list[str], timeout: float) -> int:
    """Kill the entire owned session on timeout, including a native grandchild."""
    child = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.DEVNULL, start_new_session=True)
    try:
        return child.wait(timeout=timeout)
    finally:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        # The group may outlive its leader; cleanup must not depend on poll().
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait(timeout=5)


def run(revision: str, output: Path) -> int:
    initial = {"schema": "szl.forge.minicpm5-matched-incomplete.v1", "status": "INCOMPLETE",
               "plan": plan(), "matchedExecutionObserved": False, "completedVariants": [], **BOUNDARIES}
    g.base.atomic_json(output, initial)
    reports, workers = [], []
    try:
        if not g.base.SHA40.fullmatch(revision) or g.base.file_digest(Path(g.__file__)) != GGUF_RUNNER_SHA:
            raise MatchedError("source revision or admitted evaluator mismatch")
        g.baseline()
        start = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="szl-matched-") as directory:
            for variant in ORDER:
                workers.append(worker_identity())
                if len(workers) == 2 and workers[0] != workers[1]:
                    raise MatchedError("same-worker binding failed before second evaluation")
                path = Path(directory) / (variant + ".json")
                remaining = MAX_SECONDS - (time.monotonic() - start)
                if remaining <= 0:
                    raise TimeoutError("pair budget exhausted")
                code = execute_child([sys.executable, "-m", "inference.minicpm5_gguf", "run",
                    "--variant", variant, "--source-revision", revision, "--output", str(path)], remaining)
                report = g.loads(path.read_bytes())
                validate_variant(report, variant, revision)
                if code != (0 if report["status"] == "SMOKE_PASS" else 2):
                    raise MatchedError("child exit and evidence disagree")
                reports.append(report)
                initial["completedVariants"].append(variant)
                initial["variantReports"] = list(reports)
                g.base.atomic_json(output, initial)
        context = {"evidenceClass": "LIVE_SAME_WORKER", "comparisonId": secrets.token_hex(32),
            "workerObservations": workers, "protocolSha256": g.base.digest([g.request_body(c) for c in g.base.suite()])}
        result = summarize_pair(reports, context, revision)
        g.base.atomic_json(output, result)
        print("SZL_MATCHED_RECORD_BASE64=" + base64.b64encode(g.base.canonical(result)).decode(), flush=True)
        return 0 if result["bothSmokePassed"] and result["outputHashParity"] else 2
    except Exception as exc:
        initial.update(errorType=type(exc).__name__, variantReports=reports)
        g.base.atomic_json(output, initial)
        print("SZL_MATCHED_INCOMPLETE_BASE64=" + base64.b64encode(g.base.canonical(initial)).decode(), flush=True)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run", "verify"), nargs="?", default="plan")
    parser.add_argument("--source-revision")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--record", type=Path)
    args = parser.parse_args()
    if args.command == "run":
        if args.output is None or args.source_revision is None:
            parser.error("run requires --source-revision and --output")
        return run(args.source_revision, args.output)
    if args.command == "verify":
        if args.record is None:
            parser.error("verify requires --record")
        try:
            validate_pair(g.loads(args.record.read_bytes()))
        except (ValueError, TypeError, KeyError, OSError):
            return 1
        print("MATCHED_RECORD_INTEGRITY_OK production=HOLD unsigned=true")
        return 0
    value = plan()
    if args.output:
        g.base.atomic_json(args.output, value)
    print(g.base.canonical(value).decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
