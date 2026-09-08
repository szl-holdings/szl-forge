"""Exact-byte MiniCPM5 GGUF smoke evaluation. No publication or tool authority.

Copyright 2026 SZL Holdings. SPDX-License-Identifier: Apache-2.0
The default command emits a plan. Only explicit `run` downloads one GGUF and
starts a temporary, loopback-only llama.cpp process. Historical GPU results are
behavioral references, never a matched quantization or speedup experiment.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import platform
import secrets
import socket
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from inference import minicpm5_qualification as base

MODEL = "openbmb/MiniCPM5-2B-GGUF"
REVISION = "f0c9de8f8e1bbffc5abf14e64ccb61bb99e4219d"
FILES = {
    "Q4_K_M": ("MiniCPM5-2B-Q4_K_M.gguf", 1561318368, "ec2d5801640099e97d8d7e8003ad4d81f336e757811f03a26173dddf386602fd"),
    "Q8_0": ("MiniCPM5-2B-Q8_0.gguf", 2679710688, "c5415f8989bf88a8288f1b55a3cc371af53c07b0faa220a63bd7a990cfaba078"),
    "F16": ("MiniCPM5-2B-F16.gguf", 5039006688, "0ffba3682a853295566b98bc38c0ab755d6b2d91b994bc031ed727a08cfcdf25"),
}
RUNTIME = {
    "repository": "ggml-org/llama.cpp", "tag": "b10809",
    "revision": "5266f24da75dc449bd56cbed7addb9c8e4a6a73e",
    "filename": "llama-b10809-bin-ubuntu-x64.tar.gz", "size": 16734586,
    "sha256": "5e34434ddc6d03cd1584f403201aff0d4bd1a5793a72ff7e286532dfd1e4b941",
    "upstreamPrerelease": True,
}
BASE_RUNNER_SHA = "b1f750253004d43fd44ead3581245920867ee775b9388ff3eec002af80a29af7"
BASE_RECORD_SHA = "177c49709d7cc6f981bdda195bcea73febd8062335ecd9d3be4393b8095ed6b7"
SUITE_SHA = "0d548618deb523d44d56c5c87ab9d9a92eeea8af98bc5af23e00efd7704ae869"
ROOT = Path(__file__).resolve().parents[1]
BASE_RECORD = ROOT / "frontier/minicpm5/evidence/2026-09-07-a10g-smoke.json"
MAX_JSON = 2 * 1024**2
ALIAS = "szl-minicpm5-gguf-evaluation"


class GGUFError(ValueError):
    """Incomplete or inconsistent evidence never becomes a passing claim."""


def loads(raw: bytes) -> Any:
    if len(raw) > MAX_JSON:
        raise GGUFError("JSON byte budget exceeded")
    result = json.loads(raw, object_pairs_hook=base._pairs, parse_constant=base._constant)
    base.canonical(result)
    return result


def plan() -> dict[str, Any]:
    return {"schema": "szl.forge.minicpm5-gguf-plan.v1", "modelId": MODEL,
        "revision": REVISION, "files": {k: dict(filename=v[0], size=v[1], sha256=v[2]) for k, v in FILES.items()},
        "runtime": dict(RUNTIME), "suiteSha256": SUITE_SHA, "baselineRecordSha256": BASE_RECORD_SHA,
        "baselineExecutionDtype": "float16", "defaultVariant": "Q4_K_M", "caseCount": 12,
        "authorityChain": ["GitHub", "Hugging Face", "a-11-oy.com", "a11oy.net"],
        "owners": {"evaluation": "szl-forge", "witness": "szl-nemo", "recipe": "szl-serve", "intake": "szl-frontier"},
        "automaticDownload": False, "weightsRehosted": False, "trainingAuthorized": False,
        "toolExecutionAuthorized": False, "productionDisposition": "HOLD", "modelOperational": False,
        "maxNewTokens": 96, "contextTokens": 4096, "maxGenerationSeconds": 360,
        "comparisonClass": "HISTORICAL_BEHAVIOR_REFERENCE", "matchedQuantizationExperiment": False,
        "conversionLineageVerified": False, "runtimeQualified": False,
        "remainingGates": ["matched same-runtime F16/quantized comparison", "held-out quality and long-context",
            "tokenizer/template and conversion lineage", "controller/Nemo/Serve/fallback witnesses",
            "license obligations and immutable deployment closure", "product/proof projection verification"]}


def remote_url(url: str) -> None:
    parsed = urlparse(url)
    host = parsed.hostname or ""
    allowed = host in {"huggingface.co", "github.com", "release-assets.githubusercontent.com"} or host.endswith(".xethub.hf.co") or host.endswith(".huggingface.co")
    if parsed.scheme != "https" or not allowed or parsed.username or parsed.password or parsed.port not in (None, 443):
        raise GGUFError("remote URL outside public artifact providers")


class Redirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        remote_url(newurl)  # Validate BEFORE following; no authorization is sent.
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise GGUFError("loopback response may not redirect")


def download(url: str, path: Path, size: int, sha: str) -> None:
    """Fixed-size streaming acquisition, no retries, unknown paths or credentials."""
    remote_url(url)
    if type(size) is not int or not 0 < size <= FILES["F16"][1] or not base.SHA64.fullmatch(sha):
        raise GGUFError("invalid artifact bound")
    opener = build_opener(ProxyHandler({}), Redirects())
    deadline = time.monotonic() + 240
    count, hasher = 0, hashlib.sha256()
    created = False
    try:
        with opener.open(Request(url, headers={"User-Agent": "SZL-GGUF-Evaluation/1"}), timeout=20) as response, path.open("xb") as output:
            created = True
            if response.status != 200:
                raise GGUFError("unexpected artifact status")
            length = response.headers.get("Content-Length")
            if length is not None and int(length) != size:
                raise GGUFError("artifact length differs from pin")
            while block := response.read(min(1024**2, size - count + 1)):
                count += len(block)
                if count > size or time.monotonic() > deadline:
                    raise GGUFError("artifact byte/time budget exceeded")
                hasher.update(block)
                output.write(block)
        if count != size or hasher.hexdigest() != sha:
            raise GGUFError("artifact digest or size mismatch")
    except Exception:
        # path is always under this run's new TemporaryDirectory, not user data.
        if created and path.exists():
            path.unlink()
        raise


def validate_inventory(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("id") != MODEL or payload.get("sha") != REVISION:
        raise GGUFError("upstream source revision mismatch")
    if payload.get("private") is not False or payload.get("gated") is not False or payload.get("disabled", False) is not False:
        raise GGUFError("source must be public and ungated")
    if (payload.get("cardData") or {}).get("license") != "apache-2.0":
        raise GGUFError("declared upstream license changed or unavailable")
    siblings = payload.get("siblings")
    if not isinstance(siblings, list) or len(siblings) > 64:
        raise GGUFError("invalid source inventory")
    rows = {}
    for item in siblings:
        name = item.get("rfilename")
        if not isinstance(name, str) or name in rows:
            raise GGUFError("duplicate or invalid source path")
        rows[name] = item
    for name, size, sha in FILES.values():
        row = rows.get(name, {})
        lfs = row.get("lfs") or {}
        if row.get("size", lfs.get("size")) != size or (lfs.get("sha256") or lfs.get("oid")) != sha:
            raise GGUFError("upstream GGUF bytes differ from reviewed pin")
    witness = {"schema": "szl.forge.gguf-source-observation.v1", "modelId": MODEL,
        "revision": REVISION, "files": plan()["files"], "license": "apache-2.0",
        "evidenceClass": "METADATA_ONLY", "weightsDownloaded": False,
        "productionDisposition": "HOLD", "observedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    witness["recordSha256"] = base.digest(witness)
    return witness


def source_witness() -> dict[str, Any]:
    url = f"https://huggingface.co/api/models/{MODEL}/revision/{REVISION}?blobs=true"
    opener = build_opener(ProxyHandler({}), NoRedirect())
    with opener.open(Request(url, headers={"Accept": "application/json"}), timeout=20) as response:
        if response.status != 200 or response.headers.get("Link"):
            raise GGUFError("incomplete upstream metadata response")
        return validate_inventory(loads(response.read(MAX_JSON + 1)))


def baseline(path: Path = BASE_RECORD) -> dict[str, Any]:
    value = loads(path.read_bytes())
    claimed = value.pop("recordSha256", None)
    if claimed != BASE_RECORD_SHA or base.digest(value) != BASE_RECORD_SHA:
        raise GGUFError("historical baseline integrity mismatch")
    if base.digest(base.suite()) != SUITE_SHA or base.file_digest(Path(base.__file__)) != BASE_RUNNER_SHA:
        raise GGUFError("original suite or grader source changed")
    if value["plan"]["modelRevision"] != base.REVISION or value["hardware"]["dtype"] != "float16":
        raise GGUFError("historical checkpoint/execution dtype mismatch")
    if value["passedCases"] != 9 or value["status"] != "SMOKE_FAIL":
        raise GGUFError("historical failures may not be upgraded")
    value["recordSha256"] = claimed
    return value


def unpack(archive: Path, root: Path) -> Path:
    """Extract the verified official archive with path/type/expansion bounds."""
    if base.file_digest(archive) != RUNTIME["sha256"]:
        raise GGUFError("runtime archive digest mismatch")
    with tarfile.open(archive, "r:gz") as bundle:
        members, total, names = [], 0, set()
        for member in bundle:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or member.name in names or not (member.isfile() or member.isdir() or member.issym()):
                raise GGUFError("unsafe runtime archive member")
            if member.issym() and (PurePosixPath(member.linkname).is_absolute() or ".." in PurePosixPath(member.linkname).parts):
                raise GGUFError("unsafe runtime archive link")
            names.add(member.name)
            total += member.size
            members.append(member)
            if len(members) > 512 or total > 256 * 1024**2:
                raise GGUFError("runtime expansion budget exceeded")
        bundle.extractall(root, members=members, filter="data")
    candidates = list(root.rglob("llama-server"))
    if len(candidates) != 1 or not candidates[0].is_file() or candidates[0].is_symlink():
        raise GGUFError("one real llama-server executable required")
    return candidates[0]


def request_body(case: dict[str, Any]) -> dict[str, Any]:
    return {"model": ALIAS, "messages": [{"role": "system", "content": base.SYSTEM},
        {"role": "user", "content": base.canonical(case["input"]).decode()}],
        "temperature": 0, "seed": 0, "max_tokens": 96, "stream": False,
        "cache_prompt": False, "chat_template_kwargs": {"enable_thinking": False}}


def response_grade(value: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    choices = value.get("choices")
    if value.get("model") != ALIAS or not isinstance(choices, list) or len(choices) != 1:
        raise GGUFError("response model/choice identity mismatch")
    choice = choices[0]
    message = choice.get("message") or {}
    usage = value.get("usage") or {}
    if message.get("role") != "assistant" or message.get("tool_calls") or not isinstance(message.get("content"), str):
        raise GGUFError("unexpected response envelope")
    for key, maximum in (("completion_tokens", 96), ("prompt_tokens", 4000)):
        if type(usage.get(key)) is not int or not 1 <= usage[key] <= maximum:
            raise GGUFError("response token count outside bound")
    result = base.grade(message["content"], case["expected"], truncated=choice.get("finish_reason") != "stop")
    result.update(id=case["id"], category=case["category"], generatedTokens=usage["completion_tokens"],
                  reasoningProduced=bool(message.get("reasoning_content")), inputTokens=usage["prompt_tokens"])
    return result


def summarize(report: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    result = dict(report)
    result.pop("recordSha256", None)
    rows = result["cases"]
    if len(rows) > 12 or [r["id"] for r in rows] != [c["id"] for c in base.suite()][:len(rows)]:
        raise GGUFError("case order/coverage mismatch")
    if any(type(r.get("passed")) is not bool for r in rows):
        raise GGUFError("case verdict must be boolean")
    for row, case in zip(rows, base.suite()):
        if row.get("category") != case["category"] or not base.SHA64.fullmatch(str(row.get("outputSha256", ""))):
            raise GGUFError("case identity or output digest missing")
        reasons = row.get("reasonCodes")
        if not isinstance(reasons, list) or row["passed"] != (not reasons) or any(r not in {"truncated_output", "output_schema", "decision_or_grounding_mismatch", "invalid_json"} for r in reasons):
            raise GGUFError("case grading verdict is inconsistent")
        timing = row.get("roundtripMs")
        if type(timing) not in (int, float) or not math.isfinite(timing) or timing < 0:
            raise GGUFError("invalid latency measurement")
    complete = len(rows) == 12 and not result.get("errorType")
    passed = sum(r["passed"] for r in rows)
    result.update(status="INCOMPLETE" if not complete else ("SMOKE_PASS" if passed == 12 else "SMOKE_FAIL"),
        completedCases=len(rows), passedCases=passed, expectedCases=12,
        p50RoundtripMs=base.quantile([r["roundtripMs"] for r in rows], .5),
        p95RoundtripMs=base.quantile([r["roundtripMs"] for r in rows], .95),
        comparisonClass="HISTORICAL_BEHAVIOR_REFERENCE", baselineRecordSha256=BASE_RECORD_SHA,
        newlyFailedCaseIds=[r["id"] for r, b in zip(rows, reference["cases"]) if b["passed"] and not r["passed"]],
        recoveredCaseIds=[r["id"] for r, b in zip(rows, reference["cases"]) if not b["passed"] and r["passed"]],
        sameOutputHashCount=sum(r["outputSha256"] == b["outputSha256"] for r, b in zip(rows, reference["cases"])),
        comparisonComplete=complete, matchedQuantizationExperiment=False, speedupClaim=None, runtimeQualified=False,
        productionDisposition="HOLD", modelOperational=False, publicationEligible=False,
        sealed=False, toolExecuted=False, trainingAuthorized=False, ttftMs=None, joules=None, costUsd=None)
    result["recordSha256"] = base.digest(result)
    return result


def run(variant: str, revision: str, output: Path) -> int:
    reference = baseline()
    if not base.SHA40.fullmatch(revision):
        raise GGUFError("exact Forge source revision required")
    filename, size, sha = FILES[variant]
    report = {"schema": "szl.forge.minicpm5-gguf-result.v1", "plan": plan(), "variant": variant,
        "sourceRepository": base.SOURCE, "sourceRevision": revision,
        "runnerSha256": base.file_digest(Path(__file__)), "cases": [], "jobId": os.environ.get("JOB_ID"),
        "host": {"system": platform.system(), "machine": platform.machine(), "python": platform.python_version(), "cpuCount": os.cpu_count()},
        "runtimeArchiveSha256": RUNTIME["sha256"], "modelSha256": sha, "artifactBytesVerified": False,
        "imageDigestVerified": False, "peakResidentBytes": None, "phase": "preflight"}
    base.atomic_json(output, summarize(report, reference))
    try:
        if platform.system() != "Linux" or platform.machine() not in {"x86_64", "AMD64"}:
            raise GGUFError("this executable profile requires Linux x86-64")
        report["sourceObservation"] = source_witness()
        with tempfile.TemporaryDirectory(prefix="szl-gguf-") as temporary:
            root = Path(temporary)
            report["phase"] = "download"
            model, archive = root / filename, root / "runtime.tar.gz"
            download(f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{filename}", model, size, sha)
            download(f"https://github.com/{RUNTIME['repository']}/releases/download/{RUNTIME['tag']}/{RUNTIME['filename']}", archive, RUNTIME["size"], RUNTIME["sha256"])
            with model.open("rb") as stream:
                if stream.read(4) != b"GGUF":
                    raise GGUFError("not a GGUF artifact")
            server = unpack(archive, root / "runtime")
            report.update(artifactBytesVerified=True, serverBinarySha256=base.file_digest(server), phase="startup")
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            nonce = secrets.token_hex(24)
            command = [str(server), "--model", str(model), "--host", "127.0.0.1", "--port", str(port),
                "--alias", ALIAS, "--api-key", nonce, "--ctx-size", "4096", "--parallel", "1",
                "--threads", "4", "--n-gpu-layers", "0", "--jinja", "--no-context-shift", "--no-warmup"]
            env = {"PATH": os.defpath, "HOME": str(root), "LANG": "C.UTF-8", "LD_LIBRARY_PATH": str(server.parent)}
            opener = build_opener(ProxyHandler({}), NoRedirect())
            def local(path: str, body=None, timeout=5):
                request = Request(f"http://127.0.0.1:{port}" + path, data=None if body is None else base.canonical(body),
                    headers={"Content-Type": "application/json", "Authorization": "Bearer " + nonce})
                with opener.open(request, timeout=timeout) as response:
                    return loads(response.read(MAX_JSON + 1))
            start = time.monotonic()
            with (root / "server.log").open("wb") as log:
                process = subprocess.Popen(command, cwd=server.parent, env=env, stdout=subprocess.DEVNULL, stderr=log)
                try:
                    while True:
                        if process.poll() is not None or time.monotonic() - start > 90:
                            raise GGUFError("owned server failed readiness")
                        try:
                            models = local("/v1/models")
                            if ALIAS in {m.get("id") for m in models.get("data", [])}:
                                break
                        except (OSError, ValueError):
                            pass
                        time.sleep(1)
                    report.update(startupMs=round((time.monotonic() - start) * 1000, 3), phase="generation")
                    deadline = time.monotonic() + 360
                    for case in base.suite():
                        remaining = deadline - time.monotonic()
                        if remaining <= 0 or process.poll() is not None:
                            raise TimeoutError("owned generation budget exhausted")
                        begin = time.perf_counter()
                        result = response_grade(local("/v1/chat/completions", request_body(case), min(30, remaining)), case)
                        result["roundtripMs"] = round((time.perf_counter() - begin) * 1000, 3)
                        report["cases"].append(result)
                        base.atomic_json(output, summarize(report, reference))
                    status_file = Path(f"/proc/{process.pid}/status")
                    for line in status_file.read_text().splitlines():
                        if line.startswith("VmHWM:"):
                            report["peakResidentBytes"] = int(line.split()[1]) * 1024
                    report["phase"] = "complete"
                finally:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
    except Exception as exc:
        report["errorType"] = type(exc).__name__
        if isinstance(exc, GGUFError):
            report["errorReason"] = str(exc)
    final = summarize(report, reference)
    base.atomic_json(output, final)
    # Base64 prevents middleware masking non-secret boolean/numeric constants.
    print("SZL_GGUF_RECORD_BASE64=" + base64.b64encode(base.canonical(final)).decode(), flush=True)
    return {"SMOKE_PASS": 0, "SMOKE_FAIL": 2, "INCOMPLETE": 1}[final["status"]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "source", "run"), nargs="?", default="plan")
    parser.add_argument("--variant", choices=tuple(FILES), default="Q4_K_M")
    parser.add_argument("--source-revision")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "run":
            if args.output is None or args.source_revision is None:
                parser.error("run requires --output and --source-revision")
            return run(args.variant, args.source_revision, args.output)
        result = source_witness() if args.command == "source" else plan()
        if args.output:
            base.atomic_json(args.output, result)
        print(base.canonical(result).decode())
        return 0
    except Exception as exc:
        result = {"schema": "szl.forge.minicpm5-gguf-failure.v1", "status": "INCOMPLETE",
                  "errorType": type(exc).__name__, "productionDisposition": "HOLD", "modelOperational": False}
        if args.output:
            base.atomic_json(args.output, result)
        print(base.canonical(result).decode())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
