"""Local, open-weight model experiments. Generated code never runs on the host.

SPDX-License-Identifier: Apache-2.0
This is a baseline harness, not a fine-tuned model or production release agent.
"""
from __future__ import annotations
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parent
ENDPOINT = "http://127.0.0.1:11439"
MODEL = "qwen3:4b-instruct"
IMAGE = "sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea"
DOCKER = ["docker", "--host", "npipe:////./pipe/dockerDesktopLinuxEngine"]
MAX_CODE = 24000
MAX_CODE_LINES = 400
MAX_LINE_LENGTH = 2000
MAX_OUTPUT = 65536
SYSTEM = (
    "You are a local software engineering model. Implement the supplied contract precisely. "
    "Return JSON with exactly two keys: explanation (brief string), code_lines (array of Python source lines). "
    "Each array item is one literal source line with indentation. Do not put newline escapes inside a line. "
    "Use empty strings for blank lines. No Markdown fences or backticks. "
    "The code must define solve(value), returning a JSON-serializable answer. "
    "Use only the Python standard library. Do not print anything or read stdin. "
    "Do not access files, network, subprocesses, credentials, or external services. "
    "Do not claim tests passed: an external runner evaluates your candidate."
)
# The inference engine expands bounded repetitions into grammar rules. Nesting
# 400 lines * 2000 chars exceeded its grammar complexity limit in a real request.
# Keep the wire grammar structural; candidate() enforces every size bound below,
# and num_predict plus api() cap generation and response bytes independently.
SCHEMA = {"type": "object", "properties": {"explanation": {"type": "string"}, "code_lines": {
    "type": "array", "items": {"type": "string"},
    "description": "One source line per item, indentation included. No Markdown."}},
    "required": ["explanation", "code_lines"], "additionalProperties": False}


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha(data):
    return hashlib.sha256(data).hexdigest()


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        raise ValueError("Redirect refused: local model requests must stay local")


def docker_environment():
    environment = dict(os.environ)
    for name in ("DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_TLS", "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH"):
        environment.pop(name, None)
    return environment


def api(path, body=None, timeout=120):
    # A fixed loopback endpoint; no environment-proxy or remote inference fallback.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    request = urllib.request.Request(ENDPOINT + path,
        data=None if body is None else canonical(body).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with opener.open(request, timeout=timeout) as response:
        data = response.read(2 * 1024 * 1024 + 1)
    if len(data) > 2 * 1024 * 1024:
        raise ValueError("Model API response exceeded limit")
    return json.loads(data)


def check_model(timeout=10):
    pin = json.loads((ROOT / "model.lock.json").read_text(encoding="utf-8"))
    model = next((m for m in api("/api/tags", timeout=timeout)["models"] if m["name"] == MODEL), None)
    if model is None or model["digest"] != pin["manifest_sha256"] or pin["model"] != MODEL:
        raise RuntimeError("Local model missing or changed; verify and repin explicitly")
    return pin


def wait_ready(timeout=60):
    if not 1 <= timeout <= 120:
        raise ValueError("Readiness deadline must be 1 to 120 seconds")
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            return check_model(timeout=min(2, max(0.1, deadline - time.monotonic())))
        except (OSError, urllib.error.URLError) as exc:
            last_error = type(exc).__name__
            time.sleep(min(0.5, max(0, deadline - time.monotonic())))
    raise RuntimeError("Local model readiness deadline exceeded: " + str(last_error))


def preflight_docker():
    """Refuse generation if the fixed local execution environment is unavailable."""
    checks = [[*DOCKER, "info", "--format", "{{.ServerVersion}}"],
              [*DOCKER, "image", "inspect", IMAGE, "--format", "{{.Id}}"]]
    for command in checks:
        result = subprocess.run(command, capture_output=True, text=True, timeout=12,
                                env=docker_environment())
        if result.returncode or not result.stdout.strip():
            raise RuntimeError("Local Docker preflight failed; no model request sent")
    if result.stdout.strip() != IMAGE:
        raise RuntimeError("Pinned local Docker image identity mismatch")


def candidate(document):
    if not isinstance(document, dict) or set(document) != {"code_lines", "explanation"}:
        raise ValueError("Expected exactly code_lines and explanation")
    lines = document["code_lines"]
    if not isinstance(document["explanation"], str) or not isinstance(lines, list):
        raise ValueError("Expected a code_lines array and explanation string")
    if not 1 <= len(lines) <= MAX_CODE_LINES:
        raise ValueError("Source line count exceeds bounds")
    for line in lines:
        if not isinstance(line, str) or len(line) > MAX_LINE_LENGTH or "\n" in line or "\r" in line:
            raise ValueError("Each code_lines item must be one bounded literal source line")
    code = "\n".join(lines) + "\n"
    if code.lstrip().startswith("```"):
        raise ValueError("Remove Markdown fences from code: return raw Python source only")
    if len(code.encode("utf-8")) > MAX_CODE or len(document["explanation"]) > 4000:
        raise ValueError("Candidate exceeds size bound")
    tree = ast.parse(code)
    if not any(isinstance(node, ast.FunctionDef) and node.name == "solve" for node in tree.body):
        raise ValueError("Top-level solve function is required")
    return code


def load_task(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(100001)
    if len(raw) > 100000:
        raise ValueError("Task too large")
    task = json.loads(raw)
    if not re.fullmatch(r"[a-z0-9_-]{1,64}", task["id"]):
        raise ValueError("Invalid task identity")
    if not isinstance(task["instruction"], str) or len(task["instruction"]) > 12000:
        raise ValueError("Invalid task instruction")
    if not isinstance(task["cases"], list) or not 1 <= len(task["cases"]) <= 100:
        raise ValueError("Expected 1 to 100 test cases")
    for case in task["cases"]:
        if not isinstance(case, dict) or set(case) != {"input", "expected"}:
            raise ValueError("Invalid case")
        canonical(case)
    if len(canonical([case["input"] for case in task["cases"]]).encode("utf-8")) > MAX_OUTPUT:
        raise ValueError("Aggregate task inputs exceed sandbox input limit")
    task["snapshot_sha256"] = sha(raw)
    return task


def docker_command(name, folder):
    folder = Path(folder).resolve()
    if "," in str(folder):
        raise ValueError("Mount path contains Docker option delimiter")
    return [*DOCKER, "run", "--interactive", "--rm", "--pull=never", "--name", name,
        "--network=none", "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges",
        "--user=65534:65534", "--pids-limit=32", "--memory=256m", "--memory-swap=256m", "--cpus=1",
        "--log-driver=none", "--tmpfs", "/tmp:rw,noexec,nosuid,size=16m",
        *[item for key in ("HTTP_PROXY", "HTTPS_PROXY", "FTP_PROXY", "ALL_PROXY", "NO_PROXY",
                          "http_proxy", "https_proxy", "ftp_proxy", "all_proxy", "no_proxy")
          for item in ("--env", key + "=")],
        "--mount", f"type=bind,source={folder},target=/candidate,readonly",
        "--workdir=/tmp", IMAGE, "python", "-I", "-B", "/candidate/worker.py"]


def sandbox(folder, inputs, timeout=25):
    """Only a fresh task-specific directory is mounted; tests stay on the host."""
    name = "szl-local-" + uuid.uuid4().hex
    command = docker_command(name, folder)
    payload = canonical(inputs).encode("utf-8")
    if len(payload) > MAX_OUTPUT:
        raise ValueError("Sandbox input exceeds limit")
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               env=docker_environment())
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    exceeded = threading.Event()
    input_errors = []

    def write_input():
        try:
            process.stdin.write(payload)
            process.stdin.close()
        except (BrokenPipeError, OSError, ValueError) as exc:
            input_errors.append(type(exc).__name__)

    def read(pipe, key):
        try:
            while True:
                part = pipe.read(4096)
                if not part:
                    return
                available = MAX_OUTPUT - len(buffers[key])
                buffers[key].extend(part[:max(available, 0)])
                if len(part) > available:
                    exceeded.set()
                    return
        finally:
            pipe.close()

    readers = [threading.Thread(target=read, args=(process.stdout, "stdout"), daemon=True),
               threading.Thread(target=read, args=(process.stderr, "stderr"), daemon=True)]
    for reader in readers:
        reader.start()
    failure = None
    cleanup_error = None
    start = time.monotonic()
    writer = threading.Thread(target=write_input, daemon=True)
    writer.start()
    try:
        while process.poll() is None:
            if exceeded.is_set():
                failure = "OUTPUT_LIMIT"
                break
            if time.monotonic() - start > timeout:
                failure = "TIMEOUT"
                break
            time.sleep(0.05)
    finally:
        try:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
            writer.join(timeout=3)
            for reader in readers:
                reader.join(timeout=3)
        finally:
            # Remove only the generated, exact container name; never user containers.
            try:
                cleanup = subprocess.run([*DOCKER, "rm", "--force", name], capture_output=True, timeout=10,
                                         env=docker_environment())
                if cleanup.returncode and b"No such container" not in cleanup.stderr:
                    cleanup_error = "Container cleanup not confirmed"
            except Exception as exc:
                cleanup_error = type(exc).__name__ + ": " + str(exc)[:300]
    if exceeded.is_set():
        failure = "OUTPUT_LIMIT"
    if input_errors and failure is None:
        failure = "STDIN_ERROR"
    if writer.is_alive() or any(reader.is_alive() for reader in readers):
        failure = "INCOMPLETE_IO"
    stdout = bytes(buffers["stdout"]).decode("utf-8", "replace")
    result = {"exit_code": process.returncode, "failure": failure,
              "seconds": round(time.monotonic() - start, 3), "container_image": IMAGE,
              "container_name": name, "cleanup_confirmed": cleanup_error is None,
              "stdout": stdout, "stderr": bytes(buffers["stderr"]).decode("utf-8", "replace")}
    if cleanup_error is not None:
        result["execution_failure"] = failure
        result["failure"] = "CLEANUP_UNCONFIRMED"
        result["cleanup_error"] = cleanup_error
    return result


def evaluate(result, cases):
    if result["failure"] or result["exit_code"] != 0:
        return {"passed": False, "correct": 0, "total": len(cases), "reason": result["failure"] or "EXECUTION_FAILED"}
    try:
        outputs = json.loads(result["stdout"])
        if not isinstance(outputs, list) or len(outputs) != len(cases):
            raise ValueError("Invalid result list")
        correct = sum(canonical(observed) == canonical(case["expected"]) for observed, case in zip(outputs, cases))
    except (ValueError, TypeError):
        return {"passed": False, "correct": 0, "total": len(cases), "reason": "INVALID_OUTPUT"}
    return {"passed": correct == len(cases), "correct": correct, "total": len(cases),
            "reason": "PASS" if correct == len(cases) else "ANSWER_MISMATCH"}


def save(path, value):
    # Reports are generated artifacts; exclusive creation preserves older results.
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def run_task(path, attempts=2):
    if not 1 <= attempts <= 3:
        raise ValueError("Attempt budget must be 1 to 3")
    if shutil.disk_usage(ROOT).free < 512 * 1024 * 1024:
        raise RuntimeError("Insufficient disk headroom; no generation started")
    run_dir = ROOT / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    run_dir.mkdir(parents=True)
    messages = []
    receipt = {"schema": "szl.local-model-experiment/v1", "task_id": None, "started_at": now(),
        "task_sha256": None, "model": MODEL, "model_manifest_sha256": None,
        "harness_sha256": sha(Path(__file__).read_bytes()), "worker_sha256": None, "attempts": [],
        "phase": "preflight_docker", "generation_started": False,
        "training_performed": False, "remote_inference": False, "production_changes": False,
        "scope": "Development smoke task. Not sealed benchmark, fine-tuning, or full-estate autonomy."}
    try:
        preflight_docker()
        receipt["phase"] = "preflight_model"
        pin = check_model()
        receipt["model_manifest_sha256"] = pin["manifest_sha256"]
        receipt["phase"] = "load_task"
        task = load_task(path)
        worker = (ROOT / "worker.py").read_bytes()
        receipt.update(task_id=task["id"], task_sha256=task["snapshot_sha256"], worker_sha256=sha(worker))
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": task["instruction"]}]
        for attempt in range(1, attempts + 1):
            receipt["phase"] = "preflight_model"
            check_model()
            receipt["phase"] = "generate"
            receipt["generation_started"] = True
            response = api("/api/chat", {"model": MODEL, "messages": messages, "stream": False,
                "format": SCHEMA, "keep_alive": "2m", "options": {"num_ctx": 4096, "num_predict": 1800,
                    "temperature": 0.2, "seed": 37 + attempt}})
            save(run_dir / f"generation-{attempt}.json", response)
            entry = {"attempt": attempt, "eval_count": response.get("eval_count"),
                "eval_duration_ns": response.get("eval_duration"), "total_duration_ns": response.get("total_duration")}
            messages.append({"role": "assistant", "content": response["message"]["content"]})
            try:
                receipt["phase"] = "evaluate"
                if response.get("done") is not True or response.get("done_reason") == "length":
                    raise ValueError("Generation incomplete")
                document = json.loads(response["message"]["content"])
                code = candidate(document)
                folder = run_dir / f"candidate-{attempt}"
                folder.mkdir()
                # Model controls only these source bytes, never a destination path.
                with (folder / "solution.py").open("xb") as stream:
                    stream.write(code.encode("utf-8"))
                with (folder / "worker.py").open("xb") as stream:
                    stream.write(worker)
                entry["code_sha256"] = sha(code.encode("utf-8"))
                result = sandbox(folder, [case["input"] for case in task["cases"]])
                save(run_dir / f"execution-{attempt}.json", result)
                if result.get("cleanup_confirmed") is False:
                    raise RuntimeError("Cleanup unconfirmed for " + result["container_name"])
                entry["evaluation"] = evaluate(result, task["cases"])
            except (ValueError, SyntaxError, TypeError) as exc:
                entry["evaluation"] = {"passed": False, "reason": "INVALID_CANDIDATE", "detail": str(exc)[:250]}
            except Exception as exc:
                entry["evaluation"] = {"passed": False, "reason": "INFRASTRUCTURE_ERROR", "detail": str(exc)[:500]}
                receipt["attempts"].append(entry)
                raise
            receipt["attempts"].append(entry)
            print(json.dumps({"task": task["id"], **entry}), flush=True)
            if entry["evaluation"]["passed"]:
                break
            messages.append({"role": "user", "content": "The candidate failed external development tests. "
                + canonical(entry["evaluation"]) + ". Recheck the original contract and return a corrected complete candidate."})
        receipt["passed"] = bool(receipt["attempts"] and receipt["attempts"][-1]["evaluation"]["passed"])
        receipt["phase"] = "complete"
        # Telemetry cannot rewrite the externally scored candidate outcome.
        try:
            receipt["gpu_runtime_readback"] = api("/api/ps", timeout=10)
        except Exception as exc:
            receipt["telemetry_error"] = {"type": type(exc).__name__, "message": str(exc)[:500]}
    except Exception as exc:
        receipt["passed"] = False
        receipt["error"] = {"type": type(exc).__name__, "message": str(exc)[:500]}
    receipt["completed_at"] = now()
    save(run_dir / "receipt.json", receipt)
    save(run_dir / "trajectory.json", {"messages": messages, "result": receipt,
        "training_eligible": False, "reason": "Development trace; requires review, rights and held-out separation before training"})
    print(json.dumps({"receipt": str(run_dir / "receipt.json"), "passed": receipt["passed"]}), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status")
    ready = commands.add_parser("ready")
    ready.add_argument("--timeout", type=int, default=60)
    chat = commands.add_parser("ask")
    chat.add_argument("prompt")
    run = commands.add_parser("run")
    run.add_argument("task", type=Path)
    run.add_argument("--attempts", type=int, default=2)
    commands.add_parser("smoke")
    args = parser.parse_args()
    if args.command == "ready":
        pin = wait_ready(args.timeout)
        print(json.dumps({"ready": True, "endpoint": ENDPOINT, "model": MODEL,
                          "model_manifest_sha256": pin["manifest_sha256"]}))
    elif args.command == "status":
        print(json.dumps({"pin": check_model(), "runtime": api("/api/ps", timeout=10)}, indent=2))
    elif args.command == "ask":
        check_model()
        if len(args.prompt) > 12000:
            parser.error("Prompt too large for this bounded configuration")
        response = api("/api/chat", {"model": MODEL, "messages": [{"role": "system", "content":
            "You are a local SZL engineering research assistant. Distinguish proposals from measurements. "
            "You do not have repository or deployment access in this chat mode."},
            {"role": "user", "content": args.prompt}], "stream": False, "keep_alive": "2m",
            "options": {"num_ctx": 4096, "num_predict": 1200, "temperature": 0.4}})
        print(response["message"]["content"])
    elif args.command == "run":
        return 0 if run_task(args.task, args.attempts)["passed"] else 1
    elif args.command == "smoke":
        paths = sorted((ROOT / "tasks").glob("*.json"))
        if not paths:
            raise RuntimeError("No smoke tasks found")
        results = []
        for path in paths:
            result = run_task(path)
            results.append(result)
            if "error" in result:
                break  # No further generations after infrastructure/cleanup failure.
        summary = {"passed": sum(r["passed"] for r in results), "completed": len(results),
                   "planned": len(paths), "aborted": len(results) < len(paths)}
        print(json.dumps(summary))
        return 0 if len(results) == len(paths) and all(r["passed"] for r in results) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
