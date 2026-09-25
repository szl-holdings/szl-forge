"""Witness real loopback HTTP and model inference for a completed local run.

Starts and stops only its own server child. Replays already evaluated positive
and negative given-context cases: a wiring smoke, NOT new independent quality
evidence. Keeps logs and an exclusive receipt. No provider/network writes.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import urllib.error

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from external_retrieval_lab import digest, load_json, save_new, utc


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def exchange(base, route, payload=None, *, origin=None):
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"}
    if origin:
        headers["Origin"] = origin
    request = urllib.request.Request(base + route, data=data, headers=headers)
    # Direct loopback only: ignore ambient proxy settings.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=90) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        return exc.code, json.load(exc)


def verify(labroot, run_id, port):
    run = labroot / "runs" / run_id
    bindings = {name: digest(run / name) for name in
                ("result.json", "payload.json", "calibration.json", "freeze.json")}
    result, payload = load_json(run / "result.json"), load_json(run / "payload.json")
    calibration = load_json(run / "calibration.json")
    threshold = calibration["threshold"]
    cases = {row["id"]: row for row in payload["evaluation"]}
    documents = {row["id"]: row for row in payload["documents"]}
    positive = next(row for row in result["records"]["given_context"]
                    if row["answerable"] and row["margin"] > threshold and row["prediction"] in row["gold"])
    negative = next(row for row in result["records"]["given_context"]
                    if not row["answerable"] and row["margin"] <= threshold)
    # Refuse accidental interference with an already bound local server.
    import socket
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", port))
    save_new(run / "http-attempt.json", {"started_at": utc(), "port": port,
        "quality_evidence": "REPLAY_WIRING_ONLY", "artifact_sha256": bindings})
    checks = []
    command = [sys.executable, "-I", "-B", str(HERE / "external_retrieval_lab.py"), "serve",
               "--lab-root", str(labroot), "--run-id", run_id, "--port", str(port)]
    child = None
    base = f"http://127.0.0.1:{port}"
    try:
        with (run / "http-stdout.log").open("xb") as stdout, (run / "http-stderr.log").open("xb") as stderr:
            child = subprocess.Popen(command, stdout=stdout, stderr=stderr,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                     "HF_HUB_DISABLE_PROGRESS_BARS": "1"})
            # Native Transformers discovery can take several minutes on a cold
            # Windows filesystem. This bounds startup, not an inference timeout.
            deadline = time.monotonic() + 900
            while True:
                if child.poll() is not None:
                    raise RuntimeError("Local server failed; inspect retained stderr log")
                try:
                    status, health = exchange(base, "/health")
                    if status == 200:
                        break
                except (OSError, urllib.error.URLError):
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError("Local server readiness deadline exceeded")
                time.sleep(1)
            require(health["status"] == "LOCAL_READY_NOT_PRODUCTION", "Incorrect readiness status")
            require(health["freeze_sha256"] == result["freeze_sha256"], "Readiness source mismatch")
            checks.append({"check": "live_readiness_source_binding", "passed": True, "response": health})
            for label, row, expected in (("positive_context_replay", positive, "ANSWER"),
                                         ("unsupported_context_replay", negative, "ABSTAIN")):
                case = cases[row["id"]]
                doc = documents[case["context_id"]]
                status, answer = exchange(base, "/answer-context", {"question": case["question"], "context": doc["text"]})
                require(status == 200 and answer["status"] == expected, "Context answer contract failed")
                if expected == "ANSWER":
                    require(answer["answer"] in case["answers"], "Context replay answer mismatch")
                    evidence = answer["evidence"]
                    require(doc["text"][evidence["start"]:evidence["end"]] == answer["answer"], "Context span mismatch")
                else:
                    require(answer["answer"] is None and answer["evidence"] is None, "Abstention contains an answer")
                checks.append({"check": label, "passed": True, "case_id": case["id"], "response": answer})
            question = cases[positive["id"]]["question"]
            status, hits = exchange(base, "/search", {"question": question, "k": 3})
            require(status == 200 and len(hits) == 3 and all(hit["citation"] for hit in hits), "Search contract failed")
            checks.append({"check": "hybrid_search_with_citations", "passed": True, "ids": [d["id"] for d in hits]})
            status, answer = exchange(base, "/query", {"question": question})
            require(status == 200 and answer["status"] in ("ANSWER", "ABSTAIN"), "Retrieval-reader contract failed")
            require(answer["global_unanswerability"] == "UNVERIFIED", "Unsupported global proof claim")
            if answer["status"] == "ANSWER":
                doc = next(d for d in answer["passages"] if d["id"] == answer["evidence"]["document_id"])
                require(doc["text"][answer["evidence"]["start"]:answer["evidence"]["end"]] == answer["answer"], "Retrieved span mismatch")
            checks.append({"check": "live_retrieval_reader_citation_contract", "passed": True,
                           "status": answer["status"], "answer": answer["answer"]})
            for label, route, body, origin, expected in (
                ("empty_question", "/query", {"question": ""}, None, 422),
                ("invalid_k", "/search", {"question": question, "k": 99}, None, 422),
                ("oversized_context", "/answer-context", {"question": question, "context": "x" * 16001}, None, 422),
                ("body_limit", "/query", {"question": "x" * 70000}, None, 413),
                ("cross_origin", "/query", {"question": question}, "https://example.invalid", 403)):
                status, _ = exchange(base, route, body, origin=origin)
                require(status == expected, f"{label}: expected {expected}, received {status}")
                checks.append({"check": label, "passed": True, "status": status})
    finally:
        if child is not None:
            child.terminate()
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=10)
    require(all(digest(run / name) == value for name, value in bindings.items()), "Run artifacts changed during HTTP verification")
    receipt = {"status": "LIVE_LOOPBACK_HTTP_VERIFIED_SERVER_STOPPED", "completed_at": utc(),
        "scope": "local_replay_wiring_only_not_independent_quality_evidence", "checks": checks,
        "result_sha256": bindings["result.json"], "artifact_sha256": bindings, "verifier_sha256": digest(__file__),
        "child_stopped": child.poll() is not None, "port": port, "production_deployed": False}
    save_new(run / "http-verification.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lab-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    # Reuse the safe run identifier and path contract.
    from external_retrieval_lab import paths
    root, _ = paths(args.lab_root, args.run_id)
    if not 1024 <= args.port <= 65535:
        parser.error("port must be in 1024..65535")
    print(json.dumps(verify(root, args.run_id, args.port), indent=2, allow_nan=False))
