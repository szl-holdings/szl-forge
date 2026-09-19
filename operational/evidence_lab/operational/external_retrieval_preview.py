"""Same-origin local web preview over the unchanged, source-bound retrieval lab.

No model training, calibration changes, external publication or cloud API calls.
This separate presentation adapter leaves all four frozen evaluation modules intact.
"""
from __future__ import annotations
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from external_retrieval_lab import LocalService, load_json, digest, paths, CONFIG
sys.path.insert(0, str(HERE.parent / "risk_kernel"))
from unresolved_identifier_gate_v3 import corpus_vocabulary, gate_decision


def apply_identifier_guard(question, output, vocabulary):
    """Preserve the base result unless the optional lexical policy abstains."""
    decision = gate_decision(question, output, vocabulary)
    guarded = {**output, "identifier_guard": {
        "enabled": True, "reason": decision["reason"],
        "unresolved_identifiers": decision["unresolved_identifiers"],
        "base_status": output["status"],
        "scope": "experimental_lexical_identifier_filter_not_semantic_validation"}}
    if decision["status"] == "ABSTAIN":
        guarded.update(status="ABSTAIN", answer=None, evidence=None)
    return guarded


def run_summary(service, run):
    result_bytes = (run / "result.json").read_bytes()
    result = json.loads(result_bytes)
    if (result["freeze_sha256"] != service.freeze_sha
            or result["status"] != "LOCAL_EVALUATION_COMPLETED_NOT_PRODUCTION"
            or result["index_sha256"] != digest(run / "documents.npy")
            or result["calibration_sha256"] != digest(run / "calibration.json")):
        raise ValueError("Preview/result source mismatch")
    cases = {row["id"]: row for row in service.payload["evaluation"]}
    docs = {row["id"]: row for row in service.documents}
    examples = []
    for answerable in (True, False):
        row = next(row for row in result["records"]["given_context"]
            if row["answerable"] is answerable
            and ((row["margin"] > service.threshold and row["prediction"] in row["gold"])
                 if answerable else row["margin"] <= service.threshold))
        case = cases[row["id"]]
        examples.append({"label": "Answerable passage" if answerable else "Unsupported passage",
            "question": case["question"], "context": docs[case["context_id"]]["text"],
            "source_id": case["id"], "scope": "Previously evaluated example, not new quality evidence"})
    return {"status": "LOCAL_READY_NOT_PRODUCTION", "run_id": service.run_id,
        "freeze_sha256": service.freeze_sha, "result_sha256": hashlib.sha256(result_bytes).hexdigest(),
        "documents": len(service.documents), "models": {"encoder": CONFIG["encoder"], "reader": CONFIG["reader"]},
        "device": result["runtime"]["gpu"],
        "execution_device": result["runtime"].get("device", "unknown"), "threshold": service.threshold,
        "calibration_questions": len(service.payload["calibration"]),
        "evaluation_questions": len(service.payload["evaluation"]),
        "hotpot_cases": len(service.payload["hotpot"]),
        "metrics": {key: result[key] for key in ("squad_retrieval", "hotpot_support_retrieval",
            "given_context_answerability", "retrieved_context_qa_answerable_only")},
        "examples": examples, "completed_at": result["completed_at"],
        "limitations": result["limitations"], "publication_eligible": False}


def verify_runtime(run):
    """Reject silent dependency drift before expensive model construction."""
    recorded = load_json(run / "result.json")["runtime"]
    expected = {"torch": recorded["torch"], "transformers": recorded["transformers"],
                "huggingface-hub": "1.29.0", "tokenizers": "0.23.2"}
    actual = {name: version(name) for name in expected}
    if actual != expected:
        raise ValueError(f"Preview dependency drift; use the documented isolated environment. Expected {expected}; got {actual}")
    return actual


def create_preview(service, summary, port=8766):
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import FileResponse, JSONResponse
    from starlette.concurrency import run_in_threadpool
    app = FastAPI(title="SZL Evidence Lab", docs_url=None, redoc_url=None, openapi_url=None)
    hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
    assets = HERE / "retrieval_web"
    bound = {name: digest(assets / name) for name in ("index.html", "app.css", "app.js")}
    preview_sha = digest(__file__)
    code_paths = {"preview": Path(__file__)}
    code_paths.update({name: HERE / name for name in (
        "external_retrieval_lab.py", "external_retrieval_core.py",
        "external_retrieval_reader.py", "external_retrieval_data.py")})
    code_paths.update({name: HERE.parent / "risk_kernel" / name for name in (
        "unresolved_identifier_gate_v3.py", "retrieval_risk_kernel_v2.py",
        "retrieval_risk_kernel.py")})
    code_bound = {name: digest(path) for name, path in code_paths.items()}
    vocabulary = (corpus_vocabulary({"documents": service.documents})
                  if hasattr(service, "documents") else None)

    @app.middleware("http")
    async def guard(request, call_next):
        if any(digest(path) != code_bound[name] for name, path in code_paths.items()):
            response = JSONResponse({"detail": "Runtime source changed; restart required"}, status_code=503)
        elif request.headers.get("host") not in hosts:
            response = JSONResponse({"detail": "Unexpected local host"}, status_code=400)
        elif request.headers.get("origin") is not None and request.headers["origin"] != "http://" + request.headers["host"]:
            response = JSONResponse({"detail": "Cross-origin access disabled"}, status_code=403)
        elif request.headers.get("sec-fetch-site") not in (None, "none", "same-origin"):
            response = JSONResponse({"detail": "Cross-site requests disabled"}, status_code=403)
        else:
            try:
                response = await call_next(request)
            except Exception:
                response = JSONResponse({"detail": "Local inference failed; inspect the runtime and retry."}, status_code=500)
        response.headers.update({
            "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; font-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'",
            "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer",
            "X-Frame-Options": "DENY", "Cache-Control": "no-store",
            "X-Robots-Tag": "noindex, nofollow, noarchive",
            "Permissions-Policy": "camera=(), microphone=(), geolocation=()"})
        return response

    def static(name):
        if digest(assets / name) != bound[name]:
            raise HTTPException(status_code=503, detail="Preview assets changed; restart required")
        return FileResponse(assets / name)

    @app.get("/")
    def home():
        return static("index.html")

    @app.get("/app.css")
    def css():
        return static("app.css")

    @app.get("/app.js")
    def javascript():
        return static("app.js")

    @app.get("/api/status")
    def status():
        return {**summary, "preview_sha256": preview_sha, "asset_sha256": bound,
            "runtime_source_sha256": code_bound,
            "capabilities": {"identifier_guard": vocabulary is not None},
            "identifier_guard_scope": "Optional lexical filter; general answer quality remains unverified"}

    @app.get("/api/notices")
    def notices():
        return FileResponse(HERE / "EXTERNAL_RETRIEVAL_NOTICES.md", media_type="text/plain")

    async def inference(request):
        if request.headers.get("x-szl-preview") != "1":
            raise HTTPException(status_code=403, detail="Preview request header required")
        if request.headers.get("content-type", "").split(";", 1)[0].strip() != "application/json":
            raise HTTPException(status_code=415, detail="application/json required")
        raw = bytearray()
        async for chunk in request.stream():
            raw.extend(chunk)
            if len(raw) > 65536:
                raise HTTPException(status_code=413, detail="Request exceeds 64 KiB")
        try:
            def unique_object(pairs):
                obj = {}
                for key, value in pairs:
                    if key in obj:
                        raise ValueError("Duplicate JSON field")
                    obj[key] = value
                return obj

            def reject_constant(value):
                raise ValueError("Non-finite JSON value")

            body = json.loads(raw, object_pairs_hook=unique_object, parse_constant=reject_constant)
            operation = request.url.path.rsplit("/", 1)[1]
            fields = {"query": {"question"}, "query-guarded": {"question"}, "search": {"question", "k"},
                      "answer-context": {"question", "context"}}[operation]
            if not isinstance(body, dict) or not set(body) <= fields or "question" not in body:
                raise ValueError("Invalid fields")
            if not service.lock.acquire(blocking=False):
                raise HTTPException(status_code=429, detail="Local inference is busy. Wait for the current request to finish.")
            started = time.monotonic()
            try:
                if operation in ("query", "query-guarded"):
                    output = await run_in_threadpool(service.query, body["question"])
                    if operation == "query-guarded":
                        if vocabulary is None:
                            raise HTTPException(status_code=503, detail="Identifier guard unavailable")
                        output = apply_identifier_guard(body["question"], output, vocabulary)
                elif operation == "search":
                    output = {"status": "RETRIEVED", "passages": await run_in_threadpool(
                        service.search, body["question"], body.get("k", 5)), "scope": "retrieval_only_not_answer_validation"}
                else:
                    output = await run_in_threadpool(service.answer_context, body["question"], body.get("context"))
                    context = body["context"]
                    output["passages"] = [{"id": "provided-" + hashlib.sha256(context.encode()).hexdigest(),
                        "title": "Your provided passage", "text": context, "citation": "Provided locally; source rights not reviewed", "rank": 1}]
                return {**output, "elapsed_seconds": round(time.monotonic() - started, 3),
                    "run_id": service.run_id, "freeze_sha256": service.freeze_sha,
                    "preview_sha256": preview_sha, "publication_eligible": False}
            finally:
                service.lock.release()
        except (ValueError, TypeError, KeyError, RecursionError):
            raise HTTPException(status_code=422, detail="Check the question and passage limits; this input could not be processed.") from None
    inference.__annotations__["request"] = Request
    for route in ("query", "query-guarded", "search", "answer-context"):
        app.add_api_route("/api/" + route, inference, methods=["POST"])
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lab-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    root, run = paths(args.lab_root, args.run_id)
    if not 1024 <= args.port <= 65535:
        parser.error("port must be 1024..65535")
    print("Verifying the frozen run and loading local models...", flush=True)
    bindings = {name: digest(run / name) for name in (
        "freeze.json", "payload.json", "result.json", "calibration.json", "documents.npy")}
    runtime_packages = verify_runtime(run)
    service = LocalService(root, args.run_id)
    summary = run_summary(service, run)
    summary["runtime_packages"] = runtime_packages
    if any(digest(run / name) != value for name, value in bindings.items()):
        raise ValueError("Run artifacts changed while loading; restart with a stable run")
    app = create_preview(service, summary, args.port)
    print(f"Preview ready: http://127.0.0.1:{args.port}/", flush=True)
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=args.port, workers=1, access_log=False)
