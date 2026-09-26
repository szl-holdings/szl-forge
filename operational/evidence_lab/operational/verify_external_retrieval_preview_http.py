"""Witness the already-running, source-bound local preview without owning its process.

Stdlib only. Connects exclusively to http://127.0.0.1:8766, ignores proxies and
refuses redirects. Replays frozen examples plus a Unicode-prefix wiring probe;
this is not independent model-quality evidence, training, or production proof.
The output is exclusive and includes complete actual HTTP response bodies.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.error
import urllib.request

HERE = Path(__file__).resolve().parent
DEFAULT_LAB_ROOT = HERE.parents[3] / "retrieval-integration-2026-09-07"
DEFAULT_RUN_ID = "baseline-20260908-1"
ONLY_URL = "http://127.0.0.1:8766"
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
SOURCE_NAMES = ("external_retrieval_core.py", "external_retrieval_data.py",
                "external_retrieval_lab.py", "external_retrieval_reader.py")
ARTIFACT_NAMES = ("freeze.json", "payload.json", "result.json", "calibration.json", "documents.npy")
ASSET_NAMES = ("index.html", "app.css", "app.js")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def reject_constant(value):
    raise ValueError(f"Non-finite JSON constant: {value}")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate field in received JSON")
        result[key] = value
    return result


def decode_json(raw):
    return json.loads(raw, parse_constant=reject_constant, object_pairs_hook=unique_object)


def load_json(path):
    return decode_json(Path(path).read_bytes())


def validate_url(url):
    require(url in (ONLY_URL, ONLY_URL + "/"), "Only http://127.0.0.1:8766 is permitted")
    return ONLY_URL


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        return None


def exchange(url, route, *, payload=None, raw=None, headers=None, timeout=90):
    url = validate_url(url)
    require(route.startswith("/") and not route.startswith("//"), "Invalid local route")
    require(payload is None or raw is None, "Ambiguous request body")
    data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8") if payload is not None else raw
    request_headers = {"Origin": url}
    if data is not None:
        request_headers.update({"Content-Type": "application/json", "X-SZL-Preview": "1"})
    request_headers.update(headers or {})
    request = urllib.request.Request(url + route, data=data, headers=request_headers)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    started = time.monotonic()
    try:
        response = opener.open(request, timeout=timeout)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        content = response.read(MAX_RESPONSE_BYTES + 1)
        require(len(content) <= MAX_RESPONSE_BYTES, "Response exceeds the verifier safety limit")
        body_text = content.decode("utf-8")
        response_headers = {key.lower(): value for key, value in response.headers.items()}
        body = decode_json(content) if "application/json" in response_headers.get("content-type", "") else body_text
        return {"route": route, "method": request.get_method(), "status_code": response.status,
                "elapsed_seconds": round(time.monotonic() - started, 3), "received_at": utc(),
                "request_headers": request_headers,
                "request_body_bytes": len(data) if data is not None else 0,
                "request_body_sha256": sha256(data) if data is not None else None,
                "request_payload": payload, "response_headers": response_headers,
                "response_body": body, "response_body_text": body_text,
                "response_body_sha256": sha256(content)}


def security_headers(response):
    headers = response["response_headers"]
    require(headers.get("x-content-type-options") == "nosniff", "Missing nosniff")
    require(headers.get("cache-control") == "no-store", "Missing no-store")
    require(headers.get("referrer-policy") == "no-referrer", "Missing no-referrer")
    require(headers.get("x-frame-options") == "DENY", "Missing frame denial")
    require(headers.get("x-robots-tag") == "noindex, nofollow, noarchive", "Missing indexing denial")
    require("frame-ancestors 'none'" in headers.get("content-security-policy", ""), "Missing CSP frame denial")
    require("access-control-allow-origin" not in headers, "Unexpected CORS permission")


def bound_answer(response, before):
    require(response["status_code"] == 200, "Inference did not return HTTP 200")
    security_headers(response)
    body = response["response_body"]
    for key in ("run_id", "freeze_sha256", "preview_sha256"):
        require(body.get(key) == before[key], f"Inference {key} binding mismatch")
    require(body.get("publication_eligible") is False, "Inference claimed publication eligibility")
    return body


def exact_evidence(body, document):
    evidence = body["evidence"]
    require(isinstance(evidence, dict), "Answered result lacks evidence")
    start, end = evidence.get("start"), evidence.get("end")
    require(type(start) is int and type(end) is int, "Evidence offsets are not strict integers")
    require(0 <= start < end <= len(document["text"]), "Evidence offsets out of character bounds")
    require(document["text"][start:end] == body["answer"], "Answer is not the exact source substring")
    require(evidence.get("document_id") == document["id"], "Evidence document ID mismatch")
    require(evidence.get("context_sha256") == sha256(document["text"].encode("utf-8")), "Evidence context digest mismatch")
    require(bool(document.get("citation")), "Source citation is missing")


def verify(url, lab_root, run_id, receipt):
    url = validate_url(url)
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_id) is not None, "Unsafe run identifier")
    run = Path(lab_root).resolve() / "runs" / run_id
    bindings = {name: digest(run / name) for name in ARTIFACT_NAMES}
    source_bindings = {name: digest(HERE / name) for name in SOURCE_NAMES}
    assets = {name: digest(HERE / "retrieval_web" / name) for name in ASSET_NAMES}
    wrapper_sha = digest(HERE / "external_retrieval_preview.py")
    freeze, payload, result, calibration = (load_json(run / name) for name in
        ("freeze.json", "payload.json", "result.json", "calibration.json"))
    receipt.update({"url": url, "run_id": run_id, "run_directory": str(run),
        "artifact_sha256": bindings, "frozen_source_sha256": source_bindings,
        "preview_sha256": wrapper_sha, "asset_sha256": assets,
        "result_sha256": bindings["result.json"], "freeze_sha256": bindings["freeze.json"]})
    require(freeze["code"] == source_bindings, "Frozen evaluation source binding mismatch")
    require(freeze["payload_sha256"] == bindings["payload.json"], "Frozen payload binding mismatch")
    require(calibration["freeze_sha256"] == bindings["freeze.json"], "Calibration freeze binding mismatch")
    require(result["freeze_sha256"] == bindings["freeze.json"], "Result freeze binding mismatch")
    require(result["index_sha256"] == bindings["documents.npy"], "Result index binding mismatch")
    require(result["calibration_sha256"] == bindings["calibration.json"], "Result calibration binding mismatch")
    require(result["status"] == "LOCAL_EVALUATION_COMPLETED_NOT_PRODUCTION", "Result is not a completed local evaluation")
    documents = {row["id"]: row for row in payload["documents"]}
    cases = {row["id"]: row for row in payload["evaluation"]}
    records = {row["id"]: row for row in result["records"]["given_context"]}

    def check(name, explanation, response=None):
        receipt["checks"].append({"check": name, "passed": True, "explanation": explanation,
                                  **({"http": response} if response is not None else {})})
        print(f"PASS {name}", flush=True)

    def request(route, **kwargs):
        response = exchange(url, route, **kwargs)
        receipt["http_responses"].append(response)
        return response

    deadline = time.monotonic() + 240
    while True:
        remaining = deadline - time.monotonic()
        require(remaining > 0, "Local preview readiness exceeded 240 seconds")
        try:
            initial = request("/api/status", timeout=min(3, remaining))
            if initial["status_code"] == 200:
                break
            receipt["readiness_attempts"].append({"at": utc(), "status_code": initial["status_code"]})
        except (OSError, urllib.error.URLError) as error:
            receipt["readiness_attempts"].append({"at": utc(), "error_type": type(error).__name__})
        remaining = deadline - time.monotonic()
        require(remaining > 0, "Local preview readiness exceeded 240 seconds")
        time.sleep(min(3, remaining))
    security_headers(initial)
    before = initial["response_body"]
    require(before.get("status") == "LOCAL_READY_NOT_PRODUCTION", "Incorrect preview readiness claim")
    require(before.get("publication_eligible") is False, "Preview claimed publication eligibility")
    require(before.get("run_id") == run_id, "Preview run mismatch")
    require(before.get("freeze_sha256") == bindings["freeze.json"], "Preview freeze mismatch")
    require(before.get("result_sha256") == bindings["result.json"], "Preview result mismatch")
    require(before.get("preview_sha256") == wrapper_sha, "Preview wrapper mismatch")
    require(before.get("asset_sha256") == assets, "Preview asset mismatch")
    require(before.get("threshold") == calibration["threshold"], "Preview calibration threshold mismatch")
    expected_packages = {"torch": result["runtime"]["torch"], "transformers": result["runtime"]["transformers"],
                         "huggingface-hub": "1.29.0", "tokenizers": "0.23.2"}
    require(before.get("runtime_packages") == expected_packages, "Preview runtime-package binding mismatch")
    require(before.get("documents") == len(documents), "Preview document count mismatch")
    check("readiness_current_source_and_runtime_binding", "Live status matches current wrapper, assets, frozen result and runtime versions.")

    for name, route in (("index.html", "/"), ("app.css", "/app.css"), ("app.js", "/app.js")):
        response = request(route)
        require(response["status_code"] == 200, "Static asset request failed")
        security_headers(response)
        require(response["response_body_sha256"] == assets[name], "Served asset does not match source bytes")
        check("served_" + name, "Served bytes match the status-bound current asset digest.")

    positive = next(example for example in before["examples"] if example["label"] == "Answerable passage")
    negative = next(example for example in before["examples"] if example["label"] == "Unsupported passage")
    for example, answerable in ((positive, True), (negative, False)):
        case = cases[example["source_id"]]
        record = records[case["id"]]
        require(record["answerable"] is answerable, "Example answerability differs from frozen record")
        require(example["question"] == case["question"], "Example question differs from frozen evaluation")
        require(example["context"] == documents[case["context_id"]]["text"], "Example passage differs from frozen evaluation")
        require((record["margin"] > calibration["threshold"]) is answerable, "Example does not follow frozen threshold")
    require(records[positive["source_id"]]["prediction"] == "justice and prosperity", "Unexpected baseline positive fixture")

    for label, example, expected, prefix in (
        ("positive_context_replay", positive, "ANSWER", ""),
        ("unicode_context_span_wiring", positive, "ANSWER", "\U0001f34e caf\u00e9 \u2014 "),
        ("unsupported_context_replay", negative, "ABSTAIN", "")):
        context = prefix + example["context"]
        body = bound_answer(request("/api/answer-context", payload={"question": example["question"], "context": context}), before)
        require(body.get("status") == expected, f"{label} status mismatch")
        require(body.get("scope") == "provided_context_only", "Provided-context scope mismatch")
        require(len(body.get("passages", [])) == 1 and body["passages"][0]["text"] == context, "Provided passage mismatch")
        passage = body["passages"][0]
        require(passage["id"] == "provided-" + sha256(context.encode("utf-8")), "Provided passage identity mismatch")
        if expected == "ANSWER":
            require(body["answer"] == "justice and prosperity", "Positive replay answer mismatch")
            exact_evidence(body, passage)
            record = records[positive["source_id"]]
            require(body["evidence"]["start"] == record["start"] + len(prefix), "Replayed start character offset mismatch")
            require(body["evidence"]["end"] == record["end"] + len(prefix), "Replayed end character offset mismatch")
        else:
            require(body.get("answer") is None and body.get("evidence") is None, "Abstention contains answer or evidence")
        check(label, "Live local model replay and exact character-span wiring only; not independent quality evidence.")

    search = bound_answer(request("/api/search", payload={"question": positive["question"], "k": 5}), before)
    require(search.get("status") == "RETRIEVED", "Search status mismatch")
    require(search.get("scope") == "retrieval_only_not_answer_validation", "Search overstates answer validation")
    require(len(search.get("passages", [])) == 5, "Search did not return five passages")
    require([doc["rank"] for doc in search["passages"]] == [1, 2, 3, 4, 5], "Search rank sequence mismatch")
    require(len({doc["id"] for doc in search["passages"]}) == 5, "Search returned duplicate sources")
    for doc in search["passages"]:
        require(doc["id"] in documents, "Search returned an unknown source")
        require(doc["text"] == documents[doc["id"]]["text"], "Search source text mismatch")
        require(doc["citation"] == documents[doc["id"]]["citation"] and bool(doc["citation"]), "Search citation mismatch")
    check("search_five_ranked_cited_sources", "Five distinct ranked passages match the frozen corpus and citations.")

    query = bound_answer(request("/api/query", payload={"question": positive["question"]}), before)
    require(query.get("status") in ("ANSWER", "ABSTAIN"), "Query status mismatch")
    require(query.get("scope") == "retrieved_passages_only", "Query scope mismatch")
    require(query.get("global_unanswerability") == "UNVERIFIED", "Query claims global unanswerability proof")
    require(query.get("false_answer_rate_after_retrieval_selection") == "UNMEASURED", "Query claims measured selected-context risk")
    for doc in query["passages"]:
        require(doc["id"] in documents and doc["text"] == documents[doc["id"]]["text"], "Query source text mismatch")
        require(doc["citation"] == documents[doc["id"]]["citation"] and bool(doc["citation"]), "Query source citation mismatch")
    if query["status"] == "ANSWER":
        doc = next(doc for doc in query["passages"] if doc["id"] == query["evidence"]["document_id"])
        exact_evidence(query, doc)
    else:
        require(query.get("answer") is None and query.get("evidence") is None, "Query abstention contains answer or evidence")
    check("retrieval_reader_exact_citation_contract", "Live answer or abstention preserves bounded claims and exact frozen-source evidence.")

    for label, expected, kwargs in (
        ("cross_origin_denied", 403, {"payload": {"question": positive["question"]}, "headers": {"Origin": "https://example.invalid"}}),
        ("wrong_host_denied", 400, {"payload": {"question": positive["question"]}, "headers": {"Host": "unexpected.invalid:8766"}}),
        ("duplicate_json_denied", 422, {"raw": b'{"question":"first","question":"second"}'}),
        ("body_over_64_kib_denied", 413, {"raw": b'{"question":"Which?"}' + b" " * 65536})):
        response = request("/api/query", **kwargs)
        require(response["status_code"] == expected, f"{label}: expected {expected}, got {response['status_code']}")
        security_headers(response)
        check(label, f"Actual loopback request rejected with HTTP {expected} and security headers.")

    final = request("/api/status")
    require(final["status_code"] == 200, "Final readiness request failed")
    security_headers(final)
    after = final["response_body"]
    require(after == before, "Preview status binding changed during verification")
    require({name: digest(run / name) for name in ARTIFACT_NAMES} == bindings, "Frozen run changed during verification")
    require({name: digest(HERE / name) for name in SOURCE_NAMES} == source_bindings, "Frozen evaluation source changed during verification")
    require({name: digest(HERE / "retrieval_web" / name) for name in ASSET_NAMES} == assets, "Preview assets changed during verification")
    require(digest(HERE / "external_retrieval_preview.py") == wrapper_sha, "Preview wrapper changed during verification")
    check("stable_before_after_binding", "Live status and all bound local files remained byte-identical; server left running.")
    receipt.update({"status": "LIVE_LOOPBACK_PREVIEW_HTTP_VERIFIED", "verified_at": utc(),
        "server_left_running": True, "server_process_owned": False})
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=ONLY_URL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lab-root", type=Path, default=DEFAULT_LAB_ROOT)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    args = parser.parse_args()
    validate_url(args.url)
    output = args.output.resolve()
    require(output.parent.is_dir(), "Output parent directory must already exist")
    require(not output.exists(), "Refusing to overwrite an existing verification receipt")
    started = time.monotonic()
    receipt = {"status": "LOCAL_PREVIEW_HTTP_VERIFICATION_IN_PROGRESS", "started_at": utc(),
        "scope": "local_replay_and_unicode_span_wiring_only_not_independent_quality_evidence",
        "production_deployed": False, "publication_eligible": False, "training_performed": False,
        "server_process_owned": False, "verifier_sha256": digest(__file__),
        "checks": [], "http_responses": [], "readiness_attempts": []}
    failure = None
    try:
        verify(args.url, args.lab_root, args.run_id, receipt)
        require(digest(__file__) == receipt["verifier_sha256"], "Verifier source changed during verification")
    except Exception as error:
        failure = error
        receipt.update({"status": "LOCAL_PREVIEW_HTTP_VERIFICATION_FAILED", "failed_at": utc(),
                        "error_type": type(error).__name__, "error": str(error)})
    receipt["elapsed_seconds"] = round(time.monotonic() - started, 3)
    serialized = (json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    with output.open("xb") as handle:
        handle.write(serialized)
    print(json.dumps({"status": receipt["status"], "output": str(output), "sha256": sha256(serialized),
                      "checks_passed": len(receipt["checks"]), "http_responses": len(receipt["http_responses"]),
                      "elapsed_seconds": receipt["elapsed_seconds"]}, indent=2), flush=True)
    if failure is not None:
        raise RuntimeError(f"Preview verification failed; actual responses retained at {output}") from failure


if __name__ == "__main__":
    main()
