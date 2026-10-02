"""Witness a declared Foundation runtime without credentials or state mutation beyond three trials."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

HOST = "szlholdings-szl-foundation-confirmation.hf.space"
ARCHIVE = "869e318dd5f328205dd181ee836ef267bd2ae278f6430a9e8ddc661fbc689d03"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def fetch(origin, path, payload=None):
    headers = {"Host": HOST, "Accept": "application/json", "Cache-Control": "no-cache, no-store"}
    if payload is not None:
        headers.update({"Origin": "https://" + HOST, "Content-Type": "application/json"})
    request = Request(origin + path, data=canonical(payload) if payload is not None else None, headers=headers)
    try:
        response = urlopen(request, timeout=20)
    except HTTPError as exc:
        # Preserve an actionable bounded witness without retrying failed trials
        # or retaining arbitrary response headers (which can contain credentials).
        try:
            body = exc.read(4097)
            captured, truncated = body[:4096], len(body) > 4096
            body_unavailable = False
        except (OSError, ValueError):
            captured, truncated, body_unavailable = b"", False, True
        finally:
            exc.close()
        def header(name):
            value = exc.headers.get(name)
            return value[:256] if value is not None else None
        exc.runtime_http_evidence = {
            "method": "POST" if payload is not None else "GET", "endpoint": path,
            "status": exc.code, "content_type": header("Content-Type"),
            "server": header("Server"), "connection": header("Connection"),
            "captured_body_bytes": len(captured), "body_truncated": truncated,
            "body_unavailable": body_unavailable,
            "captured_body_sha256": hashlib.sha256(captured).hexdigest() if not body_unavailable else None,
            "body_utf8": captured.decode("utf-8", "replace") if not body_unavailable else None,
        }
        raise
    with response:
        if response.geturl() != origin + path:
            raise ValueError("Runtime evidence redirected outside the declared exact endpoint")
        raw = response.read(131073)
        if len(raw) > 131072:
            raise ValueError("Runtime response exceeds evidence bound")
        return response.status, json.loads(raw)


def verify_receipt(receipt, request, source):
    if (receipt.get("status") != "COMPLETE" or receipt.get("request") != request
            or receipt.get("request_sha256") != digest(request)
            or receipt.get("result_sha256") != digest(receipt.get("result"))
            or receipt.get("archive_sha256") != ARCHIVE
            or receipt.get("scientific_overall_gate") != "FAILED"
            or receipt.get("unsigned") is not True
            or receipt.get("authenticity_established") is not False):
        raise ValueError("Exploratory execution receipt contract failed")
    unhashed = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if receipt.get("receipt_sha256") != digest(unhashed):
        raise ValueError("Exploratory receipt byte contract failed")
    if not re.fullmatch(r"[0-9a-f]{32}", receipt.get("id", "")):
        raise ValueError("Unexpected receipt identifier")
    if receipt.get("source", {}).get("revision") != source:
        raise ValueError("Exploratory source observation mismatch")
    binding = receipt.get("binding", {})
    if not re.fullmatch(r"[0-9a-f]{64}", binding.get("checkpoint_sha256", "")) or not binding.get("model_fingerprint"):
        raise ValueError("Trained checkpoint witness missing")
    if receipt["result"].get("policy") != "learned" or not isinstance(receipt["result"].get("history"), list):
        raise ValueError("Actual learned trial trace missing")


def witness(origin, source, evidence):
    parsed = urlsplit(origin)
    public = parsed.scheme == "https" and parsed.netloc == HOST
    local = parsed.scheme == "http" and parsed.hostname == "127.0.0.1" and parsed.port is not None
    if not (public or local) or parsed.path or parsed.query or parsed.fragment or not re.fullmatch(r"[0-9a-f]{40}", source):
        raise ValueError("Declare the public Space or local container and an exact source SHA")
    last = None
    for _ in range(60):
        try:
            status, readiness = fetch(origin, "/readyz")
            if status == 200 and readiness.get("ready") is True:
                break
        except (HTTPError, URLError, TimeoutError, ConnectionError) as exc:
            last = type(exc).__name__
        time.sleep(1)
    else:
        raise ValueError("Runtime startup did not qualify: " + str(last))
    if (readiness.get("state") != "READY" or readiness.get("models_loaded") != [17, 23, 41]
            or readiness.get("checkpoints_verified") != 3
            or readiness.get("archive_sha256") != ARCHIVE
            or readiness.get("scientific_overall_gate") != "FAILED"):
        raise ValueError("Runtime readiness witness failed")
    probes = readiness.get("startup_execution_probes", [])
    if [p.get("model_seed") for p in probes] != [17, 23, 41]:
        raise ValueError("Three startup execution witnesses required")
    for probe in probes:
        if not probe.get("binding", {}).get("checkpoint_sha256") or not probe.get("result_sha256"):
            raise ValueError("Incomplete startup model execution witness")
    evidence["readiness"] = readiness
    _, build = fetch(origin, "/api/build-info")
    evidence["build"] = build
    if (build.get("build", {}).get("state") != "OBSERVED" or build["build"].get("revision") != source
            or build.get("receipt_minted") is not False):
        raise ValueError("Observed runtime source mismatch")
    _, benchmark = fetch(origin, "/api/results")
    if benchmark.get("verification", {}).get("rows_verified") != 5184 or benchmark["data"]["primary"]["overall_pass"] is not False:
        raise ValueError("Frozen failed benchmark was not independently recomputed")
    evidence["benchmark_verification"] = benchmark["verification"]
    evidence["trials"] = []
    for seed in (17, 23, 41):
        request = {"seed": 20260929, "index": 9837, "family": "shared_bias", "policy": "learned", "model_seed": seed}
        for attempt in range(12):
            try:
                status, receipt = fetch(origin, "/api/trial", request)
                break
            except HTTPError as exc:
                if exc.code != 429 or attempt == 11:
                    raise
                delay = int(exc.headers.get("Retry-After", "5"))
                time.sleep(min(max(delay, 1), 10))
        if status != 201:
            raise ValueError("Trial was not created")
        verify_receipt(receipt, request, source)
        _, retained = fetch(origin, "/api/trials/" + receipt["id"])
        if canonical(retained) != canonical(receipt):
            raise ValueError("Receipt retention readback mismatch")
        evidence["trials"].append(receipt)
    evidence["ok"] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin", required=True)
    parser.add_argument("--expected-source", required=True)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    evidence = {"schema": "szl.foundation-public-runtime-witness/v1", "ok": False,
                "origin": args.origin, "expected_source": args.expected_source,
                "observed_at": datetime.now(timezone.utc).isoformat(), "independent_authenticity": False}
    try:
        witness(args.origin.rstrip("/"), args.expected_source, evidence)
    except Exception as exc:
        evidence["error"] = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, HTTPError) and hasattr(exc, "runtime_http_evidence"):
            evidence["http_failure"] = exc.runtime_http_evidence
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {"ok": evidence["ok"], "trials": len(evidence.get("trials", [])), "error": evidence.get("error")}
    if "http_failure" in evidence:
        summary["http_failure"] = evidence["http_failure"]
    print(json.dumps(summary))
    return 0 if evidence["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
