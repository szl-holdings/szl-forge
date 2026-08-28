#!/usr/bin/env python3
"""HTTP rerank serve for Tinku. UNAVAILABLE without a local reranker. No Hub pin."""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
BASE_MODEL = "Qwen/Qwen3-Reranker-0.6B"
RERANKER = HERE / "tinku-reranker"


def availability() -> dict[str, Any]:
    ready = RERANKER.is_dir() and any(RERANKER.rglob("*.safetensors"))
    return {
        "base_model": BASE_MODEL,
        "status": "READY" if ready else "UNAVAILABLE",
        "jobs": "UNKNOWN",
        "serve_pin": False,
    }


def check() -> int:
    print(json.dumps(availability(), indent=2))
    return 0


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        self._send(200 if self.path in ("/", "/health") else 404, availability())

    def do_POST(self) -> None:  # noqa: N802
        state = availability()
        if state["status"] != "READY":
            self._send(503, {"error": "UNAVAILABLE", "base_model": BASE_MODEL})
            return
        from sentence_transformers import CrossEncoder

        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        query = body.get("query") or ""
        documents = body.get("documents") or []
        model = CrossEncoder(str(RERANKER))
        scores = model.predict([(query, doc) for doc in documents])
        ranked = sorted(
            zip(documents, [float(s) for s in scores]),
            key=lambda item: item[1],
            reverse=True,
        )
        self._send(
            200,
            {
                "base_model": BASE_MODEL,
                "results": [{"document": d, "score": s} for d, s in ranked],
            },
        )

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        print("[tinku-serve]", args[0] if args else format)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8093)
    args = parser.parse_args()
    if args.check:
        return check()
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
