#!/usr/bin/env python3
"""Proposal-only Chaski generate server. No Hub adapter pin. No inference-lab pin.

base_model = Qwen/Qwen3.5-0.8B
CUTTING until a local adapter file lands. House CPU lab serves Khipu GGUF only.
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
BASE_MODEL = "Qwen/Qwen3.5-0.8B"
HUB = "SZLHOLDINGS/chaski"
ADAPTER_DIR = HERE / "chaski-adapter"


def availability() -> dict[str, Any]:
    adapter_present = ADAPTER_DIR.is_dir() and any(ADAPTER_DIR.glob("*.safetensors"))
    return {
        "model": HUB,
        "base_model": BASE_MODEL,
        "proposal_only": True,
        "serve_pin": False,
        "inference_lab_pin": False,
        "adapter": "LOCAL" if adapter_present else "UNAVAILABLE",
        "weights": "UNAVAILABLE" if not adapter_present else "LOCAL",
        "card_status": "CUTTING",
        "status": "READY" if adapter_present else "UNAVAILABLE",
    }


def check() -> int:
    payload = availability()
    print(json.dumps(payload, indent=2))
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
        if self.path in ("/", "/health", "/v1/models"):
            self._send(200, availability())
            return
        self._send(404, {"detail": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        state = availability()
        if state["status"] != "READY":
            self._send(
                503,
                {
                    "error": "UNAVAILABLE",
                    "detail": "No local adapter file. CUTTING. No Hub serve pin.",
                    "base_model": BASE_MODEL,
                },
            )
            return
        self._send(
            501,
            {
                "error": "NOT_PINNED",
                "detail": "Local adapter present but this server does not pin Hub serve.",
                "base_model": BASE_MODEL,
            },
        )

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        print("[chaski-serve]", args[0] if args else format)


def serve(host: str, port: int) -> int:
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"[chaski-serve] {host}:{port} base_model={BASE_MODEL} pin=false")
    httpd.serve_forever()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    args = parser.parse_args()
    if args.check:
        return check()
    return serve(args.host, args.port)


if __name__ == "__main__":
    raise SystemExit(main())
