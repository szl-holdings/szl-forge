#!/usr/bin/env python3
"""Proposal-only Chaski-5050 generate server.

Pins SZLHOLDINGS/chaski-5050. Does not pin live SZLHOLDINGS/chaski.
House CPU lab serves Khipu GGUF, not this kit.
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
CANONICAL_BASE = "Qwen/Qwen3.5-0.8B"
HUB = "SZLHOLDINGS/chaski-5050"
LIVE_CHASKI_HUB = "SZLHOLDINGS/chaski"
ADAPTER_DIR = HERE / "chaski-5050-adapter"


def availability() -> dict[str, Any]:
    adapter_present = ADAPTER_DIR.is_dir() and any(ADAPTER_DIR.glob("*.safetensors"))
    return {
        "model": HUB,
        "serve_pin": HUB,
        "not_live_chaski": LIVE_CHASKI_HUB,
        "live_chaski_pin": False,
        "base_model": CANONICAL_BASE,
        "proposal_only": True,
        "inference_lab_pin": False,
        "adapter": "LOCAL" if adapter_present else "UNAVAILABLE",
        "weights": "LOCAL" if adapter_present else "UNAVAILABLE",
        "jobs": "not-an-hf-job",
        "hardware": "local-RTX-5050",
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
                    "detail": (
                        "No local chaski-5050 adapter. "
                        f"Pin is {HUB}, not {LIVE_CHASKI_HUB}."
                    ),
                    "base_model": CANONICAL_BASE,
                    "serve_pin": HUB,
                },
            )
            return
        self._send(
            501,
            {
                "error": "NOT_PINNED_TO_LIVE",
                "detail": (
                    f"Local 5050 adapter present. Serve pin is {HUB}. "
                    f"Does not pin {LIVE_CHASKI_HUB}."
                ),
                "base_model": CANONICAL_BASE,
                "serve_pin": HUB,
            },
        )

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        print("[chaski-5050-serve]", args[0] if args else format)


def serve(host: str, port: int) -> int:
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(
        f"[chaski-5050-serve] {host}:{port} pin={HUB} "
        f"not={LIVE_CHASKI_HUB} base_model={CANONICAL_BASE}"
    )
    httpd.serve_forever()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8092)
    args = parser.parse_args()
    if args.check:
        return check()
    return serve(args.host, args.port)


if __name__ == "__main__":
    raise SystemExit(main())
