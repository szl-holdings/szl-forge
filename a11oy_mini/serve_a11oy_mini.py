#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# (c) 2026 Lutar, Stephen P. - SZL Holdings
"""Proposal-only A11OY-MINI llama.cpp/GGUF serve pin.

Pins SZLHOLDINGS/A11OY-MINI. Refuses live SZLHOLDINGS/chaski overwrite.
Refuses SZLHOLDINGS/chaski-5050 as parent. House CPU lab stays Khipu GGUF.
No tok/s. No Hub PUT. ROADMAP until a local .gguf exists.
"""
from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import convert_a11oy_mini_gguf as convert  # noqa: E402


def local_gguf() -> Path | None:
    for name in (convert.Q4_NAME, convert.F16_NAME):
        path = HERE / name
        if path.is_file():
            return path
    found = sorted(HERE.glob("*.gguf"))
    return found[0] if found else None


def availability() -> dict[str, Any]:
    gguf = local_gguf()
    return {
        "model": convert.SKU,
        "serve_pin": convert.SKU,
        "parent": convert.PARENT,
        "forbidden_parent": convert.FORBIDDEN_PARENT,
        "live_chaski_overwrite": False,
        "parent_5050": False,
        "khipu_lab_pin": False,
        "inference_lab_pin": False,
        "khipu_lab": convert.KHIPU_LAB,
        "base_model": convert.CANONICAL_BASE,
        "proposal_only": True,
        "publication_eligible": False,
        "evals": "none-this-run",
        "quality": "ROADMAP",
        "gguf": "LOCAL" if gguf else "UNAVAILABLE",
        "gguf_path": str(gguf) if gguf else None,
        "status": "READY" if gguf else "UNAVAILABLE",
        "tok_s_claim": False,
        "hub_put": False,
        "backend": "llama.cpp-gguf" if gguf else None,
        "card_status": "ROADMAP",
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
                        "No local A11OY-MINI GGUF. ROADMAP until a .gguf exists. "
                        f"Pin is {convert.SKU}. Parent is live {convert.PARENT}, "
                        f"not {convert.FORBIDDEN_PARENT}. Lab stays Khipu. "
                        "No live Chaski overwrite. No Hub PUT. No tok/s."
                    ),
                    "serve_pin": convert.SKU,
                    "parent": convert.PARENT,
                    "khipu_lab_pin": False,
                    "live_chaski_overwrite": False,
                },
            )
            return
        self._send(
            501,
            {
                "error": "NOT_PINNED",
                "detail": (
                    "Local GGUF present. This server is proposal-only and does "
                    "not retarget the Khipu inference lab or overwrite live Chaski."
                ),
                "serve_pin": convert.SKU,
                "khipu_lab_pin": False,
                "live_chaski_overwrite": False,
                "tok_s_claim": False,
            },
        )

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        print("[a11oy-mini-serve]", args[0] if args else format)


def serve(host: str, port: int) -> int:
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(
        f"[a11oy-mini-serve] {host}:{port} pin={convert.SKU} "
        f"parent={convert.PARENT} lab=Khipu"
    )
    httpd.serve_forever()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8092)
    parser.add_argument("--hub-id", default=convert.SKU)
    args = parser.parse_args()
    convert.refuse_live_chaski_overwrite(args.hub_id)
    convert.refuse_5050_parent(args.hub_id)
    if args.hub_id not in {convert.SKU, "A11OY-MINI"}:
        raise convert.ConvertError(
            f"[a11oy-mini] serve pin is {convert.SKU}, not {args.hub_id}"
        )
    if args.check:
        return check()
    return serve(args.host, args.port)


if __name__ == "__main__":
    raise SystemExit(main())
