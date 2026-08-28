#!/usr/bin/env python3
"""Python generate server for signed ReceiptAgent 1.5B. Does not rewrite Hub cards.

Loads local `receiptagent-model/`. PowerShell rebirth is not this serve path.
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MERGED = HERE / "receiptagent-model"


def availability() -> dict[str, Any]:
    ready = MERGED.is_dir() and any(MERGED.glob("*.safetensors"))
    return {
        "model": "receiptagent",
        "base_model": BASE_MODEL,
        "status": "READY" if ready else "UNAVAILABLE",
        "proposal_only": True,
    }


def generate(prompt: str, max_new_tokens: int = 256) -> str:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(MERGED)
    model = AutoModelForCausalLM.from_pretrained(MERGED)
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        tokens = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(tokens[0], skip_special_tokens=True)


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
            self._send(
                503,
                {
                    "error": "UNAVAILABLE",
                    "detail": "No local receiptagent-model. Train first.",
                    "base_model": BASE_MODEL,
                },
            )
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        text = generate(body.get("prompt") or "", int(body.get("max_new_tokens") or 256))
        self._send(200, {"base_model": BASE_MODEL, "text": text, "proposal_only": True})

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        print("[receiptagent-serve]", args[0] if args else format)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8095)
    args = parser.parse_args()
    if args.check:
        return check()
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
