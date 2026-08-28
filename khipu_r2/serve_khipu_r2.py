#!/usr/bin/env python3
"""Proposal-only KHIPU-R2 generate server. Separate SKU. No Hub pin.

base_model = Qwen/Qwen2.5-1.5B-Instruct
ROADMAP until a local adapter file lands. Does not load signed SZL-Khipu-1.5B.
House CPU lab stays on signed Khipu GGUF. Jobs UNKNOWN. No Hub PUT.
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
HUB = "SZLHOLDINGS/KHIPU-R2"
FORBIDDEN_HUB = "SZLHOLDINGS/SZL-Khipu-1.5B"
ADAPTER_DIR = HERE / "khipu-r2-adapter"


def availability() -> dict[str, Any]:
    adapter_present = ADAPTER_DIR.is_dir() and any(ADAPTER_DIR.glob("*.safetensors"))
    return {
        "model": HUB,
        "sku": "KHIPU-R2",
        "separate_sku": True,
        "does_not_overwrite": FORBIDDEN_HUB,
        "base_model": BASE_MODEL,
        "proposal_only": True,
        "serve_pin": False,
        "inference_lab_pin": False,
        "hub_put": False,
        "jobs": "UNKNOWN",
        "adapter": "LOCAL" if adapter_present else "UNAVAILABLE",
        "weights": "LOCAL" if adapter_present else "UNAVAILABLE",
        "card_status": "ROADMAP",
        "publication_eligible": False,
        "status": "READY" if adapter_present else "UNAVAILABLE",
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
                        "No local khipu-r2-adapter. ROADMAP. No Hub serve pin. "
                        f"Does not load {FORBIDDEN_HUB}."
                    ),
                    "base_model": BASE_MODEL,
                    "jobs": "UNKNOWN",
                    "publication_eligible": False,
                },
            )
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        prompt = body.get("prompt") or ""
        max_new_tokens = int(body.get("max_new_tokens") or 256)
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
        model = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            tokens = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        text = tokenizer.decode(tokens[0], skip_special_tokens=True)
        self._send(
            200,
            {
                "base_model": BASE_MODEL,
                "text": text,
                "proposal_only": True,
                "does_not_overwrite": FORBIDDEN_HUB,
            },
        )

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        print("[khipu-r2-serve]", args[0] if args else format)


def serve(host: str, port: int) -> int:
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(
        f"[khipu-r2-serve] {host}:{port} base_model={BASE_MODEL} "
        "pin=false jobs=UNKNOWN"
    )
    httpd.serve_forever()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8096)
    args = parser.parse_args()
    if args.check:
        return check()
    return serve(args.host, args.port)


if __name__ == "__main__":
    raise SystemExit(main())
