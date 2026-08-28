#!/usr/bin/env python3
"""Proposal-only CHASKI-R2 generate server. Separate SKU. No Hub pin.

CANONICAL_BASE = Qwen/Qwen3.5-0.8B
Does not load SZLHOLDINGS/chaski. Does not pin the 5050 kit.
House CPU lab stays signed Khipu GGUF. A11OY-MINI stays scripts-only.
Jobs UNAVAILABLE this checkout. No ROADMAP parking. No Hub PUT.
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
CANONICAL_BASE = "Qwen/Qwen3.5-0.8B"
HUB = "SZLHOLDINGS/chaski-r2"
FORBIDDEN_HUB = "SZLHOLDINGS/chaski"
FORBIDDEN_5050 = "SZLHOLDINGS/chaski-5050"
ADAPTER_DIR = HERE / "chaski-r2-adapter"


def availability() -> dict[str, Any]:
    adapter_present = ADAPTER_DIR.is_dir() and any(ADAPTER_DIR.glob("*.safetensors"))
    return {
        "model": HUB,
        "sku": "CHASKI-R2",
        "separate_sku": True,
        "does_not_overwrite": FORBIDDEN_HUB,
        "forbidden_5050": FORBIDDEN_5050,
        "canonical_base": CANONICAL_BASE,
        "base_model": CANONICAL_BASE,
        "proposal_only": True,
        "serve_pin": False,
        "inference_lab_pin": False,
        "khipu_lab_pin": False,
        "a11oy_mini_scripts_only": True,
        "hub_put": False,
        "jobs": "UNAVAILABLE",
        "adapter": "LOCAL" if adapter_present else "UNAVAILABLE",
        "weights": "LOCAL" if adapter_present else "UNAVAILABLE",
        "quality": "UNAVAILABLE",
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
                        "No local chaski-r2-adapter. Jobs UNAVAILABLE. "
                        "No Hub serve pin. "
                        f"Does not load {FORBIDDEN_HUB} or {FORBIDDEN_5050}."
                    ),
                    "canonical_base": CANONICAL_BASE,
                    "base_model": CANONICAL_BASE,
                    "jobs": "UNAVAILABLE",
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

        tokenizer = AutoTokenizer.from_pretrained(CANONICAL_BASE)
        base = AutoModelForCausalLM.from_pretrained(CANONICAL_BASE)
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
                "canonical_base": CANONICAL_BASE,
                "base_model": CANONICAL_BASE,
                "text": text,
                "proposal_only": True,
                "does_not_overwrite": FORBIDDEN_HUB,
            },
        )

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        print("[chaski-r2-serve]", args[0] if args else format)


def serve(host: str, port: int) -> int:
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(
        f"[chaski-r2-serve] {host}:{port} canonical_base={CANONICAL_BASE} "
        "pin=false jobs=UNAVAILABLE"
    )
    httpd.serve_forever()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8098)
    args = parser.parse_args()
    if args.check:
        return check()
    return serve(args.host, args.port)


if __name__ == "__main__":
    raise SystemExit(main())
