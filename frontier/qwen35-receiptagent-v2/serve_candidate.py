#!/usr/bin/env python3
"""Python generate server for qwen35-receiptagent-v2. Does not rewrite Hub cards.

Loads a local adapter directory if present. qualify_runtime.py is not serve.
base_model from candidate.json canonical Qwen/Qwen3.5-0.8B.
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
CANDIDATE = HERE / "candidate.json"
ADAPTER = HERE / "adapter"


def canonical_base() -> str:
    payload = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    return payload["canonical_base"]["repo_id"]


def availability() -> dict[str, Any]:
    ready = ADAPTER.is_dir() and (ADAPTER / "adapter_model.safetensors").is_file()
    return {
        "model": "SZLHOLDINGS/szl-receiptagent-qwen35-0.8b-v2",
        "base_model": canonical_base(),
        "status": "READY" if ready else "UNAVAILABLE",
        "proposal_only": True,
        "hub_card_rewrite": False,
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
            self._send(
                503,
                {
                    "error": "UNAVAILABLE",
                    "detail": "No local adapter/ directory. This script does not pin Hub serve.",
                    "base_model": state["base_model"],
                },
            )
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        prompt = body.get("prompt") or ""
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        base = canonical_base()
        tokenizer = AutoTokenizer.from_pretrained(ADAPTER)
        model = AutoModelForCausalLM.from_pretrained(base)
        model = PeftModel.from_pretrained(model, ADAPTER)
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            tokens = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        text = tokenizer.decode(tokens[0], skip_special_tokens=True)
        self._send(200, {"base_model": base, "text": text, "proposal_only": True})

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        print("[qwen35-v2-serve]", args[0] if args else format)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8096)
    args = parser.parse_args()
    if args.check:
        return check()
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
