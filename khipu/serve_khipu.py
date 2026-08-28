#!/usr/bin/env python3
"""Python generate server for signed SZL-Khipu-1.5B. Does not rewrite Hub cards.

Loads local `khipu-model/` (transformers) or a local GGUF via llama-cpp-python.
PowerShell rebirth is not this serve path.
House CPU lab `spaces/szl-model-inference-lab` remains the GGUF studio pin.
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MERGED = HERE / "khipu-model"
ADAPTER = HERE / "khipu-adapter"


def availability() -> dict[str, Any]:
    merged = MERGED.is_dir() and any(MERGED.glob("*.safetensors"))
    gguf = list(HERE.glob("*.gguf"))
    return {
        "model": "khipu",
        "base_model": BASE_MODEL,
        "status": "READY" if (merged or gguf) else "UNAVAILABLE",
        "backend": "transformers" if merged else ("gguf" if gguf else None),
        "proposal_only": True,
    }


def load_backend() -> tuple[str, Any, Any]:
    state = availability()
    if state["backend"] == "transformers":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MERGED)
        model = AutoModelForCausalLM.from_pretrained(MERGED)
        return "transformers", model, tokenizer
    if state["backend"] == "gguf":
        from llama_cpp import Llama

        gguf = next(HERE.glob("*.gguf"))
        return "gguf", Llama(model_path=str(gguf), n_ctx=2048), None
    raise FileNotFoundError("no local khipu-model or GGUF")


def generate(prompt: str, max_new_tokens: int = 256) -> str:
    kind, model, tokenizer = load_backend()
    if kind == "gguf":
        out = model(prompt, max_tokens=max_new_tokens, temperature=0)
        return out["choices"][0]["text"]
    import torch

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
                    "detail": "No local khipu-model or GGUF. Train/rebirth first.",
                    "base_model": BASE_MODEL,
                },
            )
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        prompt = body.get("prompt") or ""
        text = generate(prompt, int(body.get("max_new_tokens") or 256))
        self._send(200, {"base_model": BASE_MODEL, "text": text, "proposal_only": True})

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        print("[khipu-serve]", args[0] if args else format)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8094)
    args = parser.parse_args()
    if args.check:
        return check()
    print(f"[khipu-serve] {args.host}:{args.port} base_model={BASE_MODEL}")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
