#!/usr/bin/env python3
"""Proposal-only KHIPU-R2 generate server. Separate SKU. No Hub pin.

base_model = Qwen/Qwen2.5-1.5B-Instruct
Live Hub adapter is AVAILABLE (147.8MB). This server does not pin Hub serve
and does not load signed SZL-Khipu-1.5B. House CPU lab stays signed Khipu GGUF.
This-kit jobs UNKNOWN. No Hub PUT.

CHAWPI extra lock: Hub receipt ddf6c50 publication_eligible false is the
public claim. Launcher still no --run-job. r=32 α=64 this SKU.
stale profile key dropped. Signed 1.5B stays 2/6. Do not merge #64.
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
HUB_JOB_ID = "6a91bf11984507d9db4ea104"
HUB_JOB_STATUS = "COMPLETED"
HUB_ADAPTER_STATUS = "AVAILABLE"
HUB_ADAPTER_SIZE = "147.8MB"
HUB_RECEIPT_COMMIT = "ddf6c50d8baa9f818b9f478086e7b5919eb773cf"
CHAWPI = "hub-receipt-ddf6c50-publication-eligible-false"


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
        "lab": "signed Khipu GGUF",
        "hub_put": False,
        "hub_job_id": HUB_JOB_ID,
        "hub_job_status": HUB_JOB_STATUS,
        "hub_adapter": HUB_ADAPTER_STATUS,
        "hub_adapter_size": HUB_ADAPTER_SIZE,
        "hub_receipt_commit": HUB_RECEIPT_COMMIT,
        "chawpi": CHAWPI,
        "jobs": "UNKNOWN",
        "jobs_scope": "this-kit",
        "adapter": "LOCAL" if adapter_present else "UNAVAILABLE",
        "local_adapter": "LOCAL" if adapter_present else "UNAVAILABLE",
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
                        "No local khipu-r2-adapter. Hub adapter is AVAILABLE "
                        f"({HUB_ADAPTER_SIZE}) but this server does not pin Hub. "
                        "Lab stays signed Khipu GGUF. "
                        f"Does not load {FORBIDDEN_HUB}."
                    ),
                    "base_model": BASE_MODEL,
                    "hub_adapter": HUB_ADAPTER_STATUS,
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
        f"hub_adapter={HUB_ADAPTER_STATUS} pin=false jobs=UNKNOWN"
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
