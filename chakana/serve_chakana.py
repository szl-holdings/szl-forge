#!/usr/bin/env python3
"""HTTP embed + cosine retrieve for Chakana. UNAVAILABLE without a local encoder.

query→vector. Optional Matryoshka truncate 256/512/1024. No Hub pin.
Jobs UNKNOWN. Not a11oy CHAKANA wiring / tinkuy.

base_model = Qwen/Qwen3-Embedding-0.6B
"""
from __future__ import annotations

import argparse
import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
BASE_MODEL = "Qwen/Qwen3-Embedding-0.6B"
HUB = "SZLHOLDINGS/chakana"
ENCODER = HERE / "chakana-encoder"
MATRYOSHKA_DIMS = (256, 512, 1024)

_MODEL: Any = None


def encoder_present() -> bool:
    return ENCODER.is_dir() and any(ENCODER.rglob("*.safetensors"))


def availability() -> dict[str, Any]:
    ready = encoder_present()
    return {
        "model": HUB,
        "lane": "NINA (FORGE-class)",
        "owner": "Stephen Lutar",
        "base_model": BASE_MODEL,
        "library_name": "sentence-transformers",
        "pipeline_tag": "feature-extraction",
        "matryoshka_dims": list(MATRYOSHKA_DIMS),
        "serve_pin": False,
        "hub_put": False,
        "jobs": "UNKNOWN",
        "encoder": "LOCAL" if ready else "UNAVAILABLE",
        "weights": "LOCAL" if ready else "UNAVAILABLE",
        "card_status": "ROADMAP",
        "publication_eligible": False,
        "not_a11oy_chakana_wiring": True,
        "status": "READY" if ready else "UNAVAILABLE",
    }


def check() -> int:
    print(json.dumps(availability(), indent=2))
    return 0


def load_model() -> Any:
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer

        _MODEL = SentenceTransformer(str(ENCODER))
    return _MODEL


def l2_normalize(vec: list[float]) -> list[float]:
    denom = math.sqrt(sum(value * value for value in vec)) or 1.0
    return [value / denom for value in vec]


def truncate_matryoshka(vec: list[float], dim: int | None) -> list[float]:
    if dim is None:
        return vec
    if dim not in MATRYOSHKA_DIMS:
        raise ValueError(
            f"matryoshka dim must be one of {list(MATRYOSHKA_DIMS)}, got {dim}"
        )
    if dim > len(vec):
        raise ValueError(f"vector length {len(vec)} < requested dim {dim}")
    return l2_normalize(vec[:dim])


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def encode_texts(texts: list[str], dim: int | None) -> list[list[float]]:
    model = load_model()
    raw = model.encode(texts, convert_to_numpy=True).tolist()
    return [truncate_matryoshka(vec, dim) for vec in raw]


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length) or b"{}")

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
                        "No local chakana-encoder. ROADMAP. No Hub serve pin. "
                        "Not a11oy CHAKANA wiring."
                    ),
                    "base_model": BASE_MODEL,
                    "jobs": "UNKNOWN",
                    "publication_eligible": False,
                },
            )
            return
        try:
            body = self._read_json()
            dim = body.get("dim") or body.get("matryoshka_dim")
            if dim is not None:
                dim = int(dim)
            if self.path in ("/embed", "/v1/embeddings"):
                texts = body.get("texts") or body.get("input") or []
                if isinstance(texts, str):
                    texts = [texts]
                vectors = encode_texts([str(item) for item in texts], dim)
                self._send(
                    200,
                    {
                        "base_model": BASE_MODEL,
                        "dim": dim or (len(vectors[0]) if vectors else None),
                        "embeddings": vectors,
                    },
                )
                return
            if self.path == "/retrieve":
                query = str(body.get("query") or "")
                corpus = [str(item) for item in (body.get("corpus") or [])]
                top_k = int(body.get("top_k") or 10)
                encoded = encode_texts([query, *corpus], dim)
                query_vec, doc_vecs = encoded[0], encoded[1:]
                ranked = sorted(
                    (
                        {"text": text, "score": cosine(query_vec, vec)}
                        for text, vec in zip(corpus, doc_vecs)
                    ),
                    key=lambda item: item["score"],
                    reverse=True,
                )
                self._send(
                    200,
                    {
                        "base_model": BASE_MODEL,
                        "query": query,
                        "results": ranked[:top_k],
                    },
                )
                return
        except ValueError as exc:
            self._send(400, {"error": str(exc), "base_model": BASE_MODEL})
            return
        self._send(404, {"detail": "not found"})

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        print("[chakana-serve]", args[0] if args else format)


def serve(host: str, port: int) -> int:
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(
        f"[chakana-serve] {host}:{port} base_model={BASE_MODEL} "
        "pin=false jobs=UNKNOWN"
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
