#!/usr/bin/env python3
"""Private, unpaid ReceiptAgent 1.5B draft proposer. Does not rewrite Hub cards.

Loads local `receiptagent-model/`. PowerShell rebirth is not this serve path.
POST JSON requires prompt and evidence_allowlist (sourceEndpoint/label pairs).
Only schema-valid, allowlisted drafts or structured refusals are returned.
Caller-supplied evidence is not authenticated or fact-checked by this gate.
Training/evaluation signatures do not sign or authorize these runtime outputs.
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

HERE = Path(__file__).resolve().parent
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MERGED = HERE / "receiptagent-model"
SCHEMA = json.loads((HERE / "receiptagent.schema.json").read_text(encoding="utf-8"))
DRAFT_VALIDATOR = Draft202012Validator(SCHEMA)
# Refusal is a separate closed controller envelope, not a successful draft.
# Reuse the existing schema's identity and non-execution restrictions unchanged.
REFUSAL_FIELDS = (
    "proposalId", "capabilityProfile", "provenance", "approvalRequired",
    "controllerBoundary", "receiptBinding", "executed",
)
REFUSAL_VALIDATOR = Draft202012Validator({
    "type": "object",
    "properties": {
        **{key: SCHEMA["properties"][key] for key in REFUSAL_FIELDS},
        "decision": {"const": "REFUSE"},
        "reason": {"type": "string", "minLength": 1},
    },
    "required": [*REFUSAL_FIELDS, "decision", "reason"],
    "additionalProperties": False,
})
LABELS = frozenset(
    SCHEMA["properties"]["evidence"]["items"]["properties"]["label"]["enum"]
)


class GateError(ValueError):
    """Explicit rejection without returning unvalidated model text."""

    def __init__(self, code: str, status: int = 400):
        super().__init__(code)
        self.code = code
        self.status = status


def strict_json_object(text: str) -> dict[str, Any]:
    if not isinstance(text, str):
        raise GateError("INVALID_JSON")

    def unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise GateError("INVALID_JSON")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise GateError("INVALID_JSON")

    try:
        text.encode("utf-8", errors="strict")
        result = json.loads(text, object_pairs_hook=unique_keys, parse_constant=reject_constant)
        # Reject escaped unpaired surrogates and numeric overflow (e.g. 1e309)
        # after decoding; parse_constant alone does not catch overflow floats.
        json.dumps(result, ensure_ascii=False, allow_nan=False).encode("utf-8", errors="strict")
    except (ValueError, UnicodeError, TypeError, RecursionError) as exc:
        raise GateError("INVALID_JSON") from exc
    if not isinstance(result, dict):
        raise GateError("INVALID_JSON")
    return result


def evidence_pairs(allowlist: Any) -> set[tuple[str, str]]:
    if not isinstance(allowlist, list) or not allowlist:
        raise GateError("EVIDENCE_ALLOWLIST_REQUIRED")
    pairs: set[tuple[str, str]] = set()
    endpoints: set[str] = set()
    for item in allowlist:
        if not isinstance(item, dict) or set(item) != {"sourceEndpoint", "label"}:
            raise GateError("INVALID_EVIDENCE_ALLOWLIST")
        endpoint, label = item["sourceEndpoint"], item["label"]
        if (
            not isinstance(endpoint, str) or not endpoint.strip()
            or not isinstance(label, str) or label not in LABELS
            or endpoint in endpoints
        ):
            raise GateError("INVALID_EVIDENCE_ALLOWLIST")
        endpoints.add(endpoint)
        pairs.add((endpoint, label))
    return pairs


def accept_output(text: str, allowlist: Any) -> dict[str, Any]:
    allowed = evidence_pairs(allowlist)
    try:
        result = strict_json_object(text)
    except GateError as exc:
        raise GateError("OUTPUT_REJECTED", 422) from exc
    validator = REFUSAL_VALIDATOR if result.get("decision") == "REFUSE" else DRAFT_VALIDATOR
    if not validator.is_valid(result) or result.get("provenance") != "MODEL_PROPOSED":
        raise GateError("OUTPUT_REJECTED", 422)
    for key in ("proposalId", "controllerBoundary", "reason" if validator is REFUSAL_VALIDATOR else "claim"):
        if not result[key].strip():
            raise GateError("OUTPUT_REJECTED", 422)
    for citation in result.get("evidence", []):
        if (citation["sourceEndpoint"], citation["label"]) not in allowed:
            raise GateError("OUTPUT_REJECTED", 422)
    return result


def propose(body: dict[str, Any]) -> dict[str, Any]:
    if (
        not isinstance(body, dict)
        or set(body) - {"prompt", "max_new_tokens", "evidence_allowlist"}
        or not isinstance(body.get("prompt"), str) or not body["prompt"].strip()
    ):
        raise GateError("INVALID_REQUEST")
    evidence_pairs(body.get("evidence_allowlist"))
    limit = body.get("max_new_tokens", 256)
    if type(limit) is not int or not 1 <= limit <= 256:
        raise GateError("INVALID_TOKEN_LIMIT")
    if availability()["status"] != "READY":
        raise GateError("UNAVAILABLE", 503)
    try:
        completion = generate(body["prompt"], limit)
    except (ImportError, OSError, RuntimeError) as exc:
        raise GateError("GENERATION_UNAVAILABLE", 503) from exc
    result = accept_output(completion, body["evidence_allowlist"])
    return {"base_model": BASE_MODEL, "result": result, "proposal_only": True}


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
    prompt_length = inputs["input_ids"].shape[1]
    return tokenizer.decode(tokens[0][prompt_length:], skip_special_tokens=True)


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
        try:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as exc:
                raise GateError("INVALID_CONTENT_LENGTH") from exc
            if not 1 <= length <= 65_536 or self.headers.get("Transfer-Encoding"):
                raise GateError("INVALID_CONTENT_LENGTH")
            try:
                body = strict_json_object(self.rfile.read(length).decode("utf-8"))
            except UnicodeError as exc:
                raise GateError("INVALID_JSON") from exc
            response = propose(body)
        except GateError as exc:
            self._send(exc.status, {"error": exc.code, "base_model": BASE_MODEL})
            return
        self._send(200, response)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        print("[receiptagent-serve]", args[0] if args else format)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--host", default="127.0.0.1", choices=("127.0.0.1", "localhost"))
    parser.add_argument("--port", type=int, default=8095)
    args = parser.parse_args()
    if args.check:
        return check()
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
