"""Recompute an SZL unsigned execution-record hash using only Python stdlib.

Usage:
    python verify_execution_record.py response.json
    python verify_execution_record.py response.json --request request.json
    type response.json | python verify_execution_record.py

The input may be the record itself or a complete chat-completion response.
Verification establishes internal hash consistency only. It does not establish
authenticity because the record is intentionally unsigned.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


EXCLUDED_HASH_FIELDS = {"record_sha256"}
EXPECTED_MODEL_ID = (
    "SZLHOLDINGS/SZL-Khipu-1.5B-GGUF@"
    "67d60ec577730747055491640cfb91fc4a4b5d25"
)
EXPECTED_MODEL = {
    "id": EXPECTED_MODEL_ID,
    "repo": "SZLHOLDINGS/SZL-Khipu-1.5B-GGUF",
    "revision": "67d60ec577730747055491640cfb91fc4a4b5d25",
    "file": "SZL-Khipu-1.5B-Q4_K_M.gguf",
    "sha256": "13c1a1993063e1dff92f7413ccf48eaca6d48efc8801ae9af35961ae3396623a",
}
EXPECTED_SPACE_ID = "SZLHOLDINGS/szl-model-inference-lab"
EXPECTED_RELEASE_ID = "brain13-1d3960c-controller-9f227f6"
EXPECTED_RECORD_HASH_SCOPE = (
    "canonical UTF-8 JSON with sorted keys and compact separators, "
    "excluding record_sha256"
)
EXPECTED_PERSISTENCE_BOUNDARY = (
    "platform or network logging outside this source is not asserted"
)
EXPECTED_RECORD_FIELDS = {
    "schema",
    "request_id",
    "created_unix",
    "canonical_request_sha256",
    "output_sha256",
    "model",
    "source",
    "usage",
    "elapsed_ms",
    "termination",
    "signature_status",
    "signature",
    "authenticity_not_established",
    "persistence",
    "record_sha256_scope",
    "record_sha256",
}
ALLOWED_REQUEST_FIELDS = {
    "model",
    "messages",
    "max_tokens",
    "max_completion_tokens",
    "temperature",
    "top_p",
    "n",
    "stream",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
}
HEX_64 = re.compile(r"^[0-9a-f]{64}$")


def canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def extract_record(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema") == "szl.unsigned-execution-record/v1":
        return payload
    try:
        record = payload["szl_provenance"]["execution_record"]
    except (KeyError, TypeError) as exc:
        raise ValueError("input does not contain an SZL execution record") from exc
    if not isinstance(record, dict):
        raise ValueError("execution record must be a JSON object")
    return record


def extract_output(payload: dict[str, Any]) -> str | None:
    if payload.get("object") != "chat.completion":
        return None
    try:
        output = payload["choices"][0]["message"]["content"]
    except (IndexError, KeyError, TypeError) as exc:
        raise ValueError("chat-completion response does not contain output text") from exc
    if not isinstance(output, str):
        raise ValueError("chat-completion output must be a string")
    return output


def normalize_chat_request(payload: dict[str, Any]) -> dict[str, Any]:
    """Mirror the server's canonical valid-request subset using only stdlib."""

    if not isinstance(payload, dict):
        raise ValueError("request must be a JSON object")
    extras = sorted(set(payload) - ALLOWED_REQUEST_FIELDS)
    if extras:
        raise ValueError(f"request contains unsupported field(s): {', '.join(extras)}")
    try:
        model = payload["model"]
        raw_messages = payload["messages"]
    except KeyError as exc:
        raise ValueError(f"request is missing {exc.args[0]}") from exc
    if model != EXPECTED_MODEL_ID or not isinstance(raw_messages, list):
        raise ValueError("request model must be a string and messages must be a list")
    if not 1 <= len(raw_messages) <= 12:
        raise ValueError("request messages must contain between 1 and 12 items")
    messages = []
    for raw in raw_messages:
        if not isinstance(raw, dict):
            raise ValueError("each request message must be an object")
        if set(raw) != {"role", "content"}:
            raise ValueError("each request message must contain only role and content")
        role, content = raw.get("role"), raw.get("content")
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            raise ValueError("each request message needs string role and content")
        if len(content) > 1_200:
            raise ValueError("message content exceeds 1200 characters")
        cleaned = content.strip()
        if (
            not cleaned
            or "\x00" in cleaned
            or "<|im_start|>" in cleaned
            or "<|im_end|>" in cleaned
            or "<|endoftext|>" in cleaned
        ):
            raise ValueError("message content is empty or contains a forbidden token")
        messages.append({"role": role, "content": cleaned})
    if messages[-1]["role"] != "user":
        raise ValueError("the final message must have role user")
    if sum(len(message["content"]) for message in messages) > 1_200:
        raise ValueError("combined message content exceeds 1200 characters")

    max_tokens = payload.get("max_tokens")
    max_completion_tokens = payload.get("max_completion_tokens")
    if (
        max_tokens is not None
        and max_completion_tokens is not None
        and max_tokens != max_completion_tokens
    ):
        raise ValueError("request token-limit fields do not match")
    raw_generation_limit = (
        max_completion_tokens
        if max_completion_tokens is not None
        else max_tokens if max_tokens is not None else 24
    )
    if isinstance(raw_generation_limit, bool) or not isinstance(raw_generation_limit, int):
        raise ValueError("request token limit must be a JSON integer")
    else:
        generation_limit = raw_generation_limit
    if not 1 <= generation_limit <= 32:
        raise ValueError("request token limit must be between 1 and 32")

    raw_temperature = payload.get("temperature", 0.0)
    raw_top_p = payload.get("top_p", 1.0)
    raw_n = payload.get("n", 1)
    if (
        isinstance(raw_temperature, bool)
        or not isinstance(raw_temperature, (int, float))
        or isinstance(raw_top_p, bool)
        or not isinstance(raw_top_p, (int, float))
    ):
        raise ValueError("temperature and top_p must be JSON numbers")
    if isinstance(raw_n, bool) or not isinstance(raw_n, int):
        raise ValueError("n must be a JSON integer")
    try:
        temperature = float(raw_temperature)
        top_p = float(raw_top_p)
        n = raw_n
    except (TypeError, ValueError) as exc:
        raise ValueError("temperature, top_p, and n must be numeric") from exc
    if temperature != 0.0 or top_p != 1.0 or n != 1:
        raise ValueError("request decoding parameters are outside the bounded subset")
    stream = payload.get("stream", False)
    if not isinstance(stream, bool) or stream is not False:
        raise ValueError("streaming requests are outside the bounded subset")
    for field in ("tools", "tool_choice", "parallel_tool_calls"):
        if payload.get(field) is not None:
            raise ValueError(f"{field} is outside the bounded subset")
    return {
        "schema": "szl.openai-chat-request/v1",
        "model": model,
        "messages": messages,
        "max_completion_tokens": generation_limit,
        "temperature": 0.0,
        "top_p": 1.0,
        "n": 1,
        "stream": False,
        "tools": None,
    }


def normalized_request_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(normalize_chat_request(payload))).hexdigest()


def recompute_record_sha256(record: dict[str, Any]) -> str:
    core = {key: value for key, value in record.items() if key not in EXCLUDED_HASH_FIELDS}
    return hashlib.sha256(canonical_json_bytes(core)).hexdigest()


def record_semantic_errors(record: dict[str, Any]) -> list[str]:
    """Validate the fixed semantics of this release without claiming authenticity."""

    errors: list[str] = []
    if set(record) != EXPECTED_RECORD_FIELDS:
        errors.append("execution-record fields do not match the release schema")
    if record.get("schema") != "szl.unsigned-execution-record/v1":
        errors.append("unsupported execution-record schema")
    if record.get("signature_status") != "UNSIGNED" or record.get("signature") is not None:
        errors.append("record must remain explicitly unsigned")
    if record.get("authenticity_not_established") is not True:
        errors.append("record must state that authenticity is not established")
    if record.get("record_sha256_scope") != EXPECTED_RECORD_HASH_SCOPE:
        errors.append("record hash scope does not match the release contract")
    if record.get("model") != EXPECTED_MODEL:
        errors.append("record model identity does not match the immutable release")

    source = record.get("source")
    if not isinstance(source, dict):
        errors.append("record source must be an object")
    else:
        if set(source) != {"space_id", "release_id", "release_manifest_sha256"}:
            errors.append("record source fields do not match the release schema")
        if source.get("space_id") != EXPECTED_SPACE_ID:
            errors.append("record Space identity does not match the release")
        if source.get("release_id") != EXPECTED_RELEASE_ID:
            errors.append("record release id does not match the release")
        manifest_sha = source.get("release_manifest_sha256")
        if not isinstance(manifest_sha, str) or HEX_64.fullmatch(manifest_sha) is None:
            errors.append("record release-manifest hash is malformed")
        manifest_path = Path(__file__).resolve().with_name("release.json")
        if manifest_path.exists() and isinstance(manifest_sha, str):
            local_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            if manifest_sha != local_sha:
                errors.append("record release-manifest hash does not match local release.json")

    request_id = record.get("request_id")
    if not isinstance(request_id, str) or not request_id.startswith("chatcmpl-szl-"):
        errors.append("record request id is malformed")
    created = record.get("created_unix")
    if isinstance(created, bool) or not isinstance(created, int) or created <= 0:
        errors.append("record created_unix is malformed")
    for field in ("canonical_request_sha256", "output_sha256", "record_sha256"):
        value = record.get(field)
        if not isinstance(value, str) or HEX_64.fullmatch(value) is None:
            errors.append(f"record {field} is malformed")

    usage = record.get("usage")
    if not isinstance(usage, dict):
        errors.append("record usage must be an object")
    else:
        if set(usage) != {"prompt_tokens", "completion_tokens", "total_tokens"}:
            errors.append("record usage fields do not match the release schema")
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        total = usage.get("total_tokens")
        values = (prompt, completion, total)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
            errors.append("record usage values must be non-negative integers")
        elif total != prompt + completion:
            errors.append("record usage total does not match its components")
        elif prompt > 800 or completion > 32:
            errors.append("record usage exceeds the release token limits")

    elapsed = record.get("elapsed_ms")
    if isinstance(elapsed, bool) or not isinstance(elapsed, int) or elapsed < 0:
        errors.append("record elapsed_ms must be a non-negative integer")
    termination = record.get("termination")
    if not isinstance(termination, dict):
        errors.append("record termination must be an object")
    else:
        if set(termination) != {"reason", "time_budget_reached"}:
            errors.append("record termination fields do not match the release schema")
        reason = termination.get("reason")
        if reason not in {"stop", "length", "time_budget", None}:
            errors.append("record termination reason is unsupported")
        if termination.get("time_budget_reached") is not (reason == "time_budget"):
            errors.append("record time-budget marker does not match termination reason")
    persistence = record.get("persistence")
    if not isinstance(persistence, dict):
        errors.append("record persistence boundary is missing")
    elif set(persistence) != {"application_record_storage", "boundary"}:
        errors.append("record persistence fields do not match the release schema")
    elif (
        persistence.get("application_record_storage") != "NOT_PERSISTED"
        or persistence.get("boundary") != EXPECTED_PERSISTENCE_BOUNDARY
    ):
        errors.append("record persistence boundary is invalid")
    return errors


def response_consistency_errors(
    payload: dict[str, Any], record: dict[str, Any]
) -> list[str] | None:
    """Bind a full response's public fields to the hashed execution record."""

    if payload.get("object") != "chat.completion":
        return None
    errors: list[str] = []
    if set(payload) != {
        "id",
        "object",
        "created",
        "model",
        "choices",
        "usage",
        "szl_provenance",
    }:
        errors.append("response fields do not match the bounded contract")
    if payload.get("id") != record.get("request_id"):
        errors.append("response id does not match the record request id")
    if payload.get("created") != record.get("created_unix"):
        errors.append("response created timestamp does not match the record")
    model = record.get("model") if isinstance(record.get("model"), dict) else {}
    if payload.get("model") != model.get("id"):
        errors.append("response model does not match the record")
    if payload.get("usage") != record.get("usage"):
        errors.append("response usage does not match the record")
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        errors.append("response must contain exactly one choice")
    else:
        choice = choices[0]
        message = choice.get("message")
        if set(choice) != {"index", "message", "finish_reason"}:
            errors.append("response choice contains unsupported fields")
        if (
            choice.get("index") != 0
            or not isinstance(message, dict)
            or set(message) != {"role", "content"}
            or message.get("role") != "assistant"
            or not isinstance(message.get("content"), str)
        ):
            errors.append("response choice shape does not match the bounded contract")
        termination = record.get("termination") if isinstance(record.get("termination"), dict) else {}
        internal_reason = termination.get("reason")
        expected_finish = "length" if internal_reason in {"length", "time_budget"} else internal_reason
        if choice.get("finish_reason") != expected_finish:
            errors.append("response finish reason does not match the record")

    provenance = payload.get("szl_provenance")
    if not isinstance(provenance, dict):
        errors.append("response provenance must be an object")
    else:
        if set(provenance) != {
            "schema",
            "model",
            "runtime",
            "receipts",
            "output",
            "execution_record",
        }:
            errors.append("response provenance fields do not match the release contract")
        if provenance.get("schema") != "szl.openai-compat-provenance/v1":
            errors.append("response provenance schema is unsupported")
        expected_model = {
            key: EXPECTED_MODEL[key]
            for key in ("repo", "revision", "file", "sha256")
        }
        if provenance.get("model") != expected_model:
            errors.append("response provenance model does not match the record")
        source = record.get("source") if isinstance(record.get("source"), dict) else {}
        runtime = provenance.get("runtime")
        if not isinstance(runtime, dict) or runtime != {
            "space": source.get("space_id"),
            "release_id": source.get("release_id"),
            "service_level": "BEST_EFFORT_NO_SLA",
            "native_hugging_face_provider_mapping": "NOT_CLAIMED",
        }:
            errors.append("response provenance runtime boundary does not match the record")
        if provenance.get("receipts") != {
            "status": "DECLARED_KEY_SIGNATURES_VALID",
            "scope": "repository-declared key continuity only",
            "covers_this_output": False,
        }:
            errors.append("response provenance receipt boundary is invalid")
        termination = record.get("termination") if isinstance(record.get("termination"), dict) else {}
        if provenance.get("output") != {
            "signature_status": "UNSIGNED",
            "signature": None,
            "termination_reason": termination.get("reason"),
        }:
            errors.append("response provenance output boundary does not match the record")
        if provenance.get("execution_record") != record:
            errors.append("response provenance does not contain the verified record")
    return errors


def verify_record(
    record: dict[str, Any],
    output_text: str | None = None,
    request_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    supplied = record.get("record_sha256")
    recomputed = recompute_record_sha256(record)
    output_recomputed = (
        hashlib.sha256(output_text.encode("utf-8")).hexdigest()
        if output_text is not None
        else None
    )
    request_recomputed = (
        normalized_request_sha256(request_payload)
        if request_payload is not None
        else None
    )
    semantic_errors = record_semantic_errors(record)
    return {
        "schema": "szl.unsigned-execution-record-verification/v1",
        "record_sha256_supplied": supplied,
        "record_sha256_recomputed": recomputed,
        "hash_matches": isinstance(supplied, str) and supplied == recomputed,
        "output_sha256_supplied": record.get("output_sha256"),
        "output_sha256_recomputed": output_recomputed,
        "output_hash_matches": (
            record.get("output_sha256") == output_recomputed
            if output_recomputed is not None
            else None
        ),
        "request_sha256_supplied": record.get("canonical_request_sha256"),
        "request_sha256_recomputed": request_recomputed,
        "request_hash_matches": (
            record.get("canonical_request_sha256") == request_recomputed
            if request_recomputed is not None
            else None
        ),
        "semantic_checks_pass": not semantic_errors,
        "semantic_errors": semantic_errors,
        "signature_status": record.get("signature_status"),
        "authenticity_not_established": True,
    }


def verify_payload(
    payload: dict[str, Any], request_payload: dict[str, Any] | None = None
) -> dict[str, Any]:
    record = extract_record(payload)
    result = verify_record(
        record,
        output_text=extract_output(payload),
        request_payload=request_payload,
    )
    consistency_errors = response_consistency_errors(payload, record)
    result["response_consistency_matches"] = (
        not consistency_errors if consistency_errors is not None else None
    )
    result["response_consistency_errors"] = consistency_errors or []
    return result


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Recompute hashes in an unsigned SZL execution record."
    )
    parser.add_argument(
        "response", nargs="?", help="response JSON path; omit to read standard input"
    )
    parser.add_argument(
        "--request", help="optional original request JSON path for request-hash checking"
    )
    args = parser.parse_args(argv[1:])
    raw = Path(args.response).read_text(encoding="utf-8") if args.response else sys.stdin.read()
    payload = json.loads(raw)
    request_payload = (
        json.loads(Path(args.request).read_text(encoding="utf-8"))
        if args.request
        else None
    )
    result = verify_payload(payload, request_payload)
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":"), sort_keys=True))
    checks = [result["hash_matches"], result["semantic_checks_pass"]]
    for name in (
        "output_hash_matches",
        "request_hash_matches",
        "response_consistency_matches",
    ):
        if result[name] is not None:
            checks.append(result[name])
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
