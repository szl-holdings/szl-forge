#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Advisory-only local Ollama second-opinion reviewer (LOCAL PROPOSAL).

Takes card_lint findings and MinHash near-duplicate pairs and asks a *local*
Ollama model whether a finding looks genuine and whether a pair looks like a
true duplicate, plus a short rationale.

This module never changes pass/fail. Lint and near-dup results are passed
through byte-identical; the model's opinion lives only in a separate
``advisory`` field. Every call writes a receipt with the model name, digest,
sha256 of the prompt and of the response, a timestamp, and
``advisory=true``, ``gate=false``.

The reviewer refuses any model whose tag ends in ``:cloud`` or ``-cloud``
(prompts must not leave the machine) and refuses a digest that does not
match the expected prefix. If the model is unreachable it fails open:
status UNAVAILABLE, lint/near-dup results unchanged.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

DEFAULT_MODEL = "kimi-estate-ops:latest"
EXPECTED_DIGEST_PREFIX = "ca1b2de39499"
DEFAULT_HOST = "http://127.0.0.1:11434"
DEFAULT_TIMEOUT_S = 30.0

# Transport(method, url, body) -> parsed JSON object.
Transport = Callable[[str, str, Optional[Dict[str, Any]]], Any]

_CLOUD_SUFFIXES = (":cloud", "-cloud")
_FENCE_RX = re.compile(r"^```(?:json)?\s*|\s*```$", re.I | re.M)
_SHA_PREFIX = "sha256:"

# Fields forwarded into the model prompt. Intentionally narrow so card bodies
# and secret-looking values never go to the model or into a receipt.
_FINDING_KEYS = ("code", "severity", "message", "name", "kind")
_PAIR_KEYS = ("left_index", "right_index", "jaccard", "kind")


def ollama_host(explicit: Optional[str] = None) -> str:
    raw = (explicit if explicit is not None else os.environ.get("OLLAMA_HOST", DEFAULT_HOST) or DEFAULT_HOST).strip()
    if "://" not in raw:
        raw = "http://" + raw
    return raw.rstrip("/")


def is_cloud_model(name: str) -> bool:
    n = (name or "").strip()
    return any(n.endswith(s) for s in _CLOUD_SUFFIXES)


def normalize_digest(digest: str) -> str:
    d = (digest or "").strip().lower()
    if d.startswith(_SHA_PREFIX):
        d = d[len(_SHA_PREFIX):]
    return d


def digest_matches(digest: str, prefix: str = EXPECTED_DIGEST_PREFIX) -> bool:
    want = normalize_digest(prefix)
    got = normalize_digest(digest)
    return bool(want) and got.startswith(want)


def _sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _utc_timestamp(now: Optional[Callable[[], datetime]] = None) -> str:
    dt = now() if now else datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _extract_digest(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    for key in ("digest", "sha256", "modhash"):
        val = payload.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    details = payload.get("details")
    if isinstance(details, dict):
        for key in ("digest", "sha256"):
            val = details.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    return ""


def _extract_name(payload: Any, fallback: str = "") -> str:
    if not isinstance(payload, dict):
        return fallback
    for key in ("name", "model"):
        val = payload.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return fallback


def _request(
    method: str,
    url: str,
    body: Optional[Dict[str, Any]] = None,
    *,
    timeout: float,
    transport: Optional[Transport],
) -> Any:
    if transport is not None:
        return transport(method, url, body)
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json", "Accept": "application/json", "User-Agent": "szl-evolve-advisory-review/1"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def _is_unreachable(exc: BaseException) -> bool:
    return isinstance(exc, (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ConnectionError, OSError, json.JSONDecodeError))


def _find_in_tags(payload: Any, model: str) -> Optional[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return None
    models = payload.get("models") or []
    if not isinstance(models, list):
        return None
    aliases = {model}
    if model.endswith(":latest"):
        aliases.add(model[: -len(":latest")])
    else:
        aliases.add(model + ":latest")
    for entry in models:
        if not isinstance(entry, dict):
            continue
        names = {str(entry.get("name") or ""), str(entry.get("model") or "")}
        if names & aliases:
            return entry
    return None


def verify_model(
    model: str = DEFAULT_MODEL,
    *,
    expected_digest_prefix: str = EXPECTED_DIGEST_PREFIX,
    host: Optional[str] = None,
    timeout: float = DEFAULT_TIMEOUT_S,
    transport: Optional[Transport] = None,
) -> Dict[str, Any]:
    """Confirm a local model tag and digest before any review prompt is sent.

    Returns a dict with ``ok``, ``status`` (OK / REFUSED / UNAVAILABLE),
    ``model``, ``digest`` and ``reason``. Never raises for transport failures.
    """
    model = (model or "").strip() or DEFAULT_MODEL
    if is_cloud_model(model):
        return {
            "ok": False,
            "status": "REFUSED",
            "model": model,
            "digest": "",
            "reason": "cloud_model",
        }

    base = ollama_host(host)
    shown: Any = None
    try:
        shown = _request("POST", f"{base}/api/show", {"name": model}, timeout=timeout, transport=transport)
    except Exception as exc:
        if not _is_unreachable(exc):
            raise
        # Fall back to /api/tags; a show failure alone is not decisive.
        shown = None
        show_error = exc
    else:
        show_error = None

    digest = _extract_digest(shown)
    resolved = _extract_name(shown, model) if isinstance(shown, dict) else model

    # Prefer /api/show, but /api/tags is a valid alternative when show is
    # unreachable or does not include a digest.
    if shown is None or not digest:
        try:
            tags = _request("GET", f"{base}/api/tags", None, timeout=timeout, transport=transport)
        except Exception as exc:
            if not _is_unreachable(exc):
                raise
            if not digest:
                return {
                    "ok": False,
                    "status": "UNAVAILABLE",
                    "model": model,
                    "digest": digest,
                    "reason": "unreachable",
                }
        else:
            entry = _find_in_tags(tags, model)
            if entry is not None:
                if not digest:
                    digest = _extract_digest(entry)
                resolved = _extract_name(entry, resolved)
            elif not digest:
                reason = "unreachable" if shown is None and show_error is not None else "model_not_found"
                return {
                    "ok": False,
                    "status": "UNAVAILABLE",
                    "model": model,
                    "digest": "",
                    "reason": reason,
                }

    if is_cloud_model(resolved):
        return {
            "ok": False,
            "status": "REFUSED",
            "model": resolved,
            "digest": digest,
            "reason": "cloud_model",
        }
    if not digest or not digest_matches(digest, expected_digest_prefix):
        return {
            "ok": False,
            "status": "REFUSED",
            "model": resolved,
            "digest": digest,
            "reason": "digest_mismatch",
        }
    return {
        "ok": True,
        "status": "OK",
        "model": resolved,
        "digest": digest,
        "reason": "",
    }


def _findings_from(lint_result: Any) -> List[Dict[str, Any]]:
    if lint_result is None:
        return []
    if isinstance(lint_result, list):
        return [x for x in lint_result if isinstance(x, dict)]
    if not isinstance(lint_result, dict):
        return []
    if "findings" in lint_result and isinstance(lint_result["findings"], list):
        name = lint_result.get("name")
        kind = lint_result.get("kind")
        out = []
        for f in lint_result["findings"]:
            if not isinstance(f, dict):
                continue
            item = dict(f)
            if name and "name" not in item:
                item["name"] = name
            if kind and "kind" not in item:
                item["kind"] = kind
            out.append(item)
        return out
    if "results" in lint_result and isinstance(lint_result["results"], list):
        out = []
        for row in lint_result["results"]:
            out.extend(_findings_from(row))
        return out
    if "code" in lint_result:
        return [lint_result]
    return []


def _pairs_from(near_dup_pairs: Any) -> List[Dict[str, Any]]:
    if near_dup_pairs is None:
        return []
    if isinstance(near_dup_pairs, list):
        return [x for x in near_dup_pairs if isinstance(x, dict)]
    return []


def _safe_items(rows: List[Dict[str, Any]], keys: Tuple[str, ...]) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        item = {k: row[k] for k in keys if k in row}
        out.append(item)
    return out


def build_prompt(lint_result: Any = None, near_dup_pairs: Any = None) -> str:
    payload = {
        "role": "advisory_only",
        "gate": False,
        "instructions": (
            "You are an advisory-only second-opinion reviewer. You do not decide "
            "pass or fail and you must not rewrite the findings or pairs. "
            "For each card_lint finding, say whether it appears genuine. "
            "For each near-duplicate pair, say whether it appears to be a true duplicate. "
            "Give a short rationale for each item. Respond with JSON only."
        ),
        "findings": _safe_items(_findings_from(lint_result), _FINDING_KEYS),
        "near_duplicates": _safe_items(_pairs_from(near_dup_pairs), _PAIR_KEYS),
        "response_schema": {
            "lint_opinions": [{"code": "CL000", "genuine": True, "rationale": "short reason"}],
            "near_dup_opinions": [{"left_index": 0, "right_index": 0, "true_duplicate": True, "rationale": "short reason"}],
        },
    }
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _strip_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = _FENCE_RX.sub("", t).strip()
    return t


def _parse_opinions(response_text: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Optional[str]]:
    raw = _strip_fences(response_text)
    if not raw:
        return [], [], "empty_response"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Sometimes the model wraps JSON in prose; try the first brace block.
        start, end = raw.find("{"), raw.rfind("}")
        if start >= 0 and end > start:
            try:
                data = json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                return [], [], "invalid_response"
        else:
            return [], [], "invalid_response"
    if not isinstance(data, dict):
        return [], [], "invalid_response"
    lint_op = data.get("lint_opinions") or []
    pair_op = data.get("near_dup_opinions") or []
    if not isinstance(lint_op, list):
        lint_op = []
    if not isinstance(pair_op, list):
        pair_op = []
    return [x for x in lint_op if isinstance(x, dict)], [x for x in pair_op if isinstance(x, dict)], None


def _receipt(
    *,
    model: str,
    digest: str,
    prompt: str,
    response: str,
    status: str,
    now: Optional[Callable[[], datetime]] = None,
) -> Dict[str, Any]:
    return {
        "model": model,
        "digest": digest,
        "prompt_sha256": _sha256_text(prompt),
        "response_sha256": _sha256_text(response),
        "timestamp": _utc_timestamp(now),
        "advisory": True,
        "gate": False,
        "status": status,
    }


def write_receipt(receipt: Dict[str, Any], path: str) -> None:
    # Receipts record hashes only; never the prompt or response bodies.
    payload = {
        "model": receipt.get("model", ""),
        "digest": receipt.get("digest", ""),
        "prompt_sha256": receipt.get("prompt_sha256", ""),
        "response_sha256": receipt.get("response_sha256", ""),
        "timestamp": receipt.get("timestamp", ""),
        "advisory": True,
        "gate": False,
        "status": receipt.get("status", ""),
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=True, sort_keys=True)
        fh.write("\n")


def review(
    lint_result: Any = None,
    near_dup_pairs: Any = None,
    *,
    model: str = DEFAULT_MODEL,
    expected_digest_prefix: str = EXPECTED_DIGEST_PREFIX,
    host: Optional[str] = None,
    timeout: float = DEFAULT_TIMEOUT_S,
    transport: Optional[Transport] = None,
    now: Optional[Callable[[], datetime]] = None,
    receipt_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Review lint findings and/or near-dup pairs without changing pass/fail.

    The returned ``lint`` and ``near_duplicates`` fields are deep copies of the
    inputs and are byte-identical to ``json.dumps`` of those inputs. Advisory
    content lives only under ``advisory`` and ``receipt``.
    """
    lint_out = copy.deepcopy(lint_result)
    pairs_out = copy.deepcopy(near_dup_pairs)
    prompt = build_prompt(lint_result, near_dup_pairs)
    model = (model or "").strip() or DEFAULT_MODEL

    def _done(status: str, reason: str, digest: str, response: str = "",
              lint_opinions: Optional[List[Dict[str, Any]]] = None,
              near_dup_opinions: Optional[List[Dict[str, Any]]] = None,
              resolved_model: Optional[str] = None) -> Dict[str, Any]:
        receipt = _receipt(
            model=resolved_model or model,
            digest=digest,
            prompt=prompt,
            response=response,
            status=status,
            now=now,
        )
        if receipt_path:
            write_receipt(receipt, receipt_path)
        return {
            "lint": lint_out,
            "near_duplicates": pairs_out,
            "advisory": {
                "status": status,
                "reason": reason,
                "lint_opinions": lint_opinions or [],
                "near_dup_opinions": near_dup_opinions or [],
            },
            "receipt": receipt,
        }

    verified = verify_model(
        model,
        expected_digest_prefix=expected_digest_prefix,
        host=host,
        timeout=timeout,
        transport=transport,
    )
    if not verified["ok"]:
        return _done(
            verified["status"],
            verified.get("reason") or "",
            verified.get("digest") or "",
            resolved_model=verified.get("model") or model,
        )

    resolved = verified["model"]
    digest = verified["digest"]
    # Belt-and-suspenders: never send a prompt to a cloud-tagged model.
    if is_cloud_model(resolved) or is_cloud_model(model):
        return _done("REFUSED", "cloud_model", digest, resolved_model=resolved)

    body = {
        "model": resolved,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0},
    }
    try:
        gen = _request(
            "POST",
            f"{ollama_host(host)}/api/generate",
            body,
            timeout=timeout,
            transport=transport,
        )
    except Exception as exc:
        if not _is_unreachable(exc):
            raise
        return _done("UNAVAILABLE", "unreachable", digest, resolved_model=resolved)

    response_text = ""
    if isinstance(gen, dict):
        val = gen.get("response")
        if isinstance(val, str):
            response_text = val
        elif isinstance(gen.get("message"), dict) and isinstance(gen["message"].get("content"), str):
            response_text = gen["message"]["content"]
    lint_op, pair_op, parse_err = _parse_opinions(response_text)
    if parse_err:
        # Fail open: keep pass/fail untouched; record that the opinion was unusable.
        return _done("UNAVAILABLE", parse_err, digest, response=response_text, resolved_model=resolved)
    return _done("OK", "", digest, response=response_text, lint_opinions=lint_op, near_dup_opinions=pair_op, resolved_model=resolved)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lint-json", help="Path to a card_lint lint() or lint_dir() JSON file")
    ap.add_argument("--near-dup-json", help="Path to a near_duplicates() JSON file (list of pairs)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--receipt", help="Write the receipt JSON to this path")
    ap.add_argument("--host", default=None)
    a = ap.parse_args(argv)
    lint_result = json.load(open(a.lint_json, encoding="utf-8")) if a.lint_json else None
    pairs = json.load(open(a.near_dup_json, encoding="utf-8")) if a.near_dup_json else None
    out = review(lint_result, pairs, model=a.model, host=a.host, receipt_path=a.receipt)
    # Print advisory + receipt only; the lint/near-dup bytes are echoed as
    # structured fields but this CLI does not re-score them.
    print(json.dumps({"advisory": out["advisory"], "receipt": out["receipt"],
                      "lint": out["lint"], "near_duplicates": out["near_duplicates"]}, indent=2))
    # Advisory-only: never fail the process on the model's opinion.
    return 0


if __name__ == "__main__":
    sys.exit(main())
