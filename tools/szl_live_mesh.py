#!/usr/bin/env python3
"""Read-only SZL identity observations; never a release or production authority.

This is a successor for the existing Forge PR #239, not a new publisher.
Metadata consistency, served-document byte parity, authenticated deployment,
functional readiness and production authorization are distinct claims.
No credentials, package installs, upstream execution or remote writes occur.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable

SHA40 = re.compile(r"[0-9a-f]{40}\Z")
PIN_RE = re.compile(r'^SOURCE_REVISION\s*=\s*[\"\']([0-9a-f]{40})[\"\']\s*(?:#.*)?$', re.M)
MAX_BODY_BYTES = 2 * 1024 * 1024
MAX_OBSERVATION_SECONDS = 600
UA = {"User-Agent": "SZL-Live-Mesh/2.0", "Accept": "application/json", "Accept-Encoding": "identity"}
ORIGINS = {
    "github": "https://api.github.com",
    "raw": "https://raw.githubusercontent.com",
    "product": "https://a-11-oy.com",
    "proof": "https://a11oy.net",
    "hf_a11oy": "https://szlholdings-a11oy.hf.space",
    "hf_lyte": "https://szlholdings-lyte.hf.space",
    "hf_api": "https://huggingface.co",
}
REPOS = {
    "a11oy": "szl-holdings/a11oy", "lyte": "szl-holdings/lyte-services",
    "proof": "szl-holdings/a11oy-net", "forge": "szl-holdings/szl-forge",
    "frontier": "szl-holdings/szl-frontier",
}
Fetch = Callable[..., tuple[int, Any]]


class MeshError(ValueError):
    """Invalid observations have no authority. Errors never echo response bodies."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise MeshError("redirect refused; inspect the declared origin separately")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _utc(value: Any) -> dt.datetime:
    if type(value) is not str:
        raise MeshError("timezone-aware timestamp required")
    try:
        stamp = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise MeshError("invalid timestamp") from exc
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise MeshError("timezone-aware timestamp required")
    return stamp.astimezone(dt.timezone.utc)


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise MeshError("duplicate JSON key")
        result[key] = value
    return result


def _constant(value: str) -> None:
    del value
    raise MeshError("non-finite JSON number")


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise MeshError("non-finite JSON number")
    return number


def _decode(raw: bytes, json_ok: bool) -> Any:
    if len(raw) > MAX_BODY_BYTES:
        raise MeshError("response too large")
    text = raw.decode("utf-8", errors="strict")
    if not json_ok:
        return text
    return json.loads(text, object_pairs_hook=_pairs, parse_constant=_constant, parse_float=_finite_float)


def _allowed_url(url: str) -> bool:
    """No credentials, arbitrary hosts, ports, fragments or encoded path tricks."""
    if type(url) is not str or any(ord(c) < 33 for c in url):
        return False
    if "\\" in url or "%" in url:
        return False
    try:
        part = urllib.parse.urlsplit(url)
        if part.scheme != "https" or part.username or part.password or part.fragment:
            return False
        if part.port not in (None, 443):
            return False
        origin = f"https://{part.hostname}"
        if origin not in ORIGINS.values():
            return False
        if any(p in (".", "..") for p in part.path.split("/")):
            return False
        return not part.query
    except (ValueError, TypeError):
        return False


def _get(url: str, timeout: float = 12.0, json_ok: bool = True) -> tuple[int, Any]:
    """Bounded-byte GET with no redirects, credentials, raw error bodies or retries.

    The urllib timeout bounds socket inactivity, not hostile slow-trickle total
    duration. Run this observer under the existing process/job wall-clock limit.
    """
    if not _allowed_url(url):
        return 0, {"error": "URL_REFUSED"}
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 30:
        return 0, {"error": "TIMEOUT_REFUSED"}
    req = urllib.request.Request(url, headers=UA, method="GET")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    try:
        with opener.open(req, timeout=float(timeout)) as resp:
            # No implicit gzip decompression or unknown content transformation.
            if resp.headers.get("Content-Encoding", "identity").lower() not in ("", "identity"):
                return resp.status, {"error": "ENCODING_REFUSED"}
            raw = resp.read(MAX_BODY_BYTES + 1)
            return resp.status, _decode(raw, json_ok)
    except urllib.error.HTTPError as exc:
        exc.close()
        return exc.code, {"error": "HTTP_ERROR"}
    except (MeshError, UnicodeError, ValueError, OSError, urllib.error.URLError):
        return 0, {"error": "OBSERVATION_UNAVAILABLE"}


def _path(payload: Any, keys: tuple[str, ...]) -> Any:
    cur = payload
    for key in keys:
        if type(cur) is not dict:
            return None
        cur = cur.get(key)
    return cur


def _identity(payload: Any, paths: tuple[tuple[str, ...], ...]) -> tuple[str | None, str]:
    """A malformed/conflicting recognized alias cannot be hidden by a good one."""
    observed: list[str] = []
    for path in paths:
        value = _path(payload, path)
        if value is None:
            continue
        if type(value) is not str or not SHA40.fullmatch(value):
            return None, "INVALID"
        observed.append(value)
    if not observed:
        return None, "UNAVAILABLE"
    if len(set(observed)) != 1:
        return None, "CONFLICT"
    return observed[0], "OBSERVED"


def _sha(payload: Any, *keys: str) -> str | None:
    return _identity(payload, (tuple(keys),))[0]


def _object_ok(status: Any, body: Any) -> bool:
    return type(status) is int and status == 200 and type(body) is dict and "error" not in body


def _observe(fetch: Fetch, url: str, *, json_ok: bool = True) -> tuple[int, Any]:
    try:
        status, body = fetch(url, json_ok=json_ok)
        if type(status) is not int or status < 0 or status > 599:
            return 0, {"error": "INVALID_HTTP_STATUS"}
        return status, body
    except Exception:
        # A supplied adapter failure is still an unavailable observation, not PASS.
        return 0, {"error": "FETCH_ADAPTER_UNAVAILABLE"}


def probe(fetch: Fetch | None = None, clock: Callable[[], str] | None = None) -> dict[str, Any]:
    fetch = _get if fetch is None else fetch
    clock = utc_now if clock is None else clock
    started_at = clock()
    blockers: list[str] = []
    http: dict[str, int] = {}
    planes: dict[str, str | None] = {}
    bodies: dict[str, Any] = {}
    identity_states: dict[str, str] = {}

    def observe_object(name: str, url: str) -> Any:
        status, body = _observe(fetch, url)
        http[name] = status
        bodies[name] = body
        if not _object_ok(status, body):
            blockers.append(f"{name}:UNAVAILABLE_OR_INVALID_OBJECT")
            return {}
        return body

    def identity(name: str, body: Any, paths: tuple[tuple[str, ...], ...]) -> None:
        value, state = _identity(body, paths)
        planes[name], identity_states[name] = value, state
        if state != "OBSERVED":
            blockers.append(f"{name}:{state}")

    # Freeze the source vector before reading source-owned configurations.
    for name, repo in REPOS.items():
        body = observe_object(f"{name}_github", f"{ORIGINS['github']}/repos/{repo}/commits/main")
        identity(f"{name}_github", body, (("sha",),))
    a11oy_sha = planes["a11oy_github"]
    net_sha = planes["proof_github"]
    pin_text = None
    if a11oy_sha:
        status, text = _observe(fetch, f"{ORIGINS['raw']}/szl-holdings/a11oy/{a11oy_sha}/scripts/hf_publish_lyte_enterprise.py", json_ok=False)
        http["publisher_pin_blob"] = status
        if status == 200 and type(text) is str:
            pin_text = text
    matches = PIN_RE.findall(pin_text) if pin_text is not None else []
    pin = matches[0] if len(matches) == 1 else None
    planes["lyte_publisher_pin"] = pin
    if pin is None:
        blockers.append("lyte_publisher_pin:UNAVAILABLE_OR_AMBIGUOUS")

    runtime_paths = (("source_revision",), ("build", "revision"), ("git_sha",))
    endpoints = {
        "a11oy_product": f"{ORIGINS['product']}/api/build-info",
        "a11oy_honest": f"{ORIGINS['product']}/api/a11oy/v1/honest",
        "product_healthz": f"{ORIGINS['product']}/api/a11oy/healthz",
        "product_surfaces": f"{ORIGINS['product']}/api/a11oy/v1/frontier/surfaces",
        "a11oy_hf_space": f"{ORIGINS['hf_a11oy']}/api/build-info",
        "lyte_live_space": f"{ORIGINS['hf_lyte']}/api/build-info",
        "lyte_metrics_alias": f"{ORIGINS['hf_lyte']}/api/lyte/v2/metrics",
    }
    for name, url in endpoints.items():
        body = observe_object(name, url)
        if name in ("a11oy_product", "a11oy_honest", "a11oy_hf_space", "lyte_live_space"):
            identity(name, body, runtime_paths)
    health_status = _path(bodies.get("product_healthz"), ("status",))
    if health_status != "ok":
        blockers.append("product_healthz:STATUS_NOT_OK")
    # Explicit negative or malformed source-binding evidence cannot be discarded.
    binding = _path(bodies.get("lyte_live_space"), ("source_binding",))
    if type(binding) is not dict or binding.get("bindings_agree") is not True:
        blockers.append("lyte_live_space:BINDING_AGREEMENT_NOT_OBSERVED")

    def compare(label: str, names: tuple[str, ...]) -> bool:
        values = [planes.get(name) for name in names]
        if any(value is None for value in values):
            blockers.append(f"{label}:INCOMPLETE")
            return False
        if len(set(values)) != 1:
            blockers.append(f"{label}:SOURCE_MISMATCH")
            return False
        return True

    product_parity = compare("product_source_parity", ("a11oy_github", "a11oy_product", "a11oy_honest", "a11oy_hf_space"))
    lyte_parity = compare("lyte_source_parity", ("lyte_github", "lyte_publisher_pin", "lyte_live_space"))

    # health.json is explicitly STATIC_DOCUMENT, not its own future commit SHA.
    # Compare its exact served bytes with bytes at the observed proof Git source.
    proof_document: dict[str, Any] = {"state": "UNAVAILABLE", "historical_sha": None, "deployment_verified": False}
    if net_sha:
        src_status, src_text = _observe(fetch, f"{ORIGINS['raw']}/szl-holdings/a11oy-net/{net_sha}/health.json", json_ok=False)
        live_status, live_text = _observe(fetch, f"{ORIGINS['proof']}/health.json", json_ok=False)
        http["proof_document_source"], http["proof_document_live"] = src_status, live_status
        if src_status == live_status == 200 and type(src_text) is str and type(live_text) is str:
            try:
                src_body, live_body = _decode(src_text.encode("utf-8"), True), _decode(live_text.encode("utf-8"), True)
                if type(src_body) is not dict or type(live_body) is not dict:
                    raise MeshError("proof object required")
                if src_body.get("probe_contract") != "STATIC_DOCUMENT" or live_body.get("probe_contract") != "STATIC_DOCUMENT":
                    raise MeshError("static proof contract changed")
                source_digest = hashlib.sha256(src_text.encode("utf-8")).hexdigest()
                live_digest = hashlib.sha256(live_text.encode("utf-8")).hexdigest()
                proof_document.update(state="BYTE_PARITY" if source_digest == live_digest else "DOCUMENT_MISMATCH",
                                      source_revision=net_sha, source_sha256=source_digest, served_sha256=live_digest,
                                      historical_sha=_sha(live_body, "sha"))
            except (MeshError, UnicodeError, ValueError):
                proof_document["state"] = "INVALID"
    if proof_document["state"] != "BYTE_PARITY":
        blockers.append(f"proof_document:{proof_document['state']}")

    # Do not combine measurements from changing default-branch heads into one closure.
    end_sources: dict[str, str | None] = {}
    for name, repo in REPOS.items():
        body = observe_object(f"{name}_github_end", f"{ORIGINS['github']}/repos/{repo}/commits/main")
        end_sources[name] = _sha(body, "sha")
        if end_sources[name] is None:
            blockers.append(f"{name}:END_SOURCE_UNAVAILABLE")
        elif end_sources[name] != planes[f"{name}_github"]:
            blockers.append(f"{name}:SOURCE_CHANGED_DURING_OBSERVATION")
    finished_at = clock()
    try:
        elapsed = (_utc(finished_at) - _utc(started_at)).total_seconds()
        if not 0 <= elapsed <= MAX_OBSERVATION_SECONDS:
            blockers.append("OBSERVATION_WINDOW_INVALID_OR_EXPIRED")
    except MeshError:
        elapsed = None
        blockers.append("OBSERVATION_WINDOW_INVALID_OR_EXPIRED")

    blockers = sorted(set(blockers))
    report = {
        "schema": "szl.live-mesh/v2", "observed_at": finished_at,
        "observation_started_at": started_at, "elapsed_seconds": elapsed,
        "mutations": False, "production_authorization": False, "signed_off_done": False,
        "lambda": "Conjecture 1", "planes": planes, "source_vector_end": end_sources,
        "identity_states": identity_states, "http": http,
        "product_source_parity": product_parity,
        "lyte_source_parity": lyte_parity,
        "product_aligned": False, "p01_close": False,
        "proof_document": proof_document,
        "observation_state": "METADATA_CONSISTENT" if not blockers else "INCOMPLETE_OR_DIVERGENT",
        "state": "HOLD", "blockers": blockers,
        "qualification_blockers": [
            "AUTHENTICATED_PUBLISHER_ARTIFACT_BINDING_NOT_COLLECTED",
            "EXACT_IMAGE_AND_DEPENDENCY_READBACK_NOT_COLLECTED",
            "FUNCTIONAL_AND_RENDERED_BROWSER_WITNESS_NOT_COLLECTED",
            "ROLLBACK_AND_POLICY_QUALIFICATION_NOT_COLLECTED",
            "WHOLE_ESTATE_NOT_OBSERVED_BY_THIS_BOUNDED_PROBE",
        ],
        "issue_2010": "NOT_MUTATED_OR_ADJUDICATED",
        "kernel_hub": {"state": "NOT_ASSESSED", "mirrors_deleted": False},
        "signature": "UNSIGNED_LOCAL_OBSERVATION",
        "note": "Source parity is metadata, proof byte parity is one static file, neither is deployment or production qualification. Existing native verifiers retain authority.",
    }
    return report


def _write_new(path: Path, text: str) -> None:
    """Never silently replace a previous observation or follow a destination link."""
    parent = path.parent
    if parent.is_symlink() or any(p.is_symlink() for p in parent.parents):
        raise MeshError("symlink output parent refused")
    if not parent.is_dir():
        raise MeshError("output parent must already exist")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="-")
    args = parser.parse_args(argv)
    report = probe()
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output == "-":
        print(text, end="")
    else:
        _write_new(Path(args.output), text)
    # This metadata observer never authenticates an operational release closure.
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
