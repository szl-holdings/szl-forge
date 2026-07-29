"""Typed, read-only HTTP compatibility contracts for the managed Forge Space.

The Gradio event API remains the product API.  These routes are deliberately
small operational contracts for probes that require ordinary HTTP GET JSON.
They do not claim GitHub/Hugging Face byte parity or a reproducible build.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.request
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from fastapi.responses import JSONResponse


SPACE_ID = "SZLHOLDINGS/szl-forge-lab"
SOURCE_REPOSITORY = "szl-holdings/szl-forge"
SOURCE_COMMIT = "43e0aedb176fbc8868557e56e30501d8de24578e"
SOURCE_PATH = ""
ALIGNMENT_STATE = "PENDING_GITHUB_SYNC"
_SHA = re.compile(r"^[0-9a-f]{40}$")
_CACHE_LOCK = threading.Lock()
_CACHE: dict[str, Any] = {"stored_at": 0.0, "measurement": None}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _valid_sha(value: object) -> str | None:
    candidate = str(value or "").strip().lower()
    return candidate if _SHA.fullmatch(candidate) else None


def measure_hf_repository_head(force: bool = False) -> dict[str, Any]:
    """Measure Hub repository-head evidence without calling it process identity."""

    now = time.monotonic()
    with _CACHE_LOCK:
        cached = _CACHE.get("measurement")
        if not force and cached and now - float(_CACHE["stored_at"]) < 60:
            return dict(cached)

    space_id = os.environ.get("SPACE_ID") or SPACE_ID
    observed_at = _now_iso()
    resolver = f"https://huggingface.co/api/spaces/{space_id}?expand[]=sha&expand[]=lastModified"
    revision = None
    last_modified = None
    error = None
    request = urllib.request.Request(
        resolver,
        headers={"Accept": "application/json", "User-Agent": "szl-forge-runtime-contract/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=4) as response:
            payload = json.load(response)
        revision = _valid_sha(payload.get("sha"))
        last_modified = payload.get("lastModified")
    except Exception as exc:  # Network evidence is optional and reported honestly.
        error = type(exc).__name__

    measurement: dict[str, Any] = {
        "hf_revision": revision,
        "revision_state": "MEASURED_REPOSITORY_HEAD" if revision else "UNAVAILABLE",
        "running_process_revision_state": "NOT_EXPOSED_IN_PROCESS",
        "measurement_method": "HUGGINGFACE_HUB_API" if revision else "UNAVAILABLE",
        "resolver": resolver,
        "observed_at": observed_at,
        "provider_last_modified": last_modified,
    }
    if error:
        measurement["measurement_error"] = error
    with _CACHE_LOCK:
        _CACHE.update(stored_at=time.monotonic(), measurement=dict(measurement))
    return measurement


def build_source_attestation(force: bool = False) -> tuple[dict[str, Any], int]:
    measurement = measure_hf_repository_head(force=force)
    revision = measurement["hf_revision"]
    if revision is None:
        return (
            {
                "schema": "szl.deployment-source-unavailable/v1",
                "observed_at": measurement["observed_at"],
                "transport_state": "REACHABLE",
                "evidence_state": "UNAVAILABLE",
                "authority_state": "READ_ONLY",
                "deployment": {"hf_space": SPACE_ID, **measurement},
                "limits": [
                    "Repository-head evidence was unavailable, so no deployment-source document was fabricated.",
                    "Retry this GET with ?refresh=1 after Hub API reachability recovers.",
                ],
            },
            503,
        )

    built_at = measurement.get("provider_last_modified") or measurement["observed_at"]
    return (
        {
            "schema": "szl.deployment-source/v1",
            "source": {
                "repository": SOURCE_REPOSITORY,
                "commit": SOURCE_COMMIT,
                "path": SOURCE_PATH,
            },
            "deployment": {"hf_space": SPACE_ID, "hf_revision": revision},
            "built_at": built_at,
            "alignment_state": ALIGNMENT_STATE,
            "extensions": {
                "schema": "szl.forge.deployment-source-extension/v1",
                "deployment_revision_evidence": measurement,
                "source_observation": {
                    "state": "VERIFIED_REFERENCE",
                    "relation": "showcase-derived-from-source",
                    "immutable_evidence": (
                        "https://api.github.com/repos/szl-holdings/szl-forge/commits/"
                        + SOURCE_COMMIT
                    ),
                },
                "build_metadata": {
                    "built_at_semantics": (
                        "Provider-reported lastModified for deployment.hf_revision; "
                        "not a verified process start time or reproducible-build attestation."
                    )
                },
                "claims": {
                    "github_parity": "NOT_CLAIMED",
                    "reproducible_build": "NOT_CLAIMED",
                    "build_provenance": "NOT_CLAIMED",
                },
            },
            "limits": [
                "PENDING_GITHUB_SYNC means GitHub/Hugging Face content parity is not claimed.",
                "deployment.hf_revision is repository-head evidence, not exact serving-process identity.",
                "source.commit is a verified immutable reference, not proof of deployed byte equivalence.",
                "This unsigned document does not establish SLSA provenance or binary reproducibility.",
            ],
        },
        200,
    )


def build_health(status_provider: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    observed_at = _now_iso()
    try:
        packaged_status = status_provider()
        packaged_contract = "PASS"
        evidence_state = str(packaged_status.get("evidence_state", "UNKNOWN"))
    except Exception as exc:
        packaged_contract = "FAIL"
        evidence_state = "UNAVAILABLE"
        packaged_status = {"error": type(exc).__name__}
    return {
        "schema": "szl.forge.health/v1",
        "observed_at": observed_at,
        "status": "OK" if packaged_contract == "PASS" else "DEGRADED",
        "transport_state": "REACHABLE",
        "evidence_state": evidence_state,
        "authority_state": "READ_ONLY",
        "checks": {
            "packaged_status_contract": packaged_contract,
            "external_dependencies": "NOT_REQUIRED_FOR_TRANSPORT_HEALTH",
            "model_quality": "NOT_MEASURED",
            "source_parity": ALIGNMENT_STATE,
        },
        "limits": [
            "HTTP health proves this process can answer a bounded read-only contract only.",
            "It does not prove training quality, freshness, source parity, or promotion approval.",
        ],
        "packaged_status_schema": packaged_status.get("schema"),
    }


def build_runtime_status(
    status_provider: Callable[[], dict[str, Any]], force: bool = False
) -> dict[str, Any]:
    packaged = status_provider()
    measurement = measure_hf_repository_head(force=force)
    return {
        "schema": "szl.forge.runtime-status/v1",
        "observed_at": _now_iso(),
        "transport_state": "REACHABLE",
        "evidence_state": packaged.get("evidence_state", "UNKNOWN"),
        "authority_state": "READ_ONLY",
        "interface_mode": packaged.get("interface_mode", "READ_ONLY"),
        "deployment": {"hf_space": SPACE_ID, **measurement},
        "source": {
            "repository": SOURCE_REPOSITORY,
            "commit": SOURCE_COMMIT,
            "alignment_state": ALIGNMENT_STATE,
            "github_parity": "NOT_CLAIMED",
        },
        "application": packaged,
        "limits": [
            "Runtime reachability, repository-head evidence, training state, and model quality are independent.",
            "The nested application object is packaged snapshot evidence, not a live benchmark.",
        ],
    }


def _json_response(payload: dict[str, Any], status_code: int = 200) -> JSONResponse:
    return JSONResponse(
        payload,
        status_code=status_code,
        headers={
            "Cache-Control": "no-store",
            "X-SZL-Transport-State": str(payload.get("transport_state", "REACHABLE")),
            "X-SZL-Evidence-State": str(payload.get("evidence_state", "SNAPSHOT")),
            "X-SZL-Authority-State": str(payload.get("authority_state", "READ_ONLY")),
        },
    )


def register_contract_routes(
    app: Any, status_provider: Callable[[], dict[str, Any]]
) -> dict[str, Any]:
    """Add exact GET routes ahead of Gradio's fallback after launch."""

    async def source_attestation(refresh: int = 0):  # noqa: ANN202
        payload, status_code = build_source_attestation(force=refresh == 1)
        return _json_response(payload, status_code=status_code)

    async def health():  # noqa: ANN202
        return _json_response(build_health(status_provider))

    async def runtime_status(refresh: int = 0):  # noqa: ANN202
        return _json_response(build_runtime_status(status_provider, force=refresh == 1))

    contracts = (
        ("/.well-known/szl-source.json", source_attestation),
        ("/healthz", health),
        ("/api/status", runtime_status),
    )
    existing_paths = {getattr(route, "path", None) for route in app.router.routes}
    collisions = [path for path, _ in contracts if path in existing_paths]
    if collisions:
        raise RuntimeError(f"Refusing to shadow existing routes: {collisions}")

    before = list(app.router.routes)
    for path, endpoint in contracts:
        app.add_api_route(path, endpoint, methods=["GET"], include_in_schema=True)
    added = list(app.router.routes[len(before) :])
    app.router.routes[:] = added + before
    app.openapi_schema = None
    return {
        "ok": True,
        "routes": [path for path, _ in contracts],
        "position": "BEFORE_GRADIO_FALLBACK",
    }
