"""Authenticated, Python-rendered workbench; no execution/publication API exists."""
from __future__ import annotations
import asyncio
import hmac
import os
from dataclasses import dataclass
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .artifacts import load_candidate
from .catalog import TRACKS, catalog, track_for
from .probes import inspect_node, node_config
from .pool_view import load_pool_snapshot, unavailable_pool
from .safeio import strict_json

@dataclass(frozen=True)
class Settings:
    access_token: str
    nodes: dict[str, str]
    artifacts: dict[str, Path]
    pool_snapshot: Path | None = None
    pool_snapshot_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.access_token, str) or len(self.access_token) < 32:
            raise ValueError("at_least_32_character_local_access_token_required")
        import json
        node_config(json.dumps(self.nodes))
        if not set(self.artifacts).issubset(TRACKS):
            raise ValueError("unknown_artifact_track")
        if (self.pool_snapshot is None) != (self.pool_snapshot_sha256 is None):
            raise ValueError("pool_snapshot_and_external_digest_required_together")
        if self.pool_snapshot is not None and not isinstance(self.pool_snapshot, Path):
            raise ValueError("pool_snapshot_path_required")
        if self.pool_snapshot_sha256 is not None:
            import re
            if not isinstance(self.pool_snapshot_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", self.pool_snapshot_sha256):
                raise ValueError("invalid_pool_snapshot_digest")

    @classmethod
    def from_env(cls) -> "Settings":
        artifacts = {slug: Path(value) for slug in TRACKS
                     if (value := os.environ.get(f"SZL_LAB_{slug.upper()}_ARTIFACT"))}
        return cls(os.environ.get("SZL_LAB_ACCESS_TOKEN", ""),
                   node_config(os.environ.get("SZL_LAB_OLLAMA_ENDPOINTS_JSON", "{}")), artifacts,
                   Path(pool_path) if (pool_path := os.environ.get("SZL_LAB_POOL_SNAPSHOT")) else None,
                   os.environ.get("SZL_LAB_POOL_SNAPSHOT_SHA256") or None)

class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    normalization_id: str = Field(min_length=1, max_length=128)
    features: dict[str, float]

class BoundedBody:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") not in ("POST", "PUT", "PATCH"):
            return await self.app(scope, receive, send)
        body = bytearray()
        while True:
            event = await receive()
            if event["type"] == "http.disconnect":
                return
            body.extend(event.get("body", b""))
            if len(body) > 16384:
                return await JSONResponse({"detail": "request_too_large"}, status_code=413)(scope, receive, send)
            if not event.get("more_body", False):
                break
        if body:
            try:
                strict_json(bytes(body))
            except (ValueError, TypeError):
                return await JSONResponse({"detail": "invalid_json"}, status_code=400)(scope, receive, send)
        used = False
        async def replay():
            nonlocal used
            if not used:
                used = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()
        await self.app(scope, replay, send)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    app = FastAPI(title="SZL Model Lab", docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(BoundedBody)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1"])
    security = HTTPBasic()
    env = Environment(loader=FileSystemLoader(Path(__file__).parent / "templates"),
                      autoescape=select_autoescape(["html"]))
    loaded = {}
    failures = set()
    for slug, root in settings.artifacts.items():
        try:
            model, config, digest = load_candidate(root)
            if model.track.slug != slug or config["state"] != "TRAINED_UNQUALIFIED":
                raise ValueError("candidate_not_accepted_for_research_scoring")
            loaded[slug] = (model, config, digest)
        except (OSError, ValueError, KeyError, TypeError):
            failures.add(slug)

    def authorize(credentials: HTTPBasicCredentials = Depends(security)) -> None:
        user_ok = hmac.compare_digest(credentials.username.encode(), b"operator")
        token_ok = hmac.compare_digest(credentials.password.encode(), settings.access_token.encode())
        if not (user_ok and token_ok):
            raise HTTPException(401, "authentication_required", headers={"WWW-Authenticate": "Basic"})

    @app.middleware("http")
    async def headers(request: Request, call_next):
        origin = request.headers.get("origin")
        if request.method == "POST" and origin and origin != str(request.base_url).rstrip("/"):
            return JSONResponse({"detail": "cross_origin_request_rejected"}, status_code=403)
        response = await call_next(request)
        response.headers.update({"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
                                 "Referrer-Policy": "no-referrer",
                                 "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; frame-ancestors 'none'; base-uri 'none'"})
        return response

    def state() -> list[dict]:
        rows = catalog()
        for row in rows:
            slug = row["slug"]
            if slug in loaded:
                _, config, digest = loaded[slug]
                row.update(state=config["state"], manifest_sha256=digest,
                           artifact_integrity="LOCAL_HASH_CHECKED_UNSIGNED", ready=False)
            elif slug in failures:
                row.update(state="INVALID_OR_UNAVAILABLE_CANDIDATE", ready=False)
            else:
                row["ready"] = False
        return rows

    def pool_state() -> dict:
        if settings.pool_snapshot is None:
            return unavailable_pool()
        try:
            return load_pool_snapshot(settings.pool_snapshot, settings.pool_snapshot_sha256)
        except (ValueError, OSError, TypeError, KeyError):
            return unavailable_pool("INVALID_OR_UNAVAILABLE_SNAPSHOT")

    @app.get("/api/pool", dependencies=[Depends(authorize)])
    def api_pool() -> dict:
        return pool_state()

    async def node_rows() -> list[dict]:
        return list(await asyncio.gather(*(inspect_node(name, url) for name, url in settings.nodes.items())))

    @app.get("/healthz")
    def health() -> dict:
        return {"service": "szl-model-lab", "status": "running", "model_readiness": "NOT_ASSERTED"}

    @app.get("/api/catalog", dependencies=[Depends(authorize)])
    def api_catalog() -> dict:
        return {"tracks": state(), "execution_authority": False, "publication_authority": False}

    @app.get("/api/nodes", dependencies=[Depends(authorize)])
    async def api_nodes() -> dict:
        return {"nodes": await node_rows(), "source": "EXPLICIT_READ_ONLY_OLLAMA_PROBE",
                "pool_qualification_verified": False, "ready": False}

    @app.get("/pool", response_class=HTMLResponse, dependencies=[Depends(authorize)])
    @app.get("/", response_class=HTMLResponse, dependencies=[Depends(authorize)])
    def home() -> HTMLResponse:
        return HTMLResponse(env.get_template("index.html").render(tracks=state(), nodes=None,
                            configured_nodes=len(settings.nodes), pool=pool_state()))

    @app.get("/compute", response_class=HTMLResponse, dependencies=[Depends(authorize)])
    async def compute() -> HTMLResponse:
        return HTMLResponse(env.get_template("index.html").render(tracks=state(), nodes=await node_rows(),
                            configured_nodes=len(settings.nodes), pool=pool_state()))

    @app.post("/api/score/{track}", dependencies=[Depends(authorize)])
    def score(track: str, body: ScoreRequest) -> dict:
        try:
            target = track_for(track)
        except ValueError:
            raise HTTPException(404, "unknown_track") from None
        if track not in loaded:
            raise HTTPException(409, "no_usable_local_research_candidate")
        model, config, digest = loaded[track]
        if body.normalization_id != config["normalization_id"]:
            raise HTTPException(422, "normalization_contract_mismatch")
        try:
            value = model.score(body.features)
        except ValueError:
            raise HTTPException(422, "invalid_feature_contract") from None
        return {"track": track, "target": target.target, "score": value,
                "interpretation": "UNCALIBRATED_ADVISORY_SCORE", "manifest_sha256": digest,
                "signature_verified": False, "runtime_qualified": False,
                "action_authorized": False, "execution_performed": False,
                "publication_eligible": False}

    return app
