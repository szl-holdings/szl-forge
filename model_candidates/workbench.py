"""Python-rendered local design workbench, not the public Forge Lab Space.

No inference, training, provider probing, shell, upload, filesystem-reading,
checkpoint-loading or publication endpoints are exposed. Localhost is a scope
boundary, not multi-user authentication; do not tunnel or deploy this prototype.
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import re
from html import escape

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

from .specs import candidate_spec, candidate_specs, mesh_declaration

CSS = """
:root{color-scheme:dark;font-family:system-ui,sans-serif;background:#080e18;color:#ecf6ff}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(at 80% 0%,#173344,transparent 65%)}
main{max-width:1120px;margin:auto;padding:clamp(16px,4vw,48px)}
h1{font-size:clamp(2rem,5vw,3.5rem);line-height:1.1}h2{font-size:1.2rem}
p{line-height:1.65;max-width:76ch}.muted{color:#bbc9d8}.badge{color:#81f1dc;font-weight:700}
.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}
article,section{border:1px solid #3b596c;border-radius:16px;padding:22px;background:#0a1524dd}
section{margin-top:22px}a{color:#9bdcff}article a{display:inline-block;padding:12px 0}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:.9rem;line-height:1.5}
select,button{font:inherit;color:inherit;background:#152a3b;border:1px solid #7898ac;
min-height:44px;padding:10px;border-radius:8px;max-width:100%}button{cursor:pointer}
:focus-visible{outline:3px solid #81f1dc;outline-offset:4px}
form{display:flex;flex-wrap:wrap;gap:12px;align-items:center}footer{margin-top:28px}
@media(max-width:720px){.grid{grid-template-columns:1fr}section,article{padding:16px}}
@media(prefers-contrast:more){article,section,select,button{border-color:white}}
@media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
"""


def render_page(key: str) -> str:
    spec = candidate_spec(key)
    cards = "".join(
        f'<article><h2>{escape(row["proposed_hf_id"].split("/")[-1])}</h2>'
        f'<p>{escape(row["purpose"])}</p><p class="badge">NOT TRAINED</p>'
        f'<a href="/?candidate={escape(row["key"], quote=True)}">Inspect recipe</a></article>'
        for row in candidate_specs()
    )
    options = "".join(
        f'<option value="{escape(row["key"], quote=True)}"'
        f'{" selected" if row["key"] == key else ""}>{escape(row["key"])}</option>'
        for row in candidate_specs()
    )
    detail = escape(json.dumps(spec, indent=2, allow_nan=False))
    mesh = escape(json.dumps(mesh_declaration(), indent=2, allow_nan=False))
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SZL model candidates</title><link rel="stylesheet" href="/assets/workbench.css">
</head><body><main><header><p class="badge">SZL FORGE / LOCAL RESEARCH</p>
<h1>Kernels to learned models.</h1>
<p>Three trainable architectures. No trained release is claimed. This local workbench
inspects source recipes; it does not control your hardware or change Hugging Face.</p>
</header><div class="grid">{cards}</div><section aria-labelledby="recipe-title">
<h2 id="recipe-title">Inspect a training contract</h2><form action="/" method="get">
<label for="candidate">Candidate</label><select id="candidate" name="candidate">{options}</select>
<button type="submit">Inspect</button></form><pre>{detail}</pre>
<p class="muted">Before a supervised run: admit licensed data and feature normalization,
freeze the evaluation split, verify the current source and local environment, reserve an
idle device, and use the existing Forge supervisor. UI reachability grants none of these.</p>
</section><section aria-labelledby="mesh-title"><h2 id="mesh-title">Existing mesh / source map</h2>
<p class="badge">DECLARED, NOT OBSERVED</p><details><summary>Show source connections</summary>
<pre>{mesh}</pre></details></section><footer class="muted">
GitHub → Hugging Face → a-11-oy.com → a11oy.net. No deployment or publication in this lane.
</footer></main></body></html>'''


def _local_request(request: Request) -> bool:
    """Reject non-loopback clients, DNS rebinding hosts, and cross-site requests."""
    try:
        if request.client is None or not ipaddress.ip_address(request.client.host).is_loopback:
            return False
    except ValueError:
        return False
    hosts = request.headers.getlist("host")
    if len(hosts) != 1 or not re.fullmatch(r"(?:127\.0\.0\.1|localhost)(?::[0-9]{1,5})?", hosts[0]):
        return False
    if ":" in hosts[0] and not 1 <= int(hosts[0].rsplit(":", 1)[1]) <= 65535:
        return False
    origins = request.headers.getlist("origin")
    if origins and (len(origins) != 1 or origins[0] != "http://" + hosts[0]):
        return False
    return request.headers.get("sec-fetch-site") != "cross-site"


def create_app() -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def boundary(request: Request, call_next):
        if not _local_request(request):
            response = JSONResponse({"error": "local same-origin access only"}, status_code=403)
        else:
            response = await call_next(request)
        response.headers.update({
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer",
            "Content-Security-Policy": (
                "default-src 'none'; style-src 'self'; form-action 'self'; "
                "frame-ancestors 'none'; base-uri 'none'"
            ),
        })
        return response

    @app.get("/", response_class=HTMLResponse)
    async def index(candidate: str = "router"):
        try:
            return HTMLResponse(render_page(candidate))
        except ValueError as exc:
            raise HTTPException(404, "unknown candidate") from exc

    @app.get("/assets/workbench.css")
    async def stylesheet():
        return Response(CSS, media_type="text/css")

    @app.get("/healthz")
    async def health():
        return {"app": "reachable", "models": "SOURCE_ONLY_NOT_TRAINED", "mesh": "NOT_OBSERVED"}

    @app.get("/api/candidates")
    async def candidates():
        return {"scope": "three source recipes, not the HF inventory", "candidates": candidate_specs()}

    @app.get("/api/candidates/{key}")
    async def candidate_detail(key: str):
        try:
            return candidate_spec(key)
        except ValueError as exc:
            raise HTTPException(404, "unknown candidate") from exc

    @app.get("/api/mesh")
    async def mesh():
        return mesh_declaration()

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("port must be between 1024 and 65535")
    import uvicorn
    uvicorn.run(create_app(), host="127.0.0.1", port=args.port,
                proxy_headers=False, access_log=False)


if __name__ == "__main__":
    main()
