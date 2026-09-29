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
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

from .specs import candidate_spec, candidate_specs, mesh_declaration
from .blueprint import blueprint_plan

SZL_DIR = Path(__file__).resolve().parent / "assets" / "szl"
# Vendored SZL KANCHAY v1.1.0 (byte-for-byte, see SOURCE.json). Fixed allowlist:
# no path parameter reaches the filesystem, no directory mount, no listing.
SZL_ASSETS = {
    "/assets/szl/szl-design-system.css": ("szl-design-system.css", "text/css; charset=utf-8"),
}

CSS = """
main{max-width:1120px;margin:auto;padding:clamp(var(--space-4),4vw,var(--space-12))}
h1{font-size:clamp(var(--text-2xl),5vw,var(--text-4xl));text-wrap:balance}
h2{font-size:var(--text-lg);line-height:var(--leading-snug);overflow-wrap:anywhere}
.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:var(--space-4)}
section{margin-top:var(--space-6);padding:var(--space-6);background:var(--surface);
border:var(--border-hairline) solid var(--border);border-radius:var(--radius-lg)}
article a{display:inline-block;padding:var(--space-3) 0;min-block-size:var(--target-size-coarse)}
p.chip{display:flex;inline-size:fit-content;max-inline-size:100%}
.chip-sample{color:var(--text-sub)}
pre.code{white-space:pre-wrap;overflow-wrap:anywhere;margin:0 0 var(--space-4)}
label{font-size:var(--text-xs);font-weight:var(--weight-semibold);letter-spacing:var(--tracking-caps);
text-transform:uppercase;color:var(--text-sub)}
select{font:inherit;min-block-size:var(--target-size-coarse);max-inline-size:100%;padding:var(--space-2) var(--space-3);
color:var(--text);background:var(--bg-deep);border:var(--border-hairline) solid var(--color-gray-400);
border-radius:var(--radius-md)}
select option{color:var(--text);background:var(--surface)}
form .btn{min-block-size:var(--target-size-coarse);max-inline-size:100%}
summary{cursor:pointer;min-block-size:var(--target-size-coarse);padding:var(--space-3) 0;
font-weight:var(--weight-medium);color:var(--text)}
:where(a,button,select,summary):focus-visible{outline:var(--border-focus) solid var(--focus);outline-offset:2px}
.journey{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:var(--space-3);margin-bottom:var(--space-4)}
.journey div{padding:var(--space-3);overflow-wrap:anywhere;background:var(--surface-alt);
border:var(--border-hairline) solid var(--border);border-radius:var(--radius-md)}
.journey strong{display:block;font-family:var(--font-mono);font-size:var(--text-xs);font-weight:var(--weight-semibold);
letter-spacing:var(--tracking-caps);text-transform:uppercase;font-variant-numeric:tabular-nums lining-nums;
color:var(--text)}
.journey p{margin:var(--space-2) 0 0}
@media(max-width:720px){.journey{grid-template-columns:repeat(2,minmax(0,1fr))}}
form{display:flex;flex-wrap:wrap;gap:var(--space-3);align-items:center}
footer{margin-top:var(--space-8);font-size:var(--text-sm)}
@media(max-width:720px){.grid{grid-template-columns:1fr}section,.card{padding:var(--space-4)}}
@media(prefers-contrast:more){.card,section,select,.btn,.journey div{border-color:var(--text)}}
@media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
"""


def render_page(key: str) -> str:
    spec = candidate_spec(key)
    cards = "".join(
        f'<article class="card"><h2 class="card-title">{escape(row["proposed_hf_id"].split("/")[-1])}</h2>'
        f'<p>{escape(row["purpose"])}</p><p class="chip chip-unavailable">NOT TRAINED</p>'
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
    delivery = escape(json.dumps(blueprint_plan(key), indent=2, allow_nan=False))
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SZL model candidates</title><link rel="stylesheet" href="/assets/szl/szl-design-system.css">
<link rel="stylesheet" href="/assets/workbench.css">
</head><body><a class="skip-link" href="#main">Skip to content</a><main id="main"><header><p class="eyebrow">SZL FORGE / LOCAL RESEARCH</p>
<h1>Kernels to learned models.</h1>
<p>Three trainable architectures. No trained release is claimed. This local workbench
inspects source recipes; it does not control your hardware or change Hugging Face.
The admitted Model Lab remains the operator interface; these are separate research variants.</p>
</header><div class="grid">{cards}</div><section aria-labelledby="recipe-title">
<h2 id="recipe-title">Inspect a training contract</h2><form action="/" method="get">
<label for="candidate">Candidate</label><select id="candidate" name="candidate">{options}</select>
<button class="btn btn-primary" type="submit">Inspect</button></form><details><summary>Architecture and feature contract</summary><pre class="code">{detail}</pre></details>
<p class="muted">Before a supervised run: admit licensed data and feature normalization,
freeze the evaluation split, verify the current source and local environment, reserve an
idle device, and use the existing Forge supervisor. UI reachability grants none of these.</p>
</section><section aria-labelledby="delivery-title"><h2 id="delivery-title">GitHub-first delivery</h2>
<div class="journey" aria-label="Publication sequence">
<div><strong>01 / GitHub</strong><p>Reviewed source</p></div>
<div><strong>02 / Hugging Face</strong><p>Exact code projection</p></div>
<div><strong>03 / Product</strong><p>a-11-oy.com</p></div>
<div><strong>04 / Proof</strong><p>a11oy.net</p></div></div>
<p class="chip chip-unavailable">NO TRAINED WEIGHTS</p><p>Sequence, not completion status.
This view has not observed admission, publication or either deployment.</p>
<details><summary>Projection contract and remaining gates</summary><pre class="code">{delivery}</pre></details>
</section><section aria-labelledby="mesh-title"><h2 id="mesh-title">Existing mesh / source map</h2>
<p class="chip chip-sample">DECLARED, NOT OBSERVED</p><details><summary>Show source connections</summary>
<pre class="code">{mesh}</pre></details></section><footer class="muted">
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


def _static_asset(body: bytes, media_type: str):
    async def asset() -> Response:
        return Response(body, media_type=media_type)
    return asset


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

    for route, (relative, media_type) in SZL_ASSETS.items():
        path = SZL_DIR / relative
        # Bytes are bound at startup. A missing or symlinked file gets no route: normal 404.
        if path.is_file() and not path.is_symlink():
            app.add_api_route(route, _static_asset(path.read_bytes(), media_type),
                              methods=["GET"], include_in_schema=False)

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

    @app.get("/api/blueprints/{key}")
    async def blueprint_detail(key: str):
        try:
            return blueprint_plan(key)
        except ValueError as exc:
            raise HTTPException(404, "unknown candidate") from exc

    @app.get("/api/mesh")
    async def mesh():
        return mesh_declaration()

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("port must be between 1024 and 65535")
    import uvicorn
    uvicorn.run(create_app(), host="127.0.0.1", port=args.port,
                proxy_headers=False, access_log=False)


if __name__ == "__main__":
    main()
