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

KANCHAY_DIR = Path(__file__).resolve().parent / "assets" / "kanchay"
# Vendored SZL Kanchay v1.0.0 (byte-for-byte, see SOURCE.json). Fixed allowlist:
# no path parameter reaches the filesystem, no directory mount, no listing.
KANCHAY_ASSETS = {
    "/assets/kanchay/kanchay.css": ("kanchay.css", "text/css; charset=utf-8"),
    "/assets/kanchay/fonts/SpaceGrotesk-latin.woff2": ("fonts/SpaceGrotesk-latin.woff2", "font/woff2"),
    "/assets/kanchay/fonts/Inter-latin.woff2": ("fonts/Inter-latin.woff2", "font/woff2"),
    "/assets/kanchay/fonts/JetBrainsMono-latin.woff2": ("fonts/JetBrainsMono-latin.woff2", "font/woff2"),
}

CSS = """
:root{font-family:var(--font-sans);background:var(--color-a11oy-bg);color:var(--color-a11oy-text)}
*{box-sizing:border-box}body{margin:0;background:var(--color-a11oy-bg);-webkit-font-smoothing:antialiased}
main{max-width:1120px;margin:auto;padding:clamp(var(--space-4),4vw,var(--space-12))}
h1,h2{font-family:var(--font-display);color:var(--color-a11oy-text);text-wrap:balance}
h1{font-size:clamp(2rem,5vw,3.5rem);line-height:1.1;font-weight:300;letter-spacing:-.035em}
h2{font-size:1.2rem;line-height:1.3;font-weight:600;letter-spacing:-.02em;overflow-wrap:anywhere}
article h2{color:var(--color-a11oy-gold)}
p{line-height:1.65;max-width:76ch;color:var(--color-a11oy-text-sub);text-wrap:pretty}
.muted{color:var(--color-a11oy-text-sub)}footer.muted{color:var(--color-a11oy-text-ghost);font-size:.875rem}
.badge{display:flex;align-items:center;gap:var(--space-2);width:fit-content;max-width:100%;
padding:var(--space-1) var(--space-3);border-radius:var(--radius-full);
border:var(--border-hairline) solid color-mix(in srgb,var(--color-ink-caution) 28%,transparent);
background:color-mix(in srgb,var(--color-ink-caution) 12%,transparent);color:var(--color-a11oy-text-sub);
font-family:var(--font-mono);font-size:.75rem;font-weight:500;line-height:1.4;letter-spacing:.15em}
.badge::before{content:"";flex:none;width:6px;height:6px;border-radius:var(--radius-full);background:var(--color-warning)}
header .badge{padding:0;border:0;background:none;color:var(--color-a11oy-gold);letter-spacing:.3em}
header .badge::before{content:none}
.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:var(--space-4)}
article,section{border:var(--border-hairline) solid var(--color-a11oy-border-subtle);border-radius:var(--radius-md);
padding:var(--space-6)}article{background:var(--color-a11oy-surface)}
section{margin-top:var(--space-6);background:var(--color-a11oy-deep)}
a{color:var(--color-a11oy-gold);text-underline-offset:4px}a:hover{color:var(--gold-bright)}
article a{display:inline-block;padding:var(--space-3) 0;min-height:var(--szl-touch-target)}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font-family:var(--font-mono);font-size:.8125rem;line-height:1.6;
margin:0 0 var(--space-4);padding:var(--space-4);color:var(--color-a11oy-text-sub);background:var(--color-a11oy-bg);
border:var(--border-hairline) solid var(--color-a11oy-border-subtle);border-radius:var(--radius-md)}
label{font-family:var(--font-mono);font-size:.75rem;font-weight:500;letter-spacing:.15em;text-transform:uppercase;
color:var(--color-a11oy-text-sub)}
select,button{font:inherit;min-height:var(--szl-touch-target);padding:var(--space-2) var(--space-3);
border-radius:var(--radius-sm);max-width:100%}
select{color:var(--color-a11oy-text);background:var(--color-a11oy-bg);
border:var(--border-hairline) solid var(--color-control-border)}
select option{color:var(--color-a11oy-text);background:var(--color-a11oy-surface)}
button{cursor:pointer;padding-inline:var(--space-6);font-weight:600;color:var(--color-on-accent);
background:var(--color-a11oy-gold);border:var(--border-hairline) solid var(--color-a11oy-gold)}
button:hover{background:var(--gold-bright);border-color:var(--gold-bright)}
:focus-visible{outline:var(--border-focus) solid var(--color-focus);outline-offset:2px}
summary{cursor:pointer;min-height:var(--szl-touch-target);padding:var(--space-3) 0;font-weight:500;
color:var(--color-a11oy-text)}
.journey{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:var(--space-3)}
.journey div{border:var(--border-hairline) solid var(--color-a11oy-border-subtle);border-radius:var(--radius-md);
padding:var(--space-3);overflow-wrap:anywhere;background:var(--color-a11oy-surface)}
.journey strong{display:block;font-family:var(--font-mono);font-size:.75rem;font-weight:500;letter-spacing:.15em;
text-transform:uppercase;font-variant-numeric:tabular-nums;color:var(--color-a11oy-text)}
.journey p{margin:var(--space-2) 0 0}
@media(max-width:720px){.journey{grid-template-columns:repeat(2,minmax(0,1fr))}}
form{display:flex;flex-wrap:wrap;gap:var(--space-3);align-items:center}footer{margin-top:var(--space-8)}
.skip{position:absolute;left:var(--space-4);top:var(--space-4);z-index:var(--z-overlay);
padding:var(--space-3) var(--space-4);min-height:var(--szl-touch-target);border-radius:var(--radius-sm);
font-weight:600;color:var(--color-on-accent);background:var(--color-a11oy-gold);transform:translateY(-200%)}
.skip:focus,.skip:focus-visible{transform:none;color:var(--color-on-accent)}
@media(max-width:720px){.grid{grid-template-columns:1fr}section,article{padding:var(--space-4)}}
@media(prefers-contrast:more){article,section,select,button,.journey div{border-color:var(--color-a11oy-text)}}
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
    delivery = escape(json.dumps(blueprint_plan(key), indent=2, allow_nan=False))
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SZL model candidates</title><link rel="stylesheet" href="/assets/kanchay/kanchay.css">
<link rel="stylesheet" href="/assets/workbench.css">
</head><body><a class="skip" href="#main">Skip to content</a><main id="main"><header><p class="badge">SZL FORGE / LOCAL RESEARCH</p>
<h1>Kernels to learned models.</h1>
<p>Three trainable architectures. No trained release is claimed. This local workbench
inspects source recipes; it does not control your hardware or change Hugging Face.
The admitted Model Lab remains the operator interface; these are separate research variants.</p>
</header><div class="grid">{cards}</div><section aria-labelledby="recipe-title">
<h2 id="recipe-title">Inspect a training contract</h2><form action="/" method="get">
<label for="candidate">Candidate</label><select id="candidate" name="candidate">{options}</select>
<button type="submit">Inspect</button></form><details><summary>Architecture and feature contract</summary><pre>{detail}</pre></details>
<p class="muted">Before a supervised run: admit licensed data and feature normalization,
freeze the evaluation split, verify the current source and local environment, reserve an
idle device, and use the existing Forge supervisor. UI reachability grants none of these.</p>
</section><section aria-labelledby="delivery-title"><h2 id="delivery-title">GitHub-first delivery</h2>
<div class="journey" aria-label="Publication sequence">
<div><strong>01 / GitHub</strong><p>Reviewed source</p></div>
<div><strong>02 / Hugging Face</strong><p>Exact code projection</p></div>
<div><strong>03 / Product</strong><p>a-11-oy.com</p></div>
<div><strong>04 / Proof</strong><p>a11oy.net</p></div></div>
<p class="badge">NO TRAINED WEIGHTS</p><p>Sequence, not completion status.
This view has not observed admission, publication or either deployment.</p>
<details><summary>Projection contract and remaining gates</summary><pre>{delivery}</pre></details>
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
                "default-src 'none'; style-src 'self'; font-src 'self'; form-action 'self'; "
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

    for route, (relative, media_type) in KANCHAY_ASSETS.items():
        path = KANCHAY_DIR / relative
        # Bytes are bound at startup. Absent from the source-only projection: normal 404.
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
