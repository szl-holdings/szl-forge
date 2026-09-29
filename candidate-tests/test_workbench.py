"""Local ASGI HTTP integration; browser rendering is a separate acceptance gate."""
import hashlib
import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from model_candidates import workbench
from model_candidates.workbench import create_app

VENDORED = Path(workbench.__file__).resolve().parent / "assets" / "kanchay"
KANCHAY_ROUTES = {
    "/assets/kanchay/kanchay.css": ("kanchay.css", "text/css; charset=utf-8"),
    "/assets/kanchay/fonts/SpaceGrotesk-latin.woff2": ("fonts/SpaceGrotesk-latin.woff2", "font/woff2"),
    "/assets/kanchay/fonts/Inter-latin.woff2": ("fonts/Inter-latin.woff2", "font/woff2"),
    "/assets/kanchay/fonts/JetBrainsMono-latin.woff2": ("fonts/JetBrainsMono-latin.woff2", "font/woff2"),
}
CSP = ("default-src 'none'; style-src 'self'; font-src 'self'; form-action 'self'; "
       "frame-ancestors 'none'; base-uri 'none'")


@pytest.fixture
def client():
    with TestClient(create_app(), base_url="http://127.0.0.1:8765", client=("127.0.0.1", 45678)) as value:
        yield value


@pytest.mark.parametrize("key", ["router", "invariant-risk", "yarqa-causal"])
def test_python_frontend_and_backend_share_recipe(client, key):
    page = client.get("/", params={"candidate": key})
    assert page.status_code == 200
    assert "NOT TRAINED" in page.text and "DECLARED, NOT OBSERVED" in page.text
    assert 'name="viewport"' in page.text and '<label for="candidate">' in page.text
    assert '<script' not in page.text
    recipe = client.get(f"/api/candidates/{key}").json()
    assert recipe["state"] == "SOURCE_ONLY_NOT_TRAINED"
    assert recipe["runtime_eligible"] is False and recipe["benchmark"] is None
    assert recipe["architecture"] in page.text


def test_no_mutation_or_training_endpoints(client):
    assert client.post("/api/candidates").status_code == 405
    for endpoint in ["/train", "/publish", "/execute", "/api/probe", "/docs", "/openapi.json"]:
        assert client.get(endpoint).status_code == 404
    state = client.get("/api/mesh").json()
    assert state["state"] == "DECLARED_NOT_OBSERVED" and state["observed_at"] is None


@pytest.mark.parametrize("headers", [
    {"host": "evil.example"}, {"host": "127.0.0.1.evil.example"},
    {"host": "127.0.0.1:99999"}, {"origin": "https://evil.example"},
    {"origin": "null"}, {"sec-fetch-site": "cross-site"},
])
def test_host_origin_boundary(client, headers):
    response = client.get("/api/mesh", headers=headers)
    assert response.status_code == 403
    assert response.headers["cache-control"] == "no-store"


def test_nonlocal_client_is_denied_even_with_forged_forwarding():
    with TestClient(create_app(), base_url="http://127.0.0.1:8765", client=("192.0.2.8", 50000)) as client:
        response = client.get("/", headers={"x-forwarded-for": "127.0.0.1"})
        assert response.status_code == 403


def test_response_security_and_unknown_candidate(client):
    response = client.get("/", headers={"origin": "http://127.0.0.1:8765"})
    assert response.status_code == 200
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
    assert response.headers["x-content-type-options"] == "nosniff"
    assert client.get("/", params={"candidate": '<script>alert(1)</script>'}).status_code == 404
    assert client.get("/api/candidates/nonexistent").status_code == 404
    assert client.get("/assets/workbench.css").headers["content-type"].startswith("text/css")
    assert client.get("/healthz").json()["mesh"] == "NOT_OBSERVED"


def test_kanchay_allowlist_and_vendored_set_are_closed():
    assert workbench.KANCHAY_ASSETS == KANCHAY_ROUTES
    files = sorted(p.relative_to(VENDORED).as_posix() for p in VENDORED.rglob("*") if p.is_file())
    assert files == ["SOURCE.json", "fonts/Inter-latin.woff2", "fonts/JetBrainsMono-latin.woff2",
                     "fonts/SpaceGrotesk-latin.woff2", "kanchay.css"]


@pytest.mark.parametrize("route", sorted(KANCHAY_ROUTES))
def test_vendored_kanchay_asset_served_exactly_behind_same_boundary(client, route):
    relative, media_type = KANCHAY_ROUTES[route]
    response = client.get(route)
    assert response.status_code == 200
    assert response.headers["content-type"] == media_type
    assert response.content == (VENDORED / relative).read_bytes()
    pinned = json.loads((VENDORED / "SOURCE.json").read_text(encoding="utf-8"))
    assert pinned["name"] == "szl-kanchay" and pinned["version"] == "1.0.0"
    assert hashlib.sha256(response.content).hexdigest() == pinned["sha256"][relative]
    assert response.headers["content-security-policy"] == CSP
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert client.post(route).status_code == 405
    for headers in ({"host": "evil.example"}, {"origin": "null"}, {"sec-fetch-site": "cross-site"}):
        assert client.get(route, headers=headers).status_code == 403


@pytest.mark.parametrize("path", [
    "/assets/kanchay", "/assets/kanchay/", "/assets/kanchay/SOURCE.json",
    "/assets/kanchay/kanchay-components.css", "/assets/kanchay/KANCHAY.CSS", "/assets/kanchay/fonts",
    "/assets/kanchay/fonts/", "/assets/kanchay/fonts/Syncopate-400.woff2",
    "/assets/kanchay/fonts/Inter-latin.woff", "/assets/kanchay/%2e%2e/workbench.py",
    "/assets/kanchay/fonts/..%2fkanchay.css", "/assets/kanchay/fonts/%2e%2e/%2e%2e/workbench.py",
])
def test_unknown_kanchay_path_is_normal_404(client, path):
    response = client.get(path)
    assert response.status_code == 404
    assert response.headers["content-security-policy"] == CSP


def test_page_loads_kanchay_before_surface_stylesheet(client):
    response = client.get("/")
    assert response.headers["content-security-policy"] == CSP
    page = response.text
    kanchay = page.index('<link rel="stylesheet" href="/assets/kanchay/kanchay.css">')
    assert kanchay < page.index('<link rel="stylesheet" href="/assets/workbench.css">')
    assert '<a class="skip" href="#main">Skip to content</a><main id="main">' in page
    assert "style=" not in page and "fonts.googleapis" not in page and "fonts.gstatic" not in page


def test_surface_css_uses_only_kanchay_tokens():
    css = workbench.CSS
    assert not re.search(r"#[0-9a-fA-F]{3,8}\b|\b(?:rgba?|hsla?)\(|gradient|border-left", css)
    defined = set(re.findall(r"(--[A-Za-z0-9-]+)\s*:", (VENDORED / "kanchay.css").read_text(encoding="utf-8")))
    used = set(re.findall(r"var\((--[A-Za-z0-9-]+)", css))
    assert used and used <= defined
    assert all(value.startswith("var(--font-") for value in re.findall(r"font-family:([^;}]+)", css))


def test_source_only_projection_without_vendored_assets_still_serves_page(tmp_path, monkeypatch):
    monkeypatch.setattr(workbench, "KANCHAY_DIR", tmp_path)
    with TestClient(create_app(), base_url="http://127.0.0.1:8765", client=("127.0.0.1", 45679)) as local:
        assert local.get("/").status_code == 200
        for route in KANCHAY_ROUTES:
            assert local.get(route).status_code == 404
