"""Local ASGI HTTP integration; browser rendering is a separate acceptance gate."""
import hashlib
import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from model_candidates import blueprint, workbench
from model_candidates.workbench import create_app

VENDORED = Path(workbench.__file__).resolve().parent / "assets" / "szl"
SZL_ROUTES = {
    "/assets/szl/szl-design-system.css": ("szl-design-system.css", "text/css; charset=utf-8"),
}
CSP = ("default-src 'none'; style-src 'self'; form-action 'self'; "
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


def test_szl_allowlist_and_vendored_set_are_closed():
    assert workbench.SZL_ASSETS == SZL_ROUTES
    files = sorted(p.relative_to(VENDORED).as_posix() for p in VENDORED.rglob("*") if p.is_file())
    assert files == ["SOURCE.json", "szl-design-system.css"]
    # The exact-source projection carries every served file and its pin, so a workbench
    # run from a projected archive links a stylesheet that is actually present.
    for relative in files:
        assert f"model_candidates/assets/szl/{relative}" in blueprint.SOURCE_FILES


@pytest.mark.parametrize("route", sorted(SZL_ROUTES))
def test_vendored_szl_asset_served_exactly_behind_same_boundary(client, route):
    relative, media_type = SZL_ROUTES[route]
    response = client.get(route)
    assert response.status_code == 200
    assert response.headers["content-type"] == media_type
    assert response.content == (VENDORED / relative).read_bytes()
    pinned = json.loads((VENDORED / "SOURCE.json").read_text(encoding="utf-8"))
    assert pinned["name"] == "szl-kanchay" and pinned["version"] == "1.1.0"
    assert hashlib.sha256(response.content).hexdigest() == pinned["sha256"][relative]
    assert response.headers["content-security-policy"] == CSP
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert client.post(route).status_code == 405
    for headers in ({"host": "evil.example"}, {"origin": "null"}, {"sec-fetch-site": "cross-site"}):
        assert client.get(route, headers=headers).status_code == 403


@pytest.mark.parametrize("path", [
    "/assets/szl", "/assets/szl/", "/assets/szl/SOURCE.json", "/assets/szl/szl-console.css",
    "/assets/szl/SZL-DESIGN-SYSTEM.CSS", "/assets/szl/logos/szl_favicon.svg",
    "/assets/szl/%2e%2e/workbench.py", "/assets/szl/..%2f..%2fworkbench.py",
    "/assets/kanchay/kanchay.css", "/assets/kanchay/fonts/Inter-latin.woff2",
])
def test_unknown_szl_or_withdrawn_path_is_normal_404(client, path):
    response = client.get(path)
    assert response.status_code == 404
    assert response.headers["content-security-policy"] == CSP


def test_page_loads_szl_before_surface_stylesheet(client):
    response = client.get("/")
    assert response.headers["content-security-policy"] == CSP
    page = response.text
    system = page.index('<link rel="stylesheet" href="/assets/szl/szl-design-system.css">')
    assert system < page.index('<link rel="stylesheet" href="/assets/workbench.css">')
    assert '<a class="skip-link" href="#main">Skip to content</a><main id="main">' in page
    assert page.count("btn-primary") == 1
    assert "style=" not in page and "fonts.googleapis" not in page and "woff2" not in page


def test_surface_css_uses_only_szl_tokens():
    css = workbench.CSS
    assert not re.search(r"#[0-9a-fA-F]{3,8}\b|\b(?:rgba?|hsla?)\(|gradient|border-left|@font-face", css)
    defined = set(re.findall(r"(--[A-Za-z0-9-]+)\s*:",
                             (VENDORED / "szl-design-system.css").read_text(encoding="utf-8")))
    used = set(re.findall(r"var\((--[A-Za-z0-9-]+)", css))
    assert used and used <= defined
    assert all(value.startswith("var(--font-") for value in re.findall(r"font-family:([^;}]+)", css))


def test_page_still_serves_when_vendored_assets_are_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(workbench, "SZL_DIR", tmp_path)
    with TestClient(create_app(), base_url="http://127.0.0.1:8765", client=("127.0.0.1", 45679)) as local:
        assert local.get("/").status_code == 200
        for route in SZL_ROUTES:
            assert local.get(route).status_code == 404
