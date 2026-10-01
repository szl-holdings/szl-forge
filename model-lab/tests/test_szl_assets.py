# SPDX-License-Identifier: Apache-2.0
# (c) 2026 Lutar, Stephen P. - SZL Holdings - ORCID 0009-0001-0110-4173
"""Allowlisted, read-only SZL KANCHAY stylesheet routes; nothing else is served."""
import fnmatch
import hashlib
import json
import re
import tomllib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from szl_model_lab import blueprints
from szl_model_lab.app import SZL_ASSETS, Settings, create_app

TOKEN = "szl-route-test-only-" + "k" * 32
ROOT = Path(__file__).resolve().parents[1]
VENDORED = ROOT / "src/szl_model_lab/static/szl"
CSP = ("default-src 'none'; style-src 'self' 'unsafe-inline'; "
       "form-action 'self'; frame-ancestors 'none'; base-uri 'none'")
EXPECTED = {
    "/szl/szl-design-system.css": ("szl-design-system.css", "text/css; charset=utf-8"),
    "/szl/szl-console.css": ("szl-console.css", "text/css; charset=utf-8"),
}
DESIGN_SYSTEM_SHA256 = "9e5e6e3ae2a5c6f5a2a3c6bf4c607406dc39db3fbe9115e91a1865054678493e"
CONSOLE_SHA256 = "7ef6a0908302c6c2d83d637d1aa4fcac36ae2740a5c8e9cac0608c3cdb2c9928"
PAGES = ["/", "/pool", "/corpus-review", "/compute", "/storage"]
LINKS = ('<link rel="stylesheet" href="/szl/szl-design-system.css">\n'
         '<link rel="stylesheet" href="/szl/szl-console.css">\n')


@pytest.fixture
def client():
    return TestClient(create_app(Settings(TOKEN, {}, {})), base_url="http://127.0.0.1")


def auth():
    return ("operator", TOKEN)


def source_digests() -> dict:
    return json.loads((VENDORED / "SOURCE.json").read_text(encoding="utf-8"))["sha256"]


def test_allowlist_is_exactly_the_two_stylesheets():
    assert SZL_ASSETS == EXPECTED


@pytest.mark.parametrize("path", sorted(EXPECTED))
def test_asset_is_exact_vendored_bytes_with_type_and_security_headers(client, path):
    name, media_type = EXPECTED[path]
    response = client.get(path, auth=auth())
    assert response.status_code == 200
    assert response.content == (VENDORED / name).read_bytes()
    assert hashlib.sha256(response.content).hexdigest() == source_digests()[name]
    assert response.headers["content-type"] == media_type
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert response.headers["content-security-policy"] == CSP


@pytest.mark.parametrize("path", sorted(EXPECTED))
def test_asset_keeps_page_admission(client, path):
    assert client.get(path).status_code == 401
    assert client.get(path, auth=("operator", "wrong")).status_code == 401
    assert client.get(path, auth=("other", TOKEN)).status_code == 401
    assert client.get(path, auth=auth(), headers={"Host": "attacker.example"}).status_code == 400


@pytest.mark.parametrize("path", ["/szl", "/szl/", "/szl/SOURCE.json", "/szl/DESIGN_DIRECTION.md",
                                  "/szl/logos/szl_favicon.svg", "/szl/logos/szl_favicon_32.png",
                                  "/szl/szl-design-system.cssx", "/szl/SZL-DESIGN-SYSTEM.CSS",
                                  "/szl/%2e%2e/app.py", "/szl/..%2fapp.py", "/static/szl/szl-console.css",
                                  "/kanchay/kanchay.css", "/kanchay/fonts/Inter-latin.woff2"])
def test_unknown_or_withdrawn_path_is_normal_404(client, path):
    assert client.get(path, auth=auth()).status_code == 404
    assert client.get(path).status_code == 404


@pytest.mark.parametrize("method", ["post", "put", "patch", "delete"])
def test_assets_have_no_mutating_methods(client, method):
    for path in EXPECTED:
        assert getattr(client, method)(path, auth=auth()).status_code == 405


@pytest.mark.parametrize("path", PAGES)
def test_pages_load_local_design_system_before_inline_style(client, path):
    response = client.get(path, auth=auth())
    assert response.status_code == 200
    assert response.headers["content-security-policy"] == CSP
    text = response.text
    assert text.count(LINKS) == 1 and text.index(LINKS) < text.index("<style>")
    head = text.split("<body>")[0]
    assert 'href="http' not in head and "@font-face" not in text and ".woff" not in text
    assert "kanchay" not in text


@pytest.mark.parametrize("path", ["/", "/storage"])
def test_at_most_one_coral_moment_per_view(client, path):
    text = client.get(path, auth=auth()).text
    assert text.count("btn-primary") == (1 if path == "/storage" else 0)
    assert "aria-current" not in text and "badge-accent" not in text and "hero__node" not in text


@pytest.mark.parametrize("name", ["index.html", "storage.html"])
def test_template_styles_use_founder_tokens_not_literals(name):
    style = (ROOT / "src/szl_model_lab/templates" / name).read_text(encoding="utf-8")
    style = style[style.index("<style>"):style.index("</style>")]
    assert not re.search(r"#[0-9a-fA-F]{3,8}\b|\brgba?\(|\bhsla?\(", style)
    assert not re.search(r"font-family|system-ui|ui-monospace|sans-serif|monospace", style)
    assert not re.search(r"--color-a11oy|--gold|--teal|kc-", style)
    assert "var(--" in style


def test_vendored_folder_is_the_exact_export_closure():
    files = {p.relative_to(VENDORED).as_posix() for p in VENDORED.rglob("*") if p.is_file()}
    assert files == {"szl-design-system.css", "szl-console.css", "SOURCE.json"}
    source = json.loads((VENDORED / "SOURCE.json").read_text(encoding="utf-8"))
    assert source["version"] == "1.1.0"
    assert source["sha256"]["szl-design-system.css"] == DESIGN_SYSTEM_SHA256
    assert source["sha256"]["szl-console.css"] == CONSOLE_SHA256
    for name in files - {"SOURCE.json"}:
        assert hashlib.sha256((VENDORED / name).read_bytes()).hexdigest() == source["sha256"][name]
    assert not (ROOT / "src/szl_model_lab/static/kanchay").exists()


def test_wheel_package_data_and_blueprint_closure_carry_every_vendored_file():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    patterns = project["tool"]["setuptools"]["package-data"]["szl_model_lab"]
    assert not any("kanchay" in pattern or "woff" in pattern for pattern in patterns)
    assert not any("kanchay" in path for path in blueprints.SOURCE_FILES)
    for path in VENDORED.rglob("*"):
        if path.is_file():
            relative = path.relative_to(ROOT / "src/szl_model_lab").as_posix()
            assert any(fnmatch.fnmatch(relative, pattern) for pattern in patterns), relative
            assert "src/szl_model_lab/" + relative in blueprints.SOURCE_FILES
