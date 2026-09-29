# SPDX-License-Identifier: Apache-2.0
# (c) 2026 Lutar, Stephen P. - SZL Holdings - ORCID 0009-0001-0110-4173
"""Allowlisted, read-only SZL Kanchay stylesheet and font routes; nothing else is served."""
import fnmatch
import hashlib
import json
import re
import tomllib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from szl_model_lab import blueprints
from szl_model_lab.app import KANCHAY_ASSETS, Settings, create_app

TOKEN = "kanchay-route-test-only-" + "k" * 32
ROOT = Path(__file__).resolve().parents[1]
VENDORED = ROOT / "src/szl_model_lab/static/kanchay"
CSP = ("default-src 'none'; style-src 'self' 'unsafe-inline'; font-src 'self'; "
       "form-action 'self'; frame-ancestors 'none'; base-uri 'none'")
EXPECTED = {
    "/kanchay/kanchay.css": ("kanchay.css", "text/css; charset=utf-8"),
    "/kanchay/fonts/SpaceGrotesk-latin.woff2": ("fonts/SpaceGrotesk-latin.woff2", "font/woff2"),
    "/kanchay/fonts/Inter-latin.woff2": ("fonts/Inter-latin.woff2", "font/woff2"),
    "/kanchay/fonts/JetBrainsMono-latin.woff2": ("fonts/JetBrainsMono-latin.woff2", "font/woff2"),
}
PAGES = ["/", "/pool", "/corpus-review", "/compute", "/storage"]


@pytest.fixture
def client():
    return TestClient(create_app(Settings(TOKEN, {}, {})), base_url="http://127.0.0.1")


def auth():
    return ("operator", TOKEN)


def source_digests() -> dict:
    return json.loads((VENDORED / "SOURCE.json").read_text(encoding="utf-8"))["sha256"]


def test_allowlist_is_exactly_the_stylesheet_and_three_fonts():
    assert KANCHAY_ASSETS == EXPECTED


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


@pytest.mark.parametrize("path", ["/kanchay", "/kanchay/", "/kanchay/SOURCE.json",
                                  "/kanchay/kanchay-components.css", "/kanchay/tokens.json",
                                  "/kanchay/fonts", "/kanchay/fonts/",
                                  "/kanchay/fonts/Syncopate-400.woff2", "/kanchay/fonts/Inter-latin.woff2x",
                                  "/kanchay/%2e%2e/app.py", "/kanchay/fonts/..%2fkanchay.css",
                                  "/kanchay/KANCHAY.CSS", "/static/kanchay/kanchay.css",
                                  "/kanchay/marks/szl-mark.svg"])
def test_unknown_kanchay_path_is_normal_404(client, path):
    assert client.get(path, auth=auth()).status_code == 404
    assert client.get(path).status_code == 404


@pytest.mark.parametrize("method", ["post", "put", "patch", "delete"])
def test_assets_have_no_mutating_methods(client, method):
    for path in EXPECTED:
        assert getattr(client, method)(path, auth=auth()).status_code == 405


@pytest.mark.parametrize("path", PAGES)
def test_pages_load_local_kanchay_before_inline_style(client, path):
    response = client.get(path, auth=auth())
    assert response.status_code == 200
    assert response.headers["content-security-policy"] == CSP
    text = response.text
    link = '<link rel="stylesheet" href="/kanchay/kanchay.css">'
    assert text.count(link) == 1 and text.index(link) < text.index("<style>")
    assert "fonts.googleapis" not in text and "fonts.gstatic" not in text
    assert 'href="http' not in text.split("<body>")[0]


@pytest.mark.parametrize("name", ["index.html", "storage.html"])
def test_template_styles_use_tokens_not_literals(name):
    style = (ROOT / "src/szl_model_lab/templates" / name).read_text(encoding="utf-8")
    style = style[style.index("<style>"):style.index("</style>")]
    assert not re.search(r"#[0-9a-fA-F]{3,8}\b|\brgba?\(|\bhsla?\(", style)
    assert not re.search(r"system-ui|ui-monospace|sans-serif|monospace", style)
    assert "var(--color-a11oy-bg)" in style and "var(--font-display)" in style


def test_vendored_folder_is_the_exact_export_closure():
    files = {p.relative_to(VENDORED).as_posix() for p in VENDORED.rglob("*") if p.is_file()}
    assert files == {"kanchay.css", "SOURCE.json", "fonts/SpaceGrotesk-latin.woff2",
                     "fonts/Inter-latin.woff2", "fonts/JetBrainsMono-latin.woff2"}
    source = json.loads((VENDORED / "SOURCE.json").read_text(encoding="utf-8"))
    assert source["version"] == "1.0.0"
    assert source["sha256"]["kanchay.css"] == "d083a2ca219f29164d793e243b466320f27a37080b30384f93a8c0db6313d98b"
    for name in files - {"SOURCE.json"}:
        assert hashlib.sha256((VENDORED / name).read_bytes()).hexdigest() == source["sha256"][name]


def test_wheel_package_data_and_blueprint_closure_carry_every_vendored_file():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    patterns = project["tool"]["setuptools"]["package-data"]["szl_model_lab"]
    for path in VENDORED.rglob("*"):
        if path.is_file():
            relative = path.relative_to(ROOT / "src/szl_model_lab").as_posix()
            assert any(fnmatch.fnmatch(relative, pattern) for pattern in patterns), relative
            assert "src/szl_model_lab/" + relative in blueprints.SOURCE_FILES
