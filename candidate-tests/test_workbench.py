"""Local ASGI HTTP integration; browser rendering is a separate acceptance gate."""
import pytest
from fastapi.testclient import TestClient

from model_candidates.workbench import create_app


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
