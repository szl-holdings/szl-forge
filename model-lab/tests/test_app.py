import pytest
from fastapi.testclient import TestClient
from szl_model_lab.app import Settings, create_app
from szl_model_lab.artifacts import write_candidate
from szl_model_lab.models import AdvisoryMLP

TOKEN = "unit-test-only-" + "x"*32

@pytest.fixture
def client():
    return TestClient(create_app(Settings(TOKEN, {}, {})), base_url="http://localhost")

def auth():
    return ("operator", TOKEN)

def test_health_is_not_readiness(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["model_readiness"] == "NOT_ASSERTED"

@pytest.mark.parametrize("path", ["/", "/compute", "/api/catalog", "/api/nodes"])
def test_authentication_required(client, path):
    assert client.get(path).status_code == 401

def test_wrong_credentials(client):
    assert client.get("/", auth=("operator", "wrong")).status_code == 401

def test_token_required():
    with pytest.raises(ValueError):
        Settings("", {}, {})

def test_no_arbitrary_hosts(client):
    assert client.get("/", auth=auth(), headers={"Host": "attacker.example"}).status_code == 400

def test_truthful_no_weight_state(client):
    response = client.get("/api/catalog", auth=auth())
    assert response.status_code == 200
    data = response.json()
    assert not data["execution_authority"] and not data["publication_authority"]
    assert data["tracks"][0]["state"] == "BLUEPRINT_NOT_TRAINED"
    assert all(t["ready"] is False and t["publication_eligible"] is False for t in data["tracks"])

def test_no_fabricated_node_state(client):
    data = client.get("/api/nodes", auth=auth()).json()
    assert data["nodes"] == [] and data["ready"] is False

def test_missing_weights_does_not_run_random_model(client):
    response = client.post("/api/score/router", auth=auth(), json={"normalization_id": "a", "features": {}})
    assert response.status_code == 409

@pytest.mark.parametrize("path", ["/api/train", "/api/publish", "/api/execute", "/api/jobs", "/api/merge"])
def test_no_execution_routes(client, path):
    assert client.post(path, auth=auth(), json={}).status_code == 404

def test_bounded_body_without_trusting_content_length(client):
    response = client.post("/api/score/router", auth=auth(), content=b"x"*20000)
    assert response.status_code == 413

def test_cross_origin_rejected(client):
    response = client.post("/api/score/router", auth=auth(), headers={"Origin": "https://attacker.example"}, json={})
    assert response.status_code == 403

def test_python_frontend_has_semantics_and_security_headers(client):
    response = client.get("/", auth=auth())
    assert "<main>" in response.text and 'lang="en"' in response.text
    assert 'name="viewport"' in response.text and 'aria-labelledby="tracks"' in response.text
    assert "prefers-reduced-motion" in response.text and "prefers-contrast" in response.text
    assert response.headers["cache-control"] == "no-store"
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
    assert TOKEN not in response.text

def test_invalid_artifact_is_not_operational(tmp_path):
    c = TestClient(create_app(Settings(TOKEN, {}, {"router": tmp_path / "missing"})), base_url="http://localhost")
    data = c.get("/api/catalog", auth=auth()).json()
    assert data["tracks"][0]["state"] == "INVALID_OR_UNAVAILABLE_CANDIDATE"

def test_unqualified_advice_never_authorizes(tmp_path, dataset):
    root = tmp_path / "candidate"
    # Simulation of an owner-run unsigned candidate. This test is not provenance.
    write_candidate(root, AdvisoryMLP("router"), source={"verification": "GIT_CLEAN"},
                    dataset=dataset.summary(), validation={}, recipe={"test_fixture": True})
    c = TestClient(create_app(Settings(TOKEN, {}, {"router": root})), base_url="http://localhost")
    response = c.post("/api/score/router", auth=auth(), json={"normalization_id": dataset.normalization_id,
                       "features": dataset.rows[0].features})
    assert response.status_code == 200
    data = response.json()
    assert 0 <= data["score"] <= 1
    for field in ("action_authorized", "execution_performed", "runtime_qualified", "publication_eligible", "signature_verified"):
        assert data[field] is False
    wrong = c.post("/api/score/router", auth=auth(), json={"normalization_id": "wrong", "features": dataset.rows[0].features})
    assert wrong.status_code == 422

def test_fixture_weights_are_refused_by_workbench(tmp_path, dataset):
    root = tmp_path / "fixture"
    write_candidate(root, AdvisoryMLP("router"), source={"verification": "TEST_FIXTURE_ONLY"},
                    dataset=dataset.summary(), validation={}, recipe={})
    c = TestClient(create_app(Settings(TOKEN, {}, {"router": root})), base_url="http://localhost")
    assert c.get("/api/catalog", auth=auth()).json()["tracks"][0]["state"] == "INVALID_OR_UNAVAILABLE_CANDIDATE"

def test_model_name_escaped_in_html(client, monkeypatch):
    import szl_model_lab.app as module
    async def fake(*args, **kwargs):
        return {"node_id": "laptop", "state": "INVENTORY_OBSERVED_NOT_QUALIFIED", "qualification": "NOT_VERIFIED",
                "error": None, "stored_models": [{"name": "<script>alert(1)</script>", "digest": "a"*64}], "running_models": []}
    monkeypatch.setattr(module, "inspect_node", fake)
    c = TestClient(create_app(Settings(TOKEN, {"laptop": "http://127.0.0.1:11434"}, {})), base_url="http://localhost")
    text = c.get("/compute", auth=auth()).text
    assert "<script>" not in text and "&lt;script&gt;" in text
