import asyncio
import json
import httpx
import pytest
from fastapi.testclient import TestClient
from szl_model_lab.app import Settings, create_app
from szl_model_lab.data import Dataset
from szl_model_lab.probes import inspect_node, node_config
from szl_model_lab.safeio import strict_json

@pytest.mark.parametrize("raw", ['{"x":1,"x":2}', '{"x":{"y":1,"y":2}}', '[NaN]', '[Infinity]', '[-Infinity]'])
def test_ambiguous_json_rejected(raw):
    with pytest.raises(ValueError):
        strict_json(raw)

def test_node_alias_collision_rejected():
    with pytest.raises(ValueError, match="duplicate_json_key"):
        node_config('{"a":"http://127.0.0.1:11434","a":"http://100.80.1.2:11434"}')

def test_duplicate_dataset_label_rejected(rows):
    raw = "\n".join(json.dumps(row).replace('"label": false', '"label": true, "label": false') for row in rows)
    with pytest.raises(ValueError, match="duplicate_json_key"):
        Dataset(raw.encode(), "router")

def test_duplicate_inventory_rejected():
    transport = httpx.MockTransport(lambda _: httpx.Response(200, content=b'{"models":[],"models":[]}'))
    result = asyncio.run(inspect_node("a", "http://127.0.0.1:11434", transport=transport))
    assert result["error"] and result["ready"] is False

def test_api_duplicate_json_rejected():
    token = "fixture-only-" + "x" * 32
    client = TestClient(create_app(Settings(token, {}, {})), base_url="http://localhost")
    response = client.post("/api/score/router", auth=("operator", token),
                           content=b'{"features":{},"features":{},"normalization_id":"x"}',
                           headers={"Content-Type": "application/json"})
    assert response.status_code == 400
