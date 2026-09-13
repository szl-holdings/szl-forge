import asyncio
import json
import pytest
import httpx
from szl_model_lab.probes import endpoint, inspect_node, node_config

@pytest.mark.parametrize("url", ["http://127.0.0.1:11434", "http://100.80.1.2:11434/"])
def test_safe_endpoints(url):
    assert endpoint(url).startswith("http://")

@pytest.mark.parametrize("url", ["http://1.1.1.1:11434", "http://169.254.169.254:11434", "http://192.168.1.2:11434",
    "http://evil.example:11434", "http://127.0.0.1:11434@evil.example", "http://127.0.0.1:11434/api/generate",
    "http://127.0.0.1:11434?token=secret", "http://127.0.0.1:11434#fragment", "https://127.0.0.1:11434",
    "http://user:password@127.0.0.1:11434", "http://127.0.0.1:22", "file:///etc/passwd"])
def test_unsafe_endpoints(url):
    with pytest.raises(ValueError):
        endpoint(url)

def test_node_count_bounded():
    with pytest.raises(ValueError):
        node_config(json.dumps({f"node-{n}": "http://127.0.0.1:11434" for n in range(9)}))

def test_no_nodes_is_not_an_invented_fleet():
    assert node_config("{}") == {}

def test_probe_never_generates_or_marks_ready():
    paths = []
    def handler(request):
        paths.append(request.url.path)
        assert request.method == "GET"
        assert "authorization" not in request.headers
        return httpx.Response(200, json={"models": [{"name": "test-model", "digest": "a"*64}]})
    result = asyncio.run(inspect_node("laptop", "http://127.0.0.1:11434", transport=httpx.MockTransport(handler)))
    assert paths == ["/api/tags", "/api/ps"]
    assert result["ready"] is False and result["qualification"] == "NOT_VERIFIED"
    assert result["running_models"][0]["size_vram_bytes"] is None

def test_redirect_not_followed():
    calls = []
    def handler(request):
        calls.append(request.url)
        return httpx.Response(302, headers={"Location": "http://169.254.169.254/"})
    r = asyncio.run(inspect_node("laptop", "http://127.0.0.1:11434", transport=httpx.MockTransport(handler)))
    assert len(calls) == 1 and r["error"] and not r["ready"]

@pytest.mark.parametrize("response", [
    httpx.Response(200, json={"models": "bad"}),
    httpx.Response(200, json={"models": [{"name": "x", "digest": "unknown"}]}),
    httpx.Response(200, content=b"x"*(1024*1024+1)),
    httpx.Response(503, text="sensitive upstream detail")])
def test_bad_responses_are_bounded_and_sanitized(response):
    result = asyncio.run(inspect_node("laptop", "http://127.0.0.1:11434", transport=httpx.MockTransport(lambda r: response)))
    assert result["error"] == "INVENTORY_PROBE_INCOMPLETE_OR_INVALID"
    assert "sensitive" not in json.dumps(result)
    assert result["ready"] is False

def test_partial_probe_not_promoted():
    def handler(request):
        return httpx.Response(200, json={"models": []}) if request.url.path == "/api/tags" else httpx.Response(500)
    result = asyncio.run(inspect_node("laptop", "http://127.0.0.1:11434", transport=httpx.MockTransport(handler)))
    assert result["inventory_observed"] and result["error"] and not result["ready"]
