"""Read-only bounded Ollama inventory probes, not node qualification.

Only operator-configured literal loopback or tailnet IPs are accepted. No DNS,
public URLs, proxy inheritance, redirects, generation, pulls or model deletion.
A tags/ps response establishes API observation only, never inference readiness.
"""
from __future__ import annotations
from .safeio import strict_json
import ipaddress
import re
from urllib.parse import urlsplit
import httpx

_TAILNET = ipaddress.ip_network("100.64.0.0/10")
_MAX_BODY = 1024 * 1024
_NODE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


def endpoint(value: str) -> str:
    parsed = urlsplit(value)
    if (parsed.scheme != "http" or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path not in ("", "/") or parsed.port != 11434):
        raise ValueError("only_private_ollama_http_port_11434_allowed")
    ip = ipaddress.ip_address(parsed.hostname or "")
    if not ip.is_loopback and not (ip.version == 4 and ip in _TAILNET):
        raise ValueError("loopback_or_tailnet_literal_ip_required")
    if ip.version != 4:
        raise ValueError("ipv4_only_in_first_slice")
    return f"http://{ip}:11434"


def node_config(raw: str) -> dict[str, str]:
    if len(raw) > 4096:
        raise ValueError("oversized_node_configuration")
    obj = strict_json(raw)
    if not isinstance(obj, dict) or len(obj) > 8:
        raise ValueError("at_most_eight_configured_nodes")
    for name, url in obj.items():
        if not isinstance(name, str) or not _NODE.fullmatch(name) or not isinstance(url, str):
            raise ValueError("invalid_node_configuration")
    return {name: endpoint(url) for name, url in obj.items()}


async def _body(client: httpx.AsyncClient, url: str) -> dict:
    async with client.stream("GET", url) as response:
        if response.status_code != 200:
            raise ValueError("non_200_inventory_response")
        # Reject compressed responses: a decompression bomb can allocate before our bound.
        if response.headers.get("content-encoding", "identity") not in ("", "identity"):
            raise ValueError("compressed_inventory_rejected")
        buf = bytearray()
        async for chunk in response.aiter_bytes():
            buf.extend(chunk)
            if len(buf) > _MAX_BODY:
                raise ValueError("oversized_inventory_response")
        data = strict_json(buf)
    if not isinstance(data, dict) or not isinstance(data.get("models"), list) or len(data["models"]) > 128:
        raise ValueError("invalid_inventory_shape")
    return data


def _models(data: dict, *, running: bool) -> list[dict]:
    rows = []
    for item in data["models"]:
        if not isinstance(item, dict):
            raise ValueError("invalid_model_entry")
        name, digest = item.get("name"), item.get("digest")
        if not isinstance(name, str) or not 0 < len(name) <= 256 or not isinstance(digest, str) or not re.fullmatch(r"(?:sha256:)?[a-f0-9]{64}", digest):
            raise ValueError("invalid_model_name_or_digest")
        raw_size = item.get("size_vram" if running else "size")
        if raw_size is not None and (type(raw_size) is not int or not 0 <= raw_size <= 10**15):
            raise ValueError("invalid_model_size")
        rows.append({"name": name, "digest": digest, "size_vram_bytes" if running else "size_bytes": raw_size,
                     "identity_evidence": "OLLAMA_REPORTED_NOT_CRYPTOGRAPHICALLY_ATTESTED"})
    return rows


async def inspect_node(name: str, url: str, *, transport: httpx.AsyncBaseTransport | None = None) -> dict:
    # Validate before creating a client: all caller paths obey the same boundary.
    base = endpoint(url)
    if not _NODE.fullmatch(name):
        raise ValueError("invalid_node_name")
    result = {"node_id": name, "state": "CONFIGURED", "ready": False,
              "qualification": "NOT_VERIFIED", "inventory_observed": False,
              "stored_models": [], "running_models": [], "error": None}
    try:
        import asyncio
        async with asyncio.timeout(5):
            async with httpx.AsyncClient(timeout=2.0, follow_redirects=False, trust_env=False,
                                         headers={"Accept-Encoding": "identity"}, transport=transport) as client:
                tags = await _body(client, base + "/api/tags")
                result["stored_models"] = _models(tags, running=False)
                result["inventory_observed"] = True
                result["state"] = "INVENTORY_OBSERVED_NOT_QUALIFIED"
                ps = await _body(client, base + "/api/ps")
                result["running_models"] = _models(ps, running=True)
    except (httpx.HTTPError, ValueError, UnicodeError, TimeoutError):
        # Do not expose URLs, response bodies or provider error strings to the UI.
        result["error"] = "INVENTORY_PROBE_INCOMPLETE_OR_INVALID"
    return result
