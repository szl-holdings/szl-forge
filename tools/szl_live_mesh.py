#!/usr/bin/env python3
"""Live identity mesh for SZL Holdings.

Probes GitHub, Hugging Face, a-11-oy.com, and a11oy.net.
Does not merge, publish, train, delete Hub assets, or mint DONE.
Exit 0 only when contracted identities that must match actually match
AND no listed HOLD blockers remain. That is not the current estate.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import urllib.error
import urllib.request
from typing import Any

SHA40 = re.compile(r"^[0-9a-f]{40}$")
UA = {"User-Agent": "SZL-Live-Mesh/1.0", "Accept": "application/json"}
PIN_RE = re.compile(r'SOURCE_REVISION\s*=\s*"([0-9a-f]{40})"')

ORIGINS = {
    "github": "https://api.github.com",
    "raw": "https://raw.githubusercontent.com",
    "product": "https://a-11-oy.com",
    "proof": "https://a11oy.net",
    "hf_a11oy": "https://szlholdings-a11oy.hf.space",
    "hf_lyte": "https://szlholdings-lyte.hf.space",
    "hf_api": "https://huggingface.co",
}


class MeshError(RuntimeError):
    pass


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _get(url: str, timeout: float = 12.0, json_ok: bool = True) -> tuple[int, Any]:
    req = urllib.request.Request(url, headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            if json_ok:
                try:
                    return resp.status, json.loads(raw.decode("utf-8"))
                except json.JSONDecodeError:
                    return resp.status, raw.decode("utf-8", "replace")[:2000]
            return resp.status, raw.decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:500]
        try:
            return exc.code, json.loads(body)
        except json.JSONDecodeError:
            return exc.code, {"body_excerpt": body}
    except Exception as exc:
        return 0, {"error": type(exc).__name__, "detail": str(exc)[:240]}


def _sha(payload: Any, *keys: str) -> str | None:
    cur: Any = payload
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    if isinstance(cur, str) and SHA40.fullmatch(cur):
        return cur
    return None


def probe() -> dict[str, Any]:
    a11oy_gh, a11oy_gh_body = _get(f"{ORIGINS['github']}/repos/szl-holdings/a11oy/commits/main")
    lyte_gh, lyte_gh_body = _get(f"{ORIGINS['github']}/repos/szl-holdings/lyte-services/commits/main")
    net_gh, net_gh_body = _get(f"{ORIGINS['github']}/repos/szl-holdings/a11oy-net/commits/main")
    forge_gh, forge_gh_body = _get(f"{ORIGINS['github']}/repos/szl-holdings/szl-forge/commits/main")
    frontier_gh, frontier_gh_body = _get(f"{ORIGINS['github']}/repos/szl-holdings/szl-frontier/commits/main")
    pin_status, pin_body = _get(
        f"{ORIGINS['raw']}/szl-holdings/a11oy/main/scripts/hf_publish_lyte_enterprise.py",
        json_ok=False,
    )
    pin = None
    if isinstance(pin_body, str):
        found = PIN_RE.search(pin_body)
        if found:
            pin = found.group(1)
    product, product_body = _get(f"{ORIGINS['product']}/api/build-info")
    honest, honest_body = _get(f"{ORIGINS['product']}/api/a11oy/v1/honest")
    healthz, healthz_body = _get(f"{ORIGINS['product']}/api/a11oy/healthz")
    surfaces, surfaces_body = _get(f"{ORIGINS['product']}/api/a11oy/v1/frontier/surfaces")
    hf_a11oy, hf_a11oy_body = _get(f"{ORIGINS['hf_a11oy']}/api/build-info")
    lyte_live, lyte_live_body = _get(f"{ORIGINS['hf_lyte']}/api/build-info")
    lyte_alias, lyte_alias_body = _get(f"{ORIGINS['hf_lyte']}/api/lyte/v2/metrics")
    lyte_prom, _ = _get(f"{ORIGINS['hf_lyte']}/metrics", json_ok=False)
    proof, proof_body = _get(f"{ORIGINS['proof']}/health.json")
    a11oy = _sha(a11oy_gh_body, "sha")
    lyte_producer = _sha(lyte_gh_body, "sha")
    net = _sha(net_gh_body, "sha")
    forge = _sha(forge_gh_body, "sha")
    frontier = _sha(frontier_gh_body, "sha")
    product_sha = _sha(product_body, "build", "revision")
    honest_sha = _sha(honest_body, "git_sha")
    hf_a11oy_sha = _sha(hf_a11oy_body, "build", "revision")
    lyte_live_sha = _sha(lyte_live_body, "source_revision") or _sha(lyte_live_body, "build", "revision")
    proof_pub = _sha(proof_body, "sha") if isinstance(proof_body, dict) else None
    blockers: list[str] = []
    if a11oy and product_sha and a11oy != product_sha:
        blockers.append("PRODUCT_RUNTIME_NE_GITHUB")
    if a11oy and hf_a11oy_sha and a11oy != hf_a11oy_sha:
        blockers.append("HF_A11OY_NE_GITHUB")
    if pin and lyte_producer and pin != lyte_producer:
        blockers.append("PUBLISHER_PIN_NE_PRODUCER")
    if pin and lyte_live_sha and pin != lyte_live_sha:
        blockers.append("LYTE_SPACE_NE_PUBLISHER_PIN")
    if lyte_alias != 200:
        blockers.append("LYTE_METRICS_ALIAS_NOT_200")
    if net and proof_pub and net != proof_pub:
        blockers.append("PROOF_PUBLISHED_NE_GITHUB")
    if healthz != 200:
        blockers.append("PRODUCT_HEALTHZ_NOT_200")
    planes = {
        "a11oy_github": a11oy,
        "a11oy_product": product_sha,
        "a11oy_honest": honest_sha,
        "a11oy_hf_space": hf_a11oy_sha,
        "lyte_producer": lyte_producer,
        "lyte_publisher_pin": pin,
        "lyte_live_space": lyte_live_sha,
        "proof_github": net,
        "proof_published": proof_pub,
        "forge_github": forge,
        "frontier_github": frontier,
    }
    p01_close = bool(lyte_producer and pin and lyte_live_sha and lyte_producer == pin == lyte_live_sha and lyte_alias == 200)
    product_aligned = bool(a11oy and product_sha and hf_a11oy_sha and a11oy == product_sha == hf_a11oy_sha)
    return {
        "schema": "szl.live-mesh/v1",
        "observed_at": utc_now(),
        "mutations": False,
        "production_authorization": False,
        "signed_off_done": False,
        "lambda": "Conjecture 1",
        "planes": planes,
        "http": {
            "a11oy_github": a11oy_gh,
            "lyte_github": lyte_gh,
            "a11oy_net_github": net_gh,
            "product_build_info": product,
            "product_honest": honest,
            "product_healthz": healthz,
            "product_surfaces": surfaces,
            "hf_a11oy": hf_a11oy,
            "lyte_live": lyte_live,
            "lyte_metrics_alias": lyte_alias,
            "lyte_prometheus": lyte_prom,
            "proof_health": proof,
            "publisher_pin_blob": pin_status,
        },
        "p01_close": p01_close,
        "product_aligned": product_aligned,
        "blockers": blockers,
        "state": "HOLD" if blockers or not p01_close else "ALIGNED",
        "issue_2010": "OPEN" if not p01_close else "CONVERGED_PENDING_NATIVE_VERIFIER",
        "kernel_hub": {
            "takedown": "2026-09-13",
            "census": "huggingface.co/api/kernels",
            "mirrors_deleted": False,
            "state": "HOLD",
        },
        "note": "Native issue 2010 is not closed by this probe. Proof published SHA is a static Pages document. Evidence cannot grant DONE.",
        "samples": {
            "honest_lambda": (honest_body or {}).get("doctrine_lock", {}).get("lambda") if isinstance(honest_body, dict) else None,
            "lyte_bindings_agree": (lyte_live_body or {}).get("source_binding", {}).get("bindings_agree") if isinstance(lyte_live_body, dict) else None,
            "proof_contract": (proof_body or {}).get("probe_contract") if isinstance(proof_body, dict) else None,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="-")
    args = parser.parse_args(argv)
    report = probe()
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output == "-":
        print(text, end="")
    else:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text)
    return 0 if report["state"] == "ALIGNED" and report["p01_close"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
