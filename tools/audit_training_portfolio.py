#!/usr/bin/env python3
"""Public metadata inventory and research queue; never a training authorizer.

Read-only HTTPS, no credentials, weight/data downloads, model imports or launch.
A repository license tag is a declaration, not source-level rights clearance.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import parse_qs, urlencode, urlsplit
from urllib.request import Request, build_opener, HTTPRedirectHandler
from concurrent.futures import ThreadPoolExecutor

KINDS = ("models", "datasets", "spaces")
SHA = re.compile(r"[0-9a-f]{40}")
OWNER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
MAX_BYTES = 8_000_000

def utc():
    return datetime.now(timezone.utc).isoformat()

def validate_url(url, kind=None, owner=None):
    u = urlsplit(url)
    if u.scheme != "https" or u.netloc != "huggingface.co" or u.fragment:
        raise ValueError("Only credential-free HTTPS huggingface.co URLs are allowed")
    if kind:
        if u.path != "/api/" + kind or parse_qs(u.query).get("author") != [owner]:
            raise ValueError("Pagination left the selected owner/type scope")
    return url

class SameOrigin(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)

def get(url):
    validate_url(url)
    request = Request(url, headers={"User-Agent": "SZL-Public-Training-Audit/1", "Cache-Control": "no-cache"})
    with build_opener(SameOrigin()).open(request, timeout=25) as response:
        raw = response.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise ValueError("Metadata response exceeds byte bound")
        return raw, dict(response.headers.items())

def header(headers, name):
    return next((v for k, v in headers.items() if k.lower() == name.lower()), None)

def next_link(value):
    if not value:
        return None
    candidates = []
    for item in value.split(","):
        match = re.fullmatch(r'\s*<([^>]+)>\s*;\s*rel=["\x27]?next["\x27]?\s*', item)
        if match:
            candidates.append(match.group(1))
        elif re.search(r'\bnext\b', item):
            raise ValueError("Malformed pagination continuation")
    if len(candidates) > 1:
        raise ValueError("Ambiguous pagination continuation")
    return candidates[0] if candidates else None

def discover(owner, kind, fetch=get, max_pages=25):
    if not OWNER.fullmatch(owner) or kind not in KINDS:
        raise ValueError("Invalid owner/type")
    url = "https://huggingface.co/api/" + kind + "?" + urlencode({"author": owner, "limit": 100, "full": "true"})
    seen_urls, seen_ids, rows, pages = set(), set(), [], []
    started = utc()
    while url:
        validate_url(url, kind, owner)
        if url in seen_urls or len(pages) >= max_pages:
            raise ValueError("Repeated pagination URL or page bound exceeded")
        seen_urls.add(url)
        raw, headers = fetch(url)
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError("Repository list must be an array")
        for row in data:
            if not isinstance(row, dict):
                raise ValueError("Invalid repository entry")
            rid = row.get("id", "")
            if (not isinstance(rid, str) or rid.count("/") != 1
                    or rid.split("/")[0].lower() != owner.lower()
                    or not OWNER.fullmatch(rid.split("/")[1])):
                raise ValueError("Repository is outside selected owner")
            if rid.lower() in seen_ids:
                raise ValueError("Duplicate repository across pages")
            if not isinstance(row.get("sha"), str) or not SHA.fullmatch(row["sha"]):
                raise ValueError("Repository lacks immutable revision")
            if row.get("private") is not False:
                raise ValueError("Public collection returned non-public or ambiguous visibility")
            seen_ids.add(rid.lower())
            rows.append(row)
        continuation = next_link(header(headers, "link"))
        if continuation:
            validate_url(continuation, kind, owner)
        pages.append({"url": url, "at": utc(), "sha256": hashlib.sha256(raw).hexdigest(),
                      "count": len(data), "next": continuation})
        url = continuation
    return {"owner": owner, "kind": kind, "started_at": started, "finished_at": utc(),
            "scope": "PUBLIC_API_VISIBILITY_ONLY", "pagination_complete": True,
            "pages": pages, "repositories": sorted(rows, key=lambda r: r["id"])}

def read_card(kind, row, fetch=get):
    prefix = {"models": "", "datasets": "datasets/", "spaces": "spaces/"}[kind]
    url = "https://huggingface.co/" + prefix + row["id"] + "/resolve/" + row["sha"] + "/README.md"
    try:
        raw, headers = fetch(url)
        commit = header(headers, "x-repo-commit")
        if commit != row["sha"]:
            raise ValueError("Immutable card revision readback unavailable or mismatched")
        return {"kind": kind, "repo": row["id"], "revision": row["sha"], "url": url, "status": "READ_BACK",
                "sha256": hashlib.sha256(raw).hexdigest(), "text": raw.decode("utf-8")}
    except Exception as exc:
        # Status/type only: no exception URL/headers or authentication material.
        return {"kind": kind, "repo": row["id"], "revision": row["sha"], "url": url, "status": "UNAVAILABLE",
                "error_type": type(exc).__name__}

def classify(kind, row):
    files = [x.get("rfilename", "") for x in row.get("siblings", []) or []]
    tags = row.get("tags", []) or []
    if kind == "spaces":
        return "APPLICATION_REVIEW"
    if kind == "datasets":
        return "SOURCE_AND_SPLIT_REVIEW"
    if row.get("library_name") == "kernels" or "kernel" in tags or "kernels" in tags:
        return "SOFTWARE_CONFORMANCE"
    if any(x.endswith(".gguf") for x in files):
        return "DERIVATIVE_EVALUATION"
    if any(x.endswith("adapter_model.safetensors") for x in files):
        return "PEFT_EVALUATION_BEFORE_RETRAINING"
    if any(x.endswith(".safetensors") for x in files):
        return "WEIGHT_LINEAGE_AND_BASELINE_REVIEW"
    if row.get("library_name") == "numpy":
        return "CPU_BASELINE_REVIEW"
    return "RECIPE_OR_OTHER_ARTIFACT_REVIEW"

def reconcile(first, second, cards):
    if {(x["owner"], x["kind"]) for x in first} != {(x["owner"], x["kind"]) for x in second}:
        raise ValueError("Readback coverage differs")
    other = {(x["owner"], x["kind"]): x for x in second}
    # Model, dataset and Space repositories occupy distinct Hub namespaces.
    # Older untyped card records cannot establish namespace-specific readback.
    card_map = {(c.get("kind"), c["repo"], c["revision"]): c for c in cards}
    rows = []
    for bucket in first:
        if bucket.get("pagination_complete") is not True:
            raise ValueError("Incomplete first capture")
        later = other[bucket["owner"], bucket["kind"]]
        if later.get("pagination_complete") is not True:
            raise ValueError("Incomplete readback")
        later_rows = {r["id"]: r for r in later["repositories"]}
        earlier_ids = {r["id"] for r in bucket["repositories"]}
        coverage_stable = earlier_ids == set(later_rows)
        for row in bucket["repositories"]:
            stable = coverage_stable and later_rows.get(row["id"], {}).get("sha") == row["sha"]
            card = card_map.get((bucket["kind"], row["id"], row["sha"]), {})
            badges = ["PUBLIC_REVISION_REOBSERVED"] if stable else []
            if stable and card.get("status") == "READ_BACK":
                badges.append("CARD_BYTES_READ_BACK")
            rows.append({"repo": row["id"], "kind": bucket["kind"], "revision": row["sha"],
                         "public_inventory_stable": stable, "badges": badges,
                         "lane": classify(bucket["kind"], row), "gated": row.get("gated"),
                         "declared_licenses": [t[8:] for t in row.get("tags", []) if t.startswith("license:")],
                         "card_status": card.get("status", "NOT_READ"),
                         "training_authorized": False, "promotion_authorized": False,
                         "rights_verified": False, "split_qualified": False,
                         "independent_evaluator_verified": False})
        for rid in sorted(set(later_rows) - earlier_ids):
            row = later_rows[rid]
            rows.append({"repo": rid, "kind": bucket["kind"], "revision": row["sha"],
                         "public_inventory_stable": False, "badges": [], "lane": "RECAPTURE_REQUIRED",
                         "training_authorized": False, "promotion_authorized": False,
                         "rights_verified": False, "split_qualified": False,
                         "independent_evaluator_verified": False})
    return {"schema": "szl.public-training-portfolio/v1", "observed_at": utc(),
            "scope": "PUBLIC_METADATA_AND_CARDS_NOT_MODEL_QUALIFICATION",
            "readback_independence": "Separate HTTP observations, same executor; not independent review",
            "training_authorized": False, "promotion_authorized": False,
            "rows": rows, "counts_by_kind": dict(Counter(r["kind"] for r in rows)),
            "counts_by_lane": dict(Counter(r["lane"] for r in rows))}

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--owner", action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--live", action="store_true", required=True, help="Explicit public metadata network reads")
    args = p.parse_args(argv)
    if len(args.owner) != len(set(args.owner)) or len(args.owner) > 10:
        p.error("Use at most ten distinct owners")
    args.output.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        (args.output / name).write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    save("execution.json", {"source_sha256": source_hash, "started_at": utc(), "owners": args.owner})
    keys = [(owner, kind) for owner in args.owner for kind in KINDS]
    first = []
    for owner, kind in keys:
        try:
            first.append(discover(owner, kind))
        except Exception as exc:
            save("capture-incomplete.json", {"owner": owner, "kind": kind, "error_type": type(exc).__name__,
                                             "training_authorized": False, "captured": first})
            return 2
    save("first-capture.json", first)
    requests = [(b["kind"], r) for b in first for r in b["repositories"]]
    with ThreadPoolExecutor(max_workers=4) as pool:
        cards = list(pool.map(lambda kr: read_card(*kr), requests))
    save("cards.json", cards)
    second = []
    for owner, kind in keys:
        try:
            second.append(discover(owner, kind))
        except Exception as exc:
            save("readback-incomplete.json", {"owner": owner, "kind": kind, "error_type": type(exc).__name__,
                                              "training_authorized": False, "captured": second})
            return 2
    save("second-capture.json", second)
    report = reconcile(first, second, cards)
    save("portfolio.json", report)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.iterdir() if p.is_file()}
    save("SHA256.json", hashes)
    print(json.dumps({"output": str(args.output), "counts": report["counts_by_kind"],
                      "lanes": report["counts_by_lane"], "cards": dict(Counter(c["status"] for c in cards)),
                      "source_sha256": source_hash}, ensure_ascii=True))
    return 0 if all(r["public_inventory_stable"] for r in report["rows"]) else 2

if __name__ == "__main__":
    raise SystemExit(main())

