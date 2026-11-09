#!/usr/bin/env python3
"""Generate a scoped estate inventory with declared GitHub source mappings,
explicit missing metadata, and NOT_BOUND gaps with a next-inspection owner.
GitHub szl-holdings is the source authority; HF SZLHOLDINGS is the artifact
estate. This crosswalk does not verify source/artifact equivalence,
reproducible builds, qualification, or operational readiness.

Usage: python scripts/generate_estate_manifest.py --output /tmp/estate-review.json
Writes a local review artifact, not a publication or qualification result.
Authenticated results can include private assets; review visibility before sharing.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import urllib.error
import urllib.parse
import urllib.request

HF_ORG = "SZLHOLDINGS"
GH_ORG = "szl-holdings"
SCHEMA = "szl.estate-alignment-manifest/v2"
KINDS = {"model": "models", "dataset": "datasets", "space": "spaces", "kernel": "kernels"}
REGISTRY = Path(__file__).resolve().parents[1] / "publishing/kernel-hub-ownership.json"

# Known GitHub source bindings (canonical + today's launch-contract waves).
MODEL_SOURCES = {
    "SZL-Forge-1.5B-ReceiptAgent": ("szl-forge", "TRAINED_MEASURED_LIMITED"),
    "szl-receiptagent-qwen35-0.8b-v2": ("szl-forge", "TRAINED_MEASURED_LIMITED"),
    "SZL-Khipu-1.5B": ("szl-forge", "TRAINED_RESEARCH_ONLY"),
    "SZL-Khipu-1.5B-GGUF": ("szl-forge", "QUANTIZED_DERIVATIVE"),
    "SZL-GovernedDistill-0.8B-v1": ("szl-forge/training/w30", "TRAINED_RESEARCH_ONLY"),
    "willay-r1-candidate": ("szl-forge/training/w29", "TRAINED_LAUNCH_CONTRACT"),
    "willay-r2-candidate": ("szl-forge/training/w29", "TRAINED_LAUNCH_CONTRACT"),
    "forge-v3": ("szl-forge/training/w29", "TRAINED_LAUNCH_CONTRACT"),
    "abstain-doctrine-r1": ("szl-forge/training/w29-campaign", "TRAINED_LAUNCH_CONTRACT"),
    "SZL-Nemotron-Mini-4B-Estate-v1": ("szl-forge", "TRAINED_RESEARCH_ONLY"),
    "SZL-Nemotron-Mini-4B-Estate-DPO-v1": ("szl-forge", "TRAINED_RESEARCH_ONLY"),
    "SZL-Nemotron-Mini-4B-Doctrine-v1": ("szl-forge", "TRAINED_RESEARCH_ONLY"),
    "Nemotron-Mini-4B-Instruct-st": (None, "UPSTREAM_MIRROR_LICENSE_OTHER"),
}
KERNEL_NAMES = {"szl-kernels","szl-formulas","szl-govsign","szl-provctl","szl-ouroboros",
    "szl-lambda-gate","governed-inference-meter","szl-energy-attest","szl-block-kv",
    "szl-maskmod","szl-receipt-attn","szl-specdecode","YARQA-ATTN","szl-blocked",
    "szl-governed-norm","szl-invariants","szl-khipu","szl-khipu-kernels","szl-nemo",
    "szl-receiptagent-qwen35-0.8b-v3-authenticated","a11oy-v19-substrate"}
FIXTURE_NAMES = {"TinyKhipu-Nano","MiniEmbed-Nano","Moons-Nano","ReceiptAgent-Nano",
    "KILLINCHU-EYE","chakana","qantu","tinku","waman","SZLHOLDINGS"}
SPACE_SOURCES = {
    "a11oy": "a11oy", "anatomy": "anatomy", "killinchu": "killinchu",
    "szl-second-brain": "szl-second-brain", "szl-frontier": "szl-frontier",
    "curation-atlas": "curation-atlas", "lyte": "lyte-lattice", "nexus": "nexus",
    "immune": "immune", "yarqa": "yarqa", "terra": "terra", "finance": "finance",
    "counsel": "counsel", "david-leads": "david-leads", "ayllu": "ayllu",
    "agents-studio": "agents-studio", "llm-router-live": "szl-router",
    "szl-khipu": "szl-khipu", "szl-command-lab": "szl-command-lab",
    "szl-forge-lab": "szl-forge", "szl-model-inference-lab": "szl-forge",
    "szl-router-control": "szl-router", "the-grid": "the-grid",
    "vertical-services": "vertical-services", "szl-foundation-confirmation": "szl-foundation-confirmation",
    "holographic-unify": "holographic-unify", "gdw-frontier": "gdw-frontier",
}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # API redirects are unavailable evidence, never permission to forward a token.
        return None


def hf(url, token):
    """Read one API page. Never forward credentials to a pagination host."""
    if not url.startswith("/api/") or url.startswith("//"):
        raise ValueError("expected a relative Hub API path")
    req = urllib.request.Request("https://huggingface.co" + url,
        headers={"Authorization": f"Bearer {token}"} if token else {})
    opener = urllib.request.build_opener(NoRedirect())
    with opener.open(req, timeout=60) as response:
        return json.load(response), ", ".join(response.headers.get_all("Link", []))


def next_page(link, api):
    if not link:
        return None
    target = None
    for part in re.split(r",(?=\s*<)", link):
        match = re.fullmatch(r"\s*<([^<>]+)>\s*(.*)", part)
        if not match:
            raise ValueError("malformed pagination Link header")
        remaining, params = match.group(2), {}
        while remaining:
            param = re.match(r';\s*([A-Za-z][A-Za-z0-9_-]*)\s*=\s*(?:"([^"\\]*)"|([^;,\s]+))\s*', remaining)
            if not param or param.group(1).lower() in params:
                raise ValueError("malformed or duplicate pagination parameter")
            params[param.group(1).lower()] = param.group(2) if param.group(2) is not None else param.group(3)
            remaining = remaining[param.end():]
        if "rel" not in params:
            raise ValueError("pagination Link is missing rel")
        if "next" in params["rel"].lower().split():
            if target is not None:
                raise ValueError("multiple next pages")
            parsed = urllib.parse.urlsplit(match.group(1))
            if (parsed.scheme not in ("", "https")
                    or parsed.netloc not in ("", "huggingface.co")
                    or parsed.path != f"/api/{api}" or parsed.fragment):
                raise ValueError("untrusted pagination target")
            query = urllib.parse.parse_qs(parsed.query)
            if query.get("author") != [HF_ORG] or query.get("full") != ["true"]:
                raise ValueError("pagination lost organization or full metadata filter")
            target = parsed.path + "?" + parsed.query
    return target


def collect_inventory(token, fetch=hf):
    """Record collection failures separately from an observed empty namespace."""
    inventory, coverage = {}, {}
    for kind, api in KINDS.items():
        items, visited, seen = [], set(), set()
        url = f"/api/{api}?author={HF_ORG}&limit=100&full=true"
        state = {"state": "COMPLETE", "pages": 0}
        try:
            while url:
                if url in visited:
                    raise ValueError("pagination loop")
                visited.add(url)
                page, link = fetch(url, token)
                if not isinstance(page, list):
                    raise ValueError("inventory page is not a list")
                for item in page:
                    if (not isinstance(item, dict) or not isinstance(item.get("id"), str)
                            or not re.fullmatch(r"SZLHOLDINGS/[A-Za-z0-9_.-]+", item["id"])):
                        raise ValueError("malformed inventory asset id")
                    if item["id"] in seen:
                        raise ValueError("duplicate asset in namespace")
                    seen.add(item["id"])
                    items.append(item)
                state["pages"] += 1
                url = next_page(link, api)
        except (OSError, ValueError) as error:
            # Do not serialize exception text, URLs or tokens into a review artifact.
            state["state"] = "PARTIAL" if items else "UNAVAILABLE"
            state["error_type"] = type(error).__name__
            if isinstance(error, urllib.error.HTTPError):
                state["http_status"] = error.code
        state["observed_rows"] = len(items)
        inventory[kind], coverage[kind] = items, state
    return inventory, coverage


def listed_files(item, kind):
    key = "files" if kind == "kernel" else "siblings"
    raw = item.get(key)
    if not isinstance(raw, list):
        return None
    if kind == "kernel":
        return raw if all(isinstance(name, str) and name for name in raw) else None
    if not all(isinstance(entry, dict) and isinstance(entry.get("rfilename"), str)
               and entry["rfilename"] for entry in raw):
        return None
    return [entry["rfilename"] for entry in raw]


def classify_model(name, card, siblings, kernel_sources=None):
    """Keep declared labels separate from file-presence observations."""
    base = name.split("/")[-1]
    if base in KERNEL_NAMES:
        return "SOFTWARE_OR_SPEC_MIRROR", (kernel_sources or {}).get(f"{HF_ORG}/{base}")
    if base in FIXTURE_NAMES:
        return "SYNTHETIC_FIXTURE_OR_PLACEHOLDER", None
    if base in MODEL_SOURCES:
        src, cls = MODEL_SOURCES[base]
        return cls, f"{GH_ORG}/{src}" if src else None
    files = listed_files({"siblings": siblings}, "model")
    if files is None:
        return "UNAVAILABLE_FILE_METADATA", None
    if any(name.lower().endswith((".safetensors", ".bin", ".gguf")) for name in files):
        return "WEIGHTS_LISTED_UNBOUND_REQUIRES_BINDING", None
    return "NO_RECOGNIZED_WEIGHT_FILES_LISTED", None


def load_kernel_sources(path=REGISTRY):
    """Use the existing declared owner registry, not name-based repo guesses."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "szl.kernel-hub-ownership/v1":
        raise ValueError("unrecognized kernel ownership registry")
    sources = {}
    for target in payload["targets"]:
        asset, owner = target["repo_id"], target["owner"]
        if (not re.fullmatch(r"SZLHOLDINGS/[A-Za-z0-9_.-]+", asset)
                or not re.fullmatch(r"szl-holdings/[A-Za-z0-9_.-]+", owner)
                or asset in sources):
            raise ValueError("invalid or duplicate kernel ownership declaration")
        sources[asset] = owner
    return sources


def build_row(kind, item, kernel_sources):
    asset = item["id"]
    rid = asset.split("/")[-1]
    revision = item.get("sha")
    revision = revision if isinstance(revision, str) and re.fullmatch(r"[0-9a-fA-F]{40}", revision) else None
    files = listed_files(item, kind)
    visibility = item.get("private")
    visibility = visibility if isinstance(visibility, bool) else None
    missing = []
    if revision is None:
        missing.append("sha")
    if visibility is None:
        missing.append("private")
    if kind in ("model", "kernel") and files is None:
        missing.append("files" if kind == "kernel" else "siblings")
    row = {"asset": asset, "type": kind, "hub_revision": revision,
           "revision_state": "PINNED" if revision else "UNAVAILABLE",
           "private": visibility, "last_modified": item.get("lastModified"),
           "metadata_state": "AVAILABLE" if not missing else "UNAVAILABLE_REQUIRED_FIELDS",
           "unavailable_fields": missing, "github_source": None}
    if kind == "model":
        cls, src = classify_model(asset, item.get("cardData"), item.get("siblings"), kernel_sources)
        row["classification"], row["github_source"] = cls, src
        row["classification_basis"] = (
            "REPOSITORY_DECLARATION_NOT_REVERIFIED"
            if rid in KERNEL_NAMES | FIXTURE_NAMES or rid in MODEL_SOURCES
            else "LISTED_FILENAMES_ONLY" if files is not None else "UNAVAILABLE_FILE_METADATA")
        row["weight_files_state"] = "UNAVAILABLE" if files is None else (
            "PRESENT" if any(name.lower().endswith((".safetensors", ".bin", ".gguf")) for name in files)
            else "ABSENT_FROM_LISTING")
    elif kind == "space":
        row["classification"] = "SPACE"
        if rid in SPACE_SOURCES:
            row["github_source"] = f"{GH_ORG}/{SPACE_SOURCES[rid]}"
    elif kind == "kernel":
        row["classification"] = "KERNEL"
        row["github_source"] = kernel_sources.get(asset)
    else:
        row["classification"] = "DATASET"
    row["binding_state"] = "BOUND_DECLARED_SOURCE" if row["github_source"] else "NOT_BOUND_SOURCE_REPO_FIELD_MISSING"
    row["binding_evidence"] = (
        "publishing/kernel-hub-ownership.json"
        if (kind == "kernel" or (kind == "model" and rid in KERNEL_NAMES)) and row["github_source"]
        else "scripts/generate_estate_manifest.py" if row["github_source"] else None)
    if not row["github_source"]:
        row["next_inspection_owner"] = GH_ORG
        row["next_inspection"] = "identify the source owner and exact source/evidence path; do not infer from asset name"
    return row


def build_manifest(inventory, coverage, kernel_sources, authenticated=False):
    rows = [build_row(kind, item, kernel_sources)
            for kind in KINDS for item in inventory.get(kind, [])]
    identities = [(row["type"], row["asset"]) for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate typed asset identity")
    rows.sort(key=lambda row: (row["type"], row["asset"]))
    unavailable = sum(row["metadata_state"] != "AVAILABLE" for row in rows)
    complete = all(coverage.get(kind, {}).get("state") == "COMPLETE" for kind in KINDS)
    return {
        "schema": SCHEMA,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_authority": "github.com/szl-holdings",
        "artifact_estate": "huggingface.co/SZLHOLDINGS",
        "collection_scope": "TOKEN_VISIBLE" if authenticated else "ANONYMOUS_PUBLIC",
        "collection": coverage,
        "inventory_state": "COMPLETE_OBSERVED_SCOPE" if complete else "INCOMPLETE",
        "policy": {
            "artifact_equivalence": "NOT_CLAIMED", "reproducible_build": "NOT_CLAIMED",
            "promotion_requires": "evaluation + reproducibility + licensing + runtime gates",
            "not_bound_means": "explicit gap with owner and next inspection - never silently zero",
            "binding_means": "declared source mapping only; source revision and artifact equivalence not verified",
            "weight_files_mean": "recognized filenames only; no claim of training, payload integrity or quality",
            "qualification": "NOT_REASSESSED; existing declared labels are not new measurement or approval",
            "visibility": "review private and unknown-visibility rows before sharing or publication",
        },
        "counts": {
            "total": len(rows),
            "by_type": {kind: sum(row["type"] == kind for row in rows) for kind in KINDS},
            "not_bound": sum(row["binding_state"].startswith("NOT_BOUND") for row in rows),
            "bound": sum(row["binding_state"].startswith("BOUND") for row in rows),
            "metadata_unavailable": unavailable,
            "revision_unavailable": sum(row["hub_revision"] is None for row in rows),
        },
        "rows": rows,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--token-env", default="HF_TOKEN")
    ap.add_argument("--output", required=True, help="new local review file; never overwritten")
    args = ap.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        ap.error("output already exists; choose a new local review file")
    token = os.environ.get(args.token_env, "")
    sources = load_kernel_sources()
    inventory, coverage = collect_inventory(token)
    manifest = build_manifest(inventory, coverage, sources, authenticated=bool(token))
    with output.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=1, sort_keys=True)
        handle.write("\n")
    print(json.dumps(manifest["counts"], indent=1))
    print(f"local review manifest -> {output}")
    return 0 if manifest["inventory_state"] == "COMPLETE_OBSERVED_SCOPE" and not manifest["counts"]["metadata_unavailable"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
