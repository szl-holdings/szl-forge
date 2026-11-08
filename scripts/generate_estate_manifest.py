#!/usr/bin/env python3
"""Generate the estate alignment manifest: one row per HF asset, bound to its
GitHub source where known, explicitly NOT_BOUND with a next-inspection owner
where not. GitHub szl-holdings is the source authority; HF SZLHOLDINGS is the
artifact estate. This manifest is the machine-readable crosswalk that makes
'fully aligned' a checkable claim.

Usage: python scripts/generate_estate_manifest.py [--token-env HF_TOKEN]
Emits: publishing/estate-alignment-manifest.json
"""
from __future__ import annotations
import argparse, datetime as dt, json, os, urllib.request

HF_ORG = "SZLHOLDINGS"
GH_ORG = "szl-holdings"
SCHEMA = "szl.estate-alignment-manifest/v1"

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

def hf(url, token):
    req = urllib.request.Request("https://huggingface.co"+url,
        headers={"Authorization": f"Bearer {token}"} if token else {})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)

def classify_model(name, card, siblings):
    base = name.split("/")[-1]
    if base in KERNEL_NAMES: return "SOFTWARE_OR_SPEC_MIRROR", "szl-"+base if base!="szl-khipu" else "szl-khipu"
    if base in FIXTURE_NAMES: return "SYNTHETIC_FIXTURE_OR_PLACEHOLDER", None
    if base in MODEL_SOURCES:
        src, cls = MODEL_SOURCES[base]
        return cls, src
    has_weights = any(s.get("rfilename","").endswith((".safetensors",".bin",".gguf")) for s in siblings)
    if has_weights:
        return "TRAINED_UNBOUND_REQUIRES_BINDING", None
    return "WEIGHTLESS_OTHER", None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token-env", default="HF_TOKEN")
    args = ap.parse_args()
    token = os.environ.get(args.token_env, "")
    rows = []
    for kind, api in (("model","models"),("dataset","datasets"),("space","spaces")):
        items = hf(f"/api/{api}?author={HF_ORG}&limit=1000", token)
        for it in items:
            rid = it["id"].split("/")[-1]
            row = {"asset": f"{HF_ORG}/{rid}", "type": kind, "hub_revision": it.get("sha"),
                   "private": it.get("private", False), "last_modified": it.get("lastModified","")}
            if kind == "model":
                cls, src = classify_model(rid, it.get("cardData") or {}, it.get("siblings") or [])
                row["classification"], row["github_source"] = cls, src
                if src is None and cls.startswith("TRAINED"):
                    row["binding_state"] = "NOT_BOUND"
                    row["next_inspection"] = "backfill wave record: config+admission+receipt in szl-forge/training/"
            elif kind == "dataset":
                row["classification"] = "DATASET"
                row["binding_state"] = "NOT_BOUND" if rid not in (
                    "doctrine-corpus-v2","szl-1-doctrine-sft","free-energy-reasoning-sft-v1",
                    "abstention-doctrine-sft-v1","szl-frontier-sft-mix-v2","rag-corpus-v1",
                    "thesis-corpus-v18","k-verify-benchmark-v1","canonical-formulas-v1",
                    "model-eval-receipts","a11oy-verifiable-corpus","szl-lake") else "BOUND_GENERATOR_KNOWN"
            else:
                row["classification"] = "SPACE"
                row["github_source"] = GH_ORG + "/" + SPACE_SOURCES.get(rid, "") if SPACE_SOURCES.get(rid) else None
                row["binding_state"] = "BOUND" if SPACE_SOURCES.get(rid) else "NOT_BOUND_SOURCE_REPO_FIELD_MISSING"
            rows.append(row)
    manifest = {
        "schema": SCHEMA,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_authority": "github.com/szl-holdings",
        "artifact_estate": "huggingface.co/SZLHOLDINGS",
        "policy": {
            "artifact_equivalence": "NOT_CLAIMED", "reproducible_build": "NOT_CLAIMED",
            "promotion_requires": "evaluation + reproducibility + licensing + runtime gates",
            "not_bound_means": "explicit gap with owner and next inspection - never silently zero"
        },
        "counts": {
            "total": len(rows),
            "by_type": {t: sum(1 for r in rows if r["type"]==t) for t in ("model","dataset","space")},
            "not_bound": sum(1 for r in rows if r.get("binding_state")=="NOT_BOUND"),
            "bound": sum(1 for r in rows if str(r.get("binding_state","")).startswith("BOUND") or r.get("github_source")),
        },
        "rows": rows,
    }
    out = "publishing/estate-alignment-manifest.json"
    with open(out, "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    print(json.dumps(manifest["counts"], indent=1))
    print(f"manifest -> {out}")

if __name__ == "__main__":
    main()
