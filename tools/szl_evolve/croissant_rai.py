#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Croissant 1.0 + RAI 1.0 metadata generator for local JSONL datasets (LOCAL PROPOSAL; writes only the JSON-LD you ask for).

Facts come from the files (sha256, row count, field names/types). Responsible-AI statements
(collection, biases, sensitive info, limits) are copied ONLY from the metadata you supply, never invented;
anything missing is listed under `missing_rai` and is NOT silently filled.
Specs: https://docs.mlcommons.org/croissant/docs/croissant-spec.html ; https://docs.mlcommons.org/croissant/docs/croissant-rai-spec.html
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, sys
from typing import Any, Dict, List

CONTEXT = {
    "@language": "en", "@vocab": "https://schema.org/", "citeAs": "cr:citeAs", "column": "cr:column",
    "conformsTo": "dct:conformsTo", "cr": "http://mlcommons.org/croissant/", "rai": "http://mlcommons.org/croissant/RAI/",
    "data": {"@id": "cr:data", "@type": "@json"}, "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
    "dct": "http://purl.org/dc/terms/", "examples": {"@id": "cr:examples", "@type": "@json"},
    "extract": "cr:extract", "field": "cr:field", "fileProperty": "cr:fileProperty", "fileObject": "cr:fileObject",
    "fileSet": "cr:fileSet", "format": "cr:format", "includes": "cr:includes", "isLiveDataset": "cr:isLiveDataset",
    "jsonPath": "cr:jsonPath", "key": "cr:key", "md5": "cr:md5", "parentField": "cr:parentField", "path": "cr:path",
    "recordSet": "cr:recordSet", "references": "cr:references", "regex": "cr:regex", "repeated": "cr:repeated",
    "replace": "cr:replace", "sc": "https://schema.org/", "separator": "cr:separator", "source": "cr:source",
    "subField": "cr:subField", "transform": "cr:transform", "equivalentProperty": "cr:equivalentProperty",
    "samplingRate": "cr:samplingRate",
}
RAI_KEYS = ["dataCollection", "dataCollectionType", "dataPreprocessingProtocol", "dataBiases", "dataLimitations",
            "dataUseCases", "dataSocialImpact", "personalSensitiveInformation", "dataReleaseMaintenancePlan"]
HEX40 = re.compile(r"^[0-9a-f]{40}$")


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _dtype(v: Any) -> str:
    if isinstance(v, bool):
        return "sc:Boolean"
    if isinstance(v, int):
        return "sc:Integer"
    if isinstance(v, float):
        return "sc:Float"
    return "sc:Text"


def infer_fields(path: str, max_rows: int = 5000) -> Dict[str, Any]:
    fields: Dict[str, str] = {}
    nested: set = set()
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            n += 1
            if n <= max_rows:
                r = json.loads(line)
                for k, v in r.items():
                    if isinstance(v, (list, dict)):
                        nested.add(k); fields.setdefault(k, "sc:Text")
                    else:
                        fields.setdefault(k, _dtype(v))
    return {"rows": n, "fields": fields, "nested": sorted(nested)}


def build(data_path: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    """meta: name, description, license (URL or SPDX), url, version (semver), revision (HF commit, 40-hex), creator, date_published, rai{...}"""
    fn = os.path.basename(data_path)
    info = infer_fields(data_path)
    fid = f"file-{re.sub(r'[^A-Za-z0-9_.-]', '-', fn)}"
    rs_id = "records"
    fields = []
    for k, t in info["fields"].items():
        fields.append({"@type": "cr:Field", "@id": f"{rs_id}/{k}", "name": f"{rs_id}/{k}", "dataType": t,
                       "source": {"fileObject": {"@id": fid}, "extract": {"jsonPath": f"$.{k}"}},
                       **({"description": "nested JSON serialized as text"} if k in info["nested"] else {})})
    doc: Dict[str, Any] = {
        "@context": CONTEXT, "@type": "sc:Dataset", "name": meta["name"],
        "conformsTo": ["http://mlcommons.org/croissant/1.0", "http://mlcommons.org/croissant/RAI/1.0"],
        "description": meta["description"], "license": meta["license"], "url": meta["url"],
        "version": meta.get("version", "1.0.0"), "datePublished": meta.get("date_published", "1970-01-01"),
        "citeAs": meta.get("cite_as", meta["name"]), "creator": {"@type": "sc:Organization", "name": meta.get("creator", "SZL Holdings")},
        "isLiveDataset": False,
        "distribution": [{"@type": "cr:FileObject", "@id": fid, "name": fn, "contentUrl": meta.get("content_url", fn),
                          "encodingFormat": "application/jsonlines", "sha256": sha256_file(data_path)}],
        "recordSet": [{"@type": "cr:RecordSet", "@id": rs_id, "name": rs_id, "field": fields}],
    }
    rai = meta.get("rai", {}) or {}
    missing = []
    for k in RAI_KEYS:
        if rai.get(k):
            doc[f"rai:{k}"] = rai[k]
        else:
            missing.append(k)
    doc["_szl_provenance"] = {  # non-Croissant, prefixed; mlcroissant ignores unknown underscore keys
        "rows": info["rows"], "revision": meta.get("revision"),
        "revision_is_pinned_40hex": bool(HEX40.match(str(meta.get("revision", "")))),
        "missing_rai": missing, "generator": "szl_evolve.croissant_rai/v1",
        "note": "RAI fields not supplied are omitted, not invented. sha256 proves byte identity only."}
    return doc


def strict_check(doc: Dict[str, Any]) -> List[str]:
    """Local structural checks independent of mlcroissant."""
    errs = []
    for k in ("name", "description", "license", "url", "conformsTo", "distribution", "recordSet"):
        if not doc.get(k):
            errs.append(f"missing {k}")
    for d in doc.get("distribution", []):
        if not re.fullmatch(r"[0-9a-f]{64}", d.get("sha256", "")):
            errs.append("distribution sha256 invalid")
    prov = doc.get("_szl_provenance", {})
    if prov and not prov.get("revision_is_pinned_40hex"):
        errs.append("meta.revision is not a 40-hex commit (UNPINNED)")
    return errs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("data"); ap.add_argument("--meta", required=True, help="JSON file with name, description, license, url, version, rai")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    doc = build(a.data, json.load(open(a.meta)))
    json.dump(doc, open(a.out, "w"), indent=2, ensure_ascii=False)
    errs = strict_check(doc)
    print(json.dumps({"out": a.out, "problems": errs, "missing_rai": doc["_szl_provenance"]["missing_rai"]}, indent=2))
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
