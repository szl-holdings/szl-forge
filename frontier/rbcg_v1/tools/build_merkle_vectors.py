#!/usr/bin/env python3
"""build_merkle_vectors.py -- build tests/fixtures/merkle_vectors.json from the transparency-dev/merkle
test data at a PINNED commit (anonymous codeload tarball; sha256 checked). Only DATA is extracted:
inclusion/consistency probe JSON files and the LeafInputs/RootHashes constants. No code is copied.
Upstream licence: Apache-2.0 (LICENSE sha256 recorded). Usage:
  python3 build_merkle_vectors.py --tarball merkle.tgz --out tests/fixtures/merkle_vectors.json
  (or --fetch to download https://codeload.github.com/transparency-dev/merkle/tar.gz/<COMMIT>)"""
import argparse
import base64
import hashlib
import io
import json
import re
import tarfile
import urllib.request

COMMIT = "c39fbf350a405f68571bf4c42a6a518eb0a46a4b"
TARBALL_SHA256 = "ed0c1017ea66fc7e88e67998f7d6b9cc1eef66d30b97b58a1dbfc1aa3a1edd09"
URL = "https://codeload.github.com/transparency-dev/merkle/tar.gz/" + COMMIT


def b64hex(v):
    if v is None:
        return None
    return base64.b64decode(v).hex()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tarball")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.fetch:
        with urllib.request.urlopen(URL, timeout=60) as r:
            data = r.read()
    else:
        with open(a.tarball, "rb") as fh:
            data = fh.read()
    sha = hashlib.sha256(data).hexdigest()
    if sha != TARBALL_SHA256:
        raise SystemExit(f"REFUSED: tarball sha256 {sha} != pinned {TARBALL_SHA256}")
    tf = tarfile.open(fileobj=io.BytesIO(data), mode="r:gz")
    root = f"merkle-{COMMIT}/"
    files, incl, cons = {}, [], []
    lic = tf.extractfile(root + "LICENSE").read()
    const = tf.extractfile(root + "testonly/constants.go").read()
    files["LICENSE"] = hashlib.sha256(lic).hexdigest()
    files["testonly/constants.go"] = hashlib.sha256(const).hexdigest()
    src = const.decode("utf-8")
    leaf_block = src[src.index("func LeafInputs()"):src.index("func NodeHashes()")]
    root_block = src[src.index("func RootHashes()"):src.index("func CompactTrees()")]
    leaves = re.findall(r'hd\("([0-9a-f]*)"\)', leaf_block)
    roots = re.findall(r'hd\("([0-9a-f]{64})"\)', root_block)
    empty = re.search(r'func EmptyRootHash\(\)[^{]*\{\s*return hd\("([0-9a-f]{64})"\)', src).group(1)
    if "EmptyRootHash()," not in root_block:
        raise SystemExit("REFUSED: RootHashes layout changed")
    roots = [empty] + roots
    for m in tf.getmembers():
        n = m.name[len(root):] if m.name.startswith(root) else None
        if not n or not m.isfile() or not n.endswith(".json"):
            continue
        kind = "inclusion" if n.startswith("testdata/inclusion/") else (
            "consistency" if n.startswith("testdata/consistency/") else None)
        if kind is None:
            continue
        raw = tf.extractfile(m).read()
        files[n] = hashlib.sha256(raw).hexdigest()
        p = json.loads(raw)
        if kind == "inclusion":
            incl.append({"file": n, "leaf_index": p["leafIdx"], "tree_size": p["treeSize"], "root": b64hex(p.get("root")),
                         "leaf_hash": b64hex(p.get("leafHash")), "proof": [b64hex(x) for x in (p.get("proof") or [])],
                         "desc": p.get("desc", ""), "want_error": bool(p.get("wantErr", False))})
        else:
            cons.append({"file": n, "size1": p["size1"], "size2": p["size2"], "root1": b64hex(p.get("root1")),
                         "root2": b64hex(p.get("root2")), "proof": [b64hex(x) for x in (p.get("proof") or [])],
                         "desc": p.get("desc", ""), "want_error": bool(p.get("wantErr", False))})
    out = {"source": "https://github.com/transparency-dev/merkle", "commit": COMMIT, "tarball_url": URL,
           "tarball_sha256": sha, "licence": "Apache-2.0 (upstream LICENSE file, sha256 below)",
           "note": "test DATA only, parsed by tools/build_merkle_vectors.py; no upstream code copied",
           "files_sha256": files, "leaf_inputs_hex": leaves, "root_hashes_hex_by_size": roots,
           "inclusion": sorted(incl, key=lambda d: d["file"]), "consistency": sorted(cons, key=lambda d: d["file"])}
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print(json.dumps({"inclusion": len(incl), "consistency": len(cons), "leaves": len(leaves), "roots": len(roots),
                      "tarball_sha256": sha}))


if __name__ == "__main__":
    main()
