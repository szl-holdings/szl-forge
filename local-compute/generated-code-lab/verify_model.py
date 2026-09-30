"""Verify downloaded Ollama bytes and create an immutable local baseline pin.
SPDX-License-Identifier: Apache-2.0
"""
import hashlib
import json
from pathlib import Path
import re
from datetime import datetime, timezone

from lab import MODEL, ROOT, api, save

EXPECTED_MANIFEST = "0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0"


def main():
    store = Path.home() / ".ollama/models"
    manifest_path = store / "manifests/registry.ollama.ai/library/qwen3/4b-instruct"
    raw = manifest_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED_MANIFEST:
        raise RuntimeError("Manifest changed from the explicitly selected baseline")
    manifest = json.loads(raw)
    verified = []
    for layer in [manifest["config"], *manifest["layers"]]:
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", layer["digest"]):
            raise ValueError("Invalid blob digest")
        path = store / "blobs" / layer["digest"].replace(":", "-")
        if path.stat().st_size != layer["size"]:
            raise RuntimeError("Blob size mismatch")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != layer["digest"].split(":")[1]:
            raise RuntimeError("Blob SHA256 mismatch")
        verified.append({"digest": layer["digest"], "bytes": layer["size"], "media_type": layer["mediaType"]})
        print(json.dumps({"verified": layer["digest"], "bytes": layer["size"]}), flush=True)
    show = api("/api/show", {"model": MODEL}, timeout=15)
    if "Apache License" not in show.get("license", ""):
        raise RuntimeError("Downloaded license is not the selected Apache license")
    pin = {"model": MODEL, "manifest_sha256": EXPECTED_MANIFEST,
        "upstream": "Qwen/Qwen3-4B-Instruct-2507", "license": "Apache-2.0",
        "upstream_model_card": "https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507",
        "distribution": "https://ollama.com/library/qwen3:4b-instruct",
        "verified_at": datetime.now(timezone.utc).isoformat(), "verified_blobs": verified,
        "details": show.get("details"), "license_sha256": hashlib.sha256(show["license"].encode()).hexdigest(),
        "training_performed": False, "note": "Existing upstream weights; not an SZL fine-tune or novel architecture."}
    dest = ROOT / "model.lock.json"
    if dest.exists():
        old = json.loads(dest.read_bytes())
        if old["manifest_sha256"] != EXPECTED_MANIFEST or old["verified_blobs"] != verified:
            raise RuntimeError("Existing pin differs; will not overwrite")
        print("Existing identical pin preserved")
    else:
        save(dest, pin)
        print(str(dest))


if __name__ == "__main__":
    main()
