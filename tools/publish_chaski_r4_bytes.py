#!/usr/bin/env python3
"""chaski-r4 bytes publication (second half of the PUBLIC_EXPERIMENTAL_ARTIFACT decision).

Order is enforced: the card must already be live on the Hub and byte-identical to szl-forge main
before any adapter byte is uploaded. Every digest is checked before upload and again after
read-back. Idempotent: if the bytes are already on the Hub, it verifies instead of uploading.

usage: chaski_r4_publish_bytes.py <adapter_dir> <staging_chaski_r4_dir_from_origin_main> <license_path> <receipt_out>
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO = "SZLHOLDINGS/chaski-r4"
EXPECT = {
    "adapter_model.safetensors": "f1a2cdc313795775966280bc8648367005700327dd28010da8d2f88d2a5e2a02",
    "adapter_config.json": "d36472a3100231dfdf0c4aa16d3a80cb8ab53c86af33a15172b06f9d9a313193",
    "evidence/canonical_rerun_20261001_140615.receipt.json": "5ae3de970014726f190dfe8e4d5b35e667fd737b1be29e27205c25d143bdf403",
}
EXPECT_DIR_DIGEST = "e1abc37a5c41a82b0fc2cd98ccd6edbb2a08fceb8ca9e883bcfb76b861c221cf"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dir_digest(directory: Path) -> str:
    """Same algorithm as chaski/bakeoff_named_n.py::sha256_adapter."""
    digest = hashlib.sha256()
    files = sorted(directory.glob("*.safetensors"))
    if not files:
        raise SystemExit(f"no safetensors in {directory}")
    for file in files:
        digest.update(file.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file.read_bytes())
    return digest.hexdigest()


def main() -> int:
    adapter = Path(sys.argv[1])
    staging = Path(sys.argv[2])
    license_path = Path(sys.argv[3])
    receipt_out = Path(sys.argv[4])
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

    # ---- local preflight: the bytes we are about to publish are exactly the receipted bytes
    local = {
        "adapter_model.safetensors": adapter / "adapter_model.safetensors",
        "adapter_config.json": adapter / "adapter_config.json",
        "training_receipt.json": staging / "training_receipt.json",
        "evidence/canonical_rerun_20261001_140615.receipt.json": staging / "evidence" / "canonical_rerun_20261001_140615.receipt.json",
        "evidence/QUALIFICATION_STATE.md": staging / "QUALIFICATION_STATE.md",
        "LICENSE": license_path,
    }
    for target, path in local.items():
        if not path.is_file():
            raise SystemExit(f"PREFLIGHT: missing {target} at {path}")
    local_sha = {t: sha(p) for t, p in local.items()}
    for target, expected in EXPECT.items():
        if local_sha[target] != expected:
            raise SystemExit(f"PREFLIGHT: {target} sha256 {local_sha[target]} != expected {expected}. Stop state: nothing uploaded.")
    local_dir_digest = dir_digest(adapter)
    if local_dir_digest != EXPECT_DIR_DIGEST:
        raise SystemExit(f"PREFLIGHT: adapter directory digest {local_dir_digest} != {EXPECT_DIR_DIGEST}. Stop state: nothing uploaded.")
    if "Apache License" not in license_path.read_text(encoding="utf-8", errors="replace")[:400]:
        raise SystemExit("PREFLIGHT: LICENSE is not the Apache-2.0 text. Stop state: nothing uploaded.")
    print(f"[preflight] adapter raw {local_sha['adapter_model.safetensors'][:16]}... config {local_sha['adapter_config.json'][:16]}... dir {local_dir_digest[:16]}... receipt C {local_sha['evidence/canonical_rerun_20261001_140615.receipt.json'][:16]}... OK")

    # ---- Hub preflight: identity, repo, CARD FIRST
    api = HfApi()
    who = api.whoami()
    orgs = [o.get("name") for o in who.get("orgs", [])]
    if "SZLHOLDINGS" not in orgs:
        raise SystemExit(f"PREFLIGHT: token identity {who.get('name')} is not in SZLHOLDINGS ({orgs}). Stop state: nothing uploaded.")
    info = api.repo_info(REPO, repo_type="model")
    files = {s.rfilename for s in info.siblings}
    if "README.md" not in files:
        raise SystemExit("CARD FIRST: README.md is not on the Hub yet (publish-chaski-card.yml publishes it on merge of szl-forge #478). Rerun after it lands. Stop state: nothing uploaded.")
    hub_card = Path(hf_hub_download(REPO, "README.md", repo_type="model", force_download=True)).read_bytes()
    main_card = (staging / "card" / "README.md").read_bytes()
    if hub_card != main_card:
        raise SystemExit("CARD FIRST: Hub README.md differs from szl-forge main chaski_r4/card/README.md. Stop state: nothing uploaded.")
    card_revision = info.sha
    print(f"[card-first] card live and byte-identical to main at Hub revision {card_revision}")

    # ---- upload (idempotent)
    already = "adapter_model.safetensors" in files
    if already:
        print("[upload] adapter bytes already on the Hub; verifying instead of uploading")
        bytes_revision = card_revision
    else:
        ops = [CommitOperationAdd(path_in_repo=t, path_or_fileobj=str(p)) for t, p in local.items()]
        commit = api.create_commit(
            repo_id=REPO,
            repo_type="model",
            operations=ops,
            commit_message="chaski-r4: adapter bytes + training receipt + receipt C (PUBLIC_EXPERIMENTAL_ARTIFACT; NOT PROMOTABLE)",
            commit_description=(
                "Adapter raw sha256 f1a2cdc3...; directory digest e1abc37a... (receipts B and C adapter_sha256). "
                "Receipt C 5ae3de97...: 5/5 JSON drafts, 6/6 adversarial refusals beside the r2 control, 192/192 adapter tensors applied. "
                "publication_eligible=false by evaluator design; promotion is an owner decision not taken. Card published first via szl-forge publish-chaski-card.yml."
            ),
            parent_commit=card_revision,
        )
        bytes_revision = commit.oid
        print(f"[upload] committed {bytes_revision}")

    # ---- read-back verification at the exact revision
    with tempfile.TemporaryDirectory() as tmp:
        readback: dict[str, str] = {}
        for target in local:
            got = Path(hf_hub_download(REPO, target, repo_type="model", revision=bytes_revision, force_download=True, local_dir=tmp))
            readback[target] = sha(got)
        hub_dir_digest = dir_digest(Path(tmp))
    mismatches = [t for t in local if readback[t] != local_sha[t]]
    if mismatches or hub_dir_digest != EXPECT_DIR_DIGEST:
        raise SystemExit(f"READBACK: mismatch {mismatches}, hub dir digest {hub_dir_digest}. The commit exists; do not cite it until resolved.")
    print(f"[readback] every file byte-identical at {bytes_revision}; hub adapter directory digest {hub_dir_digest[:16]}... matches receipts B and C")

    receipt = {
        "schema": "szl.hf-artifact-publication/v1",
        "repo_id": REPO,
        "artifact_state": "PUBLIC_EXPERIMENTAL_ARTIFACT",
        "promotion": "NOT_PROMOTABLE",
        "publication_eligible": False,
        "autonomy_eligible": False,
        "card_revision": card_revision,
        "bytes_revision": bytes_revision,
        "card_sha256_main_equals_hub": True,
        "files": {t: {"sha256": local_sha[t], "readback_sha256": readback[t]} for t in local},
        "adapter_directory_digest": hub_dir_digest,
        "receipt_c_sha256": EXPECT["evidence/canonical_rerun_20261001_140615.receipt.json"],
        "published_by": who.get("name"),
        "published_at": datetime.now(timezone.utc).isoformat(),
        "claim_boundary": "Publication of bytes and evidence only. No promotion, deployment, autonomy or production authorization. Receipts A and B provenance remains unresolved. SZLHOLDINGS/chaski is not overwritten.",
    }
    receipt_out.parent.mkdir(parents=True, exist_ok=True)
    receipt_out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"PUBLISHED": REPO, "card_revision": card_revision, "bytes_revision": bytes_revision, "receipt": str(receipt_out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
