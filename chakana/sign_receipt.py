#!/usr/bin/env python3
# SZLHOLDINGS/chakana — owner-side receipt signer + unsigned stub.
# SPDX-License-Identifier: Apache-2.0
# (c) 2026 Lutar, Stephen P. - SZL Holdings
#
# Canonical JSON is a byte-for-byte mirror of khipu/sign_receipt.py /
# artifacts/api-server canonicalJson(): key-sorted, no whitespace,
# strings+ints+bools+null only. Floats are refused.
#
# This file signs an owner's attestation when A11OY_OWNER_KEY_PEM (or the
# default ~/.a11oy/chakana_owner_ed25519.pem) is present. The committed
# training_receipt.stub.json / eval_receipt.stub.json are UNSIGNED. A
# missing private key never fabricates a signature. Jobs UNKNOWN. nDCG@10
# stays UNKNOWN until a held-out run.
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OWNER_PUBKEY_FILE = "owner_pubkey.json"
TRAIN_STUB = HERE / "training_receipt.stub.json"
EVAL_STUB = HERE / "eval_receipt.stub.json"


def canonical_json(value):
    """Recursively key-sorted, whitespace-free JSON. Floats are REFUSED."""
    if value is None or isinstance(value, (bool, int, str)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, float):
        raise TypeError(
            "canonical_json refuses float values (doctrine: strings+ints only) -- "
            "put a loss/score in as a STRING or an integer count, got "
            + repr(value)
        )
    if isinstance(value, list):
        return "[" + ",".join(canonical_json(v) for v in value) + "]"
    if isinstance(value, dict):
        keys = sorted(value.keys())
        return (
            "{"
            + ",".join(
                json.dumps(k, ensure_ascii=False) + ":" + canonical_json(value[k])
                for k in keys
            )
            + "}"
        )
    raise TypeError(f"canonical_json: unsupported type {type(value)!r}")


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def key_pem_path() -> str:
    env = os.environ.get("A11OY_OWNER_KEY_PEM")
    if env:
        return env
    return os.path.join(os.path.expanduser("~"), ".a11oy", "chakana_owner_ed25519.pem")


def training_stub_payload() -> dict:
    return {
        "kind": "szl-chakana-training-receipt",
        "v": 1,
        "artifact": "SZLHOLDINGS/chakana",
        "owner": "Stephen Lutar",
        "lane": "NINA (FORGE-class)",
        "baseModel": "Qwen/Qwen3-Embedding-0.6B",
        "altBaseModel": "BAAI/bge-m3",
        "dataset": "SZLHOLDINGS/chakana-pairs",
        "seed": 11,
        "jobs": "UNKNOWN",
        "weights": "UNAVAILABLE",
        "evals": "UNKNOWN",
        "ndcg10": "UNKNOWN",
        "mtebPasted": False,
        "publication_eligible": False,
        "lambda": "Conjecture 1",
        "doctrine": "v11 LOCKED 749/14/163",
        "evidenceCeiling": "0.97",
        "signed": False,
        "keyId": "",
        "claimBoundary": (
            "UNSIGNED stub. No job id. No weights. nDCG@10 UNKNOWN. "
            "Do not paste MTEB numbers."
        ),
    }


def eval_stub_payload() -> dict:
    return {
        "kind": "szl-chakana-eval-receipt",
        "v": 1,
        "artifact": "SZLHOLDINGS/chakana",
        "owner": "Stephen Lutar",
        "lane": "NINA (FORGE-class)",
        "baseModel": "Qwen/Qwen3-Embedding-0.6B",
        "jobs": "UNKNOWN",
        "evals": "UNKNOWN",
        "ndcg10": "UNKNOWN",
        "mtebPasted": False,
        "publication_eligible": False,
        "signed": False,
        "keyId": "",
        "claimBoundary": (
            "UNSIGNED stub. Frozen in-house nDCG@10 has not run. "
            "Status stays UNKNOWN."
        ),
    }


def write_unsigned_stub(payload: dict, path: Path) -> dict:
    canonical = canonical_json(payload)
    wrapper = {
        "payload": payload,
        "canonical": canonical,
        "signatureBase64": None,
        "publicKeySpkiBase64": None,
        "keyId": "",
        "signed": False,
        "status": "STUB-UNSIGNED",
        "jobs": "UNKNOWN",
    }
    path.write_text(json.dumps(wrapper, indent=2) + "\n", encoding="utf-8")
    print(f"[chakana-sign] unsigned stub -> {path}")
    print(f"[chakana-sign]   canonical sha256 = {sha256_hex(canonical)}")
    return wrapper


def _cryptography():
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
        )
        from cryptography.hazmat.primitives import serialization
    except ImportError:
        sys.stderr.write(
            "[chakana-sign] missing dependency: pip install cryptography\n"
        )
        raise
    return Ed25519PrivateKey, serialization


def generate_owner_key(out_dir: str, force: bool = False) -> str:
    Ed25519PrivateKey, serialization = _cryptography()
    pem_path = key_pem_path()
    pub_path = os.path.join(out_dir, OWNER_PUBKEY_FILE)
    if os.path.exists(pem_path) and not force:
        raise SystemExit(
            f"[chakana-sign] private key already exists at {pem_path}\n"
            "Refusing to overwrite. Pass --force ONLY if you intend to rotate."
        )
    priv = Ed25519PrivateKey.generate()
    pem = priv.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    os.makedirs(os.path.dirname(pem_path), exist_ok=True)
    with open(pem_path, "wb") as handle:
        handle.write(pem)
    try:
        os.chmod(pem_path, 0o600)
    except OSError:
        pass
    spki = priv.public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    key_id = hashlib.sha256(spki).hexdigest()[:16]
    pub = {
        "algo": "ed25519",
        "publicKeySpkiBase64": base64.b64encode(spki).decode("ascii"),
        "keyId": key_id,
    }
    with open(pub_path, "w", encoding="utf-8") as handle:
        json.dump(pub, handle, indent=2)
        handle.write("\n")
    print(f"[chakana-sign] PRIVATE key written to {pem_path}")
    print("[chakana-sign]   -> KEEP THIS SECRET. Never commit it.")
    print(f"[chakana-sign] PUBLIC key committed to {pub_path}")
    print(f"[chakana-sign] keyId = {key_id}")
    return key_id


def sign_payload(payload: dict, out_path: str) -> dict:
    Ed25519PrivateKey, serialization = _cryptography()
    pem_path = key_pem_path()
    if not os.path.exists(pem_path):
        raise SystemExit(
            f"[chakana-sign] no private key at {pem_path} -- refusing to "
            "fabricate a signature. Run 'keygen' first, or keep the unsigned stub."
        )
    with open(pem_path, "rb") as handle:
        priv = serialization.load_pem_private_key(handle.read(), password=None)
    spki = priv.public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    key_id = hashlib.sha256(spki).hexdigest()[:16]
    payload_key_id = payload.get("keyId")
    if payload_key_id not in (None, "") and payload_key_id != key_id:
        raise SystemExit(
            f"[chakana-sign] payload keyId {payload_key_id!r} != signing key "
            f"{key_id!r}. Refusing to sign a mismatched receipt."
        )
    payload = {**payload, "keyId": key_id, "signed": True}
    canonical = canonical_json(payload)
    signature = priv.sign(canonical.encode("utf-8"))
    wrapper = {
        "payload": payload,
        "canonical": canonical,
        "signatureBase64": base64.b64encode(signature).decode("ascii"),
        "publicKeySpkiBase64": base64.b64encode(spki).decode("ascii"),
        "keyId": key_id,
        "signed": True,
    }
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(wrapper, handle, indent=2)
        handle.write("\n")
    print(f"[chakana-sign] signed -> {out_path}")
    print(f"[chakana-sign]   canonical sha256 = {sha256_hex(canonical)}")
    print(f"[chakana-sign]   keyId = {key_id}")
    return wrapper


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Chakana receipt signer / stub")
    sub = parser.add_subparsers(dest="cmd", required=True)

    kg = sub.add_parser("keygen", help="generate the owner ed25519 keypair")
    kg.add_argument("--dir", default=str(HERE))
    kg.add_argument("--force", action="store_true")

    sg = sub.add_parser("sign", help="sign a payload JSON file (requires private key)")
    sg.add_argument("payload")
    sg.add_argument("out")

    stub = sub.add_parser("stub", help="rewrite unsigned training/eval stubs")
    stub.add_argument(
        "--check",
        action="store_true",
        help="Verify committed stubs are unsigned. Do not rewrite.",
    )

    args = parser.parse_args()
    if args.cmd == "keygen":
        generate_owner_key(args.dir, force=args.force)
        return 0
    if args.cmd == "sign":
        with open(args.payload, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        sign_payload(payload, args.out)
        return 0
    if args.cmd == "stub":
        if args.check:
            for path in (TRAIN_STUB, EVAL_STUB):
                wrapper = json.loads(path.read_text(encoding="utf-8"))
                if wrapper.get("signed") is not False:
                    raise SystemExit(f"[chakana-sign] {path.name} must be unsigned")
                if wrapper.get("signatureBase64") is not None:
                    raise SystemExit(
                        f"[chakana-sign] {path.name} has a signature; stubs stay unsigned"
                    )
                if wrapper.get("jobs") != "UNKNOWN":
                    raise SystemExit(f"[chakana-sign] {path.name} jobs must be UNKNOWN")
                print(f"[chakana-sign] {path.name} STUB-UNSIGNED jobs=UNKNOWN")
            return 0
        write_unsigned_stub(training_stub_payload(), TRAIN_STUB)
        write_unsigned_stub(eval_stub_payload(), EVAL_STUB)
        return 0
    raise SystemExit(f"unknown command {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(_cli())
