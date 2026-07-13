#!/usr/bin/env python3
# SZL-Forge-1.5B-ReceiptAgent — owner-side receipt signer + keygen.
# SPDX-License-Identifier: Apache-2.0
# (c) 2026 Lutar, Stephen P. - SZL Holdings
#
# This is the trust root of the ReceiptAgent forge. It:
#   1. generates the owner's ed25519 keypair (the PRIVATE key never leaves the
#      owner's machine; the PUBLIC key is committed as owner_pubkey.json), and
#   2. signs a receipt payload into the exact signed-file wrapper the Alloy
#      backbone verifies (lib/receipts.ts SignedFileSchema).
#
# BINDING honesty doctrine:
# - The signature is computed over a CANONICAL JSON string that is byte-for-byte
#   identical to the TypeScript canonicalJson() in artifacts/api-server. If this
#   file drifts from that function, receipts will FAIL to verify server-side --
#   that is the point: an unverifiable receipt is honestly rejected, never
#   rounded up to "probably fine".
# - keyId = first 16 hex chars of sha256(SPKI DER), identical to the server.
# - Payloads carry integers and strings ONLY (never floats): a float would not
#   canonicalize identically across Python and JS. A training loss is a STRING.
# - This script signs an owner's attestation; it does NOT train, evaluate, or
#   measure anything, and NOTHING here upgrades Lambda.
import argparse
import base64
import hashlib
import json
import os
import sys

try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
    )
    from cryptography.hazmat.primitives import serialization
except ImportError:
    sys.stderr.write(
        "[sign_receipt] missing dependency: pip install cryptography\n"
    )
    raise

HERE = os.path.dirname(os.path.abspath(__file__))
OWNER_PUBKEY_FILE = "owner_pubkey.json"


def canonical_json(value):
    """Recursively key-sorted, whitespace-free JSON. A byte-for-byte mirror of
    canonicalJson() in artifacts/api-server/src/lib/receipts.ts. Scalars go
    through json.dumps (matching JSON.stringify); objects sort their keys;
    arrays preserve order. undefined does not exist in Python, so (like the TS
    which only drops undefined) every key is kept."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
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


def _spki_der(public_key) -> bytes:
    return public_key.public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def key_id_from_spki(spki_der: bytes) -> str:
    return hashlib.sha256(spki_der).hexdigest()[:16]


def key_pem_path() -> str:
    """Where the PRIVATE key lives. Defaults OUTSIDE the repo so it can never be
    committed by accident; override with A11OY_OWNER_KEY_PEM."""
    env = os.environ.get("A11OY_OWNER_KEY_PEM")
    if env:
        return env
    return os.path.join(os.path.expanduser("~"), ".a11oy", "receiptagent_owner_ed25519.pem")


def generate_owner_key(out_dir: str, force: bool = False) -> str:
    pem_path = key_pem_path()
    pub_path = os.path.join(out_dir, OWNER_PUBKEY_FILE)
    if os.path.exists(pem_path) and not force:
        raise SystemExit(
            f"[sign_receipt] private key already exists at {pem_path}\n"
            "Refusing to overwrite (rotating a key invalidates every existing "
            "receipt). Pass --force ONLY if you intend to rotate."
        )
    priv = Ed25519PrivateKey.generate()
    pem = priv.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    os.makedirs(os.path.dirname(pem_path), exist_ok=True)
    with open(pem_path, "wb") as f:
        f.write(pem)
    try:
        os.chmod(pem_path, 0o600)
    except OSError:
        pass  # best-effort on Windows

    spki = _spki_der(priv.public_key())
    key_id = key_id_from_spki(spki)
    pub = {
        "algo": "ed25519",
        "publicKeySpkiBase64": base64.b64encode(spki).decode("ascii"),
        "keyId": key_id,
    }
    with open(pub_path, "w", encoding="utf-8") as f:
        json.dump(pub, f, indent=2)
        f.write("\n")

    print(f"[sign_receipt] PRIVATE key written to {pem_path}")
    print(f"[sign_receipt]   -> KEEP THIS SECRET. Never commit it. Never paste it.")
    print(f"[sign_receipt] PUBLIC key committed to {pub_path}")
    print(f"[sign_receipt] keyId = {key_id}")
    print(
        "[sign_receipt] To PIN this key in production, set the Replit secret "
        f"A11OY_OWNER_KEYID = {key_id}"
    )
    return key_id


def _load_private_key():
    pem_path = key_pem_path()
    if not os.path.exists(pem_path):
        raise SystemExit(
            f"[sign_receipt] no private key at {pem_path} -- run 'keygen' first."
        )
    with open(pem_path, "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None)


def sign_payload(payload: dict, out_path: str) -> dict:
    """Canonicalize + ed25519-sign a payload into the signed-file wrapper the
    backbone verifies. Returns the wrapper dict (also written to out_path)."""
    priv = _load_private_key()
    spki = _spki_der(priv.public_key())
    key_id = key_id_from_spki(spki)

    # The receipt embeds its own keyId; fail loud if the caller disagrees. An
    # empty string (or absent) means "stamp it from the signing key".
    payload_key_id = payload.get("keyId")
    if payload_key_id not in (None, "") and payload_key_id != key_id:
        raise SystemExit(
            f"[sign_receipt] payload keyId {payload_key_id!r} != signing key "
            f"{key_id!r}. Refusing to sign a mismatched receipt."
        )
    payload = {**payload, "keyId": key_id}

    canonical = canonical_json(payload)
    signature = priv.sign(canonical.encode("utf-8"))
    wrapper = {
        "payload": payload,
        "canonical": canonical,
        "signatureBase64": base64.b64encode(signature).decode("ascii"),
        "publicKeySpkiBase64": base64.b64encode(spki).decode("ascii"),
        "keyId": key_id,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(wrapper, f, indent=2)
        f.write("\n")
    print(f"[sign_receipt] signed -> {out_path}")
    print(f"[sign_receipt]   canonical sha256 = {sha256_hex(canonical)}")
    print(f"[sign_receipt]   keyId = {key_id}")
    return wrapper


def _cli() -> None:
    parser = argparse.ArgumentParser(description="SZL ReceiptAgent receipt signer")
    sub = parser.add_subparsers(dest="cmd", required=True)

    kg = sub.add_parser("keygen", help="generate the owner ed25519 keypair")
    kg.add_argument("--dir", default=HERE, help="where owner_pubkey.json is written")
    kg.add_argument("--force", action="store_true", help="rotate an existing key")

    sg = sub.add_parser("sign", help="sign a payload JSON file")
    sg.add_argument("payload", help="path to a JSON payload file")
    sg.add_argument("out", help="path to write the .signed.json wrapper")

    args = parser.parse_args()
    if args.cmd == "keygen":
        generate_owner_key(args.dir, force=args.force)
    elif args.cmd == "sign":
        with open(args.payload, "r", encoding="utf-8") as f:
            payload = json.load(f)
        sign_payload(payload, args.out)


if __name__ == "__main__":
    _cli()
