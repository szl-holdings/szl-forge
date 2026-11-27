"""Ed25519 envelopes. Every receipt is {payload, payload_sha256, sig: {alg, public_key, signature}}.

Run keys are EPHEMERAL: generated in memory inside the training container, never written to disk
and never shipped as a long-lived secret. Trust is anchored after the run by an ENDORSEMENT over
the finished bundle (see endorse.py): an org-owned ed25519 key held off the GPU host (Amaru's
custody on the box) and/or Sigstore keyless via GitHub OIDC. If `cryptography` is missing,
signing is impossible and verification FAILS closed.

Fingerprints:  short  ed25519:<16 hex>          (legacy, display; 64-bit)
               full   ed25519:sha256:<64 hex>   (publish this one; sha256 of the raw 32-byte key)
"""
from __future__ import annotations

import base64
import os
import re
import stat
from pathlib import Path
from typing import Any

from .canonical import ProofError, canonical_bytes, sha256_bytes

try:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (Ed25519PrivateKey,
                                                                   Ed25519PublicKey)
    HAVE_CRYPTO = True
except Exception:  # pragma: no cover
    HAVE_CRYPTO = False

ALG = "ed25519"


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def _unb64(s: str) -> bytes:
    try:
        return base64.b64decode(s.encode("ascii"), validate=True)
    except Exception:
        raise ProofError("bad base64 in signature envelope")


def key_fingerprint(public_key_b64: str) -> str:
    return "ed25519:" + sha256_bytes(_unb64(public_key_b64))[:16]


def full_fingerprint(public_key_b64: str) -> str:
    return "ed25519:sha256:" + sha256_bytes(_unb64(public_key_b64))


SHORT_FP_RE = re.compile(r"^ed25519:[0-9a-f]{16}$")
FULL_FP_RE = re.compile(r"^ed25519:sha256:[0-9a-f]{64}$")


def valid_fingerprint(fp: Any) -> bool:
    return isinstance(fp, str) and bool(SHORT_FP_RE.match(fp) or FULL_FP_RE.match(fp))


def fp_matches(public_key_b64: str, trusted: set[str]) -> bool:
    """True when the key's short or full fingerprint is in `trusted` (exact string match)."""
    return key_fingerprint(public_key_b64) in trusted or full_fingerprint(public_key_b64) in trusted


def generate_org_key(key_dir: Path | str, name: str = "proofrun-signing") -> dict:
    """Create an org ed25519 key ON THIS HOST: dir mode 700, private key mode 600 (O_EXCL, never
    overwritten). Returns PUBLIC values only. The private key is never printed or returned."""
    if not HAVE_CRYPTO:
        raise ProofError("cryptography not installed: cannot generate a key")
    d = Path(key_dir)
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(d, 0o700)
    priv = d / f"{name}.key"
    sk = Ed25519PrivateKey.generate()
    pem = sk.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                           serialization.NoEncryption())
    fd = os.open(priv, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, pem)
    finally:
        os.close(fd)
    del pem
    raw = sk.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    pub_pem = sk.public_key().public_bytes(serialization.Encoding.PEM,
                                           serialization.PublicFormat.SubjectPublicKeyInfo)
    (d / f"{name}.pub").write_bytes(pub_pem)
    b64 = _b64(raw)
    return {"private_key_path": str(priv), "public_key_pem": pub_pem.decode("ascii"),
            "public_key_b64": b64, "fingerprint": full_fingerprint(b64), "short": key_fingerprint(b64)}


class Signer:
    """Ephemeral in-memory ed25519 signer (test/demo only)."""

    def __init__(self, role: str = "test-ephemeral", _sk=None) -> None:
        if not HAVE_CRYPTO:
            raise ProofError("cryptography not installed: cannot sign")
        self._sk = _sk if _sk is not None else Ed25519PrivateKey.generate()
        self.role = role
        raw = self._sk.public_key().public_bytes(serialization.Encoding.Raw,
                                                 serialization.PublicFormat.Raw)
        self.public_key_b64 = _b64(raw)
        self.fingerprint = key_fingerprint(self.public_key_b64)
        self.full_fingerprint = full_fingerprint(self.public_key_b64)

    def __repr__(self) -> str:  # never expose key material in logs or tracebacks
        return f"Signer(role={self.role!r}, fp={self.full_fingerprint})"

    @classmethod
    def from_key_file(cls, path: Path | str, role: str = "org-endorser") -> "Signer":
        """Load an org key from disk. Refuses keys readable by group/other (must be 0600 or stricter)
        and refuses to run when PROOFRUN_ON_GPU_HOST is set (the org key must never reach a GPU host)."""
        if not HAVE_CRYPTO:
            raise ProofError("cryptography not installed: cannot sign")
        if os.environ.get("PROOFRUN_ON_GPU_HOST"):
            raise ProofError("refusing to load an org key on a GPU host (PROOFRUN_ON_GPU_HOST is set)")
        p = Path(path)
        st = p.stat()
        if not stat.S_ISREG(st.st_mode) or st.st_mode & 0o077:
            raise ProofError(f"org key {p} must be a regular file with mode 0600")
        sk = serialization.load_pem_private_key(p.read_bytes(), password=None)
        if not isinstance(sk, Ed25519PrivateKey):
            raise ProofError("org key is not ed25519")
        return cls(role=role, _sk=sk)

    def envelope(self, kind: str, payload: dict) -> dict:
        body = {"kind": kind, "payload": payload}
        msg = canonical_bytes(body)
        return {"kind": kind, "payload": payload, "payload_sha256": sha256_bytes(msg),
                "sig": {"alg": ALG, "public_key": self.public_key_b64, "role": self.role,
                        "signature": _b64(self._sk.sign(msg))}}


def open_envelope(env: Any, kind: str, trusted: set[str] | None = None) -> tuple[dict, str]:
    """Verify an envelope and return (payload, signer_fingerprint). Raises ProofError on ANY issue."""
    if not isinstance(env, dict) or env.get("kind") != kind:
        raise ProofError(f"not a {kind} envelope")
    payload = env.get("payload")
    if not isinstance(payload, dict):
        raise ProofError(f"{kind}: payload missing")
    msg = canonical_bytes({"kind": kind, "payload": payload})
    if env.get("payload_sha256") != sha256_bytes(msg):
        raise ProofError(f"{kind}: payload_sha256 mismatch (payload edited after signing)")
    sig = env.get("sig")
    if not isinstance(sig, dict) or sig.get("alg") != ALG:
        raise ProofError(f"{kind}: unsigned or unsupported signature algorithm")
    if not HAVE_CRYPTO:
        raise ProofError("cryptography not installed: signature cannot be verified (fail closed)")
    pk_b64 = sig.get("public_key", "")
    try:
        pk = Ed25519PublicKey.from_public_bytes(_unb64(pk_b64))
        pk.verify(_unb64(sig.get("signature", "")), msg)
    except InvalidSignature:
        raise ProofError(f"{kind}: signature INVALID")
    except ProofError:
        raise
    except Exception as e:
        raise ProofError(f"{kind}: signature unverifiable ({e.__class__.__name__})")
    fp = key_fingerprint(pk_b64)
    if trusted is not None and not fp_matches(pk_b64, trusted):
        raise ProofError(f"{kind}: signer {full_fingerprint(pk_b64)} is not in the trusted key set")
    return payload, fp
