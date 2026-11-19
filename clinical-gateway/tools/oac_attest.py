#!/usr/bin/env python3
"""Unsigned in-toto Statement v1 + DSSE envelope for an OAC receipt. RESEARCH PROTOTYPE.

The existing receipt binds sha256 digests, but a digest list that the issuer wrote about itself
is not an attestation. This emits the standard shape (in-toto Statement v1 inside a DSSE
envelope) so Sigstore/cosign can sign it later. THIS TOOL NEVER SIGNS and never reports a
signed state: verify() returns UNSIGNED:* at best. Not a SLSA level claim.

Exit codes of `verify`: 0 only with --accept-unsigned-digests AND verdict UNSIGNED:DIGESTS_OK (a digest-integrity check: the listed
files still hash to the recorded values; it is NOT bound to a particular receipt and proves nothing about who wrote the envelope);
3 for UNSIGNED:* without that flag and for UNVERIFIED_SIGNATURE:* (signature verification is not implemented, so a signature is never
accepted); 2 for any REFUSE:* verdict, including an unreadable or non-JSON envelope. Exiting 0 on a bare UNSIGNED verdict would let
a shell `&&` treat an unsigned digest list as a pass.
"""
from __future__ import annotations
import argparse, base64, hashlib, json, sys
from pathlib import Path

STATEMENT_TYPE = "https://in-toto.io/Statement/v1"
PAYLOAD_TYPE = "application/vnd.in-toto+json"
PREDICATE_TYPE = "urn:szl:oac:rbcg-receipt:v1"  # local placeholder URI, not a registered type

def sha256_file(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def pae(payload_type: str, payload: bytes) -> bytes:
    """DSSE v1 pre-authentication encoding (the bytes a signer signs)."""
    t = payload_type.encode()
    return b"DSSEv1 %d %s %d %s" % (len(t), t, len(payload), payload)

def build(subject_paths, base: Path, predicate: dict) -> dict:
    subj = []
    for p in sorted(subject_paths):
        rel = str(Path(p).resolve().relative_to(base.resolve()))
        subj.append({"name": rel, "digest": {"sha256": sha256_file(p)}})
    stmt = {"_type": STATEMENT_TYPE, "subject": subj, "predicateType": PREDICATE_TYPE, "predicate": predicate}
    payload = json.dumps(stmt, sort_keys=True, separators=(",", ":")).encode()
    return {"payloadType": PAYLOAD_TYPE, "payload": base64.b64encode(payload).decode(), "signatures": []}

def verify(env: dict, base: Path) -> dict:
    try:
        if env.get("payloadType") != PAYLOAD_TYPE: return {"verdict": "REFUSE:PAYLOAD_TYPE"}
        stmt = json.loads(base64.b64decode(env["payload"], validate=True))
        if stmt.get("_type") != STATEMENT_TYPE or not stmt.get("subject"): return {"verdict": "REFUSE:STATEMENT"}
        for s in stmt["subject"]:
            f = (base / s["name"]).resolve()
            if base.resolve() not in f.parents: return {"verdict": "REFUSE:PATH_ESCAPE:" + s["name"]}
            if not f.is_file(): return {"verdict": "REFUSE:MISSING:" + s["name"]}
            if sha256_file(f) != s["digest"].get("sha256"): return {"verdict": "REFUSE:DIGEST_MISMATCH:" + s["name"]}
    except Exception as e:  # malformed input is a refusal, never an accept
        return {"verdict": "REFUSE:MALFORMED:" + type(e).__name__}
    sigs = env.get("signatures") or []
    return {"verdict": "UNSIGNED:DIGESTS_OK" if not sigs else "UNVERIFIED_SIGNATURE:DIGESTS_OK",
            "note": "signature verification is not implemented here; use cosign/sigstore. Digest check only: not bound to a receipt.",
            "signatures_present": len(sigs)}

def main(argv=None):
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--base", required=True); b.add_argument("--predicate-json", required=True); b.add_argument("subjects", nargs="+")
    v = sub.add_parser("verify"); v.add_argument("--base", required=True); v.add_argument("envelope")
    v.add_argument("--accept-unsigned-digests", action="store_true", help="exit 0 on UNSIGNED:DIGESTS_OK (digest-integrity check only)")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        print(json.dumps(build(a.subjects, Path(a.base), json.loads(Path(a.predicate_json).read_text())), indent=1)); return 0
    try: env = json.loads(Path(a.envelope).read_text())
    except (OSError, ValueError, RecursionError, MemoryError) as e: r = {"verdict": "REFUSE:MALFORMED:" + type(e).__name__}
    else: r = verify(env, Path(a.base))
    print(json.dumps(r))
    if r["verdict"].startswith("REFUSE"): return 2
    return 0 if (a.accept_unsigned_digests and r["verdict"] == "UNSIGNED:DIGESTS_OK") else 3

if __name__ == "__main__": raise SystemExit(main())
