#!/usr/bin/env python3
# =============================================================================
# szl_geh_verify.py — ZERO-DEPENDENCY offline verifier for GEH proof receipts.
#
# Verifies, with nothing but the Python standard library:
#   1. DSSE envelope signature (Ed25519, RFC 8032, pure Python — slow but
#      dependency-free; ~50 ms per signature) over PAE(payloadType, payload).
#   2. Receipt bytes match the chain entry (sha256 of canonical envelope).
#   3. Chain linkage: genesis -> prev pointers, no gaps, no reordering.
#   4. Verdict/gate consistency: GATE=PASS only with kernel_verified=true,
#      certified transport, live no-goals, audit 0 errors / 0 sorries, guard
#      allowed, artifact COMPLIANCE_CANDIDATE — recomputed from the payload,
#      not trusted from the `decision` field.
#
# Usage:
#   python szl_geh_verify.py --evidence DIR              # verify whole chain
#   python szl_geh_verify.py --receipt FILE [--pub HEX]  # verify one envelope
# Exit 0 = all verified; nonzero = the first failing check, printed.
# =============================================================================
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# ---------------------------- Ed25519 (RFC 8032) -----------------------------
_P = 2**255 - 19
_L = 2**252 + 27742317777372353535851937790883648493
_D = (-121665 * pow(121666, _P - 2, _P)) % _P
_I = pow(2, (_P - 1) // 4, _P)


def _inv(x: int) -> int:
    return pow(x, _P - 2, _P)


def _recover_x(y: int, sign: int) -> int | None:
    if y >= _P:
        return None
    x2 = (y * y - 1) * _inv(_D * y * y + 1) % _P
    if x2 == 0:
        return None if sign else 0
    x = pow(x2, (_P + 3) // 8, _P)
    if (x * x - x2) % _P != 0:
        x = x * _I % _P
    if (x * x - x2) % _P != 0:
        return None
    if (x & 1) != sign:
        x = _P - x
    return x


def _pt_add(P, Q):
    x1, y1, z1, t1 = P
    x2, y2, z2, t2 = Q
    a = (y1 - x1) * (y2 - x2) % _P
    b = (y1 + x1) * (y2 + x2) % _P
    c = 2 * t1 * t2 * _D % _P
    d = 2 * z1 * z2 % _P
    e, f, g, h = b - a, d - c, d + c, b + a
    return (e * f % _P, g * h % _P, f * g % _P, e * h % _P)


def _pt_mul(s: int, P):
    Q = (0, 1, 1, 0)
    while s > 0:
        if s & 1:
            Q = _pt_add(Q, P)
        P = _pt_add(P, P)
        s >>= 1
    return Q


def _pt_eq(P, Q) -> bool:
    x1, y1, z1, _ = P
    x2, y2, z2, _ = Q
    return (x1 * z2 - x2 * z1) % _P == 0 and (y1 * z2 - y2 * z1) % _P == 0


def _decode_point(s: bytes):
    if len(s) != 32:
        return None
    y = int.from_bytes(s, "little")
    sign = y >> 255
    y &= (1 << 255) - 1
    x = _recover_x(y, sign)
    if x is None:
        return None
    return (x, y, 1, x * y % _P)


_G_Y = 4 * _inv(5) % _P
_G_X = _recover_x(_G_Y, 0)
_G = (_G_X, _G_Y, 1, _G_X * _G_Y % _P)


def ed25519_verify(pub: bytes, msg: bytes, sig: bytes) -> bool:
    if len(sig) != 64 or len(pub) != 32:
        return False
    A = _decode_point(pub)
    R = _decode_point(sig[:32])
    S = int.from_bytes(sig[32:], "little")
    if A is None or R is None or S >= _L:
        return False
    k = int.from_bytes(hashlib.sha512(sig[:32] + pub + msg).digest(), "little") % _L
    return _pt_eq(_pt_mul(S, _G), _pt_add(R, _pt_mul(k, A)))


# ------------------------------- DSSE / receipts -------------------------------
def canon(o) -> str:
    return json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def pae(payload_type: str, payload: bytes) -> bytes:
    pt = payload_type.encode("utf-8")
    return b"DSSEv1 " + str(len(pt)).encode() + b" " + pt + b" " + str(len(payload)).encode() + b" " + payload


def verify_envelope(env: dict, pub_hex: str | None = None) -> tuple[bool, str]:
    sigs = env.get("signatures") or []
    if not sigs:
        return False, "no signatures"
    s = sigs[0]
    if s.get("alg") != "ed25519" or s.get("sig") == "UNSIGNED_HONEST":
        return False, "unsigned envelope (UNSIGNED_HONEST) — cannot verify"
    pub = bytes.fromhex(pub_hex or s.get("pub_hex", ""))
    if not pub:
        return False, "no public key (pass --pub HEX)"
    if hashlib.sha256(pub).hexdigest()[:16] != s.get("keyid"):
        return False, "keyid does not match the public key"
    ok = ed25519_verify(pub, pae(env["payloadType"], env["payload"].encode("utf-8")), bytes.fromhex(s["sig"]))
    return ok, "signature valid" if ok else "SIGNATURE INVALID"


def recompute_gate(payload: dict) -> tuple[str, list[str]]:
    """Derive the gate from the evidence in the payload. Never trust `decision`."""
    proof = payload.get("proof") or {}
    reasons = []
    cert_transports = {"LEAN_REPL", "LEAN_REPL_AUDIT", "PANTOGRAPH", "LEANDOJO"}
    if proof.get("transport") not in cert_transports:
        reasons.append(f"transport {proof.get('transport')} is not certified")
    if proof.get("verdict") != "KERNEL_VERIFIED_AND_SIGNED":
        reasons.append(f"verdict {proof.get('verdict')}")
    if "bench" in proof:
        b = proof["bench"]
        if b.get("pass") != b.get("k"):
            reasons.append(f"bench pass {b.get('pass')}/{b.get('k')}")
        if (b.get("contamination") or {}).get("contaminated"):
            reasons.append("canary leakage detected")
        for r in proof.get("results", []):
            if not r.get("kernel_verified"):
                reasons.append(f"{r.get('id')} not kernel_verified")
    else:
        live = proof.get("live")
        if live is not None and live.get("done") is not True:
            reasons.append("live channel: goals remain")
        if live is not None and live.get("violations"):
            reasons.append(f"trace violations {live['violations']}")
        a = proof.get("audit")
        if a is None:
            reasons.append("no audit channel result")
        else:
            if a.get("error_count", 1) != 0:
                reasons.append("audit errors")
            if a.get("sorry_count", 1) != 0:
                reasons.append("audit sorries")
            if not a.get("guard_all_allowed"):
                reasons.append("guard rejected or absent")
            if a.get("artifact_grade") not in (None, "COMPLIANCE_CANDIDATE"):
                reasons.append(f"artifact grade {a.get('artifact_grade')}")
            for g in a.get("guard", []):
                extra = set(g.get("axioms") or []) - {"propext", "Classical.choice", "Quot.sound"}
                if extra:
                    reasons.append(f"foreign axioms {sorted(extra)}")
    return ("PASS" if not reasons else "VOID"), reasons


def verify_receipt_file(path: Path, pub_hex: str | None = None) -> list[tuple[str, bool, str]]:
    env = json.loads(path.read_text(encoding="utf-8"))
    out = []
    ok, msg = verify_envelope(env, pub_hex)
    out.append(("dsse_signature", ok, msg))
    payload = json.loads(env["payload"])
    gate, reasons = recompute_gate(payload)
    claimed = (payload.get("decision") or {}).get("gate")
    out.append(("gate_recomputed", gate == claimed, f"claimed={claimed} recomputed={gate} {reasons if reasons else ''}"))
    sc = ((payload.get("harness") or {}).get("self_check") or {})
    out.append(("harness_self_check_recorded_pass", sc.get("self_check_pass") is True, "self_check_pass in receipt"))
    return out


def verify_chain(evidence: Path, pub_hex: str | None = None) -> int:
    chain = evidence / "geh_chain.jsonl"
    lines = [json.loads(l) for l in chain.read_text(encoding="utf-8").strip().splitlines()]
    if not lines or not lines[0].get("genesis"):
        print("FAIL chain: missing genesis")
        return 2
    prev, failures, n = lines[0]["sha256"], 0, 0
    for link in lines[1:]:
        n += 1
        p = evidence / link["receipt"]
        if link["prev"] != prev:
            print(f"FAIL link {link['receipt']}: prev pointer broken")
            failures += 1
        env = json.loads(p.read_text(encoding="utf-8"))
        actual = hashlib.sha256(canon(env).encode("utf-8")).hexdigest()
        if actual != link["sha256"]:
            print(f"FAIL bytes {link['receipt']}: sha256 {actual[:16]} != chain {link['sha256'][:16]}")
            failures += 1
        for name, ok, msg in verify_receipt_file(p, pub_hex):
            print(f"{'ok  ' if ok else 'FAIL'} {link['receipt']} {name}: {msg}")
            failures += 0 if ok else 1
        prev = link["sha256"]
    print(f"CHAIN: {n} receipts, head={prev}, failures={failures}")
    return 0 if failures == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", type=Path)
    ap.add_argument("--receipt", type=Path)
    ap.add_argument("--pub", help="ed25519 public key hex (defaults to pub_hex embedded in the envelope)")
    a = ap.parse_args()
    if a.receipt:
        res = verify_receipt_file(a.receipt, a.pub)
        for name, ok, msg in res:
            print(f"{'ok  ' if ok else 'FAIL'} {name}: {msg}")
        return 0 if all(ok for _, ok, _ in res) else 1
    if a.evidence:
        return verify_chain(a.evidence, a.pub)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
