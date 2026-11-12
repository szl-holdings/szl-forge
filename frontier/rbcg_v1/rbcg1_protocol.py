#!/usr/bin/env python3
# RBCG v1 prototype -- SZL Holdings. Licence: UNDECIDED (owner choice). See NOTICE.
"""rbcg1_protocol.py -- write, sign and verify the RBCG v1 pre-registration protocol receipt.

SYNTHETIC DATA ONLY. NOT FOR CLINICAL USE. The protocol is a DSSE envelope (ECDSA P-256 / RFC 6979,
same code as the receipts) whose payload fixes hypotheses, endpoints, seed families, R and the analysis
plan, and binds the sha256 of every file that generates or analyses evaluation data. Its digest P
(sha256 of the canonical envelope) is embedded, as P[:16], in every evaluation seed family, so no
evaluation datum can exist before the protocol. The signing key is generated for this one signature
and never written; only its public key is kept.

  python3 rbcg1_protocol.py sign --pkg DIR --spec SPEC.json --out OUTDIR
  python3 rbcg1_protocol.py verify --pkg DIR --protocol OUTDIR/protocol.dsse.json
"""
import datetime
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rbcg1  # noqa: E402

PROTOCOL_SCHEMA = "szl.rbcg.protocol/v1"
CODE_FILES = ("rbcg1.py", "rbcg1_log.py", "rbcg1_verify.py", "rbcg1_triagex.py", "rbcg1_experiments.py",
              "rbcg1_protocol.py", "v0_ref/rbcg.py", "v0_ref/rbcg_verify.py")
REQUIRED_KEYS = ("analysis_plan", "code_sha256", "design", "e3_plan", "endpoints", "families", "hypotheses",
                 "lane", "not_for_clinical_use", "pilot", "schema", "signed_utc", "stack", "synthetic_only",
                 "title")


def code_hashes(pkg):
    return {f: rbcg1.sha256_file(os.path.join(pkg, f)) for f in CODE_FILES}


def sign_protocol(pkg, spec, out_dir):
    payload = dict(spec)
    payload["schema"] = PROTOCOL_SCHEMA
    payload["code_sha256"] = code_hashes(pkg)
    payload["signed_utc"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    payload["synthetic_only"] = True
    payload["not_for_clinical_use"] = True
    missing = [k for k in REQUIRED_KEYS if k not in payload]
    if missing:
        raise SystemExit(f"REFUSED: protocol spec lacks {missing}")
    key = rbcg1.SigningKey.generate()
    env = rbcg1.dsse_envelope(payload, rbcg1.PT_PROTOCOL, key)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "protocol.dsse.json"), "wb") as fh:
        fh.write(rbcg1.envelope_bytes(env))
    with open(os.path.join(out_dir, "protocol_public_key.pem"), "w", encoding="ascii", newline="\n") as fh:
        fh.write(key.public.pem())
    del key
    return env, rbcg1.envelope_digest(env)


def verify_protocol(raw, pem, pkg):
    """VALID only if the signature verifies, the payload has every required key, declares
    synthetic-only use, and every bound code hash equals the file in `pkg`. Never raises."""
    import rbcg1_verify as V
    try:
        pub = V._public_key(pem)
        env, payload = V.parse_envelope(raw, rbcg1.PT_PROTOCOL, pub, "protocol")
        if type(payload) is not dict or any(k not in payload for k in REQUIRED_KEYS) or \
                payload["schema"] != PROTOCOL_SCHEMA:
            return {"verdict": "REFUSED:SCHEMA", "detail": "protocol payload keys/schema", "payload": None,
                    "protocol_sha256": None}
        if payload["synthetic_only"] is not True or payload["not_for_clinical_use"] is not True:
            return {"verdict": "REFUSED:CLAIM_UNSUPPORTED", "detail": "not declared synthetic-only",
                    "payload": None, "protocol_sha256": None}
        want = payload["code_sha256"]
        if set(want) != set(CODE_FILES):
            return {"verdict": "REFUSED:SCHEMA", "detail": "bound code file list differs", "payload": None,
                    "protocol_sha256": None}
        got = code_hashes(pkg)
        bad = sorted(f for f in CODE_FILES if got[f] != want[f])
        if bad:
            return {"verdict": "REFUSED:HASH_MISMATCH", "detail": f"code differs from the protocol: {bad}",
                    "payload": None, "protocol_sha256": None}
        return {"verdict": "VALID", "detail": "protocol signature and code bindings verified",
                "payload": payload, "protocol_sha256": rbcg1.envelope_digest(env)}
    except V.Refuse as r:
        return {"verdict": f"REFUSED:{r.code}", "detail": r.detail, "payload": None, "protocol_sha256": None}
    except Exception as e:
        return {"verdict": "REFUSED:INTERNAL", "detail": type(e).__name__, "payload": None,
                "protocol_sha256": None}


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sign", "verify"])
    ap.add_argument("--pkg", default=HERE)
    ap.add_argument("--spec")
    ap.add_argument("--out")
    ap.add_argument("--protocol")
    a = ap.parse_args(argv)
    if a.cmd == "sign":
        with open(a.spec, "rb") as fh:
            spec = json.loads(fh.read())
        env, dig = sign_protocol(a.pkg, spec, a.out)
        print(json.dumps({"protocol_sha256": dig, "P16": dig[:16]}))
        return 0
    with open(a.protocol, "rb") as fh:
        raw = fh.read()
    with open(os.path.join(os.path.dirname(os.path.abspath(a.protocol)), "protocol_public_key.pem"),
              encoding="ascii") as fh:
        pem = fh.read()
    res = verify_protocol(raw, pem, a.pkg)
    print(json.dumps({"verdict": res["verdict"], "detail": res["detail"],
                      "protocol_sha256": res["protocol_sha256"]}))
    return 0 if res["verdict"] == "VALID" else 2


if __name__ == "__main__":
    sys.exit(main())
