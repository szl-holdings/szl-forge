"""proofrun CLI.
  proofrun verify <dir> [--trusted-key FP]... [--trusted-identity SAN]... [--trusted-issuer URL]
                        [--accept-retroactive] [--allow-unanchored] [--json]
  proofrun statement <dir> [-o FILE]            (write proofrun_bundle.json; print its sha256)
  proofrun endorse <dir> --key-file PATH        (org-key endorsement; off the GPU host only)
  proofrun demo <dir>                           (writes a synthetic signed run with an ephemeral key)

FP is ed25519:sha256:<64 hex> (preferred) or ed25519:<16 hex>. A trusted key may be the run signer
itself (direct) or an org endorser whose endorsement.json binds the bundle digest. A trusted identity
is an exact GitHub Actions workflow SAN, checked with `gh attestation verify` against the recomputed
bundle statement. When both are given, both must pass."""
from __future__ import annotations

import argparse
import json
import sys

from .endorse import GITHUB_ISSUER
from .verify import render, verify


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="proofrun")
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify", help="verify every claim in a run directory (exit 0 only if all pass)")
    v.add_argument("dir")
    v.add_argument("--trusted-key", action="append", default=[],
                   help="run signer or org endorser fingerprint: ed25519:sha256:<64hex> or ed25519:<16hex>")
    v.add_argument("--trusted-identity", action="append", default=[],
                   help="exact Sigstore cert SAN: https://github.com/<o>/<r>/.github/workflows/<f>.yml@refs/...")
    v.add_argument("--trusted-issuer", default=GITHUB_ISSUER, help="OIDC issuer for --trusted-identity")
    v.add_argument("--accept-retroactive", action="store_true",
                   help="evaluate runs that predate proofrun (no heartbeat chain or image digest)")
    v.add_argument("--allow-unanchored", action="store_true",
                   help="DEV ONLY: accept a single self-asserted signer when no --trusted-key or "
                        "--trusted-identity is given (exit 3 otherwise)")
    v.add_argument("--json", action="store_true")
    s = sub.add_parser("statement", help="write the canonical bundle statement and print its sha256")
    s.add_argument("dir")
    s.add_argument("-o", "--out", default=None)
    e = sub.add_parser("endorse", help="endorse a finished bundle with the org key (never on a GPU host)")
    e.add_argument("dir")
    e.add_argument("--key-file", default=None, help="org key (required unless --dry-run)")
    e.add_argument("--accept-retroactive", action="store_true")
    e.add_argument("--note", default="")
    e.add_argument("--claims", default="", help="comma list of claims to endorse (each must PASS now): "
                   "steps,final_adapter,quality; omitted = custody only")
    e.add_argument("--qualification", required=True,
                   help='plain statement of what is NOT qualified, e.g. "NOT QUALIFIED per AQ-16 ..."')
    e.add_argument("--dry-run", action="store_true", help="print the exact endorsement payload; sign nothing")
    d = sub.add_parser("demo", help="write a synthetic proof-carrying run (ephemeral key)")
    d.add_argument("dir")
    d.add_argument("--steps", type=int, default=20)
    a = ap.parse_args(argv)
    if a.cmd == "verify":
        if not a.trusted_key and not a.trusted_identity and not a.allow_unanchored:
            print("proofrun: no --trusted-key or --trusted-identity given: signer is unanchored, refusing (exit 3)",
                  file=sys.stderr)
            return 3
        rep = verify(a.dir, set(a.trusted_key) or None, a.accept_retroactive,
                     set(a.trusted_identity) or None, a.trusted_issuer)
        print(json.dumps(rep, indent=1) if a.json else render(rep))
        return 0 if rep["verdict"] == "PASS" else 1
    if a.cmd == "statement":
        from .endorse import write_statement
        p, dig = write_statement(a.dir, a.out)
        print(json.dumps({"statement": str(p), "bundle_sha256": dig}))
        return 0
    if a.cmd == "endorse":
        from .canonical import ProofError
        from .endorse import endorse
        from .signing import Signer
        claims = [c for c in a.claims.split(",") if c]
        if not a.dry_run and not a.key_file:
            print("endorse refused: --key-file is required unless --dry-run", file=sys.stderr)
            return 1
        try:
            org = None if a.dry_run else Signer.from_key_file(a.key_file)
            info = endorse(a.dir, org, a.accept_retroactive, a.note, claims, a.qualification, a.dry_run)
        except ProofError as ex:
            print(f"endorse refused: {ex}", file=sys.stderr)
            return 1
        print(json.dumps(info))
        return 0
    if a.cmd == "demo":
        from .synthetic import make_run
        info = make_run(a.dir, steps=a.steps)
        print(json.dumps(info, indent=1))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
