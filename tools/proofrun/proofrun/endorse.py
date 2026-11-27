"""Post-run endorsement: the owned trust anchor for proofrun bundles (Amaru, SEC-PROOFRUN-SIGNING).

Why: run receipts are signed inside a rented GPU container (Salad) by a per-run EPHEMERAL ed25519
key. That key is never exported, and no long-lived secret is ever placed on the GPU host. Without
an anchor, an outsider can only check that one (unknown) key signed everything.

The anchor binds the finished bundle, after the run, off the GPU host:

  bundle statement  canonical JSON listing run_id, the run signer's full fingerprint, the heartbeat
                    chain head and count, and the sha256 of EVERY receipt file (claim, manifest,
                    final, checkpoints/*, evals/*, heartbeats.jsonl). Its sha256 is the bundle digest.
  endorsement.json  ed25519 envelope by the ORG key (custody: Amaru, box only) over
                    {run_id, bundle_sha256, run_signers}. Verified with --trusted-key <org fp>.
  endorsement.sigstore.json
                    Sigstore bundle from actions/attest in a pinned GitHub Actions workflow, whose
                    subject is proofrun_bundle.json (the statement bytes). Verified with
                    --trusted-identity <workflow SAN>, via `gh attestation verify` (fail closed).

Any edit to any receipt after endorsement changes the bundle digest and FAILS verification.
Honest limit: an endorsement says "SZL took custody of this exact bundle and its integrity checks
passed at that time". It does not prove the GPU host was honest; the receipts themselves carry
the per-claim evidence that the verifier re-derives.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable

from . import SCHEMA
from .canonical import ProofError, canonical_bytes, load_json, loads_strict, sha256_bytes, sha256_file, write_json
from .signing import Signer, full_fingerprint

STATEMENT_TYPE = "proofrun-bundle/v1"
ENDORSEMENT_SCHEME = "proofrun-endorsement/v1"
PREDICATE_TYPE = "https://szl-holdings.github.io/proofrun/endorsement/v1"
GITHUB_ISSUER = "https://token.actions.githubusercontent.com"
STATEMENT_FILE = "proofrun_bundle.json"
ENDORSEMENT_FILE = "endorsement.json"
SIGSTORE_FILE = "endorsement.sigstore.json"
CLAIM_NAMES = ("steps", "final_adapter", "quality")
_ENVELOPE_FILES = ("claim.json", "run_manifest.json", "final.json")
_ENVELOPE_DIRS = ("checkpoints", "evals")
IDENTITY_RE = re.compile(r"^https://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)/"
                         r"(\.github/workflows/[A-Za-z0-9_.-]+\.ya?ml)@(refs/(?:heads|tags)/[A-Za-z0-9_./-]+)$")


def _signer_full(env) -> str | None:
    try:
        return full_fingerprint(env["sig"]["public_key"])
    except Exception:
        return None


def bundle_statement(run_dir: Path | str) -> dict:
    """Recompute the bundle statement from the bytes on disk. Raises ProofError if malformed."""
    d = Path(run_dir)
    files, signers = [], set()
    paths = [d / n for n in _ENVELOPE_FILES if (d / n).is_file()]
    for sub in _ENVELOPE_DIRS:
        if (d / sub).is_dir():
            paths += sorted(p for p in (d / sub).rglob("*") if p.is_file())
    for p in paths:
        if p.is_symlink():
            raise ProofError(f"symlink in bundle: {p.relative_to(d)}")
        env = load_json(p)
        fp = _signer_full(env)
        if fp is None:
            raise ProofError(f"unsigned file in bundle: {p.relative_to(d)}")
        signers.add(fp)
        files.append({"path": p.relative_to(d).as_posix(), "sha256": sha256_file(p)})
    head, count = None, 0
    hb = d / "heartbeats.jsonl"
    if hb.is_file():
        files.append({"path": "heartbeats.jsonl", "sha256": sha256_file(hb)})
        for line in hb.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            env = loads_strict(line)
            fp = _signer_full(env)
            if fp is None:
                raise ProofError("unsigned heartbeat in bundle")
            signers.add(fp)
            head, count = env.get("payload_sha256"), count + 1
    claim = load_json(d / "claim.json") if (d / "claim.json").is_file() else {}
    run_id = (claim.get("payload") or {}).get("run_id") if isinstance(claim, dict) else None
    if not run_id:
        raise ProofError("bundle has no claim.json run_id")
    return {"_type": STATEMENT_TYPE, "schema": SCHEMA, "run_id": run_id, "run_signers": sorted(signers),
            "heartbeat_head": head, "heartbeat_count": count,
            "files": sorted(files, key=lambda f: f["path"])}


def statement_bytes(st: dict) -> bytes:
    return canonical_bytes(st)


def bundle_digest(st: dict) -> str:
    return sha256_bytes(statement_bytes(st))


def write_statement(run_dir: Path | str, out: Path | str | None = None) -> tuple[Path, str]:
    st = bundle_statement(run_dir)
    p = Path(out) if out else Path(run_dir) / STATEMENT_FILE
    p.write_bytes(statement_bytes(st))
    return p, bundle_digest(st)


def build_scope(rep: dict, endorse_claims, qualification: str) -> dict:
    """Explicit scope: custody + the named claims only. Every named claim must PASS right now; every
    other claim is listed as NOT endorsed with its current status. An endorsement never upgrades a claim."""
    claims = list(endorse_claims or ())
    bad = [c for c in claims if c not in CLAIM_NAMES]
    if bad or len(set(claims)) != len(claims):
        raise ProofError(f"refusing to endorse: unknown or duplicate claim names {bad or claims}")
    status = {k: rep["claims"][k]["status"] for k in CLAIM_NAMES}
    failing = [c for c in claims if status[c] != "PASS"]
    if failing:
        raise ProofError(f"refusing to endorse: claims not PASS at endorsement time: {failing}")
    if not isinstance(qualification, str) or not qualification.strip():
        raise ProofError("refusing to endorse: qualification statement required (e.g. NOT QUALIFIED per AQ-16 ...)")
    return {"custody": True, "integrity": "PASS", "endorsed_claims": sorted(claims),
            "not_endorsed": sorted(c for c in CLAIM_NAMES if c not in claims),
            "claim_status_at_endorsement": status, "verdict_at_endorsement": rep["verdict"],
            "qualification": qualification.strip()}


def check_scope(scope) -> str | None:
    """Return a problem string if a scope block is malformed, else None."""
    if not isinstance(scope, dict) or scope.get("custody") is not True:
        return "scope malformed or custody not asserted"
    ec, ne = scope.get("endorsed_claims"), scope.get("not_endorsed")
    st = scope.get("claim_status_at_endorsement")
    if not (isinstance(ec, list) and isinstance(ne, list) and isinstance(st, dict)):
        return "scope lists malformed"
    if set(ec) & set(ne) or not set(ec) | set(ne) <= set(CLAIM_NAMES):
        return "scope claim names invalid or overlapping"
    if any(st.get(c) != "PASS" for c in ec):
        return "scope endorses a claim that was not PASS at endorsement"
    if not isinstance(scope.get("qualification"), str) or not scope["qualification"].strip():
        return "scope qualification missing"
    return None


def endorse(run_dir: Path | str, org: Signer, accept_retroactive: bool = False, note: str = "",
            endorse_claims=(), qualification: str = "", dry_run: bool = False) -> dict:
    """Endorse a finished bundle with the org key. Refuses unless integrity PASSes (single run signer,
    unanchored) and every claim named in `endorse_claims` PASSes at endorsement time. Writes
    proofrun_bundle.json and endorsement.json. With dry_run=True returns the exact payload and
    writes/signs nothing (org may be None)."""
    from .verify import verify
    rep = verify(run_dir, accept_retroactive=accept_retroactive)
    if rep["integrity"]["status"] != "PASS":
        raise ProofError("refusing to endorse: integrity does not PASS")
    scope = build_scope(rep, endorse_claims, qualification)
    st = bundle_statement(run_dir)
    if len(st["run_signers"]) != 1:
        raise ProofError("refusing to endorse: bundle must have exactly one run signer")
    if org is not None and org.full_fingerprint in st["run_signers"]:
        raise ProofError("refusing to endorse: org key must not also be the run signer")
    payload = {"scheme": ENDORSEMENT_SCHEME, "run_id": st["run_id"], "bundle_sha256": bundle_digest(st),
               "run_signers": st["run_signers"], "endorser_role": org.role if org else "org-endorser",
               "endorsed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "integrity_at_endorsement": "PASS", "retroactive": bool(accept_retroactive),
               "scope": scope, "note": note}
    if dry_run:
        return {"dry_run": True, "payload": payload, "statement": st}
    if org is None:
        raise ProofError("org signer required")
    sp, dig = write_statement(run_dir)
    if dig != payload["bundle_sha256"]:
        raise ProofError("bundle changed during endorsement")
    env = org.envelope("endorsement", payload)
    write_json(Path(run_dir) / ENDORSEMENT_FILE, env)
    return {"bundle_sha256": dig, "run_id": st["run_id"], "endorser": org.full_fingerprint, "scope": scope}


# ---- Sigstore keyless (GitHub OIDC) -----------------------------------------------------------

def parse_identity(identity: str) -> dict:
    m = IDENTITY_RE.match(identity or "")
    if not m:
        raise ProofError("trusted identity must be an exact workflow SAN "
                         "https://github.com/<owner>/<repo>/.github/workflows/<file>.yml@refs/(heads|tags)/<ref>")
    owner, repo, wf, ref = m.groups()
    return {"owner": owner, "repo": f"{owner}/{repo}", "workflow": f"{owner}/{repo}/{wf}", "ref": ref}


def gh_command(artifact: str, bundle: str, identity: str, issuer: str) -> list[str]:
    i = parse_identity(identity)
    return ["gh", "attestation", "verify", artifact, "--bundle", bundle, "--repo", i["repo"],
            "--predicate-type", PREDICATE_TYPE, "--cert-identity", identity, "--cert-oidc-issuer", issuer,
            "--source-ref", i["ref"], "--deny-self-hosted-runners", "--format", "json"]


def _default_runner(cmd: list[str]) -> tuple[int, str, str]:
    if shutil.which(cmd[0]) is None:
        return 127, "", f"{cmd[0]} not found"
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return r.returncode, r.stdout, r.stderr


def check_sigstore(run_dir: Path | str, st: dict, identity: str, issuer: str = GITHUB_ISSUER,
                   runner: Callable[[list[str]], tuple[int, str, str]] | None = None) -> tuple[bool, str]:
    """Verify endorsement.sigstore.json against the RECOMPUTED statement bytes. Returns (ok, detail).
    Fails closed: missing bundle, gh missing, nonzero exit, unparsable output, or any field mismatch."""
    bpath = Path(run_dir) / SIGSTORE_FILE
    if not bpath.is_file():
        return False, f"{SIGSTORE_FILE} missing (fail closed)"
    try:
        parse_identity(identity)
    except ProofError as e:
        return False, str(e)
    dig = bundle_digest(st)
    with tempfile.TemporaryDirectory(prefix="proofrun-sig-") as td:
        art = Path(td) / STATEMENT_FILE
        art.write_bytes(statement_bytes(st))
        try:
            code, out, err = (runner or _default_runner)(gh_command(str(art), str(bpath), identity, issuer))
        except Exception as e:
            return False, f"gh attestation verify could not run ({e.__class__.__name__})"
    if code != 0:
        return False, f"gh attestation verify exit {code}: {(err or out).strip()[:200]}"
    try:
        results = json.loads(out)
    except Exception:
        return False, "gh output is not JSON"
    if not isinstance(results, list) or not results:
        return False, "gh returned no verified attestations"
    for r in results:
        vr = (r or {}).get("verificationResult") or {}
        cert = (vr.get("signature") or {}).get("certificate") or {}
        stmt = vr.get("statement") or {}
        subj = [((s or {}).get("digest") or {}).get("sha256") for s in stmt.get("subject") or []]
        pred = stmt.get("predicate") or {}
        if (cert.get("subjectAlternativeName") == identity and cert.get("issuer") == issuer
                and stmt.get("predicateType") == PREDICATE_TYPE and dig in subj
                and pred.get("bundle_sha256") == dig and pred.get("run_id") == st["run_id"]):
            return True, f"pinned --trusted-identity matched; bundle {dig[:16]}"
    return False, "no attestation matched identity, issuer, predicate type and bundle digest"
