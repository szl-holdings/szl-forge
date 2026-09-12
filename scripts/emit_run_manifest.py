#!/usr/bin/env python3
"""Assemble release/run-manifest.json — the object the final publish job asserts.

Emits exactly the schema the model-publish-gate publish job checks:
  eval.heldout_passed            (from release/heldout-eval.json)
  eval.refusal_no_regression     (from heldout-eval.json merged w/ refusal evidence)
  model_bom_sha256               (sha256 of the CycloneDX BOM file)
  signatures.manifest            (caller declaration, NOT signature verification;
                                  verify the signature over FINAL bytes outside
                                  this manifest before publication)
  conformance.all_vectors_passed (True only with --conformance-passed, set by
                                  the conformance job after run_vectors.py)
  release.kind                   (patch|minor|major; 'major' needs allow_major)

Fail-closed: exit 1 if required inputs are missing/malformed.
Stdlib only. Timestamps are UTC ISO-8601; canonical JSON (sorted keys) so the
DSSE PAE over this file is stable across runs (see conformance vectors).
"""
import argparse
import datetime
import hashlib
import json
import math
import pathlib
import re
import subprocess
import sys

MAX_JSON_BYTES = 1024 * 1024
ROOT = pathlib.Path(__file__).resolve().parents[1]
RUN_URL = re.compile(r"https://github\.com/szl-holdings/szl-forge/actions/runs/[1-9][0-9]{0,19}")


def die(msg: str) -> "SystemExit":
    print(f"::error::emit_run_manifest: {msg}", file=sys.stderr)
    raise SystemExit(1)


def load_json(p: str, label: str) -> dict:
    """Read bounded, unambiguous evidence; malformed input is never coerced."""
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    def invalid_number(_):
        raise ValueError("nonfinite number")

    def finite_number(text):
        value = float(text)
        if not math.isfinite(value):
            raise ValueError("nonfinite number")
        return value

    try:
        with pathlib.Path(p).open("rb") as stream:
            raw = stream.read(MAX_JSON_BYTES + 1)
        if not raw or len(raw) > MAX_JSON_BYTES:
            die(f"{label}: JSON_SIZE")
        result = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                            parse_constant=invalid_number, parse_float=finite_number)
    except (OSError, UnicodeError, ValueError, RecursionError):
        die(f"{label}: INVALID_OR_UNREADABLE_JSON")
    if type(result) is not dict:
        die(f"{label}: JSON_OBJECT_REQUIRED")
    return result


def validate_evaluation(evidence: dict) -> None:
    """Keep true/false as actual JSON booleans, never Python truthiness."""
    for field in ("heldout_passed", "refusal_no_regression"):
        if type(evidence.get(field)) is not bool:
            die(f"held-out eval result: {field} MUST_BE_BOOLEAN")
    rate = evidence.get("pass_rate")
    # bool is a subclass of int; exact types intentionally exclude it.
    if type(rate) not in (int, float) or not 0 <= rate <= 1:
        die("held-out eval result: PASS_RATE_OUT_OF_RANGE")
    if type(rate) is float and not math.isfinite(rate):
        die("held-out eval result: PASS_RATE_NOT_FINITE")
    # The existing evaluation job owns thresholds and refusal baselines.
    # This assembler cannot infer qualification from a rate or change that policy.


def validate_subject(git_sha: str, workflow_run: str, previous: str | None) -> str:
    """Bind the declaration to clean checked-out source, not a shortened label.

    A syntactically valid workflow URL is not proof of a hosted run. Independent
    run, signature, conformance and publication checks remain required.
    """
    if re.fullmatch(r"[0-9a-f]{40}", git_sha) is None:
        die("source: EXACT_LOWERCASE_SHA40_REQUIRED")
    if RUN_URL.fullmatch(workflow_run) is None:
        die("source: CANONICAL_FORGE_WORKFLOW_RUN_REQUIRED")
    previous = "genesis" if previous is None else previous
    if previous != "genesis" and re.fullmatch(r"[0-9a-f]{64}", previous) is None:
        die("chain: GENESIS_OR_EXACT_SHA256_REQUIRED")
    try:
        current = subprocess.check_output(
            ["git", "rev-parse", "--verify", "HEAD"], cwd=ROOT,
            text=True, stderr=subprocess.DEVNULL, timeout=10,
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
            text=True, stderr=subprocess.DEVNULL, timeout=10,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        die("source: CHECKOUT_UNAVAILABLE")
    if current != git_sha:
        die("source: CHECKOUT_REVISION_MISMATCH")
    if dirty:
        die("source: TRACKED_CHECKOUT_DIRTY")
    return previous


def sha256_file(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--eval", dest="eval_json", required=True, help="release/heldout-eval.json")
    ap.add_argument("--bom", required=True, help="bom/model-bom.cdx.json")
    ap.add_argument("--git-sha", required=True)
    ap.add_argument("--workflow-run", required=True, help="URL of the GH Actions run (provenance)")
    ap.add_argument("--release-kind", default="patch", choices=["patch", "minor", "major"])
    ap.add_argument("--signed-manifest", action="store_true",
                    help="Legacy caller declaration only; does not verify or preserve a signature.")
    ap.add_argument("--conformance-passed", action="store_true",
                    help="Set ONLY by the conformance job after run_vectors --require-all-pass.")
    ap.add_argument("--prev-hash", default=None,
                    help="sha256 of the previous run manifest (receipt-chain linkage); "
                         "'genesis' for the first publication of this model line.")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    md = pathlib.Path(a.model_dir)
    if not md.is_dir():
        die(f"model dir not found: {md}")
    ev = load_json(a.eval_json, "held-out eval result")
    validate_evaluation(ev)
    previous = validate_subject(a.git_sha, a.workflow_run, a.prev_hash)
    bp = pathlib.Path(a.bom)
    if not bp.is_file():
        die(f"BOM not found: {bp}")
    bom_sha = sha256_file(bp)
    git_sha = a.git_sha

    manifest = {
        "schema": "szl.run-manifest/v1",
        "issued_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "subject": {
            "model_dir": str(md),
            "git_sha": git_sha,
            "workflow_run": a.workflow_run,
        },
        "eval": {
            "pass_rate": ev["pass_rate"],
            "heldout_passed": ev["heldout_passed"],
            "refusal_no_regression": ev["refusal_no_regression"],
        },
        "model_bom_sha256": bom_sha,
        "signatures": {"manifest": bool(a.signed_manifest)},
        "conformance": {"all_vectors_passed": bool(a.conformance_passed)},
        "release": {"kind": a.release_kind},
        "chain": {"prev_hash": previous},
    }
    op = pathlib.Path(a.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(f"wrote {op} (bom sha256 {bom_sha[:16]}..., kind={a.release_kind})")
    if not a.signed_manifest:
        print("::notice::signatures.manifest=false — verify a signature over final bytes before publication")
    return 0


if __name__ == "__main__":
    sys.exit(main())
