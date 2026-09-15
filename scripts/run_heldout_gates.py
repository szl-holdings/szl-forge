#!/usr/bin/env python3
"""Re-evaluate exact published Hugging Face revisions after model publication.

This is a post-publication evidence runner. It resolves each requested Hub repo to
one immutable commit SHA, executes the repository's real held-out and refusal
harnesses against that exact revision, and emits a source-bound receipt. It does
not clear a quarantine or mutate Hugging Face state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

MODELS = {
    "chaski-r2": "SZLHOLDINGS/chaski-r2",
    "chaski-5050": "SZLHOLDINGS/chaski-5050",
    "WILLAY": "SZLHOLDINGS/WILLAY",
    "brain-navigator-r2": "SZLHOLDINGS/brain-navigator-r2",
    "khipu-r3": "SZLHOLDINGS/khipu-r3",
    "KHIPU-R2": "SZLHOLDINGS/KHIPU-R2",
    "szl-receiptagent-qwen35-0.8b-v3": "SZLHOLDINGS/szl-receiptagent-qwen35-0.8b-v3",
}

REPO_ROOT = Path(__file__).resolve().parents[1]
HELDOUT = REPO_ROOT / "eval" / "run_heldout.py"
REFUSAL = REPO_ROOT / "eval" / "run_refusal.py"
CONFIG = REPO_ROOT / "eval" / "heldout_generate.yaml"
BASELINE = REPO_ROOT / "eval" / "baselines" / "refusal.json"
OUT_DIR = REPO_ROOT / "frontier" / "evaluation" / "gate_runs"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def resolve_hub_revision(repo: str, timeout: int = 30) -> str:
    request = urllib.request.Request(
        f"https://huggingface.co/api/models/{repo}",
        headers={"User-Agent": "szl-forge-postpublish-gate/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    revision = payload.get("sha")
    if not isinstance(revision, str) or len(revision) != 40 or any(
        char not in "0123456789abcdef" for char in revision
    ):
        raise RuntimeError(f"Hub did not return an immutable 40-hex revision for {repo}")
    return revision


def run_checked(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, cwd=REPO_ROOT)


def run_gate(model_name: str, repo: str, min_pass_rate: float) -> dict[str, object]:
    revision = resolve_hub_revision(repo)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_slug = model_name.replace("/", "_")
    heldout_path = OUT_DIR / f"{model_slug}_{revision[:12]}_{stamp}_heldout.json"
    receipt_path = OUT_DIR / f"{model_slug}_{revision[:12]}_{stamp}_receipt.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    heldout_cmd = [
        sys.executable,
        str(HELDOUT),
        "--model",
        repo,
        "--revision",
        revision,
        "--config",
        str(CONFIG),
        "--out",
        str(heldout_path),
        "--assert-min-pass-rate",
        str(min_pass_rate),
    ]
    heldout_proc = run_checked(heldout_cmd)
    if heldout_proc.returncode != 0 or not heldout_path.is_file():
        receipt = {
            "schema": "szl.forge.postpublish-eval.v1",
            "model": model_name,
            "repo": repo,
            "revision": revision,
            "status": "HELDOUT_EXECUTION_FAIL",
            "heldoutStderr": heldout_proc.stderr[-4000:],
            "heldoutStdout": heldout_proc.stdout[-4000:],
            "productionAuthorization": False,
        }
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        return receipt

    heldout = json.loads(heldout_path.read_text(encoding="utf-8"))
    if heldout.get("model_revision") != revision:
        raise RuntimeError("held-out receipt revision does not match resolved Hub revision")

    refusal_cmd = [
        sys.executable,
        str(REFUSAL),
        "--model",
        repo,
        "--revision",
        revision,
        "--assert-no-regression",
        "--baseline",
        str(BASELINE),
    ]
    refusal_proc = run_checked(refusal_cmd)

    heldout_passed = bool(heldout.get("heldout_passed"))
    refusal_passed = refusal_proc.returncode == 0
    status = "PASS" if heldout_passed and refusal_passed else "FAIL"
    receipt = {
        "schema": "szl.forge.postpublish-eval.v1",
        "model": model_name,
        "repo": repo,
        "revision": revision,
        "observedAt": datetime.now(timezone.utc).isoformat(),
        "heldout": heldout,
        "heldoutReceiptSha256": sha256_bytes(heldout_path.read_bytes()),
        "refusalPassed": refusal_passed,
        "refusalStdoutSha256": sha256_bytes(refusal_proc.stdout.encode("utf-8")),
        "refusalStderr": refusal_proc.stderr[-2000:] if not refusal_passed else "",
        "status": status,
        "productionAuthorization": False,
        "publicationAuthorization": False,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS.keys(), required=True)
    parser.add_argument("--min-pass-rate", type=float, default=0.80)
    args = parser.parse_args()
    if not 0.0 <= args.min_pass_rate <= 1.0:
        parser.error("--min-pass-rate must be between 0 and 1")

    results: list[dict[str, object]] = []
    exit_code = 0
    for model_name in args.models:
        try:
            result = run_gate(model_name, MODELS[model_name], args.min_pass_rate)
        except Exception as exc:
            result = {
                "schema": "szl.forge.postpublish-eval.v1",
                "model": model_name,
                "repo": MODELS[model_name],
                "status": "PRECHECK_FAIL",
                "error": str(exc),
                "productionAuthorization": False,
            }
        results.append(result)
        print(f"{model_name}: {result['status']}")
        if result["status"] != "PASS":
            exit_code = 1

    print(json.dumps({"results": results}, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
