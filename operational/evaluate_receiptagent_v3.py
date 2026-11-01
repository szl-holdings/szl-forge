#!/usr/bin/env python3
"""ReceiptAgent v3 sealed-test runner.

ReceiptAgent v3 (`SZLHOLDINGS/szl-receiptagent-qwen35-0.8b-v3`) has a
historical dev result (11/12). This runner preserves local unseal evidence;
it cannot prove that test data have never been read elsewhere:

  - default: reports existing local evidence without rewriting it
  - --unseal: opens `receiptagent/sealed_test.jsonl` (or --test path),
    records the unseal event in the receipt, and scores a candidate model
    endpoint against it

Opening the sealed test is a one-way governance event: once a model has
seen test results, later claims about "unseen test" are void for that
candidate. The receipt therefore records unsealed_at and the candidate id
the unseal was bound to.

Scoring expects an OpenAI-compatible local endpoint (e.g. Ollama or the
lane's serve script). No endpoint -> UNKNOWN. No test file -> UNKNOWN.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "out"
DEFAULT_TEST = ROOT / "receiptagent" / "sealed_test.jsonl"
RECEIPT = OUT / "receiptagent-v3-eval-receipt.json"


def generate(endpoint: str, model: str, prompt: str) -> str:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }).encode()
    req = urllib.request.Request(f"{endpoint.rstrip('/')}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode())
    return data["choices"][0]["message"]["content"]


def write_receipt(receipt: dict) -> None:
    """Replace a receipt atomically; persist unseal evidence before reading tests."""
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=RECEIPT.parent,
                                     prefix=RECEIPT.name + ".", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(receipt, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, RECEIPT)
    finally:
        temporary.unlink(missing_ok=True)


def load_rows(data: bytes) -> list[dict]:
    rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("test set is empty")
    identifiers = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
            raise ValueError("every test row requires a nonempty prompt")
        identity = row.get("id")
        if not isinstance(identity, str) or not identity or identity in identifiers:
            raise ValueError("test row ids must be nonempty and unique")
        identifiers.add(identity)
        constraints = [row.get("must_contain", []), row.get("must_not_contain", [])]
        if any(not isinstance(group, list) or
               any(not isinstance(token, str) or not token for token in group)
               for group in constraints) or not any(constraints):
            raise ValueError("every test row requires explicit nonempty string assertions")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unseal", action="store_true", help="open the sealed test set (one-way)")
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="szl-receiptagent-qwen35-0.8b-v3")
    args = parser.parse_args()

    prior = None
    if RECEIPT.exists():
        try:
            prior = json.loads(RECEIPT.read_text(encoding="utf-8"))
            if not isinstance(prior, dict):
                raise ValueError("receipt must be an object")
        except (OSError, ValueError) as exc:
            print(f"[eval-receiptagent] UNKNOWN (unreadable prior receipt: {exc})")
            return 2
    if not args.unseal:
        # File existence cannot establish that another evaluator never read it.
        # A status check must not erase a previously recorded one-way unseal event.
        result = prior if prior is not None and prior.get("unsealed_at") else {
            "status": "UNKNOWN", "seal_intact": None,
            "reason": "No local unseal receipt; file existence does not prove a seal.",
            "sealed_test": str(args.test), "publication_eligible": False,
        }
        print(json.dumps(result, indent=2))
        return 0
    if prior is not None and prior.get("unsealed_at"):
        print("[eval-receiptagent] REFUSED: an unseal event is already recorded; preserve this evidence")
        return 2

    receipt = {
        "kind": "szl-ops-receiptagent-v3-eval",
        "candidate": args.model,
        "dev_result_prior": "MEASURED 11/12 (see receiptagent/ lane receipts)",
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "publication_eligible": False,
        "unsealed_at": datetime.now(timezone.utc).isoformat(),
        "unseal_bound_to": args.model,
        "sealed_test": str(args.test.resolve()),
        "seal_intact": False,
        "status": "UNSEALED_EVALUATION_PENDING",
    }
    # A concurrent evaluator must never overwrite this run's evidence.
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    lock = RECEIPT.with_suffix(RECEIPT.suffix + ".lock")
    try:
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        print("[eval-receiptagent] REFUSED: another evaluation owns the receipt lock")
        return 2
    try:
        # Recheck after acquiring the lock to close the competing-unseal race.
        if RECEIPT.exists():
            current = json.loads(RECEIPT.read_text(encoding="utf-8"))
            if not isinstance(current, dict) or current.get("unsealed_at"):
                print("[eval-receiptagent] REFUSED: unseal state changed")
                return 2
        write_receipt(receipt)
        results = []
        try:
            data = args.test.read_bytes()
            receipt["test_sha256"] = hashlib.sha256(data).hexdigest()
            rows = load_rows(data)
            for row in rows:
                output = generate(args.endpoint, args.model, row["prompt"])
                if not isinstance(output, str):
                    raise ValueError("endpoint returned non-text model output")
                ok = all(token in output for token in row.get("must_contain", [])) and not any(
                    token in output for token in row.get("must_not_contain", []))
                results.append({"id": row["id"], "pass": bool(ok)})
        except Exception as exc:
            receipt.update({"status": "UNKNOWN", "reason": f"evaluation unavailable: {exc}",
                            "completed_cases": len(results)})
            write_receipt(receipt)
            print(f"[eval-receiptagent] UNKNOWN ({exc})")
            return 2
        receipt.update({"status": "MEASURED", "test_pass": sum(r["pass"] for r in results),
                        "test_total": len(results), "results": results})
        write_receipt(receipt)
        print(json.dumps(receipt, indent=2))
        return 0
    finally:
        os.close(lock_fd)
        lock.unlink()


if __name__ == "__main__":
    sys.exit(main())
