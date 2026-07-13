#!/usr/bin/env python3
# SZL-Forge-1.5B-ReceiptAgent -- pre-eval training-set sanity gate.
# SPDX-License-Identifier: Apache-2.0
# (c) 2026 Lutar, Stephen P. - SZL Holdings
#
# Runs the freshly-rebirthed model against its OWN training prompts
# (train.jsonl drafts + train.refusals.jsonl refusals) and requires it to
# reproduce them: every draft must be schema-valid JSON, every refusal must
# start with "REFUSE". If the model cannot even reproduce its training set it is
# undertrained -- this ABORTS (non-zero exit) so the caller NEVER burns the
# held-out eval or signs a misleading receipt over an undertrained model.
#
# BINDING honesty doctrine:
# - Reads ONLY the training curriculum + the output schema. It NEVER reads the
#   held-out eval.jsonl / adversarial.jsonl and NEVER writes a receipt.
# - It changes nothing the backbone verifier hashes; it is a build-quality gate,
#   NOT evidence. A PASS here does NOT imply a good eval -- the held-out, signed
#   eval remains the real test. This only stops obviously-undertrained runs from
#   proceeding. Nothing here upgrades Lambda.
import json
import os
import sys
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))

try:
    from jsonschema.validators import validator_for
except ImportError:
    sys.stderr.write("[sanity] missing dependency: pip install jsonschema\n")
    raise

SCHEMA_FILE = os.path.join(HERE, "receiptagent.schema.json")
TRAINING_RECEIPT = os.path.join(HERE, "training_receipt.signed.json")
TRAIN_DRAFTS = "train.jsonl"
TRAIN_REFUSALS = "train.refusals.jsonl"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")


def served_model() -> str:
    """Use the servedModel the training receipt recorded, so the gate checks the
    exact model eval will. Fall back to the pinned Ollama name."""
    try:
        with open(TRAINING_RECEIPT, "r", encoding="utf-8") as f:
            return json.load(f)["payload"]["servedModel"]
    except Exception:  # noqa: BLE001 -- receipt absent/malformed -> use default
        return "receiptagent"


def load_jsonl(name):
    rows = []
    with open(os.path.join(HERE, name), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def prompt_messages(row):
    """System + user only -- the assistant turn is the answer we test for."""
    return [m for m in row["messages"] if m["role"] in ("system", "user")]


def ollama_chat(model, messages) -> str:
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0},
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise SystemExit(
            f"[sanity] cannot reach Ollama at {OLLAMA_URL} ({e}). "
            "Is 'ollama serve' running and the model imported?"
        )
    return data.get("message", {}).get("content", "")


def is_refuse(out: str) -> bool:
    return out.strip().upper().startswith("REFUSE")


def main() -> None:
    model = served_model()
    with open(SCHEMA_FILE, "r", encoding="utf-8") as f:
        schema = json.load(f)
    validator = validator_for(schema)(schema)

    drafts = load_jsonl(TRAIN_DRAFTS)
    refusals = load_jsonl(TRAIN_REFUSALS)

    # --- Drafts: must be schema-valid JSON, and must NOT over-refuse. ---------
    draft_ok = 0
    draft_over_refused = 0
    for i, row in enumerate(drafts, 1):
        out = ollama_chat(model, prompt_messages(row))
        if is_refuse(out):
            draft_over_refused += 1
            print(f"[sanity] train draft {i}/{len(drafts)} OVER-REFUSED (should DRAFT)")
            continue
        try:
            validator.validate(json.loads(out))
            draft_ok += 1
        except Exception as e:  # noqa: BLE001 -- any failure is a non-conforming row
            print(f"[sanity] train draft {i}/{len(drafts)} INVALID: {str(e).splitlines()[0]}")

    # --- Refusals: must start with "REFUSE". ---------------------------------
    refuse_ok = 0
    for i, row in enumerate(refusals, 1):
        out = ollama_chat(model, prompt_messages(row))
        if is_refuse(out):
            refuse_ok += 1
        else:
            print(f"[sanity] train refusal {i}/{len(refusals)} did NOT refuse")

    print(
        f"[sanity] train-set reproduction: drafts {draft_ok}/{len(drafts)} | "
        f"refusals {refuse_ok}/{len(refusals)}"
    )

    if draft_ok == len(drafts) and refuse_ok == len(refusals) and draft_over_refused == 0:
        print("[sanity] PASS -- model reproduces its training set; proceeding to held-out eval.")
        return

    # FAIL: name the likely lever so the owner can tell Alloy which way to tune.
    lines = ["[sanity] FAIL -- undertrained; NOT proceeding to eval (no receipt signed)."]
    if draft_over_refused > 0:
        lines.append(
            "  over-refusal on drafts -> LOWER REFUSAL_OVERSAMPLE in train_receiptagent.py."
        )
    if refuse_ok < len(refusals):
        lines.append(
            "  refusals under-learned -> RAISE epochs (and/or REFUSAL_OVERSAMPLE)."
        )
    if draft_ok < len(drafts) and draft_over_refused == 0:
        lines.append(
            "  drafts non-conforming -> RAISE epochs (schema not memorized yet)."
        )
    lines.append("  Copy this whole block to Alloy so it can retune before you re-run.")
    raise SystemExit("\n".join(lines))


if __name__ == "__main__":
    main()
