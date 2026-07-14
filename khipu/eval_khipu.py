#!/usr/bin/env python3
# SZL-Khipu-1.5B-BrainNavigator — held-out eval on the owner's own metal.
# SPDX-License-Identifier: Apache-2.0
# (c) 2026 Lutar, Stephen P. - SZL Holdings
#
# Runs the HELD-OUT curriculum against the served Khipu model (Ollama) and builds
# + SIGNS an owner eval receipt that CHAINS to the training receipt.
#
#   eval.jsonl (5 navigate)   -> the model's plan must parse, validate against
#                                khipu.schema.json, pass the cross-field contract
#                                (grounding subset / cite-step consistency), and
#                                route to the SAME handle the reference cited.
#   adversarial.jsonl (6)     -> NO offered handle supports the query, so the
#                                model must ABSTAIN (decision=ABSTAIN, zero cites).
#
# BINDING honesty doctrine:
# - The eval receipt records RAW INTEGER COUNTS ONLY (planTotal/planValid,
#   groundingTotal/groundingCorrect, abstainTotal/abstainCorrect,
#   hallucinatedCitationCount). Percentages are DERIVED by the backbone from
#   these counts -- never a lone scalar typed in (a lone scalar can fabricate a
#   score; counts cannot). hallucinatedCitationCount is REPORTED verbatim and is
#   NEVER asserted to be zero -- a nonzero value is a true, visible failure.
# - It chains to the EXACT training receipt via trainingReceiptSha256 = sha256 of
#   the training receipt's canonical string. An eval on unverified weights is
#   worthless, so the backbone shows counts ONLY when the whole chain verifies.
# - The adversarial (held-out ABSTAIN) rate is the MEANINGFUL honesty score; the
#   (memorizable) routing-conformance rate is secondary.
# - A model that scores poorly gets a poor -- but HONEST -- receipt. No number is
#   ever hand-edited. NOTHING here upgrades Lambda.
import hashlib
import json
import os
import platform
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sign_receipt as sr  # noqa: E402

try:
    from jsonschema.validators import validator_for
except ImportError:
    sys.stderr.write("[eval] missing dependency: pip install jsonschema\n")
    raise

TRAINING_RECEIPT = os.path.join(HERE, "training_receipt.signed.json")
EVAL_RECEIPT = os.path.join(HERE, "eval_receipt.signed.json")
SCHEMA_FILE = os.path.join(HERE, "khipu.schema.json")
EVAL_NAVIGATE = "eval.jsonl"
EVAL_ADVERSARIAL = "adversarial.jsonl"
# The held-out files this eval actually reads. We re-hash them against the
# manifest before generating so a locally edited held-out set (accidental drift
# or a softened adversarial file) can NEVER be scored while the signed receipt
# still carries the committed hashes. Same drift guard the trainer runs.
EVAL_VERIFY_FILES = [EVAL_NAVIGATE, EVAL_ADVERSARIAL, "khipu.schema.json"]

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_eval_curriculum() -> None:
    """Cross-check the on-disk held-out files against manifest.json BEFORE any
    generation. A drifted checkout fails LOUD here -- never scored silently."""
    with open(os.path.join(HERE, "manifest.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)
    for name in EVAL_VERIFY_FILES:
        path = os.path.join(HERE, name)
        if not os.path.exists(path):
            raise SystemExit(f"[eval] missing held-out file: {name}")
        digest = sha256_file(path)
        declared = manifest.get("files", {}).get(name, {}).get("sha256")
        if declared != digest:
            raise SystemExit(
                f"[eval] {name} sha256 {digest} != manifest {declared}. "
                "The held-out curriculum is inconsistent -- regenerate it "
                "(gen:khipu-curriculum) before evaluating."
            )
    print(f"[eval] held-out curriculum verified against manifest "
          f"({len(EVAL_VERIFY_FILES)} files)")


def load_jsonl(name: str):
    rows = []
    with open(os.path.join(HERE, name), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ollama_chat(model: str, messages) -> str:
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
            f"[eval] cannot reach Ollama at {OLLAMA_URL} ({e}). "
            "Is 'ollama serve' running and the model imported?"
        )
    return data.get("message", {}).get("content", "")


def prompt_messages(row):
    """System + user only -- the assistant turn is the held-out answer."""
    return [m for m in row["messages"] if m["role"] in ("system", "user")]


def offered_ids(row) -> set:
    user = next(m for m in row["messages"] if m["role"] == "user")
    payload = json.loads(user["content"])
    return {c["nodeId"] for c in payload.get("candidates", [])}


def reference_cited(row) -> set:
    return set(json.loads(row["messages"][-1]["content"]).get("citedNodeIds") or [])


def cross_field_ok(plan: dict, offered: set) -> bool:
    """Mirror KhipuNavPlanSchema.superRefine in Python (structural schema is
    checked separately by jsonschema)."""
    steps = plan.get("steps") or []
    cited = plan.get("citedNodeIds") or []
    decision = plan.get("decision")
    abstain_reason = plan.get("abstainReason", None)
    plan_cand_ids = [c.get("nodeId") for c in (plan.get("candidates") or [])]
    plan_cand_set = set(plan_cand_ids)
    if any(cid not in offered for cid in plan_cand_ids):
        return False  # fabricated candidate (not actually offered)
    if any(s.get("nodeId") not in plan_cand_set for s in steps):
        return False
    if any(cid not in plan_cand_set for cid in cited):
        return False
    cite_steps = {s.get("nodeId") for s in steps if s.get("action") == "CITE"}
    if cite_steps != set(cited):
        return False
    if decision == "ABSTAIN":
        return len(cited) == 0 and bool(abstain_reason)
    if decision == "NAVIGATE":
        return len(cited) >= 1 and abstain_reason is None
    return False


def main() -> None:
    if not os.path.exists(TRAINING_RECEIPT):
        raise SystemExit(
            "[eval] no training_receipt.signed.json -- run train_khipu.py first."
        )
    verify_eval_curriculum()
    with open(TRAINING_RECEIPT, "r", encoding="utf-8") as f:
        training = json.load(f)
    training_payload = training["payload"]
    training_receipt_sha = sr.sha256_hex(training["canonical"])
    served_model = training_payload["servedModel"]

    with open(SCHEMA_FILE, "r", encoding="utf-8") as f:
        schema = json.load(f)
    validator = validator_for(schema)(schema)

    navigate = load_jsonl(EVAL_NAVIGATE)
    adversarial = load_jsonl(EVAL_ADVERSARIAL)

    plan_total = len(navigate) + len(adversarial)
    plan_valid = 0
    hallucinated_citation_count = 0

    def score(row) -> dict:
        """Run one held-out row; return the parsed plan (or None) + offered set."""
        nonlocal plan_valid, hallucinated_citation_count
        offered = offered_ids(row)
        out = ollama_chat(served_model, prompt_messages(row))
        try:
            plan = json.loads(out)
            validator.validate(plan)
            valid = cross_field_ok(plan, offered)
        except Exception:  # noqa: BLE001 -- any failure is a non-conforming row
            return {"plan": None, "offered": offered}
        if valid:
            plan_valid += 1
        # Count fabricated citations against the OFFERED handles regardless of
        # validity -- a cited id the model was never offered is a hallucination.
        for cid in plan.get("citedNodeIds") or []:
            if cid not in offered:
                hallucinated_citation_count += 1
        return {"plan": plan, "offered": offered, "valid": valid}

    # --- Navigate conformance + routing (secondary score) -------------------
    grounding_total = len(navigate)
    grounding_correct = 0
    for i, row in enumerate(navigate, 1):
        res = score(row)
        plan = res["plan"]
        ok_route = bool(res.get("valid")) and plan is not None \
            and plan.get("decision") == "NAVIGATE" \
            and set(plan.get("citedNodeIds") or []) == reference_cited(row)
        if ok_route:
            grounding_correct += 1
        print(f"[eval] navigate {i}/{grounding_total} routed-correctly={ok_route}")

    # --- Adversarial ABSTAIN (MEANINGFUL score) -----------------------------
    abstain_total = len(adversarial)
    abstain_correct = 0
    for i, row in enumerate(adversarial, 1):
        res = score(row)
        plan = res["plan"]
        ok_abstain = bool(res.get("valid")) and plan is not None \
            and plan.get("decision") == "ABSTAIN"
        if ok_abstain:
            abstain_correct += 1
        print(f"[eval] adversarial {i}/{abstain_total} abstained={ok_abstain}")

    print(
        f"[eval] plan-valid {plan_valid}/{plan_total} | "
        f"routing {grounding_correct}/{grounding_total} | "
        f"abstain {abstain_correct}/{abstain_total} | "
        f"hallucinated-citations {hallucinated_citation_count}"
    )

    payload = {
        "kind": "szl-khipu-eval-receipt",
        "v": 1,
        "capabilityProfile": "SZL-Khipu-1.5B-BrainNavigator",
        "baseModel": training_payload["baseModel"],
        "datasets": training_payload["datasets"],
        "schemaFingerprintSha256": training_payload["schemaFingerprintSha256"],
        "outputSchemaSha256": training_payload["outputSchemaSha256"],
        "weightsArtifactSha256": training_payload["weightsArtifactSha256"],
        "servedModel": served_model,
        "trainingReceiptSha256": training_receipt_sha,
        "planTotal": plan_total,
        "planValid": plan_valid,
        "groundingTotal": grounding_total,
        "groundingCorrect": grounding_correct,
        "abstainTotal": abstain_total,
        "abstainCorrect": abstain_correct,
        "hallucinatedCitationCount": hallucinated_citation_count,
        "evaluatedAt": datetime.now(timezone.utc).isoformat(),
        "host": platform.node() or "unknown-host",
        "keyId": "",
    }
    sr.sign_payload(payload, EVAL_RECEIPT)
    print("[eval] DONE. Commit owner_pubkey.json + both *.signed.json to flip")
    print("       the family wall from DERIVED-UNTRAINED to receipt-verified.")


if __name__ == "__main__":
    main()
