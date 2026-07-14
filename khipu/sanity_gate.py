#!/usr/bin/env python3
# SZL-Khipu-1.5B-BrainNavigator -- pre-eval training-set sanity gate.
# SPDX-License-Identifier: Apache-2.0
# (c) 2026 Lutar, Stephen P. - SZL Holdings
#
# Runs the freshly-rebirthed model against its OWN training prompts
# (train.jsonl navigate plans + train.abstain.jsonl abstain plans) and requires
# it to reproduce them: every navigate row must be a schema-valid plan that
# routes to the SAME handle the reference cited, and every abstain row must
# ABSTAIN (decision=ABSTAIN, zero citations). If the model cannot even reproduce
# its training set it is undertrained -- this ABORTS (non-zero exit) so the
# caller NEVER burns the held-out eval or signs a misleading receipt over an
# undertrained model.
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

SCHEMA_FILE = os.path.join(HERE, "khipu.schema.json")
TRAINING_RECEIPT = os.path.join(HERE, "training_receipt.signed.json")
TRAIN_NAVIGATE = "train.jsonl"
TRAIN_ABSTAIN = "train.abstain.jsonl"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")


def served_model() -> str:
    """Use the servedModel the training receipt recorded, so the gate checks the
    exact model eval will. Fall back to the pinned Ollama name."""
    try:
        with open(TRAINING_RECEIPT, "r", encoding="utf-8") as f:
            return json.load(f)["payload"]["servedModel"]
    except Exception:  # noqa: BLE001 -- receipt absent/malformed -> use default
        return "khipu"


def load_jsonl(name):
    rows = []
    with open(os.path.join(HERE, name), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def prompt_messages(row):
    """System + user only -- the assistant turn is the plan we test for."""
    return [m for m in row["messages"] if m["role"] in ("system", "user")]


def offered_ids(row) -> set:
    """The candidate handle ids the model was actually offered in the prompt --
    ground truth, so a model that fabricates a candidate is still caught."""
    user = next(m for m in row["messages"] if m["role"] == "user")
    payload = json.loads(user["content"])
    return {c["nodeId"] for c in payload.get("candidates", [])}


def reference_plan(row) -> dict:
    return json.loads(row["messages"][-1]["content"])


def cross_field_ok(plan: dict, offered: set):
    """Mirror KhipuNavPlanSchema.superRefine in Python. (ok, reason)."""
    reasons = []
    steps = plan.get("steps") or []
    cited = plan.get("citedNodeIds") or []
    decision = plan.get("decision")
    abstain_reason = plan.get("abstainReason", None)
    plan_cand_ids = [c.get("nodeId") for c in (plan.get("candidates") or [])]
    plan_cand_set = set(plan_cand_ids)
    for cid in plan_cand_ids:
        if cid not in offered:
            reasons.append(f"fabricated candidate {cid}")
    for s in steps:
        if s.get("nodeId") not in plan_cand_set:
            reasons.append(f"step handle {s.get('nodeId')} not among candidates")
    for cid in cited:
        if cid not in plan_cand_set:
            reasons.append(f"cited handle {cid} not among candidates")
    cite_steps = {s.get("nodeId") for s in steps if s.get("action") == "CITE"}
    if cite_steps != set(cited):
        reasons.append("citedNodeIds != CITE-action step set")
    if decision == "ABSTAIN":
        if len(cited) != 0:
            reasons.append("ABSTAIN with citations")
        if not abstain_reason:
            reasons.append("ABSTAIN with no abstainReason")
    elif decision == "NAVIGATE":
        if len(cited) < 1:
            reasons.append("NAVIGATE with zero citations")
        if abstain_reason is not None:
            reasons.append("NAVIGATE carries an abstainReason")
    else:
        reasons.append(f"unknown decision {decision!r}")
    return (len(reasons) == 0, "; ".join(reasons))


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


def snippet(s: str, limit: int = 300) -> str:
    s = (s or "").strip().replace("\n", " ")
    if not s:
        return "<empty>"
    return s[:limit] + " ...[truncated]" if len(s) > limit else s


def main() -> None:
    model = served_model()
    with open(SCHEMA_FILE, "r", encoding="utf-8") as f:
        schema = json.load(f)
    validator = validator_for(schema)(schema)

    navigate = load_jsonl(TRAIN_NAVIGATE)
    abstain = load_jsonl(TRAIN_ABSTAIN)

    first_fail_nav = None
    first_fail_abstain = None

    # --- Navigate: must be a schema-valid plan that routes to the right handle.
    nav_ok = 0
    nav_over_abstained = 0
    for i, row in enumerate(navigate, 1):
        offered = offered_ids(row)
        ref_cited = set(reference_plan(row).get("citedNodeIds") or [])
        out = ollama_chat(model, prompt_messages(row))
        try:
            plan = json.loads(out)
            validator.validate(plan)
        except Exception as e:  # noqa: BLE001 -- any failure is non-conforming
            if first_fail_nav is None:
                first_fail_nav = out
            print(f"[sanity] train navigate {i}/{len(navigate)} INVALID: {str(e).splitlines()[0]}")
            continue
        if plan.get("decision") == "ABSTAIN":
            nav_over_abstained += 1
            if first_fail_nav is None:
                first_fail_nav = out
            print(f"[sanity] train navigate {i}/{len(navigate)} OVER-ABSTAINED (should NAVIGATE)")
            continue
        ok, reason = cross_field_ok(plan, offered)
        if ok and set(plan.get("citedNodeIds") or []) == ref_cited:
            nav_ok += 1
        else:
            if first_fail_nav is None:
                first_fail_nav = out
            print(f"[sanity] train navigate {i}/{len(navigate)} MISROUTED/{reason or 'wrong handle'}")

    # --- Abstain: must be a valid plan with decision=ABSTAIN, zero citations. --
    abstain_ok = 0
    for i, row in enumerate(abstain, 1):
        offered = offered_ids(row)
        out = ollama_chat(model, prompt_messages(row))
        try:
            plan = json.loads(out)
            validator.validate(plan)
            ok, reason = cross_field_ok(plan, offered)
        except Exception as e:  # noqa: BLE001
            ok, reason, plan = False, str(e).splitlines()[0], {}
        if ok and plan.get("decision") == "ABSTAIN":
            abstain_ok += 1
        else:
            if first_fail_abstain is None:
                first_fail_abstain = out
            print(f"[sanity] train abstain {i}/{len(abstain)} did NOT abstain cleanly ({reason})")

    print(
        f"[sanity] train-set reproduction: navigate {nav_ok}/{len(navigate)} | "
        f"abstain {abstain_ok}/{len(abstain)}"
    )

    if nav_ok == len(navigate) and abstain_ok == len(abstain) and nav_over_abstained == 0:
        print("[sanity] PASS -- model reproduces its training set; proceeding to held-out eval.")
        return

    lines = ["[sanity] FAIL -- undertrained; NOT proceeding to eval (no receipt signed)."]
    if nav_over_abstained > 0:
        lines.append(
            "  over-abstention on navigate rows -> LOWER ABSTAIN_OVERSAMPLE in train_khipu.py."
        )
    if abstain_ok < len(abstain):
        lines.append(
            "  abstentions under-learned -> RAISE epochs (and/or ABSTAIN_OVERSAMPLE)."
        )
    if nav_ok < len(navigate) and nav_over_abstained == 0:
        lines.append(
            "  navigate plans non-conforming/misrouted -> RAISE epochs (routing not memorized yet)."
        )
    if first_fail_nav is not None:
        lines.append("  first non-conforming NAVIGATE output: " + snippet(first_fail_nav))
    if first_fail_abstain is not None:
        lines.append("  first non-abstaining ABSTAIN output: " + snippet(first_fail_abstain))
    lines.append("  Copy this whole block to Alloy so it can retune before you re-run.")
    raise SystemExit("\n".join(lines))


if __name__ == "__main__":
    main()
