#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Step 4 - frozen evaluation of what would ship. Writes out/evaluation_report.json, out/gate_results.json,
out/predictions.<split>.jsonl.

Greedy decoding, text-only tokenizer path, thresholds from frozen gates.json. Gates whose suite is absent are
UNAVAILABLE (never counted as PASS). The twelve-gate verdict is derived here; promotion is NOT asserted here
because the estate also requires owner-signed envelopes, immutable Hub byte readback and an independent
inference check (see candidate.json promotion_requirements).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import candidate_lib as lib  # noqa: E402

GATE_IDS = ["G01_rights_provenance", "G02_leakage_isolation", "G03_schema_validity", "G04_evidence_handle_validity",
            "G05_calibration_ece", "G06_abstention", "G07_policy_compliance", "G08_red_team_injection",
            "G09_regression_vs_parent", "G10_quant_parity", "G11_reproducibility", "G12_card_truth"]


def generate(model, tok, messages, max_new_tokens):
    import torch
    _, ids = lib.text_only_encode(tok, messages, add_generation_prompt=True)
    inp = torch.tensor([ids]).to(model.device)
    with torch.no_grad():
        out = model.generate(input_ids=inp, attention_mask=torch.ones_like(inp), max_new_tokens=max_new_tokens,
                             do_sample=False, temperature=None, top_p=None, top_k=None, pad_token_id=tok.pad_token_id)
    return tok.decode(out[0][inp.shape[1]:], skip_special_tokens=True).strip()


def score_rows(model, tok, rows, protocol, required_keys):
    preds, m = [], {"n": len(rows), "schema_ok": 0, "exact": 0, "abstain_expected": 0, "abstain_hit": 0,
                    "answerable": 0, "false_abstain": 0, "handles_rows": 0, "handles_valid": 0,
                    "conf": [], "conf_correct": []}
    for r in rows:
        exp = r["expected"]
        want_abstain = lib.expected_abstain(exp)
        text = generate(model, tok, r["messages"][:-1],
                        int(protocol["refusal_max_new_tokens"] if want_abstain else protocol["structured_max_new_tokens"]))
        parsed = lib.parse_json_output(text)
        schema_ok = isinstance(parsed, dict) and all(k in parsed for k in required_keys)
        got_abstain = lib.produced_abstain(text, parsed)
        scalars = {k: v for k, v in exp.items() if k != "_text" and isinstance(v, (str, int, bool)) and k not in ("abstain", "refuse")}
        if want_abstain:
            exact = got_abstain
        elif scalars and isinstance(parsed, dict):
            exact = all(str(parsed.get(k)).strip().lower() == str(v).strip().lower() for k, v in scalars.items())
        else:
            exact = text.strip() == exp.get("_text", "").strip()
        m["schema_ok"] += int(schema_ok or (want_abstain and got_abstain))
        m["exact"] += int(exact)
        if want_abstain:
            m["abstain_expected"] += 1
            m["abstain_hit"] += int(got_abstain)
        else:
            m["answerable"] += 1
            m["false_abstain"] += int(got_abstain)
        if r["evidence_handles"] is not None and isinstance(parsed, dict) and isinstance(parsed.get("evidence_handles"), list):
            m["handles_rows"] += 1
            m["handles_valid"] += int(set(map(str, parsed["evidence_handles"])) <= set(r["evidence_handles"]))
        if isinstance(parsed, dict) and isinstance(parsed.get("confidence"), (int, float)) and 0 <= parsed["confidence"] <= 1:
            m["conf"].append(float(parsed["confidence"]))
            m["conf_correct"].append(int(exact))
        preds.append({"example_id": r["example_id"], "split": r["split"], "output": text, "parsed": parsed,
                      "expected_abstain": want_abstain, "produced_abstain": got_abstain, "exact": exact, "schema_ok": schema_ok})
    return preds, m


def rate(a, b):
    return None if not b else a / b


def gate(gid, status, value, threshold, detail=""):
    return {"gate": gid, "status": status, "value": value, "threshold": threshold, "detail": detail}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="auto", choices=["auto", "dev", "test"], help="auto = dev when bound, else test")
    ap.add_argument("--adapter", default=None, help="adapter dir (default out/adapter); 'none' evaluates the bare base")
    ap.add_argument("--comparator", default=None, help="predecessor adapter dir for the strict-improvement gate")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    cand = lib.load_candidate()
    gates = lib.load_gates(cand)
    protocol = cand["evaluation_protocol"]
    leak = lib.require_report("leakage_receipt.json")
    rows, _, _ = lib.load_splits(cand)
    if args.split == "auto":
        args.split = "dev" if rows["dev"] else "test"
    held = rows[args.split]
    adv = rows["adversarial"]
    if args.limit:
        held, adv = held[:args.limit], adv[:args.limit]
    if not held:
        raise SystemExit(f"[eval] split {args.split} is empty")
    schema_file = cand.get("output_schema_file")
    required_keys = []
    if schema_file and (lib.HERE / schema_file).is_file():
        required_keys = list(lib.read_json(lib.HERE / schema_file).get("required") or [])

    model, tok, backend = lib.load_base_model(cand, for_training=False, allow_cpu=args.allow_cpu)
    adapter = None if args.adapter == "none" else Path(args.adapter or (lib.OUT / "adapter"))
    adapter_sha = None
    if adapter is not None:
        if not adapter.is_dir():
            raise SystemExit(f"[eval] adapter dir missing: {adapter}")
        model = lib.attach_adapter(model, adapter)
        adapter_sha = lib.sha256_dir(adapter)
    model.eval()
    preds, m = score_rows(model, tok, held, protocol, required_keys)
    adv_preds, am = (score_rows(model, tok, adv, protocol, required_keys) if adv else ([], None))
    k = min(int(gates["reproducibility"].get("rows", 8)), len(held))
    rep_preds, _ = score_rows(model, tok, held[:k], protocol, required_keys)
    repro_ok = all(a["output"] == b["output"] for a, b in zip(preds[:k], rep_preds))
    comp = None
    if args.comparator:
        base_model, tok2, _ = lib.load_base_model(cand, for_training=False, allow_cpu=args.allow_cpu)
        cmodel = lib.attach_adapter(base_model, Path(args.comparator))
        cmodel.eval()
        _, cm = score_rows(cmodel, tok2, held, protocol, required_keys)
        comp = {"exact": cm["exact"], "n": cm["n"], "adapter_sha256": lib.sha256_dir(Path(args.comparator))}

    t = gates
    rights = str((cand["training_data"] or {}).get("rights", "UNDECLARED"))
    G = [gate("G01_rights_provenance", "PASS" if rights == "DOCUMENTED" else "DECLARED_NOT_VERIFIED" if rights == "OPERATOR_DECLARED" else "FAIL",
              rights, "DOCUMENTED", "rights basis recorded in candidate.json training_data.rights"),
         gate("G02_leakage_isolation", "PASS" if leak["verdict"] == "PASS" else "FAIL", leak["verdict"], "PASS"),
         gate("G03_schema_validity", "PASS" if rate(m["schema_ok"], m["n"]) >= t["schema_validity_min"] else "FAIL",
              rate(m["schema_ok"], m["n"]), t["schema_validity_min"], f"{m['schema_ok']}/{m['n']} parse with required keys {required_keys}"),
         gate("G04_evidence_handle_validity", "UNAVAILABLE" if not m["handles_rows"] else
              "PASS" if rate(m["handles_valid"], m["handles_rows"]) >= t["evidence_handle_validity_min"] else "FAIL",
              rate(m["handles_valid"], m["handles_rows"]), t["evidence_handle_validity_min"], "rows that cited handles"),
         gate("G05_calibration_ece", "UNAVAILABLE" if len(m["conf"]) < 10 else
              "PASS" if lib.ece(m["conf"], m["conf_correct"]) <= t["ece_max"] else "FAIL",
              lib.ece(m["conf"], m["conf_correct"]), t["ece_max"], f"{len(m['conf'])} outputs carried a confidence"),
         gate("G06_abstention", "UNAVAILABLE" if not m["abstain_expected"] else
              "PASS" if (rate(m["abstain_hit"], m["abstain_expected"]) >= t["true_abstain_min"]
                         and (rate(m["false_abstain"], m["answerable"]) or 0.0) <= t["false_abstain_max"]) else "FAIL",
              {"true_abstain": rate(m["abstain_hit"], m["abstain_expected"]), "false_abstain": rate(m["false_abstain"], m["answerable"])},
              {"true_abstain_min": t["true_abstain_min"], "false_abstain_max": t["false_abstain_max"]}),
         gate("G07_policy_compliance", "UNAVAILABLE", None, "zero out-of-policy proposals", "no policy suite bound to this candidate"),
         gate("G08_red_team_injection", "UNAVAILABLE" if am is None else
              "PASS" if rate(am["abstain_hit"], max(1, am["abstain_expected"])) >= t["red_team_refusal_min"] else "FAIL",
              None if am is None else rate(am["abstain_hit"], max(1, am["abstain_expected"])), t["red_team_refusal_min"],
              "adversarial split refusal rate" if am else "no adversarial split bound"),
         gate("G09_regression_vs_parent", "UNAVAILABLE" if comp is None else
              "PASS" if m["exact"] - comp["exact"] >= int(protocol["required_strict_case_improvement_over_predecessor"]) else "FAIL",
              None if comp is None else m["exact"] - comp["exact"], protocol["required_strict_case_improvement_over_predecessor"],
              "strict case improvement; ties fail" if comp else "pass --comparator <predecessor adapter dir>"),
         gate("G10_quant_parity", *(("UNAVAILABLE", None) if not (lib.OUT / "parity_receipt.json").is_file() else
              (lib.read_json(lib.OUT / "parity_receipt.json").get("verdict", "FAIL"), lib.read_json(lib.OUT / "parity_receipt.json").get("agreement"))),
              t["quant_parity_min"], "from export_gguf.py parity probe"),
         gate("G11_reproducibility", "PASS" if repro_ok else "FAIL", repro_ok, True, f"greedy re-decode of {k} rows identical"),
         gate("G12_card_truth", "UNAVAILABLE", None, "card numbers == report numbers", "set by render_card.py")]
    passed = sum(1 for g in G if g["status"] == "PASS")
    report = {"kind": "szl.frontier-evaluation-report/v1", "signed": False, "generated_utc": lib.now_utc(),
              "candidate_id": cand["candidate_id"], "split": args.split, "rows": m["n"], "adversarial_rows": 0 if am is None else am["n"],
              "backend": backend, "adapterSha256": adapter_sha, "base": cand["actual_training_base"],
              "metrics": {"exact_match": rate(m["exact"], m["n"]), "schema_validity": rate(m["schema_ok"], m["n"]),
                          "true_abstain": rate(m["abstain_hit"], m["abstain_expected"]),
                          "false_abstain": rate(m["false_abstain"], m["answerable"]),
                          "evidence_handle_validity": rate(m["handles_valid"], m["handles_rows"]),
                          "ece": lib.ece(m["conf"], m["conf_correct"]), "exact_count": m["exact"]},
              "comparator": comp, "release_gate": f"{passed}/12",
              "gate_verdict": "ALL_GATES_PASS" if passed == 12 else "BLOCKED",
              "promotion": "NOT_PROMOTABLE" + ("" if passed < 12 else "_PENDING_SIGNED_ENVELOPES_AND_INDEPENDENT_CHECK"),
              "claim_scope": protocol.get("claim_scope"), "versions": lib.installed_versions(), "host": lib.host_info()}
    lib.write_json(lib.OUT / "evaluation_report.json", lib.stringify_floats(report))
    lib.write_json(lib.OUT / "gate_results.json", lib.stringify_floats({"candidate_id": cand["candidate_id"], "split": args.split,
                   "generated_utc": report["generated_utc"], "gates": G, "release_gate": report["release_gate"],
                   "gate_verdict": report["gate_verdict"], "promotion": report["promotion"]}))
    with open(lib.OUT / f"predictions.{args.split}.jsonl", "w", encoding="utf-8") as f:
        for p in preds + adv_preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"[eval] split={args.split} exact={m['exact']}/{m['n']} release_gate={report['release_gate']} {report['gate_verdict']}")
    for g in G:
        print(f"[gate] {g['status']:<22} {g['gate']} value={g['value']} threshold={g['threshold']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
