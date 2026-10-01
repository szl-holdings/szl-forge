#!/usr/bin/env python3
"""Stage and validate khipu-abstain challenger receipts from the laptop runbook.

The GPU runbook (SZL-KHIPU-DPO-RUNBOOK.ps1) leaves three artifacts on the owner's
machine: <challenger>/dpo_receipt.json, <challenger>/eval/eval_receipt.json (+ the
transcripts it hashes) and <cards>/cards_receipt.json. Nothing about them lands in
the repository unless a human copies files by hand, and hand-copied receipts cannot
be trusted to agree with each other.

This tool closes that gap without touching any lifecycle gate:

  stage     read a challenger dir (+ optional cards dir), check every cross-receipt
            invariant, copy the receipts into <out>/receipts/khipu-abstain/<run_id>/,
            and write MANIFEST.json (sha256 of every staged file) + STATUS.md (the
            qualification state block derived from the receipts, never edited).
  validate  re-check a staged bundle: manifest hashes and the same invariants.

Both commands fail closed (exit 2) on any inconsistency. The tool never sets
publication_eligible, never derives PROMOTABLE, never uploads, and never needs a
token. A negative receipt (FAIL-CLOSED dpo or eval) is staged as INCOMPLETE evidence
rather than discarded; a receipt that claims publication eligibility is refused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import shutil
import sys
from datetime import datetime, timezone

SCHEMA = "szl.omen-pipeline/v2"
CHALLENGER_ID = "dpo-khipu-r4-challenger"
HELDOUT_FILES = ("adversarial.jsonl", "eval.jsonl")
HEX64 = re.compile(r"^[0-9a-f]{64}$")

# Pipeline digests known to this repository. A receipt produced by other bytes is not
# rejected (pins move), but STATUS.md says so explicitly.
KNOWN_PIPELINE_SHA256 = {
    "6a4b893b8d5141e73e2d1e91b0cc4f84c1ceb0bef5ea9230cc4f4288a1440a15": "szl-forge huggingface@c6e60fee (pipeline v2)",
    "31d5d1bb33f81d9b4ab20f9ba9054741bec8183e45788a99f0e77fe50f58d1ca": "szl-forge huggingface@6c35dbca (pipeline v2 + owner-authorized cards)",
    "c2361df7ab79315fd5934b839116a7ad3ad612329f182c5069c0ff6bc947bdff": "szl-forge huggingface@8cdb1ac3 (#467 memory ladder)",
    "ef3a9049ca8ebd9f48cfecb1c039792360d220cc101bd638f2e563ed84432da2": "szl-forge huggingface@3dee2933 (#469 pure-bf16 lane; produced bundle 20261001-a989dd523998)",
    "2e242ac68e8fe925d3cbb9c77feb9ffc908bbfc1c8f0ed841cfeeafc2bb56316": "szl-forge huggingface (DPO profiles C2/C3; this commit)",
}


class Invalid(Exception):
    """A receipt bundle that must not be staged or trusted."""


def sha256_of(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: pathlib.Path) -> dict:
    if not path.is_file():
        raise Invalid(f"missing file: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise Invalid(f"{path.name} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise Invalid(f"{path.name} must be a JSON object")
    return data


def require(cond: bool, message: str, problems: list[str]) -> None:
    if not cond:
        problems.append(message)


# ---------------------------------------------------------------- invariants

def check_common(receipt: dict, label: str, problems: list[str]) -> None:
    require(receipt.get("schema") == SCHEMA, f"{label}: schema must be {SCHEMA!r}, got {receipt.get('schema')!r}", problems)
    require(receipt.get("publication_eligible") is False, f"{label}: publication_eligible must be false (this tool never stages a receipt that claims eligibility)", problems)
    require(receipt.get("autonomy_eligible") is False, f"{label}: autonomy_eligible must be false", problems)
    require(receipt.get("promotion") == "NOT_PROMOTABLE", f"{label}: promotion must be NOT_PROMOTABLE; promotion is derived by the canonical Forge gate, never by a receipt", problems)
    require(bool(HEX64.match(str(receipt.get("pipeline_sha256", "")))), f"{label}: pipeline_sha256 must be 64 hex", problems)
    require(isinstance(receipt.get("generated_at"), str) and receipt["generated_at"], f"{label}: generated_at missing", problems)


def check_dpo(dpo: dict, problems: list[str]) -> None:
    require(dpo.get("stage") == "dpo", "dpo: stage must be 'dpo'", problems)
    status = dpo.get("status")
    if status == "FAIL-CLOSED":
        require(isinstance(dpo.get("reason"), str) and dpo["reason"], "dpo: FAIL-CLOSED receipt needs a reason", problems)
        require(bool(HEX64.match(str(dpo.get("pipeline_sha256", "")))), "dpo: pipeline_sha256 must be 64 hex", problems)
        return
    require(status == "TRAINED_CHALLENGER", f"dpo: status must be TRAINED_CHALLENGER or FAIL-CLOSED, got {status!r}", problems)
    check_common(dpo, "dpo", problems)
    require(dpo.get("artifact") == CHALLENGER_ID, f"dpo: artifact must be {CHALLENGER_ID!r}", problems)
    require(dpo.get("evaluation") == "NOT_RUN", "dpo: evaluation must be NOT_RUN inside the training receipt", problems)
    require("NOT AN EVAL" in str(dpo.get("train_loss_label", "")), "dpo: train_loss_label must say NOT AN EVAL", problems)
    hw = dpo.get("hardware") or {}
    require(bool(hw.get("cuda")) and bool(hw.get("gpu")), "dpo: hardware.cuda and hardware.gpu must be recorded (CPU lanes are forbidden)", problems)
    shas = dpo.get("adapter_sha256") or {}
    require(isinstance(shas, dict) and shas and all(HEX64.match(str(v)) for v in shas.values()), "dpo: adapter_sha256 must map adapter files to 64-hex digests", problems)
    data = dpo.get("data") or {}
    require(list(data.get("heldout_files_sealed") or []) == list(HELDOUT_FILES), f"dpo: heldout_files_sealed must be {list(HELDOUT_FILES)}", problems)
    require(data.get("disjoint_prompts_verified") is True, "dpo: disjoint_prompts_verified must be true", problems)
    digests = data.get("digests_sha256") or {}
    for f in HELDOUT_FILES:
        require(bool(HEX64.match(str(digests.get(f, "")))), f"dpo: data.digests_sha256[{f}] must be 64 hex", problems)
    counts = data.get("counts") or {}
    for f in HELDOUT_FILES:
        require(isinstance(counts.get(f), int) and counts[f] > 0, f"dpo: data.counts[{f}] must be a positive integer", problems)


def check_eval(ev: dict, dpo: dict, eval_dir: pathlib.Path, problems: list[str]) -> None:
    require(ev.get("stage") == "eval-abstain", "eval: stage must be 'eval-abstain'", problems)
    status = ev.get("status")
    if status == "FAIL-CLOSED":
        require(isinstance(ev.get("reason"), str) and ev["reason"], "eval: FAIL-CLOSED receipt needs a reason", problems)
        return
    require(status == "MEASURED", f"eval: status must be MEASURED or FAIL-CLOSED, got {status!r}", problems)
    check_common(ev, "eval", problems)
    require(ev.get("artifact") == CHALLENGER_ID, f"eval: artifact must be {CHALLENGER_ID!r}", problems)
    if dpo.get("status") != "TRAINED_CHALLENGER":
        problems.append("eval: MEASURED evaluation cannot accompany a dpo receipt that is not TRAINED_CHALLENGER")
        return
    require(ev.get("challenger_adapter_sha256") == dpo.get("adapter_sha256"), "eval: challenger_adapter_sha256 must equal dpo.adapter_sha256 (the evaluated bytes must be the trained bytes)", problems)
    require(ev.get("policy") == dpo.get("policy"), "eval: policy block must equal the dpo policy block", problems)
    require(ev.get("pipeline_sha256") == dpo.get("pipeline_sha256"), "eval: pipeline_sha256 must equal the dpo pipeline_sha256 (same tool bytes for both stages)", problems)
    edata, ddata = ev.get("data") or {}, dpo.get("data") or {}
    require(edata.get("revision") == ddata.get("revision"), "eval: data.revision must equal dpo data.revision", problems)
    require(edata.get("disjoint_prompts_verified") is True, "eval: disjoint_prompts_verified must be true", problems)
    hd = edata.get("heldout_digests_sha256") or {}
    dd = ddata.get("digests_sha256") or {}
    for f in HELDOUT_FILES:
        require(hd.get(f) == dd.get(f), f"eval: heldout digest for {f} must equal the digest sealed at training time", problems)
    counts = edata.get("counts") or {}
    require(counts == ddata.get("counts"), "eval: data.counts must equal dpo data.counts", problems)
    results = ev.get("results") or {}
    total_rows = 0
    for f in HELDOUT_FILES:
        block = results.get(f) or {}
        for side in ("baseline", "challenger"):
            s = block.get(side) or {}
            total, correct = s.get("total"), s.get("correct")
            require(isinstance(total, int) and isinstance(correct, int) and 0 <= correct <= total, f"eval: results[{f}][{side}] needs integer total/correct with 0 <= correct <= total", problems)
            require(total == counts.get(f), f"eval: results[{f}][{side}].total must equal data.counts[{f}]", problems)
            require(isinstance(s.get("invented_handle_rows"), int) and s["invented_handle_rows"] >= 0, f"eval: results[{f}][{side}].invented_handle_rows must be a non-negative integer", problems)
        if isinstance(block.get("baseline", {}).get("total"), int):
            total_rows += block["baseline"]["total"]
    require(isinstance(ev.get("advisory_challenger_improves"), bool), "eval: advisory_challenger_improves must be a boolean", problems)
    decode = ev.get("decode") or {}
    require(decode.get("greedy") is True, "eval: decode.greedy must be true (measured counts require deterministic decoding)", problems)
    tpath = eval_dir / "eval_transcripts.jsonl"
    if not tpath.is_file():
        problems.append("eval: eval_transcripts.jsonl missing next to eval_receipt.json")
    else:
        require(sha256_of(tpath) == ev.get("transcripts_sha256"), "eval: eval_transcripts.jsonl sha256 must equal transcripts_sha256", problems)
        lines = [ln for ln in tpath.read_text(encoding="utf-8").splitlines() if ln.strip()]
        require(len(lines) == total_rows, f"eval: transcript line count {len(lines)} must equal held-out rows {total_rows}", problems)
        for i, ln in enumerate(lines):
            try:
                row = json.loads(ln)
            except json.JSONDecodeError:
                problems.append(f"eval: transcript line {i} is not JSON")
                break
            if not ({"baseline", "challenger", "file", "index"} <= set(row)):
                problems.append(f"eval: transcript line {i} lacks baseline/challenger/file/index")
                break


def check_cards(cards: dict, problems: list[str]) -> None:
    require(cards.get("stage") == "cards", "cards: stage must be 'cards'", problems)
    require(cards.get("publication_eligible") is False, "cards: publication_eligible must be false", problems)
    require(cards.get("schema") == SCHEMA, f"cards: schema must be {SCHEMA!r}", problems)


# ---------------------------------------------------------------- state block

def state_block(dpo: dict, ev: dict | None, cards: dict | None, run_id: str) -> str:
    trained = dpo.get("status") == "TRAINED_CHALLENGER"
    training = "TRAINED_CHALLENGER" if trained else "FAILED"
    if ev is None:
        evaluation = "NOT_RUN"
    elif ev.get("status") == "MEASURED":
        evaluation = "MEASURED"
    else:
        evaluation = "FAILED"
    adapter_shas = dpo.get("adapter_sha256") or {}
    adapter_id = ", ".join(f"{k}={v[:12]}" for k, v in sorted(adapter_shas.items())) or "UNAVAILABLE"
    pipeline = str(dpo.get("pipeline_sha256", "UNAVAILABLE"))
    pin = KNOWN_PIPELINE_SHA256.get(pipeline, "NOT A PIPELINE DIGEST KNOWN TO THIS REPOSITORY (verify before trusting)")
    policy = dpo.get("policy") or {}
    ddata = dpo.get("data") or {}
    digests = ddata.get("digests_sha256") or {}
    protected = (
        f"{policy.get('adapter', 'UNAVAILABLE')} @ {policy.get('adapter_revision', 'UNAVAILABLE')} remains the artifact of record; "
        f"held-out digests adversarial={str(digests.get('adversarial.jsonl', 'UNAVAILABLE'))[:12]} eval={str(digests.get('eval.jsonl', 'UNAVAILABLE'))[:12]} unchanged"
    )
    if evaluation == "MEASURED" and ev is not None:
        r = ev["results"]
        a, n = r["adversarial.jsonl"], r["eval.jsonl"]
        measured = (
            f"advisory_challenger_improves={ev['advisory_challenger_improves']}; "
            f"ABSTAIN held-out challenger {a['challenger']['correct']}/{a['challenger']['total']} vs baseline {a['baseline']['correct']}/{a['baseline']['total']}; "
            f"NAVIGATE held-out challenger {n['challenger']['correct']}/{n['challenger']['total']} vs baseline {n['baseline']['correct']}/{n['baseline']['total']}; "
            f"invented-handle rows challenger {n['challenger']['invented_handle_rows']} vs baseline {n['baseline']['invented_handle_rows']}"
        )
        blocking = f"canonical Forge promotion gate + owner decision (not run here). Measured: {measured}"
        next_action = (
            "Owner reviews this bundle. If the advisory improves, run the canonical Forge gate on the unchanged challenger "
            "against an untouched challenge set before any publication decision; otherwise retire the challenger and keep the negative receipt."
        )
    elif evaluation == "FAILED" and ev is not None:
        blocking = f"eval-abstain FAIL-CLOSED: {ev.get('reason', 'UNAVAILABLE')}"
        next_action = "Fix the named preflight condition and rerun eval-abstain on the same challenger bytes; do not retrain to dodge the failure."
    elif not trained:
        blocking = f"dpo FAIL-CLOSED: {dpo.get('reason', 'UNAVAILABLE')}"
        next_action = "Fix the named preflight condition and rerun dpo with a fresh --out; challengers are immutable."
    else:
        blocking = "eval-abstain not run"
        next_action = "Run eval-abstain on this challenger (the only unmet gate)."
    cards_line = "UNAVAILABLE (no cards receipt in bundle)"
    if cards is not None:
        cards_line = f"cards stage recorded (apply={cards.get('apply', 'UNAVAILABLE')}, mode={cards.get('mode', 'UNAVAILABLE')}); card publication is a separate gate"
    return "\n".join([
        "```text",
        f"Artifact: {CHALLENGER_ID} (adapter {adapter_id})",
        f"Canonical source: szl-holdings/szl-forge tools/szl_omen_pipeline.py sha256 {pipeline} — {pin}",
        f"Training: {training}",
        f"Evaluation: {evaluation}",
        "Publication: UNPUBLISHED",
        "Promotion: NOT_PROMOTABLE",
        f"Blocking gate: {blocking}",
        f"Next bounded action: {next_action}",
        f"Protected state: {protected}",
        f"Completion evidence: receipts/khipu-abstain/{run_id}/MANIFEST.json",
        f"Cards: {cards_line}",
        "```",
    ])


# ---------------------------------------------------------------- commands

def collect(challenger: pathlib.Path, cards_dir: pathlib.Path | None) -> tuple[dict, dict | None, dict | None, list[tuple[pathlib.Path, str]]]:
    """Load receipts and return (dpo, eval, cards, files-to-stage) or raise Invalid."""
    problems: list[str] = []
    dpo = load_json(challenger / "dpo_receipt.json")
    check_dpo(dpo, problems)
    files: list[tuple[pathlib.Path, str]] = [(challenger / "dpo_receipt.json", "dpo_receipt.json")]
    ev = None
    eval_dir = challenger / "eval"
    if (eval_dir / "eval_receipt.json").is_file():
        ev = load_json(eval_dir / "eval_receipt.json")
        check_eval(ev, dpo, eval_dir, problems)
        files.append((eval_dir / "eval_receipt.json", "eval/eval_receipt.json"))
        if (eval_dir / "eval_transcripts.jsonl").is_file():
            files.append((eval_dir / "eval_transcripts.jsonl", "eval/eval_transcripts.jsonl"))
    cards = None
    if cards_dir is not None:
        cards = load_json(cards_dir / "cards_receipt.json")
        check_cards(cards, problems)
        files.append((cards_dir / "cards_receipt.json", "cards/cards_receipt.json"))
        for diff in sorted(cards_dir.glob("*.diff")):
            files.append((diff, f"cards/{diff.name}"))
    if problems:
        raise Invalid("\n".join(f"  - {p}" for p in problems))
    return dpo, ev, cards, files


def run_id_for(dpo: dict) -> str:
    shas = dpo.get("adapter_sha256") or {}
    first = min(shas.values())[:12] if shas else "failclosed"
    stamp = str(dpo.get("generated_at", ""))[:10].replace("-", "") or datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"{stamp}-{first}"


def cmd_stage(challenger: pathlib.Path, cards_dir: pathlib.Path | None, out: pathlib.Path) -> int:
    try:
        dpo, ev, cards, files = collect(challenger, cards_dir)
    except Invalid as exc:
        print(f"REFUSED: receipts are not internally consistent; nothing staged.\n{exc}")
        return 2
    run_id = run_id_for(dpo)
    bundle = out / "receipts" / "khipu-abstain" / run_id
    if bundle.exists():
        print(f"REFUSED: {bundle} already exists; bundles are immutable (choose a different --out or remove it deliberately).")
        return 2
    bundle.mkdir(parents=True)
    manifest = {"schema": "szl.khipu-abstain-bundle/v1", "run_id": run_id, "staged_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "complete": dpo.get("status") == "TRAINED_CHALLENGER" and ev is not None and ev.get("status") == "MEASURED",
                "publication_eligible": False, "promotion": "NOT_PROMOTABLE", "files": {}}
    for src, rel in files:
        dst = bundle / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        manifest["files"][rel] = sha256_of(dst)
    status_md = "# khipu-abstain challenger bundle " + run_id + "\n\n" + state_block(dpo, ev, cards, run_id) + "\n"
    (bundle / "STATUS.md").write_text(status_md, encoding="utf-8")
    manifest["files"]["STATUS.md"] = sha256_of(bundle / "STATUS.md")
    (bundle / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"STAGED {bundle}\n  complete={manifest['complete']} files={len(manifest['files'])}\n")
    print(status_md)
    return 0


def cmd_validate(bundle: pathlib.Path) -> int:
    problems: list[str] = []
    try:
        manifest = load_json(bundle / "MANIFEST.json")
        require(manifest.get("schema") == "szl.khipu-abstain-bundle/v1", "manifest schema mismatch", problems)
        require(manifest.get("publication_eligible") is False and manifest.get("promotion") == "NOT_PROMOTABLE", "manifest must keep publication_eligible=false and NOT_PROMOTABLE", problems)
        for rel, digest in (manifest.get("files") or {}).items():
            p = bundle / rel
            if not p.is_file():
                problems.append(f"manifest lists missing file {rel}")
            elif sha256_of(p) != digest:
                problems.append(f"{rel} sha256 differs from MANIFEST.json (bundle was edited after staging)")
        dpo = load_json(bundle / "dpo_receipt.json")
        check_dpo(dpo, problems)
        ev = None
        if (bundle / "eval" / "eval_receipt.json").is_file():
            ev = load_json(bundle / "eval" / "eval_receipt.json")
            check_eval(ev, dpo, bundle / "eval", problems)
        cards = load_json(bundle / "cards" / "cards_receipt.json") if (bundle / "cards" / "cards_receipt.json").is_file() else None
        if cards is not None:
            check_cards(cards, problems)
        if not problems:
            expected = "# khipu-abstain challenger bundle " + str(manifest.get("run_id")) + "\n\n" + state_block(dpo, ev, cards, str(manifest.get("run_id"))) + "\n"
            require((bundle / "STATUS.md").read_text(encoding="utf-8") == expected, "STATUS.md does not match the state derived from the receipts (status blocks are derived, never edited)", problems)
    except Invalid as exc:
        problems.append(str(exc))
    if problems:
        print("INVALID bundle " + str(bundle) + ":\n" + "\n".join(f"  - {p}" for p in problems))
        return 2
    print(f"VALID bundle {bundle} ({len(manifest['files'])} files, complete={manifest.get('complete')})")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stage", help="validate laptop receipts and stage an immutable bundle")
    s.add_argument("challenger", type=pathlib.Path, help="challenger dir holding dpo_receipt.json and eval/")
    s.add_argument("--cards", type=pathlib.Path, default=None, help="cards dir holding cards_receipt.json (optional)")
    s.add_argument("--out", type=pathlib.Path, required=True, help="repository root (or any dir); bundle lands under receipts/khipu-abstain/<run_id>/")
    v = sub.add_parser("validate", help="re-validate a staged bundle")
    v.add_argument("bundle", type=pathlib.Path)
    args = ap.parse_args(argv)
    if args.cmd == "stage":
        return cmd_stage(args.challenger, args.cards, args.out)
    return cmd_validate(args.bundle)


if __name__ == "__main__":
    sys.exit(main())
