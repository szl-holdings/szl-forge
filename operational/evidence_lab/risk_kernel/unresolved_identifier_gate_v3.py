"""Evaluate a fail-closed unresolved-identifier gate on a fresh anchored suite.

The gate is defined before this run. It abstains when a question contains an
identifier-like token that is absent from the frozen corpus vocabulary. This is
an experimental local policy for evidence-bound QA, not a trained model.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import secrets
import sys
import unicodedata


HERE = Path(__file__).resolve().parent


def load_exact(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE_SOURCE = HERE / "retrieval_risk_kernel.py"
V2_SOURCE = HERE / "retrieval_risk_kernel_v2.py"
BASE = load_exact("szl_retrieval_risk_kernel_v1_for_v3", BASE_SOURCE)
V2 = load_exact("szl_retrieval_risk_kernel_v2_for_v3", V2_SOURCE)


VARIANT = "UNRESOLVED_IDENTIFIER_GATE_V3"
TOKEN = re.compile(r"[^\W_][\w-]*", re.UNICODE)


def normalized_token(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold().strip("-")


def corpus_vocabulary(payload: dict[str, object]) -> set[str]:
    documents = payload.get("documents")
    if not isinstance(documents, list):
        raise ValueError("payload documents are missing")
    vocabulary: set[str] = set()
    for document in documents:
        if not isinstance(document, dict):
            raise ValueError("invalid document record")
        for field in ("title", "text"):
            value = document.get(field)
            if not isinstance(value, str):
                raise ValueError(f"document {field} is not text")
            vocabulary.update(normalized_token(match.group(0)) for match in TOKEN.finditer(value))
    vocabulary.discard("")
    return vocabulary


def identifier_like(token: str) -> bool:
    if len(token) < 16:
        return False
    has_ascii_letter = any(character in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" for character in token)
    has_digit = any(character.isdigit() for character in token)
    internal_upper = sum(character.isupper() for character in token[1:])
    mixed_case = any(character.islower() for character in token) and internal_upper >= 2
    return has_ascii_letter and (has_digit or mixed_case)


def unresolved_identifiers(question: str, vocabulary: set[str]) -> list[str]:
    if not isinstance(question, str):
        raise ValueError("question must be text")
    unresolved = []
    seen: set[str] = set()
    for match in TOKEN.finditer(question):
        raw = match.group(0)
        normalized = normalized_token(raw)
        if identifier_like(raw) and normalized not in vocabulary and normalized not in seen:
            unresolved.append(raw)
            seen.add(normalized)
    return unresolved


def gate_decision(
    question: str,
    response: dict[str, object],
    vocabulary: set[str],
) -> dict[str, object]:
    unresolved = unresolved_identifiers(question, vocabulary)
    base_answer = response.get("status") == "ANSWER" and isinstance(response.get("answer"), str)
    selected = bool(base_answer) and not unresolved
    return {
        "status": "ANSWER" if selected else "ABSTAIN",
        "answer": response.get("answer") if selected else None,
        "reason": "BASE_POLICY_ABSTAINED"
        if not base_answer
        else ("UNRESOLVED_IDENTIFIER" if unresolved else "SOURCE_BOUND_BASE_ANSWER"),
        "unresolved_identifiers": unresolved,
    }


def summarize_gate(records: list[dict[str, object]]) -> dict[str, object]:
    false_answers = sum(record["gate"]["status"] == "ANSWER" for record in records)
    lower, upper = BASE.wilson_interval(false_answers, len(records))
    return {
        "queries": len(records),
        "false_answers": false_answers,
        "false_answer_rate": false_answers / len(records),
        "wilson_95_lower": lower,
        "wilson_95_upper": upper,
        "forced_abstentions": sum(
            record["response"]["status"] == "ANSWER"
            and record["gate"]["status"] == "ABSTAIN"
            for record in records
        ),
    }


def validate_live_gate(response, base, expected):
    """Require the live filter to match the fixed policy, without hidden answers."""
    details = response.get("identifier_guard", {})
    if (response.get("status") != expected["status"]
            or response.get("answer") != expected["answer"]
            or details.get("enabled") is not True
            or details.get("reason") != expected["reason"]
            or details.get("unresolved_identifiers") != expected["unresolved_identifiers"]
            or details.get("base_status") != base.get("status")):
        raise ValueError("Live identifier guard does not match its source-defined policy")
    if expected["status"] == "ABSTAIN":
        if response.get("evidence") is not None:
            raise ValueError("Live abstention exposes rejected evidence")
    elif response.get("evidence") != base.get("evidence"):
        raise ValueError("Live identifier guard changed accepted source evidence")


def score_positive_gate(records: list[dict[str, object]]) -> dict[str, object]:
    if not records:
        raise ValueError("positive records are empty")
    base_answers = gate_answers = 0
    base_exact = base_f1 = gate_exact = gate_f1 = 0.0
    forced = 0
    for record in records:
        response = record["response"]
        gate = record["gate"]
        assert isinstance(response, dict) and isinstance(gate, dict)
        base_prediction = response.get("answer") if response.get("status") == "ANSWER" else ""
        gate_prediction = gate.get("answer") if gate.get("status") == "ANSWER" else ""
        if not isinstance(base_prediction, str):
            base_prediction = ""
        if not isinstance(gate_prediction, str):
            gate_prediction = ""
        base_answers += bool(base_prediction)
        gate_answers += bool(gate_prediction)
        forced += bool(base_prediction) and not bool(gate_prediction)
        gold = record["gold"]
        assert isinstance(gold, list) and gold
        base_scores = [BASE._exact_f1(base_prediction, str(answer)) for answer in gold]
        gate_scores = [BASE._exact_f1(gate_prediction, str(answer)) for answer in gold]
        base_exact += max(score[0] for score in base_scores)
        base_f1 += max(score[1] for score in base_scores)
        gate_exact += max(score[0] for score in gate_scores)
        gate_f1 += max(score[1] for score in gate_scores)
    total = len(records)
    return {
        "scope": "PUBLIC_DEVELOPMENT_SET_REPLAY_NOT_FRESH_QUALITY_EVIDENCE",
        "queries": total,
        "base": {
            "coverage": base_answers / total,
            "exact_match": base_exact / total,
            "f1": base_f1 / total,
        },
        "unresolved_identifier_gate": {
            "coverage": gate_answers / total,
            "exact_match": gate_exact / total,
            "f1": gate_f1 / total,
            "forced_abstentions": forced,
        },
    }


def report_markdown(receipt: dict[str, object], receipt_sha: str) -> str:
    calibration = receipt["calibration"]
    holdout = receipt["holdout"]
    replay = receipt["positive_replay"]
    bindings = receipt["bindings"]
    assert isinstance(calibration, dict)
    assert isinstance(holdout, dict)
    assert isinstance(replay, dict)
    assert isinstance(bindings, dict)
    base_cal = calibration["base"]
    gate_cal = calibration["gate"]
    base_hold = holdout["base"]
    gate_hold = holdout["gate"]
    replay_base = replay["base"]
    replay_gate = replay["unresolved_identifier_gate"]
    outcome = (
        "MEETS_FRESH_ANCHORED_HOLDOUT_TARGET"
        if holdout["target_met"]
        else "FRESH_ANCHORED_HOLDOUT_TARGET_NOT_MET"
    )
    lines = [
        "# SZL Unresolved-Identifier Gate V3 — Fresh Local Evaluation",
        "",
        f"Outcome: `{outcome}`. The gate was fixed in source before this fresh run. It is an experimental inference policy, not training, new weights, publication, or production deployment.",
        "",
        "## Exact bindings",
        "",
        f"- Run: `{bindings['run_id']}`",
        f"- Freeze SHA-256: `{bindings['artifacts']['freeze.json']}`",
        f"- Preview SHA-256: `{bindings['preview_sha256']}`",
        f"- V3 kernel SHA-256: `{bindings['kernel_v3_sha256']}`",
        f"- Receipt SHA-256: `{receipt_sha}`",
        "",
        "## Fresh corpus-anchored unsupported queries",
        "",
        "| Split / policy | False answers | Queries | Rate | Wilson 95% upper | Forced abstentions |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Calibration / base | {base_cal['false_answers']} | {base_cal['queries']} | {base_cal['false_answer_rate']:.2%} | {base_cal['wilson_95_upper']:.2%} | — |",
        f"| Calibration / gate | {gate_cal['false_answers']} | {gate_cal['queries']} | {gate_cal['false_answer_rate']:.2%} | {gate_cal['wilson_95_upper']:.2%} | {gate_cal['forced_abstentions']} |",
        f"| Holdout / base | {base_hold['false_answers']} | {base_hold['queries']} | {base_hold['false_answer_rate']:.2%} | {base_hold['wilson_95_upper']:.2%} | — |",
        f"| Holdout / gate | {gate_hold['false_answers']} | {gate_hold['queries']} | {gate_hold['false_answer_rate']:.2%} | {gate_hold['wilson_95_upper']:.2%} | {gate_hold['forced_abstentions']} |",
        "",
        f"Holdout anchor recall@3 was {receipt['anchor_retrieval']['holdout']['anchor_recall_at_3']:.2%}. Every synthetic subject/event identifier was absent from the frozen vocabulary and every anchor phrase was source-offset/hash bound.",
        "",
        "## Public positive replay",
        "",
        "| Policy | Coverage | Exact match | F1 | Forced abstentions |",
        "|---|---:|---:|---:|---:|",
        f"| Base | {replay_base['coverage']:.2%} | {replay_base['exact_match']:.2%} | {replay_base['f1']:.2%} | — |",
        f"| Identifier gate | {replay_gate['coverage']:.2%} | {replay_gate['exact_match']:.2%} | {replay_gate['f1']:.2%} | {replay_gate['forced_abstentions']} |",
        "",
        "This public replay is only a regression signal. The operational next gate is a larger rights-reviewed set of legitimate unseen identifiers, aliases, Unicode names, and ordinary unsupported questions.",
        "",
        "## Decision boundary",
        "",
        "The source and receipt are locally operational, but the gate remains experimental. It should be integrated into a protected release only after broader false-abstention and bypass testing establishes acceptable behavior.",
        "",
        "## Explicit limits",
        "",
    ]
    lines.extend(f"- {item}" for item in receipt["limitations"])
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, object]:
    run_dir = args.run_dir.resolve()
    preview_source = args.preview_source.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    if args.calibration_count < 80 or args.holdout_count < 80:
        raise ValueError("calibration and holdout counts must each be at least 80")
    if args.positive_count <= 0 or not 0 < args.target_upper_bound < 1:
        raise ValueError("invalid positive replay count or risk target")
    seed_hex = args.seed or secrets.token_hex(32)
    seed_origin = "caller_supplied" if args.seed else "fresh_os_csprng_at_run_start"
    BASE.validate_seed(seed_hex)
    created_at = BASE.utc_now()
    hashes_before = BASE.artifact_hashes(run_dir)
    freeze = BASE.load_json(run_dir / "freeze.json")
    payload = BASE.load_json(run_dir / "payload.json")
    if not isinstance(freeze, dict) or not isinstance(payload, dict):
        raise ValueError("invalid frozen run")
    if datetime.fromisoformat(created_at) <= datetime.fromisoformat(str(freeze["created_at"])):
        raise ValueError("suite was not generated after the selected local freeze")
    vocabulary = corpus_vocabulary(payload)
    client = BASE.PreviewClient(args.base_url, args.timeout)
    status_before = client.request("/api/status")
    BASE.verify_status(status_before, run_dir, hashes_before, preview_source)
    live_guard = getattr(args, "live_guard", False)
    if live_guard and not status_before.get("capabilities", {}).get("identifier_guard"):
        raise ValueError("Preview does not expose the live identifier guard")

    def live_observation(question, response, gate):
        if not live_guard:
            return {}
        guarded = client.request("/api/query-guarded", {"question": question})
        BASE.validate_response(guarded, status_before)
        validate_live_gate(guarded, response, gate)
        return {"live_guard_response": guarded}

    cases = V2.generate_anchored_cases(
        seed_hex, args.calibration_count, args.holdout_count, payload
    )
    proof = V2.verify_anchors(cases, payload)
    positives = BASE.select_positive_cases(payload, seed_hex, args.positive_count)
    output_dir.mkdir(parents=True, exist_ok=False)
    BASE.save_new(
        output_dir / "attempt.json",
        {
            "schema_version": BASE.SCHEMA_VERSION,
            "started_at": created_at,
            "scope": BASE.SCOPE,
            "variant": VARIANT,
            "seed_sha256": hashlib.sha256(bytes.fromhex(seed_hex)).hexdigest(),
            "bindings": hashes_before,
        },
    )
    negative_records: list[dict[str, object]] = []
    positive_records: list[dict[str, object]] = []
    for index, case in enumerate(cases):
        response = client.request("/api/query", {"question": case["question"]})
        BASE.validate_response(response, status_before)
        gate = gate_decision(str(case["question"]), response, vocabulary)
        if set(gate["unresolved_identifiers"]) != {
            case["subject_nonce"],
            case["event_nonce"],
        }:
            raise RuntimeError("fresh synthetic identifiers were not detected exactly")
        negative_records.append({**case, "response": response, "gate": gate,
                                 **live_observation(case["question"], response, gate)})
        if (index + 1) % 20 == 0:
            print(f"v3 anchored queries: {index + 1}/{len(cases)}", flush=True)
    for index, case in enumerate(positives):
        response = client.request("/api/query", {"question": case["question"]})
        BASE.validate_response(response, status_before)
        gate = gate_decision(str(case["question"]), response, vocabulary)
        positive_records.append(
            {
                "id": case["id"],
                "question": case["question"],
                "gold": case["answers"],
                "response": response,
                "gate": gate,
                **live_observation(case["question"], response, gate),
            }
        )
        if (index + 1) % 10 == 0:
            print(f"v3 positive replays: {index + 1}/{len(positives)}", flush=True)
    base_thresholds = {
        float(record["response"]["threshold"])
        for record in negative_records + positive_records
    }
    if len(base_thresholds) != 1:
        raise RuntimeError("base threshold changed during evaluation")
    base_threshold = base_thresholds.pop()
    calibration_records = [
        record for record in negative_records if record["split"] == "calibration"
    ]
    holdout_records = [
        record for record in negative_records if record["split"] == "holdout"
    ]
    base_calibration = BASE.summarize_negative(
        [record["response"] for record in calibration_records], base_threshold
    )
    base_holdout = BASE.summarize_negative(
        [record["response"] for record in holdout_records], base_threshold
    )
    gate_calibration = summarize_gate(calibration_records)
    gate_holdout = summarize_gate(holdout_records)
    status_after = client.request("/api/status")
    hashes_after = BASE.artifact_hashes(run_dir)
    if status_after != status_before or hashes_after != hashes_before:
        raise RuntimeError("frozen artifacts or preview status changed during evaluation")
    receipt: dict[str, object] = {
        "schema_version": BASE.SCHEMA_VERSION,
        "status": BASE.STATUS,
        "scope": BASE.SCOPE,
        "variant": VARIANT,
        "live_guard_verified": live_guard,
        "completed_at": BASE.utc_now(),
        "suite": {
            "seed_hex": seed_hex,
            "seed_origin": seed_origin,
            "generated_after_local_freeze": True,
            "calibration_cases": args.calibration_count,
            "holdout_cases": args.holdout_count,
            "families": len(V2.TEMPLATES),
            "proof": proof,
            "corpus_vocabulary_tokens": len(vocabulary),
        },
        "gate_specification": {
            "normalization": "Unicode NFKC plus casefold",
            "identifier_min_characters": 16,
            "identifier_shape": "ASCII letter and either a digit or at least two internal uppercase characters",
            "decision": "force abstain when any identifier-like question token is absent from frozen title/text vocabulary",
            "threshold_tuning_performed": False,
        },
        "bindings": {
            "run_id": status_before["run_id"],
            "artifacts": hashes_before,
            "preview_sha256": status_before["preview_sha256"],
            "preview_assets": status_before["asset_sha256"],
            "kernel_v3_sha256": BASE.sha256_file(Path(__file__).resolve()),
            "kernel_v2_sha256": BASE.sha256_file(V2_SOURCE),
            "kernel_v1_sha256": BASE.sha256_file(BASE_SOURCE),
        },
        "base_threshold": base_threshold,
        "calibration": {"base": base_calibration, "gate": gate_calibration},
        "holdout": {
            "target_wilson_upper": args.target_upper_bound,
            "base": base_holdout,
            "gate": gate_holdout,
            "target_met": gate_holdout["wilson_95_upper"]
            <= args.target_upper_bound,
        },
        "anchor_retrieval": {
            "calibration": V2.anchor_recall(calibration_records),
            "holdout": V2.anchor_recall(holdout_records),
        },
        "positive_replay": score_positive_gate(positive_records),
        "training_performed": False,
        "new_weights_created": False,
        "publication_eligible": False,
        "limitations": [
            "This lexical gate recognizes the constructed identifier distribution. Zero errors on that family do not demonstrate general semantic safety.",
            "Wilson intervals here are nominal binomial summaries; template dependence and design overlap prevent treating them as general risk guarantees.",
            "The gate covers identifier-like single tokens; ordinary unseen names, multiword entities, aliases, homoglyphs, and paraphrases remain unmeasured.",
            "A legitimate new identifier absent from the corpus will be forced to abstain by design; broader false-abstention testing is required.",
            "The unsupported suite is synthetic and template-based, not a complete distribution of real user requests.",
            "Exact vocabulary absence proves no direct token support in this payload, not global real-world nonexistence.",
            "The public positive replay may be contaminated and is not hidden-test quality evidence.",
            "No training, paid cloud job, external publication, deployment, credential change, firewall change, or independent witnessing occurred.",
        ],
        "records": {
            "corpus_anchored_nonce": negative_records,
            "public_positive_replay": positive_records,
        },
    }
    BASE.save_new(output_dir / "receipt.json", receipt)
    receipt_sha = BASE.sha256_file(output_dir / "receipt.json")
    with (output_dir / "REPORT.md").open("xb") as handle:
        handle.write(report_markdown(receipt, receipt_sha).encode("utf-8"))
    return {
        "status": BASE.STATUS,
        "variant": VARIANT,
        "output_dir": str(output_dir),
        "receipt_sha256": receipt_sha,
        "holdout_target_met": receipt["holdout"]["target_met"],
        "positive_forced_abstentions": receipt["positive_replay"][
            "unresolved_identifier_gate"
        ]["forced_abstentions"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--preview-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8766")
    parser.add_argument("--seed")
    parser.add_argument("--calibration-count", type=int, default=80)
    parser.add_argument("--holdout-count", type=int, default=80)
    parser.add_argument("--positive-count", type=int, default=24)
    parser.add_argument("--target-upper-bound", type=float, default=0.05)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--live-guard", action="store_true", help="Compare actual guarded API responses with the fixed lexical policy")
    return parser


def main() -> int:
    try:
        result = run(build_parser().parse_args())
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
