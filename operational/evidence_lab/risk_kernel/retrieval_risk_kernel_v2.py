"""Harder corpus-anchored nonce evaluation for retrieval-selected QA.

Each unsupported question contains a six-token phrase copied from one frozen
document so retrieval has a strong lexical target, while its subject and event
identifiers are freshly generated and proven absent from the complete payload.
Calibration and holdout anchor documents are disjoint.
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


HERE = Path(__file__).resolve().parent
BASE_SOURCE = HERE / "retrieval_risk_kernel.py"
SPEC = importlib.util.spec_from_file_location("szl_retrieval_risk_kernel_v1", BASE_SOURCE)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load the bound v1 kernel primitives")
BASE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BASE
SPEC.loader.exec_module(BASE)


VARIANT = "CORPUS_ANCHORED_NONCE_V2"
TEMPLATES = (
    "In the passage containing the phrase ‘{anchor}’, which city hosted the {event} ceremony for {subject}?",
    "The account mentioning ‘{anchor}’ is under review. Who founded {subject} before the {event} charter?",
    "Using the passage that says ‘{anchor}’, in what year did {subject} demonstrate the {event} process?",
    "The source containing ‘{anchor}’ is the retrieval anchor. Which material powered the {subject} device in the {event} protocol?",
    "Within the account that mentions ‘{anchor}’, what treaty did {subject} sign at the {event} summit?",
    "In the text containing ‘{anchor}’, which language was used for the {event} constitution of {subject}?",
    "Consult the passage that includes ‘{anchor}’. What orbit period does the {event} catalogue assign to {subject}?",
    "According to the passage containing ‘{anchor}’, which court decided the {event} dispute involving {subject}?",
)
WORD = re.compile(r"[^\W_][\w’'-]{2,}", re.UNICODE)


def _anchor(text: str, selector: int) -> tuple[str, int, int]:
    matches = list(WORD.finditer(text))
    if len(matches) < 12:
        raise ValueError("document has too few anchor tokens")
    width = 6
    start_token = selector % (len(matches) - width + 1)
    chosen = matches[start_token : start_token + width]
    phrase = " ".join(match.group(0) for match in chosen)
    return phrase, chosen[0].start(), chosen[-1].end()


def generate_anchored_cases(
    seed_hex: str,
    calibration_count: int,
    holdout_count: int,
    payload: dict[str, object],
) -> list[dict[str, object]]:
    seed = BASE.validate_seed(seed_hex)
    documents = payload.get("documents")
    if not isinstance(documents, list):
        raise ValueError("payload documents are missing")
    eligible = [
        document
        for document in documents
        if isinstance(document, dict)
        and isinstance(document.get("id"), str)
        and isinstance(document.get("text"), str)
        and len(WORD.findall(str(document["text"]))) >= 12
    ]
    eligible.sort(
        key=lambda document: hashlib.sha256(
            seed + b"\0anchor-doc\0" + str(document["id"]).encode()
        ).hexdigest()
    )
    total = calibration_count + holdout_count
    if len(eligible) < total:
        raise ValueError(f"need {total} disjoint anchor documents; found {len(eligible)}")
    cases: list[dict[str, object]] = []
    offset = 0
    for split, count in (("calibration", calibration_count), ("holdout", holdout_count)):
        for index, document in enumerate(eligible[offset : offset + count]):
            selector_bytes = hashlib.sha256(
                seed + b"\0anchor-span\0" + str(document["id"]).encode()
            ).digest()
            phrase, start, end = _anchor(
                str(document["text"]), int.from_bytes(selector_bytes[:8], "big")
            )
            subject = BASE._nonce(seed, "anchored-" + split, index, "subject")
            event = BASE._nonce(seed, "anchored-" + split, index, "event")
            family = index % len(TEMPLATES)
            question = TEMPLATES[family].format(
                anchor=phrase, subject=subject, event=event
            )
            if len(question) > 512:
                raise ValueError("anchored question exceeds preview limit")
            cases.append(
                {
                    "id": f"anchored-nonce-{split}-{index + 1:04d}",
                    "split": split,
                    "family": family,
                    "subject_nonce": subject,
                    "event_nonce": event,
                    "question": question,
                    "label": "UNSUPPORTED_WITHIN_FROZEN_CORPUS_BY_NONCE_ABSENCE",
                    "anchor": {
                        "document_id": document["id"],
                        "phrase": phrase,
                        "source_start": start,
                        "source_end": end,
                        "source_sha256": hashlib.sha256(
                            str(document["text"]).encode()
                        ).hexdigest(),
                    },
                }
            )
        offset += count
    return cases


def verify_anchors(
    cases: list[dict[str, object]], payload: dict[str, object]
) -> dict[str, object]:
    documents = payload["documents"]
    assert isinstance(documents, list)
    by_id = {
        str(document["id"]): document
        for document in documents
        if isinstance(document, dict) and isinstance(document.get("id"), str)
    }
    seen: set[str] = set()
    for case in cases:
        anchor = case["anchor"]
        assert isinstance(anchor, dict)
        document_id = str(anchor["document_id"])
        if document_id in seen:
            raise ValueError("anchor document reused across calibration and holdout")
        seen.add(document_id)
        document = by_id.get(document_id)
        if not isinstance(document, dict) or not isinstance(document.get("text"), str):
            raise ValueError("anchor document is not in the frozen payload")
        text = str(document["text"])
        start, end = anchor["source_start"], anchor["source_end"]
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or not 0 <= start < end <= len(text)
        ):
            raise ValueError("anchor offsets are invalid")
        phrase_tokens = " ".join(match.group(0) for match in WORD.finditer(text[start:end]))
        if phrase_tokens != anchor["phrase"]:
            raise ValueError("anchor phrase is not bound to its document span")
        if hashlib.sha256(text.encode()).hexdigest() != anchor["source_sha256"]:
            raise ValueError("anchor document hash mismatch")
    nonce = BASE.verify_nonce_absence(cases, payload)
    return {**nonce, "unique_anchor_documents": len(seen), "anchor_reuse": 0}


def anchor_recall(records: list[dict[str, object]]) -> dict[str, object]:
    retrieved = 0
    ranks: list[int] = []
    for record in records:
        anchor = record["anchor"]
        response = record["response"]
        assert isinstance(anchor, dict) and isinstance(response, dict)
        passages = response["passages"]
        assert isinstance(passages, list)
        ids = [passage["id"] for passage in passages if isinstance(passage, dict)]
        if anchor["document_id"] in ids:
            retrieved += 1
            ranks.append(ids.index(anchor["document_id"]) + 1)
    return {
        "queries": len(records),
        "anchor_retrieved_at_3": retrieved,
        "anchor_recall_at_3": retrieved / len(records),
        "mean_anchor_rank_when_retrieved": sum(ranks) / len(ranks) if ranks else None,
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
    baseline = holdout["base_threshold"]
    candidate = holdout["retrieval_conditioned_threshold"]
    assert isinstance(baseline, dict) and isinstance(candidate, dict)
    replay_base = replay["base"]
    replay_new = replay["retrieval_conditioned_kernel"]
    assert isinstance(replay_base, dict) and isinstance(replay_new, dict)
    outcome = (
        "MEETS_CORPUS_ANCHORED_HOLDOUT_TARGET"
        if holdout["target_met"]
        else "CORPUS_ANCHORED_HOLDOUT_TARGET_NOT_MET"
    )
    lines = [
        "# SZL Corpus-Anchored Nonce Kernel — Local Evaluation",
        "",
        f"Outcome: `{outcome}`. This is a harder local evaluation, not production, publication, training, new weights, or global unanswerability proof.",
        "",
        "## Exact bindings",
        "",
        f"- Run: `{bindings['run_id']}`",
        f"- Freeze SHA-256: `{bindings['artifacts']['freeze.json']}`",
        f"- Preview SHA-256: `{bindings['preview_sha256']}`",
        f"- V2 kernel SHA-256: `{bindings['kernel_v2_sha256']}`",
        f"- Primitive kernel SHA-256: `{bindings['kernel_v1_sha256']}`",
        f"- Receipt SHA-256: `{receipt_sha}`",
        "",
        "## Hard-negative result",
        "",
        "| Slice | Threshold | False answers | Queries | Rate | Wilson 95% upper | Anchor recall@3 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Calibration | {calibration['threshold']:.6f} | {calibration['false_answers']} | {calibration['queries']} | {calibration['false_answer_rate']:.2%} | {calibration['wilson_upper']:.2%} | {receipt['anchor_retrieval']['calibration']['anchor_recall_at_3']:.2%} |",
        f"| Holdout, base | {receipt['base_threshold']:.6f} | {baseline['false_answers']} | {baseline['queries']} | {baseline['false_answer_rate']:.2%} | {baseline['wilson_95_upper']:.2%} | {receipt['anchor_retrieval']['holdout']['anchor_recall_at_3']:.2%} |",
        f"| Holdout, candidate | {calibration['threshold']:.6f} | {candidate['false_answers']} | {candidate['queries']} | {candidate['false_answer_rate']:.2%} | {candidate['wilson_95_upper']:.2%} | {receipt['anchor_retrieval']['holdout']['anchor_recall_at_3']:.2%} |",
        "",
        "Each question includes a six-token lexical phrase bound by offsets and SHA-256 to one real frozen document. It simultaneously asks about fresh subject and event identifiers proven absent from the complete payload. Calibration and holdout use different anchor documents.",
        "",
        "## Public positive replay",
        "",
        "| Policy | Coverage | Exact match | F1 |",
        "|---|---:|---:|---:|",
        f"| Base | {replay_base['coverage']:.2%} | {replay_base['exact_match']:.2%} | {replay_base['f1']:.2%} |",
        f"| Candidate | {replay_new['coverage']:.2%} | {replay_new['exact_match']:.2%} | {replay_new['f1']:.2%} |",
        "",
        "This replay exposes selectivity cost but is public development data, not fresh quality proof.",
        "",
        "## Decision boundary",
        "",
        "The receipt supports only this corpus-anchored synthetic risk slice. Broader rights-reviewed adversarial judgments, protected publication, deployment, provider readback, and witnessed runtime remain separate gates.",
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
    client = BASE.PreviewClient(args.base_url, args.timeout)
    status_before = client.request("/api/status")
    BASE.verify_status(status_before, run_dir, hashes_before, preview_source)
    cases = generate_anchored_cases(
        seed_hex, args.calibration_count, args.holdout_count, payload
    )
    proof = verify_anchors(cases, payload)
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
        negative_records.append({**case, "response": response})
        if (index + 1) % 20 == 0:
            print(f"anchored nonce queries: {index + 1}/{len(cases)}", flush=True)
    for index, case in enumerate(positives):
        response = client.request("/api/query", {"question": case["question"]})
        BASE.validate_response(response, status_before)
        positive_records.append(
            {
                "id": case["id"],
                "question": case["question"],
                "gold": case["answers"],
                "response": response,
            }
        )
        if (index + 1) % 10 == 0:
            print(f"positive replays: {index + 1}/{len(positives)}", flush=True)
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
    calibration = BASE.select_threshold(
        [record["response"] for record in calibration_records],
        base_threshold,
        args.target_upper_bound,
    )
    candidate_threshold = float(calibration["threshold"])
    base_holdout = BASE.summarize_negative(
        [record["response"] for record in holdout_records], base_threshold
    )
    candidate_holdout = BASE.summarize_negative(
        [record["response"] for record in holdout_records], candidate_threshold
    )
    status_after = client.request("/api/status")
    hashes_after = BASE.artifact_hashes(run_dir)
    if status_after != status_before or hashes_after != hashes_before:
        raise RuntimeError("frozen artifacts or preview status changed during evaluation")
    receipt: dict[str, object] = {
        "schema_version": BASE.SCHEMA_VERSION,
        "status": BASE.STATUS,
        "scope": BASE.SCOPE,
        "variant": VARIANT,
        "completed_at": BASE.utc_now(),
        "suite": {
            "seed_hex": seed_hex,
            "seed_origin": seed_origin,
            "generated_after_local_freeze": True,
            "calibration_cases": args.calibration_count,
            "holdout_cases": args.holdout_count,
            "families": len(TEMPLATES),
            "proof": proof,
        },
        "bindings": {
            "run_id": status_before["run_id"],
            "artifacts": hashes_before,
            "preview_sha256": status_before["preview_sha256"],
            "preview_assets": status_before["asset_sha256"],
            "kernel_v2_sha256": BASE.sha256_file(Path(__file__).resolve()),
            "kernel_v1_sha256": BASE.sha256_file(BASE_SOURCE),
        },
        "base_threshold": base_threshold,
        "calibration": calibration,
        "holdout": {
            "target_wilson_upper": args.target_upper_bound,
            "base_threshold": base_holdout,
            "retrieval_conditioned_threshold": candidate_holdout,
            "target_met": candidate_holdout["wilson_95_upper"]
            <= args.target_upper_bound,
        },
        "anchor_retrieval": {
            "calibration": anchor_recall(calibration_records),
            "holdout": anchor_recall(holdout_records),
        },
        "positive_replay": BASE.score_positive_replay(
            positive_records, candidate_threshold
        ),
        "training_performed": False,
        "new_weights_created": False,
        "publication_eligible": False,
        "limitations": [
            "This is a synthetic, template-based unsupported-query distribution, not a complete model of real user requests.",
            "Nonce absence establishes lack of direct support in this finite payload, not global real-world nonexistence.",
            "The anchor phrase intentionally biases retrieval but does not guarantee its source document reaches top three; recall is reported.",
            "Shared templates and one frozen inference stack create dependence beyond a simple binomial model.",
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
        "candidate_threshold": candidate_threshold,
        "holdout_target_met": receipt["holdout"]["target_met"],
        "anchor_recall_at_3": receipt["anchor_retrieval"]["holdout"][
            "anchor_recall_at_3"
        ],
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
