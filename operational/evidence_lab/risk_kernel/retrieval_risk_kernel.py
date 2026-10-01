"""Evaluate a retrieval-conditioned abstention kernel on fresh nonce queries.

This is a local, evaluation-only control. It does not train or publish weights.
It measures a narrow, defensible slice of unsupported-query risk: questions whose
freshly generated subject and event identifiers are absent from the entire frozen
corpus. It then chooses an additional margin threshold on a calibration split and
evaluates that threshold once on a disjoint holdout split.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import secrets
import sys
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


SCHEMA_VERSION = 1
SCOPE = "SYNTHETIC_NONCE_UNSUPPORTED_AFTER_LOCAL_FREEZE"
STATUS = "LOCAL_EVALUATION_COMPLETED_NOT_PRODUCTION"
ARTIFACT_NAMES = (
    "freeze.json",
    "payload.json",
    "result.json",
    "calibration.json",
    "documents.npy",
)
TEMPLATES = (
    "Which city hosted the {event} ratification ceremony for the {subject} consortium?",
    "Who founded the {subject} institute before it issued the {event} charter?",
    "In what year did {subject} first demonstrate the {event} process?",
    "Which material powers the {subject} reactor described in the {event} protocol?",
    "What treaty was signed by {subject} during the {event} summit?",
    "Which language was used to draft the {event} constitution of {subject}?",
    "What is the measured orbit period of {subject} in the {event} catalogue?",
    "Which court decided the {event} dispute involving {subject}?",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def save_new(path: Path, value: object) -> None:
    with path.open("xb") as handle:
        handle.write(canonical_bytes(value))


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON value: {value}")


def parse_json(data: bytes) -> object:
    return json.loads(
        data.decode("utf-8"),
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )


def load_json(path: Path) -> object:
    return parse_json(path.read_bytes())


def normalize_for_absence(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def validate_seed(seed_hex: str) -> bytes:
    if not re.fullmatch(r"[0-9a-fA-F]{64}", seed_hex):
        raise ValueError("seed must be exactly 64 hexadecimal characters")
    return bytes.fromhex(seed_hex)


def _nonce(seed: bytes, split: str, index: int, role: str) -> str:
    material = b"szl-nonce-suite-v1\0" + seed + b"\0" + f"{split}:{index}:{role}".encode()
    token = hashlib.sha256(material).hexdigest()[:20]
    prefix = "SzlNqSubject" if role == "subject" else "SzlNqEvent"
    return prefix + token


def generate_cases(
    seed_hex: str, calibration_count: int, holdout_count: int
) -> list[dict[str, object]]:
    seed = validate_seed(seed_hex)
    cases: list[dict[str, object]] = []
    for split, count in (("calibration", calibration_count), ("holdout", holdout_count)):
        for index in range(count):
            subject = _nonce(seed, split, index, "subject")
            event = _nonce(seed, split, index, "event")
            family = index % len(TEMPLATES)
            cases.append(
                {
                    "id": f"nonce-{split}-{index + 1:04d}",
                    "split": split,
                    "family": family,
                    "subject_nonce": subject,
                    "event_nonce": event,
                    "question": TEMPLATES[family].format(subject=subject, event=event),
                    "label": "UNSUPPORTED_WITHIN_FROZEN_CORPUS_BY_NONCE_ABSENCE",
                }
            )
    return cases


def verify_nonce_absence(
    cases: list[dict[str, object]], payload: object
) -> dict[str, object]:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
    searchable = normalize_for_absence(serialized)
    seen: set[str] = set()
    for case in cases:
        for field in ("subject_nonce", "event_nonce"):
            nonce = str(case[field])
            normalized = normalize_for_absence(nonce)
            if normalized in seen:
                raise ValueError("Generated nonce collision")
            seen.add(normalized)
            if normalized in searchable:
                raise ValueError(f"Generated nonce already occurs in frozen payload: {nonce}")
    return {
        "method": "Unicode NFKC plus casefold exact-substring scan of canonical frozen payload",
        "payload_characters_scanned": len(serialized),
        "unique_nonces_checked": len(seen),
        "occurrences_found": 0,
    }


def wilson_interval(
    errors: int, total: int, z: float = 1.96
) -> tuple[float, float]:
    if (
        isinstance(errors, bool)
        or isinstance(total, bool)
        or not 0 <= errors <= total
        or total <= 0
    ):
        raise ValueError("invalid binomial counts")
    proportion = errors / total
    z2 = z * z
    denominator = 1 + z2 / total
    center = (proportion + z2 / (2 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            (proportion * (1 - proportion) + z2 / (4 * total)) / total
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def is_answer(response: dict[str, object], threshold: float) -> bool:
    margin = response.get("margin")
    answer = response.get("answer")
    return (
        response.get("status") == "ANSWER"
        and isinstance(answer, str)
        and bool(answer.strip())
        and not isinstance(margin, bool)
        and isinstance(margin, (int, float))
        and math.isfinite(float(margin))
        and float(margin) > threshold
    )


def select_threshold(
    responses: list[dict[str, object]],
    base_threshold: float,
    target_upper_bound: float,
    z: float = 1.96,
) -> dict[str, object]:
    if not responses:
        raise ValueError("calibration responses are empty")
    if not math.isfinite(base_threshold):
        raise ValueError("base threshold is not finite")
    candidates = {float(base_threshold)}
    for response in responses:
        margin = response.get("margin")
        if (
            isinstance(margin, bool)
            or not isinstance(margin, (int, float))
            or not math.isfinite(float(margin))
        ):
            raise ValueError("response margin is invalid")
        if response.get("status") == "ANSWER" and float(margin) >= base_threshold:
            candidates.add(float(margin))
    selected: dict[str, object] | None = None
    for threshold in sorted(candidates):
        errors = sum(is_answer(response, threshold) for response in responses)
        lower, upper = wilson_interval(errors, len(responses), z)
        row = {
            "threshold": threshold,
            "false_answers": errors,
            "queries": len(responses),
            "false_answer_rate": errors / len(responses),
            "wilson_lower": lower,
            "wilson_upper": upper,
            "target_wilson_upper": target_upper_bound,
            "target_met": upper <= target_upper_bound,
        }
        if row["target_met"]:
            selected = row
            break
        selected = row
    assert selected is not None
    return selected


def summarize_negative(
    responses: list[dict[str, object]], threshold: float, z: float = 1.96
) -> dict[str, object]:
    errors = sum(is_answer(response, threshold) for response in responses)
    lower, upper = wilson_interval(errors, len(responses), z)
    return {
        "queries": len(responses),
        "false_answers": errors,
        "false_answer_rate": errors / len(responses),
        "wilson_95_lower": lower,
        "wilson_95_upper": upper,
    }


def _normalize_answer(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = "".join(
        character
        if not unicodedata.category(character).startswith("P")
        else " "
        for character in value
    )
    tokens = [token for token in value.split() if token not in {"a", "an", "the"}]
    return " ".join(tokens)


def _exact_f1(prediction: str, gold: str) -> tuple[float, float]:
    predicted = _normalize_answer(prediction)
    expected = _normalize_answer(gold)
    exact = float(predicted == expected)
    predicted_tokens = predicted.split()
    expected_tokens = expected.split()
    if not predicted_tokens or not expected_tokens:
        return exact, float(predicted_tokens == expected_tokens)
    expected_counts: dict[str, int] = {}
    for token in expected_tokens:
        expected_counts[token] = expected_counts.get(token, 0) + 1
    common = 0
    for token in predicted_tokens:
        count = expected_counts.get(token, 0)
        if count:
            common += 1
            expected_counts[token] = count - 1
    if not common:
        return exact, 0.0
    precision = common / len(predicted_tokens)
    recall = common / len(expected_tokens)
    return exact, 2 * precision * recall / (precision + recall)


def score_positive_replay(
    records: list[dict[str, object]], threshold: float
) -> dict[str, object]:
    if not records:
        raise ValueError("positive replay records are empty")
    base_exact = base_f1 = new_exact = new_f1 = 0.0
    base_answers = new_answers = 0
    for record in records:
        response = record["response"]
        assert isinstance(response, dict)
        gold = record["gold"]
        assert isinstance(gold, list) and gold
        base_prediction = (
            response.get("answer") if response.get("status") == "ANSWER" else ""
        )
        if not isinstance(base_prediction, str):
            base_prediction = ""
        new_prediction = base_prediction if is_answer(response, threshold) else ""
        base_answers += bool(base_prediction)
        new_answers += bool(new_prediction)
        base_scores = [_exact_f1(base_prediction, str(answer)) for answer in gold]
        new_scores = [_exact_f1(new_prediction, str(answer)) for answer in gold]
        base_exact += max(score[0] for score in base_scores)
        base_f1 += max(score[1] for score in base_scores)
        new_exact += max(score[0] for score in new_scores)
        new_f1 += max(score[1] for score in new_scores)
    total = len(records)
    return {
        "scope": "PUBLIC_DEVELOPMENT_SET_REPLAY_NOT_FRESH_QUALITY_EVIDENCE",
        "queries": total,
        "base": {
            "coverage": base_answers / total,
            "exact_match": base_exact / total,
            "f1": base_f1 / total,
        },
        "retrieval_conditioned_kernel": {
            "coverage": new_answers / total,
            "exact_match": new_exact / total,
            "f1": new_f1 / total,
        },
    }


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(
        self, request, file_pointer, code, message, headers, new_url
    ):  # noqa: ANN001
        return None


class PreviewClient:
    def __init__(self, base_url: str, timeout: float):
        split = urlsplit(base_url)
        if split.scheme != "http" or split.hostname not in {"127.0.0.1", "localhost"}:
            raise ValueError("base URL must be an explicit loopback HTTP endpoint")
        if (
            split.username
            or split.password
            or split.query
            or split.fragment
            or split.path not in {"", "/"}
        ):
            raise ValueError(
                "base URL must not contain credentials, path, query, or fragment"
            )
        if split.port is None:
            raise ValueError("base URL must include an explicit port")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.opener = build_opener(ProxyHandler({}), NoRedirect())

    def request(
        self, route: str, body: dict[str, object] | None = None
    ) -> dict[str, object]:
        url = self.base_url + route
        data = canonical_bytes(body) if body is not None else None
        headers = {"Accept": "application/json"}
        if data is not None:
            headers.update(
                {"Content-Type": "application/json", "X-SZL-Preview": "1"}
            )
        request = Request(
            url,
            data=data,
            headers=headers,
            method="POST" if data is not None else "GET",
        )
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                if response.status != 200 or response.geturl() != url:
                    raise RuntimeError(
                        f"unexpected preview response: status={response.status}"
                    )
                content_type = (
                    response.headers.get("Content-Type", "").split(";", 1)[0].strip()
                )
                if content_type != "application/json":
                    raise RuntimeError("preview returned a non-JSON content type")
                raw = response.read(4 * 1024 * 1024 + 1)
                if len(raw) > 4 * 1024 * 1024:
                    raise RuntimeError("preview response exceeded 4 MiB")
        except (HTTPError, URLError, TimeoutError) as exc:
            raise RuntimeError(f"preview request failed for {route}: {exc}") from exc
        parsed = parse_json(raw)
        if not isinstance(parsed, dict):
            raise RuntimeError("preview response is not a JSON object")
        return parsed


def artifact_hashes(run_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name in ARTIFACT_NAMES:
        path = run_dir / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"required frozen artifact is missing or unsafe: {name}")
        hashes[name] = sha256_file(path)
    return hashes


def validate_completed_run(
    run_dir: Path, hashes: dict[str, str] | None = None
) -> dict[str, object]:
    """Verify the completed artifact graph without loading models or NumPy.

    ``hashes`` may bind a caller's earlier snapshot. Readback on both sides of
    parsing rejects files changed before or during validation. The index bytes
    are bound to the completed result; array shape is checked by LocalService.
    """
    run_dir = Path(run_dir).resolve()
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", run_dir.name):
        raise ValueError("run directory must end with a simple run identifier")
    observed = artifact_hashes(run_dir)
    if hashes is not None and hashes != observed:
        raise RuntimeError("frozen artifacts changed since the supplied snapshot")
    freeze = load_json(run_dir / "freeze.json")
    payload = load_json(run_dir / "payload.json")
    result = load_json(run_dir / "result.json")
    calibration = load_json(run_dir / "calibration.json")
    if not all(isinstance(value, dict) for value in (freeze, payload, result, calibration)):
        raise ValueError("frozen artifacts have invalid top-level types")
    if result.get("status") != STATUS:
        raise RuntimeError("selected run does not contain a completed evaluation")
    links = (
        (freeze.get("payload_sha256"), observed["payload.json"]),
        (result.get("freeze_sha256"), observed["freeze.json"]),
        (result.get("index_sha256"), observed["documents.npy"]),
        (result.get("calibration_sha256"), observed["calibration.json"]),
        (calibration.get("freeze_sha256"), observed["freeze.json"]),
    )
    if any(recorded != actual for recorded, actual in links):
        raise RuntimeError("completed run artifact binding failed")
    documents = payload.get("documents")
    if not isinstance(documents, list) or not documents:
        raise ValueError("completed run has no document list")
    threshold = calibration.get("threshold")
    if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold)):
        raise ValueError("completed run has an invalid calibration threshold")
    if artifact_hashes(run_dir) != observed:
        raise RuntimeError("frozen artifacts changed while validating the completed run")
    return {"artifacts": observed, "run_id": run_dir.name,
            "documents": len(documents), "threshold": threshold}


def validate_response(
    response: dict[str, object], status: dict[str, object]
) -> None:
    if (
        response.get("run_id") != status.get("run_id")
        or response.get("freeze_sha256") != status.get("freeze_sha256")
    ):
        raise RuntimeError("response is not bound to the expected frozen run")
    if response.get("preview_sha256") != status.get("preview_sha256"):
        raise RuntimeError("preview source changed during inference")
    if response.get("publication_eligible") is not False:
        raise RuntimeError(
            "preview response incorrectly claims publication eligibility"
        )
    if response.get("status") not in {"ANSWER", "ABSTAIN"}:
        raise RuntimeError("invalid answer status")
    if (
        response.get("scope") != "retrieved_passages_only"
        or response.get("global_unanswerability") != "UNVERIFIED"
    ):
        raise RuntimeError("preview response lost its narrow evidence scope")
    margin = response.get("margin")
    base_threshold = response.get("threshold")
    if (
        isinstance(margin, bool)
        or not isinstance(margin, (int, float))
        or not math.isfinite(float(margin))
    ):
        raise RuntimeError("invalid response margin")
    if (
        isinstance(base_threshold, bool)
        or not isinstance(base_threshold, (int, float))
        or not math.isfinite(float(base_threshold))
    ):
        raise RuntimeError("invalid response threshold")
    if base_threshold != status.get("threshold"):
        raise RuntimeError("response threshold does not match the verified calibration")
    passages = response.get("passages")
    if not isinstance(passages, list) or not passages:
        raise RuntimeError("response contains no retrieved passages")
    passage_by_id: dict[str, dict[str, object]] = {}
    for passage in passages:
        if (
            not isinstance(passage, dict)
            or not isinstance(passage.get("id"), str)
            or not isinstance(passage.get("text"), str)
        ):
            raise RuntimeError("invalid passage record")
        if passage["id"] in passage_by_id:
            raise RuntimeError("duplicate passage identifier")
        passage_by_id[str(passage["id"])] = passage
    if response["status"] == "ABSTAIN":
        if response.get("answer") is not None or response.get("evidence") is not None:
            raise RuntimeError("abstention leaked an answer or evidence claim")
        return
    answer = response.get("answer")
    evidence = response.get("evidence")
    if (
        not isinstance(answer, str)
        or not answer.strip()
        or not isinstance(evidence, dict)
    ):
        raise RuntimeError("answer lacks a non-empty source binding")
    document_id = evidence.get("document_id")
    if document_id not in passage_by_id:
        raise RuntimeError("answer evidence does not name a retrieved passage")
    passage_text = str(passage_by_id[str(document_id)]["text"])
    if evidence.get("context_sha256") != hashlib.sha256(
        passage_text.encode()
    ).hexdigest():
        raise RuntimeError("answer evidence hash does not match its passage")
    start, end = evidence.get("start"), evidence.get("end")
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
    ):
        raise RuntimeError("answer offsets are not integers")
    if not 0 <= start < end <= len(passage_text) or passage_text[start:end] != answer:
        raise RuntimeError("answer offsets do not select the returned source span")


def verify_status(
    status: dict[str, object],
    run_dir: Path,
    hashes: dict[str, str],
    preview_source: Path,
) -> None:
    completed = validate_completed_run(run_dir, hashes)
    if status.get("run_id") != completed["run_id"]:
        raise RuntimeError("preview run identifier does not match the selected run")
    if status.get("freeze_sha256") != hashes["freeze.json"]:
        raise RuntimeError("preview freeze hash does not match the selected run")
    if status.get("result_sha256") != hashes["result.json"]:
        raise RuntimeError("preview result hash does not match the selected run")
    if (isinstance(status.get("documents"), bool)
            or not isinstance(status.get("documents"), int)
            or status.get("documents") != completed["documents"]):
        raise RuntimeError(
            "preview document count does not match the frozen payload"
        )
    threshold = status.get("threshold")
    if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold) or threshold != completed["threshold"]):
        raise RuntimeError("preview threshold does not match the verified calibration")
    if status.get("publication_eligible") is not False:
        raise RuntimeError("preview incorrectly claims publication eligibility")
    if (
        not preview_source.is_file()
        or preview_source.is_symlink()
        or status.get("preview_sha256") != sha256_file(preview_source)
    ):
        raise RuntimeError(
            "preview status is not bound to the supplied source file"
        )


def select_positive_cases(
    payload: dict[str, object], seed_hex: str, count: int
) -> list[dict[str, object]]:
    rows = payload.get("evaluation")
    if not isinstance(rows, list):
        raise ValueError("payload evaluation split is missing")
    positives = [
        row
        for row in rows
        if isinstance(row, dict) and row.get("answerable") is True
    ]
    positives.sort(
        key=lambda row: hashlib.sha256(
            (seed_hex + "\0" + str(row.get("id"))).encode()
        ).hexdigest()
    )
    if count > len(positives):
        raise ValueError(
            f"requested {count} positive replays but only {len(positives)} exist"
        )
    return positives[:count]


def report_markdown(receipt: dict[str, object], receipt_sha256: str) -> str:
    calibration = receipt["calibration"]
    holdout = receipt["holdout"]
    replay = receipt["positive_replay"]
    assert isinstance(calibration, dict)
    assert isinstance(holdout, dict)
    assert isinstance(replay, dict)
    baseline = holdout["base_threshold"]
    candidate = holdout["retrieval_conditioned_threshold"]
    assert isinstance(baseline, dict)
    assert isinstance(candidate, dict)
    replay_base = replay["base"]
    replay_new = replay["retrieval_conditioned_kernel"]
    assert isinstance(replay_base, dict)
    assert isinstance(replay_new, dict)
    outcome = (
        "MEETS_SYNTHETIC_HOLDOUT_TARGET"
        if holdout["target_met"]
        else "SYNTHETIC_HOLDOUT_TARGET_NOT_MET"
    )
    bindings = receipt["bindings"]
    assert isinstance(bindings, dict)
    artifacts = bindings["artifacts"]
    assert isinstance(artifacts, dict)
    lines = [
        "# SZL Retrieval-Conditioned Abstention Kernel — Local Evaluation",
        "",
        f"Outcome: `{outcome}`. This is a local, evaluation-only result—not production, publication, new weights, or a corpus-wide unanswerability proof.",
        "",
        "## Bound run",
        "",
        f"- Frozen run: `{bindings['run_id']}`",
        f"- Freeze SHA-256: `{artifacts['freeze.json']}`",
        f"- Preview source SHA-256: `{bindings['preview_sha256']}`",
        f"- Receipt SHA-256: `{receipt_sha256}`",
        "",
        "## Unsupported-query result",
        "",
        "| Slice | Threshold | False answers | Queries | Rate | Wilson 95% upper |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Calibration | {calibration['threshold']:.6f} | {calibration['false_answers']} | {calibration['queries']} | {calibration['false_answer_rate']:.2%} | {calibration['wilson_upper']:.2%} |",
        f"| Holdout, base policy | {receipt['base_threshold']:.6f} | {baseline['false_answers']} | {baseline['queries']} | {baseline['false_answer_rate']:.2%} | {baseline['wilson_95_upper']:.2%} |",
        f"| Holdout, candidate kernel | {calibration['threshold']:.6f} | {candidate['false_answers']} | {candidate['queries']} | {candidate['false_answer_rate']:.2%} | {candidate['wilson_95_upper']:.2%} |",
        "",
        "Every synthetic question contains independently derived subject and event identifiers verified absent from the complete frozen payload. The calibration and holdout identifiers are disjoint. Any extracted answer is therefore false for this finite-corpus test slice.",
        "",
        "## Public positive replay trade-off",
        "",
        "| Policy | Coverage | Exact match | F1 |",
        "|---|---:|---:|---:|",
        f"| Base threshold | {replay_base['coverage']:.2%} | {replay_base['exact_match']:.2%} | {replay_base['f1']:.2%} |",
        f"| Candidate kernel | {replay_new['coverage']:.2%} | {replay_new['exact_match']:.2%} | {replay_new['f1']:.2%} |",
        "",
        "These positives are a deterministic replay of public development examples. They expose the selectivity trade-off but are not fresh quality evidence.",
        "",
        "## Decision boundary",
        "",
        "The candidate is fit for further local evaluation only. It must not be described as production-ready unless broader unsupported-query families, rights-reviewed relevance judgments, current-source publication, deployment, and independently witnessed runtime are separately proven.",
        "",
        "## Explicit limits",
        "",
    ]
    limitations = receipt["limitations"]
    assert isinstance(limitations, list)
    lines.extend(f"- {item}" for item in limitations)
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, object]:
    run_dir = args.run_dir.resolve()
    preview_source = args.preview_source.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    if args.calibration_count < 80 or args.holdout_count < 80:
        raise ValueError(
            "calibration and holdout counts must each be at least 80 for this 95% risk-bound design"
        )
    if args.positive_count <= 0:
        raise ValueError("positive replay count must be positive")
    if not 0 < args.target_upper_bound < 1:
        raise ValueError("target upper bound must be in (0, 1)")
    seed_origin = "caller_supplied"
    seed_hex = args.seed
    if seed_hex is None:
        seed_hex = secrets.token_hex(32)
        seed_origin = "fresh_os_csprng_at_run_start"
    validate_seed(seed_hex)
    created_at = utc_now()
    pre_hashes = artifact_hashes(run_dir)
    freeze = load_json(run_dir / "freeze.json")
    payload = load_json(run_dir / "payload.json")
    if not isinstance(freeze, dict) or not isinstance(payload, dict):
        raise ValueError("invalid frozen run")
    if datetime.fromisoformat(created_at) <= datetime.fromisoformat(
        str(freeze["created_at"])
    ):
        raise ValueError("nonce suite was not created after the selected local freeze")
    client = PreviewClient(args.base_url, args.timeout)
    status_before = client.request("/api/status")
    verify_status(status_before, run_dir, pre_hashes, preview_source)
    cases = generate_cases(seed_hex, args.calibration_count, args.holdout_count)
    absence = verify_nonce_absence(cases, payload)
    positives = select_positive_cases(payload, seed_hex, args.positive_count)
    output_dir.mkdir(parents=True, exist_ok=False)
    save_new(
        output_dir / "attempt.json",
        {
            "schema_version": SCHEMA_VERSION,
            "started_at": created_at,
            "scope": SCOPE,
            "seed_sha256": hashlib.sha256(bytes.fromhex(seed_hex)).hexdigest(),
            "bindings": pre_hashes,
        },
    )
    negative_records: list[dict[str, object]] = []
    positive_records: list[dict[str, object]] = []
    for index, case in enumerate(cases):
        response = client.request("/api/query", {"question": case["question"]})
        validate_response(response, status_before)
        negative_records.append({**case, "response": response})
        if (index + 1) % 20 == 0:
            print(f"nonce queries: {index + 1}/{len(cases)}", flush=True)
    for index, case in enumerate(positives):
        response = client.request("/api/query", {"question": case["question"]})
        validate_response(response, status_before)
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
    calibration_responses = [
        record["response"]
        for record in negative_records
        if record["split"] == "calibration"
    ]
    holdout_responses = [
        record["response"]
        for record in negative_records
        if record["split"] == "holdout"
    ]
    calibration = select_threshold(
        calibration_responses, base_threshold, args.target_upper_bound
    )
    candidate_threshold = float(calibration["threshold"])
    base_holdout = summarize_negative(holdout_responses, base_threshold)
    candidate_holdout = summarize_negative(
        holdout_responses, candidate_threshold
    )
    status_after = client.request("/api/status")
    post_hashes = artifact_hashes(run_dir)
    if post_hashes != pre_hashes or status_after != status_before:
        raise RuntimeError(
            "frozen artifacts or preview status changed during evaluation"
        )
    receipt: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "scope": SCOPE,
        "completed_at": utc_now(),
        "suite": {
            "seed_hex": seed_hex,
            "seed_origin": seed_origin,
            "generated_after_local_freeze": True,
            "calibration_cases": args.calibration_count,
            "holdout_cases": args.holdout_count,
            "families": len(TEMPLATES),
            "nonce_absence": absence,
        },
        "bindings": {
            "run_id": status_before["run_id"],
            "artifacts": pre_hashes,
            "preview_sha256": status_before["preview_sha256"],
            "preview_assets": status_before["asset_sha256"],
            "kernel_sha256": sha256_file(Path(__file__).resolve()),
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
        "positive_replay": score_positive_replay(
            positive_records, candidate_threshold
        ),
        "training_performed": False,
        "new_weights_created": False,
        "publication_eligible": False,
        "limitations": [
            "The unsupported-query distribution is synthetic and narrow; it is not a proxy for every real unsupported request.",
            "Exact nonce absence proves no direct subject or event mention in this finite corpus, not global real-world nonexistence.",
            "Fresh generation occurred after the local freeze; upstream model-training contamination is not globally audited.",
            "The Wilson interval assumes independent Bernoulli trials, while shared templates and one frozen pipeline create dependence.",
            "Public positive examples may be contaminated and are replayed only to expose answer-coverage trade-offs.",
            "No training, paid cloud job, external publication, deployment, credential change, firewall change, or independent witnessing occurred.",
        ],
        "records": {
            "synthetic_nonce": negative_records,
            "public_positive_replay": positive_records,
        },
    }
    save_new(output_dir / "receipt.json", receipt)
    receipt_sha = sha256_file(output_dir / "receipt.json")
    with (output_dir / "REPORT.md").open("xb") as handle:
        handle.write(report_markdown(receipt, receipt_sha).encode("utf-8"))
    return {
        "status": STATUS,
        "output_dir": str(output_dir),
        "receipt_sha256": receipt_sha,
        "candidate_threshold": candidate_threshold,
        "synthetic_holdout_target_met": receipt["holdout"]["target_met"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--preview-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8766")
    parser.add_argument(
        "--seed", help="optional replay seed: exactly 64 hexadecimal characters"
    )
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
