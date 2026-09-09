"""Bounded public Hub observations, not model qualification or promotion.

Copyright 2026 SZL Holdings. SPDX-License-Identifier: Apache-2.0
v2 preserves explicit dispositions, verifies receipts before projection, and
reports partial outages. Hashes detect corruption; they do NOT authenticate a
publisher or make local storage immutable. No weights, remote code, credentials,
training, provider writes, or production route changes are used here.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

HF_API = "https://huggingface.co/api/models/"
SOURCE_REPOSITORY = "szl-holdings/szl-forge"
AUTHORITY_CHAIN = ("GitHub", "Hugging Face", "a-11-oy.com", "a11oy.net")
SHA40 = re.compile(r"[0-9a-f]{40}")
MAX_JSON_BYTES = 2 * 1024 * 1024
MAX_CANDIDATES = 64
ALLOWED_LICENSES = frozenset({"apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause", "isc"})
REMOTE_CODE_TAGS = frozenset({"custom_code", "trust_remote_code"})
DENIED_AUTHORITY = {
    "weightDownloadAuthorized": False, "remoteCodeExecutionAuthorized": False,
    "trainingAuthorized": False, "publicationAuthorized": False,
    "deploymentAuthorized": False, "automaticPromotionAuthorized": False,
}
REMAINING_GATES = (
    "explicit license/data-rights admission",
    "pinned artifact inventory and cryptographic digests",
    "category-specific independent benchmark",
    "runtime/dependency closure and sandbox qualification",
    "SZL Nemo/A11oy policy witness",
    "rollback and product/proof projection",
    "human promotion approval",
)


class FrontierError(ValueError):
    """Invalid or incomplete evidence must not become permission."""


class Disposition(str, Enum):
    WATCH = "WATCH"
    EVALUATE = "EVALUATE"
    HOLD = "HOLD"
    REJECT = "REJECT"


def _text(value: Any, limit: int = 4096) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise FrontierError("invalid or oversized text field")
    return value


def _repo_id(value: Any) -> str:
    _text(value, 193)
    parts = value.split("/")
    if len(parts) != 2 or any(
        not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,95}", part)
        or part.endswith((".", "-", ".git")) or ".." in part or "--" in part
        for part in parts
    ):
        raise FrontierError("invalid Hugging Face repository id")
    return value


def _revision(value: Any) -> str:
    if not isinstance(value, str) or not SHA40.fullmatch(value):
        raise FrontierError("expected a lowercase 40-character commit SHA")
    return value


@dataclasses.dataclass(frozen=True, slots=True)
class Candidate:
    repo_id: str
    category: str
    disposition: Disposition
    rationale: str
    production_eligible: bool = False

    def __post_init__(self) -> None:
        _repo_id(self.repo_id)
        _text(self.category, 128)
        _text(self.rationale)
        if not isinstance(self.disposition, Disposition):
            raise FrontierError("disposition must be a Disposition enum")
        if self.production_eligible is not False:
            raise FrontierError("frontier discovery cannot pre-authorize production")


CANDIDATES = (
    Candidate("bodhan-ai/indic-transcribe-core", "multilingual-asr", Disposition.EVALUATE,
              "Evaluate multilingual/code-mixed ASR, language identification, "
              "latency, WER/CER, resource footprint, and data/license constraints."),
    Candidate("bodhan-ai/indic-speak", "multilingual-tts", Disposition.EVALUATE,
              "Evaluate multilingual/code-mixed TTS for intelligibility, speaker "
              "quality, latency, safety, resource footprint, and license constraints."),
    Candidate("gdiamos/amx-reasoning-v1-instruct", "cpu-kernel-architecture", Disposition.WATCH,
              "Study AMX/CPU-oriented architecture, linear/sliding-window attention "
              "and efficiency patterns; do not treat the checkpoint as a production LLM."),
)


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (ValueError, TypeError, RecursionError, UnicodeError) as exc:
        raise FrontierError("value is not finite canonical JSON") from exc


def sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FrontierError("duplicate JSON key")
        result[key] = value
    return result


def _constant(_: str) -> None:
    raise FrontierError("nonfinite JSON constant")


def strict_json(body: bytes | str) -> dict[str, Any]:
    """Reject duplicates, nonfinite numbers, oversize bodies and non-objects."""
    try:
        data = body.encode("utf-8") if isinstance(body, str) else body
        if not isinstance(data, bytes) or len(data) > MAX_JSON_BYTES:
            raise FrontierError("JSON body exceeded bound")
        value = json.loads(data.decode("utf-8"), object_pairs_hook=_pairs,
                           parse_constant=_constant)
        _canonical(value)  # Also catches exponent overflow, e.g. 1e999.
        if not isinstance(value, dict):
            raise FrontierError("JSON must be an object")
        return value
    except (ValueError, TypeError, RecursionError, UnicodeError) as exc:
        raise FrontierError("invalid bounded JSON object") from exc


def _timestamp(value: str | None = None) -> str:
    if value is None:
        value = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    try:
        parsed = dt.datetime.fromisoformat(_text(value, 64).replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise FrontierError("timestamp needs an explicit timezone")
        return parsed.astimezone(dt.timezone.utc).isoformat()
    except (ValueError, OverflowError) as exc:
        raise FrontierError("invalid observation timestamp") from exc


def _strings(value: Any, limit: int = 1024) -> list[str]:
    if not isinstance(value, list) or len(value) > limit:
        raise FrontierError("invalid string-list metadata")
    return sorted({_text(item, 512) for item in value})


def normalize_metadata(candidate: Candidate, raw: dict[str, Any]) -> dict[str, Any]:
    """Never interpret missing metadata as an affirmative security decision."""
    if not isinstance(raw, dict) or len(_canonical(raw)) > MAX_JSON_BYTES:
        raise FrontierError("invalid or oversized Hub metadata")
    identities = [raw[key] for key in ("modelId", "id") if key in raw]
    if not identities or any(value != candidate.repo_id for value in identities):
        raise FrontierError("upstream repository identity mismatch")
    revision = _revision(raw.get("sha"))
    tags = _strings(raw.get("tags", []))
    card = raw.get("cardData")
    if card is None:
        card = {}
    if not isinstance(card, dict):
        raise FrontierError("invalid card metadata")
    declarations = [tag[8:].strip().lower() for tag in tags if tag.startswith("license:")]
    card_license = card.get("license")
    if card_license is not None:
        values = [card_license] if isinstance(card_license, str) else _strings(card_license)
        declarations.extend(_text(item, 128).strip().lower() for item in values)
    declarations = sorted(set(declarations))
    if any(not value or len(value) > 128 for value in declarations):
        raise FrontierError("invalid license declaration")
    license_id = declarations[0] if len(declarations) == 1 else None

    # Signals are conservative metadata observations, not a source-code audit.
    signals = ["tag:" + tag for tag in tags if tag in REMOTE_CODE_TAGS]
    for key in ("config", "transformersInfo"):
        config = raw.get(key)
        if config is not None and not isinstance(config, dict):
            raise FrontierError("invalid configuration metadata")
        if config and config.get("auto_map"):
            signals.append(key + ":auto_map")
    siblings = raw.get("siblings", [])
    if not isinstance(siblings, list) or len(siblings) > 10000:
        raise FrontierError("invalid file inventory")
    for row in siblings:
        if not isinstance(row, dict):
            raise FrontierError("invalid inventory row")
        name = _text(row.get("rfilename"), 512)
        if name.lower().endswith(".py"):
            signals.append("repository_python_file")
    signals = sorted(set(signals))
    gated = raw.get("gated")
    if not (type(gated) is bool or gated is None or
            isinstance(gated, str) and gated in {"auto", "manual"}):
        gated = "unknown"
    for field in ("private", "disabled"):
        if raw.get(field) is not None and type(raw[field]) is not bool:
            raise FrontierError("invalid visibility metadata")
    for field in ("pipeline_tag", "library_name", "lastModified"):
        if raw.get(field) is not None:
            _text(raw[field], 256)
    return {
        "repoId": candidate.repo_id, "revision": revision,
        "category": candidate.category, "requestedDisposition": candidate.disposition.value,
        "productionEligible": False, "rationale": candidate.rationale,
        "license": license_id, "licenseDeclarations": declarations,
        "licenseAdmitted": license_id in ALLOWED_LICENSES,
        "gated": gated, "private": raw.get("private"), "disabled": raw.get("disabled"),
        "pipelineTag": raw.get("pipeline_tag"), "libraryName": raw.get("library_name"),
        "tags": tags, "remoteCodeSignal": bool(signals), "codeSignals": signals,
        "lastModified": raw.get("lastModified"),
        "sourceUrl": f"https://huggingface.co/{candidate.repo_id}/tree/{revision}",
    }


def gate(metadata: dict[str, Any]) -> dict[str, Any]:
    """HOLD/REJECT are absorbing; a WATCH is never a runtime authorization."""
    try:
        requested = Disposition(metadata["requestedDisposition"])
    except (KeyError, ValueError, TypeError) as exc:
        raise FrontierError("invalid requested disposition") from exc
    reasons = []
    declarations = metadata.get("licenseDeclarations", [])
    if len(declarations) > 1:
        reasons.append("license_conflict")
    elif not metadata.get("license"):
        reasons.append("license_unknown")
    elif metadata["license"] not in ALLOWED_LICENSES:
        reasons.append("license_not_admitted")
    if metadata.get("remoteCodeSignal") is not False:
        reasons.append("remote_code_signal")
    gated = metadata.get("gated")
    if gated is None or gated == "unknown":
        reasons.append("gating_unknown")
    elif gated is not False:
        reasons.append("gated_repository")
    if metadata.get("private") is True:
        reasons.append("private_repository")
    if metadata.get("disabled") is True:
        reasons.append("disabled_repository")
    if requested in {Disposition.REJECT, Disposition.HOLD, Disposition.WATCH}:
        effective = requested
    else:
        effective = Disposition.HOLD if reasons else Disposition.EVALUATE
    return {
        "effectiveDisposition": effective.value,
        "evaluationMetadataAdmitted": effective != Disposition.REJECT,
        **DENIED_AUTHORITY, "reasonCodes": sorted(reasons),
        "remainingGates": list(REMAINING_GATES),
    }


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise FrontierError("Hub metadata redirects are not admitted")


class HuggingFaceMetadataClient:
    """Fixed-origin, unauthenticated metadata GETs with bounded transient retries."""

    def __init__(self, *, timeout_seconds: float = 15.0, user_agent: str = "szl-forge/frontier-v2"):
        if type(timeout_seconds) not in (float, int) or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 60:
            raise FrontierError("timeout must be finite and within (0, 60]")
        if not isinstance(user_agent, str) or not re.fullmatch(r"[\x20-\x7e]{1,128}", user_agent):
            raise FrontierError("invalid user agent")
        self.timeout_seconds = timeout_seconds
        self.user_agent = user_agent
        self._opener = urllib.request.build_opener(_NoRedirect())

    def fetch(self, repo_id: str) -> dict[str, Any]:
        url = HF_API + urllib.parse.quote(_repo_id(repo_id), safe="/")
        request = urllib.request.Request(url, headers={"Accept": "application/json",
            "User-Agent": self.user_agent}, method="GET")
        for attempt in range(3):
            try:
                with self._opener.open(request, timeout=self.timeout_seconds) as response:
                    if response.status != 200 or response.geturl() != url:
                        raise FrontierError("unexpected Hub metadata response")
                    if response.headers.get_content_type() != "application/json":
                        raise FrontierError("Hub response is not JSON content")
                    return strict_json(response.read(MAX_JSON_BYTES + 1))
            except urllib.error.HTTPError as exc:
                # Never retry auth/permission failures or sleep past the job budget.
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                delay = float(2 ** attempt)
                retryable = exc.code in {429, 500, 502, 503, 504}
                exc.close()
                if retry_after is not None:
                    if not retry_after.isdigit() or len(retry_after) > 2 or int(retry_after) > 30:
                        raise FrontierError("Hub retry deferred beyond bounded scan") from exc
                    delay = max(delay, float(retry_after))
                if not retryable or attempt == 2:
                    raise FrontierError("Hub metadata HTTP failure") from exc
                time.sleep(delay)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                if attempt == 2:
                    raise FrontierError("Hub metadata transport failure") from exc
                time.sleep(float(2 ** attempt))
        raise FrontierError("Hub retry budget exhausted")


def build_receipt(candidate: Candidate, raw_metadata: dict[str, Any], *,
                  observed_at: str | None = None) -> dict[str, Any]:
    normalized = normalize_metadata(candidate, raw_metadata)
    receipt = {
        "schema": "szl.forge.hf-frontier-receipt.v2", "observedAt": _timestamp(observed_at),
        "authorityChain": list(AUTHORITY_CHAIN), "candidate": normalized,
        "gate": gate(normalized), "rawMetadataSha256": sha256(raw_metadata),
        "evidenceClass": "PUBLIC_HUGGING_FACE_METADATA_ONLY", "productionDisposition": "HOLD",
    }
    return {**receipt, "receiptSha256": sha256(receipt)}


def _registry(candidates: Iterable[Candidate]) -> tuple[Candidate, ...]:
    rows = []
    for candidate in candidates:
        if not isinstance(candidate, Candidate) or len(rows) >= MAX_CANDIDATES:
            raise FrontierError("invalid or oversized candidate registry")
        rows.append(candidate)
    if not rows or len({row.repo_id for row in rows}) != len(rows):
        raise FrontierError("empty or duplicate candidate registry")
    return tuple(sorted(rows, key=lambda row: row.repo_id))


def _spec(candidate: Candidate) -> dict[str, Any]:
    return {"repoId": candidate.repo_id, "category": candidate.category,
            "requestedDisposition": candidate.disposition.value,
            "productionEligible": False, "rationale": candidate.rationale}


def _semantic(manifest: dict[str, Any]) -> str:
    # Timestamp and popularity-only changes must not create new frontier events.
    return sha256({"registry": manifest["registry"], "errors": manifest["errors"],
        "receipts": [{"candidate": row["candidate"], "gate": row["gate"]}
                     for row in manifest["receipts"]]})


def refresh(*, candidates: Iterable[Candidate] = CANDIDATES,
            client: HuggingFaceMetadataClient | None = None,
            observed_at: str | None = None, source_revision: str | None = None) -> dict[str, Any]:
    registry = _registry(candidates)
    timestamp = _timestamp(observed_at)
    if source_revision is not None:
        _revision(source_revision)
    hf = client or HuggingFaceMetadataClient()
    receipts, errors = [], []
    for candidate in registry:
        try:
            receipts.append(build_receipt(candidate, hf.fetch(candidate.repo_id), observed_at=timestamp))
        except FrontierError:
            # Do not publish exception messages, tokens, response bodies, or stale success.
            errors.append({"repoId": candidate.repo_id, "reasonCode": "metadata_unavailable_or_invalid"})
    manifest = {
        "schema": "szl.forge.hf-frontier-manifest.v2", "observedAt": timestamp,
        "sourceRepository": SOURCE_REPOSITORY, "sourceRevision": source_revision,
        "authorityChain": list(AUTHORITY_CHAIN), "registry": [_spec(c) for c in registry],
        "candidateCount": len(registry), "receipts": receipts, "errors": errors,
        "scanStatus": "COMPLETE" if not errors else "PARTIAL" if receipts else "UNAVAILABLE",
        **DENIED_AUTHORITY, "productionDisposition": "HOLD",
    }
    manifest["semanticSha256"] = _semantic(manifest)
    manifest["manifestSha256"] = sha256(manifest)
    # Keep every manifest bounded so a writer cannot emit what a reader rejects.
    if len(_canonical(manifest)) > MAX_JSON_BYTES // 2:
        raise FrontierError("frontier manifest exceeded bound")
    return manifest


def _hash_check(value: dict[str, Any], field: str) -> None:
    if not isinstance(value, dict) or value.get(field) != sha256({k: v for k, v in value.items() if k != field}):
        raise FrontierError("evidence digest mismatch")


def verify_manifest(manifest: dict[str, Any], *, candidates: Iterable[Candidate] = CANDIDATES,
                    expected_source_revision: str | None = None) -> None:
    """Verify integrity and policy, not publisher authenticity or benchmark quality."""
    try:
        _verify_manifest(manifest, candidates, expected_source_revision)
    except (KeyError, TypeError, ValueError, AttributeError, RecursionError) as exc:
        raise FrontierError("frontier manifest failed integrity/policy verification") from exc


def _verify_manifest(manifest, candidates, expected_source_revision) -> None:
    if set(manifest) != {
        "schema", "observedAt", "sourceRepository", "sourceRevision", "authorityChain",
        "registry", "candidateCount", "receipts", "errors", "scanStatus",
        "productionDisposition", "semanticSha256", "manifestSha256", *DENIED_AUTHORITY,
    }:
        raise FrontierError("unexpected manifest fields")
    _hash_check(manifest, "manifestSha256")
    registry = _registry(candidates)
    expected = {row.repo_id: row for row in registry}
    if manifest["schema"] != "szl.forge.hf-frontier-manifest.v2" or manifest["sourceRepository"] != SOURCE_REPOSITORY:
        raise FrontierError("unexpected manifest identity")
    if manifest["authorityChain"] != list(AUTHORITY_CHAIN) or manifest["productionDisposition"] != "HOLD":
        raise FrontierError("invalid authority boundary")
    if any(manifest.get(key) is not False for key in DENIED_AUTHORITY):
        raise FrontierError("discovery authority escalation")
    if _canonical(manifest["registry"]) != _canonical([_spec(c) for c in registry]):
        raise FrontierError("registry does not match trusted candidates")
    if type(manifest["candidateCount"]) is not int or manifest["candidateCount"] != len(registry):
        raise FrontierError("candidate count mismatch")
    timestamp = _timestamp(_text(manifest["observedAt"], 64))
    if manifest["sourceRevision"] is not None:
        _revision(manifest["sourceRevision"])
    if expected_source_revision is not None and manifest["sourceRevision"] != _revision(expected_source_revision):
        raise FrontierError("source revision mismatch")
    if not isinstance(manifest["receipts"], list) or not isinstance(manifest["errors"], list):
        raise FrontierError("invalid observation lists")
    seen = set()
    for receipt in manifest["receipts"]:
        if set(receipt) != {"schema", "observedAt", "authorityChain", "candidate", "gate",
                            "rawMetadataSha256", "evidenceClass", "productionDisposition", "receiptSha256"}:
            raise FrontierError("unexpected receipt fields")
        _hash_check(receipt, "receiptSha256")
        if receipt["schema"] != "szl.forge.hf-frontier-receipt.v2" or receipt["observedAt"] != timestamp:
            raise FrontierError("invalid receipt identity or timestamp")
        if receipt["authorityChain"] != list(AUTHORITY_CHAIN) or receipt["productionDisposition"] != "HOLD" or receipt["evidenceClass"] != "PUBLIC_HUGGING_FACE_METADATA_ONLY":
            raise FrontierError("receipt authority mismatch")
        if not re.fullmatch(r"[0-9a-f]{64}", receipt["rawMetadataSha256"]):
            raise FrontierError("invalid raw metadata digest")
        metadata = receipt["candidate"]
        repo_id = metadata["repoId"]
        if repo_id in seen or repo_id not in expected:
            raise FrontierError("duplicate or unknown receipt")
        seen.add(repo_id)
        candidate = expected[repo_id]
        if _canonical({key: metadata[key] for key in _spec(candidate)}) != _canonical(_spec(candidate)):
            raise FrontierError("candidate policy changed")
        # Re-normalize observed facts: this catches numeric booleans, license/tag
        # conflicts, malformed fields, unknown signals, and extra authority keys.
        signals = _strings(metadata["codeSignals"])
        reconstructed = normalize_metadata(candidate, {
            "id": repo_id, "sha": metadata["revision"], "tags": metadata["tags"],
            "cardData": {"license": metadata["licenseDeclarations"]},
            "gated": metadata["gated"], "private": metadata["private"],
            "disabled": metadata["disabled"], "pipeline_tag": metadata["pipelineTag"],
            "library_name": metadata["libraryName"], "lastModified": metadata["lastModified"],
            "config": {"auto_map": {"observed": "untrusted"}} if "config:auto_map" in signals else None,
            "transformersInfo": {"auto_map": {"observed": "untrusted"}} if "transformersInfo:auto_map" in signals else None,
            "siblings": [{"rfilename": "observed.py"}] if "repository_python_file" in signals else [],
        })
        if _canonical(metadata) != _canonical(reconstructed):
            raise FrontierError("normalized evidence is inconsistent")
        if _canonical(receipt["gate"]) != _canonical(gate(metadata)):
            raise FrontierError("gate decision does not match evidence")
    for error in manifest["errors"]:
        if set(error) != {"repoId", "reasonCode"} or error["reasonCode"] != "metadata_unavailable_or_invalid":
            raise FrontierError("invalid error receipt")
        if error["repoId"] in seen or error["repoId"] not in expected:
            raise FrontierError("duplicate or unknown failure")
        seen.add(error["repoId"])
    status = "COMPLETE" if not manifest["errors"] else "PARTIAL" if manifest["receipts"] else "UNAVAILABLE"
    if seen != set(expected) or manifest["scanStatus"] != status or manifest["semanticSha256"] != _semantic(manifest):
        raise FrontierError("coverage or semantic digest mismatch")


def project_manifest(manifest: dict[str, Any], *, now: str | None = None,
                     max_age_seconds: int = 86400) -> dict[str, Any]:
    """Build a read-only consumer contract; writing it is NOT deploying a website."""
    verify_manifest(manifest)
    _revision(manifest["sourceRevision"])
    if type(max_age_seconds) is not int or not 1 <= max_age_seconds <= 86400:
        raise FrontierError("invalid freshness budget")
    age = (dt.datetime.fromisoformat(_timestamp(now)) -
           dt.datetime.fromisoformat(manifest["observedAt"])).total_seconds()
    if age < -60 or age > max_age_seconds:
        raise FrontierError("observation is stale or future-dated")
    projection = {
        "schema": "szl.forge.hf-frontier-projection.v2", "authorityChain": list(AUTHORITY_CHAIN),
        "sourceRepository": SOURCE_REPOSITORY, "sourceRevision": manifest["sourceRevision"],
        "manifestSha256": manifest["manifestSha256"], "semanticSha256": manifest["semanticSha256"],
        "observedAt": manifest["observedAt"], "scanStatus": manifest["scanStatus"],
        "candidates": [{"repoId": row["candidate"]["repoId"], "revision": row["candidate"]["revision"],
                        "effectiveDisposition": row["gate"]["effectiveDisposition"],
                        "reasonCodes": list(row["gate"]["reasonCodes"])} for row in manifest["receipts"]],
        "errors": [dict(row) for row in manifest["errors"]], **DENIED_AUTHORITY,
        "modelEvaluation": "NOT_PERFORMED", "runtimeVerification": "NOT_PERFORMED",
        "productionDisposition": "HOLD",
    }
    return {**projection, "projectionSha256": sha256(projection)}


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Unique temporary file + fsync + atomic replacement, not immutable storage."""
    text = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    if len(text.encode("utf-8")) > MAX_JSON_BYTES:
        raise FrontierError("serialized evidence exceeded bound")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".hf-frontier-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def plan() -> dict[str, Any]:
    return {"schema": "szl.forge.hf-frontier-plan.v2", "authorityChain": list(AUTHORITY_CHAIN),
            "candidates": [_spec(c) for c in CANDIDATES], "networkAction": "metadata-only",
            **DENIED_AUTHORITY}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SZL bounded Hugging Face frontier observations")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--refresh", action="store_true", help="fetch live public Hub metadata")
    modes.add_argument("--verify", type=Path, help="verify a v2 manifest offline")
    parser.add_argument("--source-revision", help="source commit; asserted, not authenticated here")
    parser.add_argument("--output", type=Path, default=Path("evidence/hf-frontier/latest.json"))
    parser.add_argument("--projection-output", type=Path, help="write a source-bound, non-serving consumer contract")
    args = parser.parse_args(argv)
    if args.projection_output and not (args.refresh or args.verify):
        parser.error("projection requires --refresh or --verify")
    if args.projection_output and args.projection_output.resolve() == (args.verify or args.output).resolve():
        parser.error("projection must not overwrite the source manifest")
    try:
        if not args.refresh and not args.verify:
            print(json.dumps(plan(), indent=2, sort_keys=True))
            return 0
        if args.refresh:
            manifest = refresh(source_revision=args.source_revision)
        else:
            with args.verify.open("rb") as stream:
                manifest = strict_json(stream.read(MAX_JSON_BYTES + 1))
        verify_manifest(manifest, expected_source_revision=args.source_revision)
        if args.refresh:
            write_manifest(args.output, manifest)
        if args.projection_output:
            write_manifest(args.projection_output, project_manifest(manifest))
        print(json.dumps({"scanStatus": manifest["scanStatus"], "manifestSha256": manifest["manifestSha256"]}))
        return 0 if manifest["scanStatus"] == "COMPLETE" else 2
    except (FrontierError, OSError):
        print("frontier observation failed; no qualification or promotion authorized", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
