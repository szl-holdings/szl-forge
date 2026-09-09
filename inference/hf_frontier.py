"""Fail-closed Hugging Face frontier candidate registry and evidence receipts.

Copyright 2026 SZL Holdings. SPDX-License-Identifier: Apache-2.0

This module discovers metadata only. It never downloads weights, executes remote
code, trains, publishes, deploys, or promotes a candidate. Promotion remains an
explicit governed action outside this module.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

HF_API = "https://huggingface.co/api/models/"
AUTHORITY_CHAIN = ("GitHub", "Hugging Face", "a-11-oy.com", "a11oy.net")
SHA40 = re.compile(r"^[0-9a-f]{40}$")
ALLOWED_LICENSES = frozenset({
    "apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause", "isc",
})
REMOTE_CODE_TAGS = frozenset({"custom_code", "trust_remote_code"})


class FrontierError(ValueError):
    """Raised when upstream evidence is incomplete or violates an SZL boundary."""


class Disposition(str, Enum):
    WATCH = "WATCH"
    EVALUATE = "EVALUATE"
    HOLD = "HOLD"
    REJECT = "REJECT"


@dataclasses.dataclass(frozen=True, slots=True)
class Candidate:
    repo_id: str
    category: str
    disposition: Disposition
    rationale: str
    production_eligible: bool = False

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", self.repo_id):
            raise FrontierError("invalid Hugging Face repository id")
        if not self.category.strip() or not self.rationale.strip():
            raise FrontierError("candidate metadata cannot be blank")
        if self.production_eligible:
            raise FrontierError("frontier discovery cannot pre-authorize production")


CANDIDATES: tuple[Candidate, ...] = (
    Candidate(
        repo_id="bodhan-ai/indic-transcribe-core",
        category="multilingual-asr",
        disposition=Disposition.EVALUATE,
        rationale=(
            "Evaluate multilingual/code-mixed ASR, language identification, "
            "latency, WER/CER, resource footprint, and data/license constraints."
        ),
    ),
    Candidate(
        repo_id="bodhan-ai/indic-speak",
        category="multilingual-tts",
        disposition=Disposition.EVALUATE,
        rationale=(
            "Evaluate multilingual/code-mixed TTS for intelligibility, speaker "
            "quality, latency, safety, resource footprint, and license constraints."
        ),
    ),
    Candidate(
        repo_id="gdiamos/amx-reasoning-v1-instruct",
        category="cpu-kernel-architecture",
        disposition=Disposition.WATCH,
        rationale=(
            "Study AMX/CPU-oriented architecture, linear/sliding-window attention "
            "and efficiency patterns; do not treat the checkpoint as a production LLM."
        ),
    ),
)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _license_from(metadata: dict[str, Any]) -> str | None:
    card = metadata.get("cardData")
    if isinstance(card, dict):
        value = card.get("license")
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    for tag in metadata.get("tags") or []:
        if isinstance(tag, str) and tag.startswith("license:"):
            return tag.split(":", 1)[1].strip().lower() or None
    return None


def normalize_metadata(candidate: Candidate, raw: dict[str, Any]) -> dict[str, Any]:
    """Reduce mutable Hub metadata to the bounded evidence SZL actually admits."""
    if not isinstance(raw, dict):
        raise FrontierError("Hugging Face metadata must be an object")
    model_id = raw.get("modelId") or raw.get("id")
    if model_id != candidate.repo_id:
        raise FrontierError("upstream repository identity mismatch")
    revision = raw.get("sha")
    if not isinstance(revision, str) or not SHA40.fullmatch(revision):
        raise FrontierError("upstream revision is absent or not a 40-char commit SHA")

    tags = sorted({tag for tag in (raw.get("tags") or []) if isinstance(tag, str)})
    license_id = _license_from(raw)
    gated = raw.get("gated")
    if gated not in (True, False, "auto", "manual", None):
        gated = "unknown"

    return {
        "repoId": candidate.repo_id,
        "revision": revision,
        "category": candidate.category,
        "requestedDisposition": candidate.disposition.value,
        "productionEligible": False,
        "license": license_id,
        "licenseAdmitted": license_id in ALLOWED_LICENSES if license_id else False,
        "gated": gated,
        "pipelineTag": raw.get("pipeline_tag"),
        "libraryName": raw.get("library_name"),
        "tags": tags,
        "remoteCodeSignal": any(tag in REMOTE_CODE_TAGS for tag in tags),
        "lastModified": raw.get("lastModified"),
        "sourceUrl": f"https://huggingface.co/{candidate.repo_id}",
        "rationale": candidate.rationale,
    }


def gate(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return a deterministic, fail-closed evaluation admission decision."""
    reasons: list[str] = []
    if not metadata.get("license"):
        reasons.append("license_unknown")
    elif not metadata.get("licenseAdmitted"):
        reasons.append("license_not_admitted")
    if metadata.get("remoteCodeSignal"):
        reasons.append("remote_code_signal")
    if metadata.get("gated") not in (False, None):
        reasons.append("gated_repository")

    requested = metadata["requestedDisposition"]
    if requested == Disposition.WATCH.value:
        effective = Disposition.WATCH.value
    elif reasons:
        effective = Disposition.HOLD.value
    else:
        effective = Disposition.EVALUATE.value

    return {
        "effectiveDisposition": effective,
        "evaluationMetadataAdmitted": effective in {
            Disposition.WATCH.value,
            Disposition.EVALUATE.value,
            Disposition.HOLD.value,
        },
        "weightDownloadAuthorized": False,
        "remoteCodeExecutionAuthorized": False,
        "trainingAuthorized": False,
        "publicationAuthorized": False,
        "deploymentAuthorized": False,
        "automaticPromotionAuthorized": False,
        "reasonCodes": sorted(reasons),
        "remainingGates": [
            "explicit license/data-rights admission",
            "pinned artifact inventory and cryptographic digests",
            "category-specific independent benchmark",
            "runtime/dependency closure and sandbox qualification",
            "SZL Nemo/A11oy policy witness",
            "rollback and product/proof projection",
            "human promotion approval",
        ],
    }


class HuggingFaceMetadataClient:
    """Tiny stdlib-only client for public model metadata; no artifact downloads."""

    def __init__(self, *, timeout_seconds: float = 15.0, user_agent: str = "szl-forge/0.2"):
        if timeout_seconds <= 0 or timeout_seconds > 60:
            raise FrontierError("timeout must be within (0, 60]")
        self.timeout_seconds = timeout_seconds
        self.user_agent = user_agent

    def fetch(self, repo_id: str) -> dict[str, Any]:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo_id):
            raise FrontierError("invalid Hugging Face repository id")
        url = HF_API + urllib.parse.quote(repo_id, safe="/")
        request = urllib.request.Request(
            url,
            headers={"Accept": "application/json", "User-Agent": self.user_agent},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                if response.status != 200:
                    raise FrontierError(f"Hub metadata HTTP status {response.status}")
                body = response.read(2 * 1024 * 1024 + 1)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise FrontierError("Hub metadata fetch failed") from exc
        if len(body) > 2 * 1024 * 1024:
            raise FrontierError("Hub metadata exceeded size bound")
        try:
            parsed = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise FrontierError("Hub metadata is not valid JSON") from exc
        if not isinstance(parsed, dict):
            raise FrontierError("Hub metadata response must be an object")
        return parsed


def build_receipt(
    candidate: Candidate,
    raw_metadata: dict[str, Any],
    *,
    observed_at: str | None = None,
) -> dict[str, Any]:
    normalized = normalize_metadata(candidate, raw_metadata)
    decision = gate(normalized)
    timestamp = observed_at or dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    receipt: dict[str, Any] = {
        "schema": "szl.forge.hf-frontier-receipt.v1",
        "observedAt": timestamp,
        "authorityChain": list(AUTHORITY_CHAIN),
        "candidate": normalized,
        "gate": decision,
        "evidenceClass": "PUBLIC_HUGGING_FACE_METADATA_ONLY",
        "productionDisposition": "HOLD",
    }
    receipt["receiptSha256"] = sha256(receipt)
    return receipt


def refresh(
    *,
    candidates: Iterable[Candidate] = CANDIDATES,
    client: HuggingFaceMetadataClient | None = None,
) -> dict[str, Any]:
    """Fetch candidate metadata and return one deterministic frontier manifest."""
    hf = client or HuggingFaceMetadataClient()
    receipts = [build_receipt(candidate, hf.fetch(candidate.repo_id)) for candidate in candidates]
    manifest: dict[str, Any] = {
        "schema": "szl.forge.hf-frontier-manifest.v1",
        "authorityChain": list(AUTHORITY_CHAIN),
        "candidateCount": len(receipts),
        "receipts": receipts,
        "automaticPromotionAuthorized": False,
        "productionDisposition": "HOLD",
    }
    manifest["manifestSha256"] = sha256(manifest)
    return manifest


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    text = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)


def plan() -> dict[str, Any]:
    return {
        "schema": "szl.forge.hf-frontier-plan.v1",
        "authorityChain": list(AUTHORITY_CHAIN),
        "candidates": [
            {
                "repoId": candidate.repo_id,
                "category": candidate.category,
                "requestedDisposition": candidate.disposition.value,
                "productionEligible": False,
                "rationale": candidate.rationale,
            }
            for candidate in CANDIDATES
        ],
        "networkAction": "metadata-only",
        "weightDownloadAuthorized": False,
        "remoteCodeExecutionAuthorized": False,
        "trainingAuthorized": False,
        "deploymentAuthorized": False,
        "automaticPromotionAuthorized": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SZL fail-closed Hugging Face frontier evaluator")
    parser.add_argument("--refresh", action="store_true", help="fetch live public Hub metadata")
    parser.add_argument("--output", type=Path, default=Path("evidence/hf-frontier/latest.json"))
    args = parser.parse_args(argv)
    payload = refresh() if args.refresh else plan()
    if args.refresh:
        write_manifest(args.output, payload)
        print(str(args.output))
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
