"""Public metadata-only acquisition and conservative material-delta detection.

No weights, datasets, archives, upstream Python, or models are downloaded by this
module. An observed upstream SHA is not an admitted production recipe. Materiality
is about changed evidence, not popularity or an edited collection timestamp.
"""
from __future__ import annotations

import dataclasses
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import PurePosixPath
from typing import Any

from .core import MAX_JSON, EvidenceError, digest, observation_shell, parse_json, receipt, require, sha256, text, verify_receipt

ALLOWED_HOSTS = {"huggingface.co", "api.github.com", "a-11-oy.com", "a11oy.net", "szlholdings-a11oy.hf.space"}
REPO_PATTERN = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9_][A-Za-z0-9_.-]{0,95}")


@dataclasses.dataclass(frozen=True)
class Candidate:
    key: str
    repo: str
    kind: str
    lane: str
    disposition: str = "EVALUATE"

    def __post_init__(self) -> None:
        require(type(self.key) is str and bool(re.fullmatch(r"[a-z][a-z0-9-]{1,79}", self.key)), "invalid_candidate_key")
        require(isinstance(self.repo, str) and bool(REPO_PATTERN.fullmatch(self.repo)), "invalid_repository")
        require(".." not in self.repo and "--" not in self.repo and not self.repo.endswith(".git"), "invalid_repository")
        require(self.kind in {"model", "dataset"}, "unsupported_repository_kind")
        text(self.lane, 80)
        require(self.disposition in {"WATCH", "EVALUATE", "HOLD", "REJECT"}, "invalid_disposition")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise EvidenceError("redirect_not_admitted")


class PublicClient:
    """Unauthenticated fixed-origin GET, no ambient token and finite retry budget."""
    def __init__(self, timeout: float = 12.0):
        require(type(timeout) in (int, float) and 0 < timeout <= 30, "invalid_timeout")
        self.timeout = timeout
        self.opener = urllib.request.build_opener(NoRedirect())

    def get(self, url: str) -> tuple[bytes, str]:
        parsed = urllib.parse.urlsplit(url)
        require(parsed.scheme == "https" and parsed.hostname in ALLOWED_HOSTS,
                "origin_not_admitted")
        require(not parsed.username and not parsed.password and parsed.port in (None, 443)
                and not parsed.fragment, "invalid_url")
        require(len(url) <= 2048 and not any(ord(c) < 32 for c in url), "url_bound")
        request = urllib.request.Request(url, method="GET", headers={
            "User-Agent": "SZL-Handoff-Metadata/1", "Accept": "application/json",
            "Cache-Control": "no-cache"})
        for attempt in range(2):
            try:
                with self.opener.open(request, timeout=self.timeout) as response:
                    require(response.status == 200 and response.geturl() == url, "unexpected_response")
                    raw = response.read(MAX_JSON + 1)
                    require(len(raw) <= MAX_JSON, "response_size_bound")
                    return raw, response.headers.get_content_type()
            except urllib.error.HTTPError as exc:
                code = exc.code
                retry = exc.headers.get("Retry-After") if exc.headers else None
                exc.close()
                # Auth/access and unknown retry dates are never bypassed.
                if code not in {429, 500, 502, 503, 504} or attempt == 1:
                    raise EvidenceError("http_" + str(code)) from exc
                if retry is not None and (not re.fullmatch(r"\d{1,2}", retry) or int(retry) > 10):
                    raise EvidenceError("retry_deferred") from exc
                time.sleep(max(1, int(retry or "1")))
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                if attempt == 1:
                    raise EvidenceError("public_transport_unavailable") from exc
                time.sleep(1)
        raise EvidenceError("retry_budget_exhausted")

    def json(self, url: str) -> dict[str, Any]:
        raw, content_type = self.get(url)
        require(content_type == "application/json" or content_type.endswith("+json"), "json_content_type_required")
        return parse_json(raw)


def _identity(raw: dict[str, Any], candidate: Candidate) -> None:
    identities = [raw[key] for key in ("id", "modelId") if key in raw]
    require(bool(identities) and all(identity == candidate.repo for identity in identities), "upstream_identity_mismatch")
    digest(raw.get("sha"), 40)


def _file_facts(raw: dict[str, Any]) -> tuple[list[dict[str, Any]], bool, bool]:
    """Interpret declared file identities only; never call them locally hashed bytes."""
    siblings = raw.get("siblings")
    require(type(siblings) is list and len(siblings) <= 8192, "file_inventory_missing_or_large")
    results, complete, has_code, seen = [], True, False, set()
    for entry in siblings:
        require(type(entry) is dict, "invalid_file_entry")
        name = text(entry.get("rfilename"), 512)
        path = PurePosixPath(name)
        require(not path.is_absolute() and ".." not in path.parts and "\\" not in name
                and str(path) == name and name not in seen, "invalid_or_duplicate_file_path")
        seen.add(name)
        lower = name.lower()
        has_code |= lower.endswith((".py", ".so", ".dll", ".dylib", ".sh"))
        license_document = any(part.startswith(("license", "licence", "notice", "terms")) for part in (path.name.lower(),))
        if lower.endswith((".md", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg")) and not license_document:
            continue
        lfs = entry.get("lfs")
        identity = entry.get("blobId")
        size = entry.get("size")
        kind = "git_blob_sha1"
        if lfs is not None:
            require(type(lfs) is dict, "invalid_lfs_metadata")
            values = [lfs[key] for key in ("sha256", "oid") if key in lfs]
            for value in values:
                digest(value, 64)
            require(len(set(values)) <= 1, "lfs_identity_conflict")
            identity = values[0] if values else None
            size = lfs.get("size", size)
            kind = "lfs_sha256"
        if identity is None or size is None:
            complete = False
        else:
            digest(identity, 64 if kind == "lfs_sha256" else 40)
            require(type(size) is int and 0 <= size <= 10**15, "invalid_declared_file_size")
        results.append({"path": name, "declaredIdentity": identity, "identityKind": kind, "declaredBytes": size})
    return sorted(results, key=lambda row: row["path"]), complete and bool(results), has_code


def materiality(candidate: Candidate, raw: dict[str, Any]) -> dict[str, Any]:
    _identity(raw, candidate)
    card = {} if raw.get("cardData") is None else raw["cardData"]
    require(type(card) is dict, "invalid_card_metadata")
    tags = raw.get("tags", [])
    require(type(tags) is list and len(tags) <= 2048 and all(type(t) is str for t in tags), "invalid_tags")
    card_license = card.get("license")
    declarations = [] if card_license is None else [card_license] if type(card_license) is str else card_license
    require(type(declarations) is list and all(type(v) is str for v in declarations), "invalid_license_metadata")
    licenses = sorted(set(declarations + [tag[8:] for tag in tags if tag.startswith("license:")]))
    # Do not decide that two licenses mean OR vs AND from metadata tags alone.
    files, complete, file_code = _file_facts(raw)
    config = {} if raw.get("config") is None else raw["config"]
    require(type(config) is dict, "invalid_config_metadata")
    gate = raw.get("gated")
    if not (type(gate) is bool or gate is None or type(gate) is str and gate in {"auto", "manual"}):
        gate = "UNKNOWN"
    for flag in ("private", "disabled"):
        require(raw.get(flag) is None or type(raw[flag]) is bool, "invalid_visibility_flag")
    facts = {"repoId": candidate.repo, "repoType": candidate.kind,
             "licenseDeclarations": licenses, "gated": gate, "private": raw.get("private"),
             "disabled": raw.get("disabled"), "pipeline": raw.get("pipeline_tag"),
             "library": raw.get("library_name"), "config": config,
             "remoteCodeSignals": file_code or "custom_code" in tags or bool(config.get("auto_map")),
             "materialFiles": files}
    blockers = ["rights_and_license_review_required", "model_runtime_not_qualified"]
    if gate is not False:
        blockers.append("access_gated_or_unknown")
    if not complete:
        blockers.append("file_inventory_identity_incomplete")
    if facts["remoteCodeSignals"]:
        blockers.append("source_code_review_required")
    if not licenses:
        blockers.append("license_unknown")
    disposition = candidate.disposition if candidate.disposition in {"WATCH", "HOLD", "REJECT"} else "HOLD"
    return {"key": candidate.key, "repoId": candidate.repo, "repoType": candidate.kind,
            "upstreamRevision": raw["sha"], "lane": candidate.lane,
            "requestedDisposition": candidate.disposition, "effectiveDisposition": disposition,
            "materialIdentityComplete": complete, "materialFacts": facts,
            "materialSha256": sha256(facts), "rawMetadataSha256": sha256(raw),
            "blockers": blockers, "evidenceClass": "UPSTREAM_METADATA_ONLY",
            "downloadedModelBytes": False, "productionPromotion": False}


def observe(candidate: Candidate, client: PublicClient) -> dict[str, Any]:
    noun = "models" if candidate.kind == "model" else "datasets"
    base = "https://huggingface.co/api/" + noun + "/" + urllib.parse.quote(candidate.repo, safe="/")
    current = client.json(base)
    _identity(current, candidate)
    # A second lookup binds observation to immutable revision, not a moving main.
    pinned = client.json(base + "/revision/" + current["sha"] + "?blobs=true")
    _identity(pinned, candidate)
    require(pinned["sha"] == current["sha"], "pinned_lookup_revision_mismatch")
    return materiality(candidate, pinned)


def compare_observations(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    require(old["repoId"] == new["repoId"] and old["repoType"] == new["repoType"], "comparison_identity_mismatch")
    if old["materialIdentityComplete"] is not True or new["materialIdentityComplete"] is not True:
        return {"state": "INDETERMINATE", "notify": False, "reason": "incomplete_file_identity"}
    if old["materialSha256"] == new["materialSha256"]:
        return {"state": "NO_MATERIAL_CHANGE", "notify": False, "reason": "artifact_and_policy_facts_unchanged"}
    return {"state": "REVIEW_MATERIAL_DELTA", "notify": True, "reason": "artifact_or_policy_facts_changed"}


def scan(candidates: list[Candidate], source: str, client: PublicClient,
         previous: dict[str, Any] | None = None) -> dict[str, Any]:
    require(type(candidates) is list and 0 < len(candidates) <= 32, "candidate_registry_bound")
    require(all(isinstance(c, Candidate) for c in candidates), "invalid_candidate_registry")
    require(len({c.key for c in candidates}) == len(candidates), "duplicate_candidate_key")
    require(len({(c.kind, c.repo) for c in candidates}) == len(candidates), "duplicate_repository")
    baseline = {}
    if previous is not None:
        verify_receipt(previous)
        require(previous.get("schema") == "szl.handoff.frontier-scan.v1", "baseline_schema")
        baseline = {row["key"]: row for row in previous["observations"]}
    observations, failures, notices = [], [], []
    for candidate in candidates:
        try:
            result = observe(candidate, client)
            observations.append(result)
            delta = compare_observations(baseline[candidate.key], result) if candidate.key in baseline else {
                "state": "BASELINE_CAPTURED_NOT_A_NEW_RELEASE_CLAIM", "notify": False}
            if delta["notify"]:
                notices.append({"key": candidate.key, **delta,
                                "sourceUrl": "https://huggingface.co/" + ("datasets/" if candidate.kind == "dataset" else "") + candidate.repo})
        except EvidenceError as exc:
            # Only locally defined reason codes; response bodies/tokens never enter receipts.
            failures.append({"key": candidate.key, "state": "UNAVAILABLE", "reasonCode": str(exc)})
    report = {**observation_shell("szl.handoff.frontier-scan.v1", source),
              "observations": observations, "failures": failures, "notifications": notices,
              "scanStatus": "COMPLETE" if not failures else "PARTIAL" if observations else "UNAVAILABLE",
              "completeMaterialityCoverage": not failures and all(r["materialIdentityComplete"] for r in observations)}
    return receipt(report)


def observe_release(repo: str, tag: str, expected_commit: str, client: PublicClient) -> dict[str, Any]:
    """Resolve exact upstream release/tag; annotated tag SHA != source commit SHA."""
    allowed = {("huggingface/trl", "v1.13.0"), ("huggingface/huggingface_hub", "v1.31.0"),
               ("huggingface/tau", "v0.4.2")}
    require((repo, tag) in allowed, "release_not_in_handoff_allowlist")
    digest(expected_commit, 40)
    root = "https://api.github.com/repos/" + repo
    release = client.json(root + "/releases/tags/" + tag)
    require(release.get("tag_name") == tag and release.get("draft") is False, "release_identity_or_draft")
    require(type(release.get("prerelease")) is bool, "prerelease_status_missing")
    ref = client.json(root + "/git/ref/tags/" + tag)
    require(ref.get("ref") == "refs/tags/" + tag, "tag_ref_mismatch")
    obj = ref.get("object")
    chain, seen = [], set()
    for _ in range(5):
        require(type(obj) is dict, "invalid_git_object")
        oid = digest(obj.get("sha"), 40)
        require(oid not in seen, "tag_cycle")
        seen.add(oid)
        chain.append({"sha": oid, "type": obj.get("type")})
        if obj.get("type") == "commit":
            require(oid == expected_commit, "release_commit_changed_review_required")
            return {"repository": repo, "tag": tag, "commit": oid, "gitObjectChain": chain,
                    "publishedAt": release.get("published_at"), "prerelease": release["prerelease"],
                    "upstreamReleaseMetadataSha256": sha256(release),
                    "signatureVerifiedLocally": False, "productionPromotion": False}
        require(obj.get("type") == "tag", "tag_not_a_commit")
        obj = client.json(root + "/git/tags/" + oid).get("object")
    raise EvidenceError("tag_depth_exceeded")
