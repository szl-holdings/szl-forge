#!/usr/bin/env python3
"""Read-only, immutable-revision Hugging Face evidence consistency auditor.

PASS means only that the configured documentation consistency checks passed.
It never means model qualification, safety, production readiness, or promotion.
Network mode uses only Hub read APIs. Offline snapshots preserve fetched source
bytes and hashes; an edited local README must be separately labelled a proposal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import posixpath
import re
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote, urlsplit
from urllib.request import Request, urlopen

import yaml


VALID_TYPES = {"model", "dataset", "space"}
SEVERITY = {"PASS": 0, "MISSING_BINDING": 1, "HOLD": 2, "CONFLICT": 3}
ARTIFACT_SUFFIXES = (".safetensors", ".gguf", ".bin", ".pt", ".pth", ".onnx", ".joblib")
HEX_SHA256 = re.compile(r"\b[a-fA-F0-9]{64}\b")
HEX_REVISION = re.compile(r"[a-fA-F0-9]{40,64}\Z")


def safe_relative(path: str) -> str:
    """Accept only literal, relative Hub file paths, never host filesystem paths."""
    if not isinstance(path, str) or not path or "\\" in path or "\x00" in path:
        raise ValueError(f"Invalid repository-relative path: {path!r}")
    if ":" in path or path.startswith("/") or any(p in {"..", ""} for p in path.split("/")):
        raise ValueError(f"Unsafe repository-relative path: {path!r}")
    if any(p == "." for p in path.split("/")):
        raise ValueError(f"Non-canonical repository-relative path: {path!r}")
    return str(PurePosixPath(path))


def safe_destination(root: Path, relative: str) -> Path:
    candidate = root.joinpath(*safe_relative(relative).split("/")).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError("Snapshot path leaves the selected snapshot directory")
    return candidate


def snapshot_key(entry: dict[str, Any]) -> str:
    repo = entry["repo"]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
        raise ValueError(f"Invalid repository id: {repo!r}")
    if any(part in {".", ".."} for part in repo.split("/")):
        raise ValueError(f"Invalid repository id: {repo!r}")
    return f"{entry.get('type', 'model')}--{repo.replace('/', '--')}"


def evidence_specs(entry: dict[str, Any]) -> list[dict[str, Any]]:
    specs = []
    for item in entry.get("evidence_files", []):
        spec = {"path": item} if isinstance(item, str) else dict(item)
        spec["path"] = safe_relative(spec["path"])
        specs.append(spec)
    return specs


def read_inventory(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8-sig")
    data = json.loads(raw) if path.suffix.lower() == ".json" else yaml.safe_load(raw)
    rows = data.get("repositories") if isinstance(data, dict) else data
    if not isinstance(rows, list) or not rows:
        raise ValueError("Inventory must contain a nonempty repository list")
    seen = set()
    for entry in rows:
        if not isinstance(entry, dict) or not isinstance(entry.get("repo"), str):
            raise ValueError("Every repository entry needs a repo string")
        entry.setdefault("type", "model")
        if entry["type"] not in VALID_TYPES:
            raise ValueError(f"Unsupported repository type: {entry['type']}")
        key = snapshot_key(entry)
        if key in seen:
            raise ValueError(f"Duplicate repository: {key}")
        seen.add(key)
        safe_relative(entry.get("readme", "README.md"))
        evidence_specs(entry)
        for spec in entry.get("external_evidence", []):
            safe_relative(spec["path"])
            revision = spec.get("revision", "")
            url = urlsplit(spec.get("url", ""))
            if not HEX_REVISION.fullmatch(revision) or revision not in url.path.split("/"):
                raise ValueError("External evidence URL must name its declared immutable revision as a path component")
            if url.scheme != "https" or url.hostname not in {"raw.githubusercontent.com", "huggingface.co"}:
                raise ValueError("External evidence requires an HTTPS GitHub raw or Hugging Face URL")
        for path in entry.get("required_files", []):
            safe_relative(path)
        if not isinstance(entry.get("checks", {}), dict):
            raise ValueError("checks must be an object")
    return rows


def source_url(entry: dict[str, Any], revision: str | None, path: str | None = None) -> str:
    prefix = {"model": "", "dataset": "datasets/", "space": "spaces/"}[entry.get("type", "model")]
    base = f"https://huggingface.co/{prefix}{entry['repo']}"
    return f"{base}/blob/{revision}/{path}" if revision and path else base


def fetch_snapshot(entry: dict[str, Any], cache_root: Path | None, max_bytes: int) -> dict[str, Any]:
    from huggingface_hub import HfApi, hf_hub_download

    snap: dict[str, Any] = {
        "schema_version": 1, "repo": entry["repo"], "type": entry["type"],
        "requested_revision": entry.get("revision", "main"), "revision": None,
        "fetched_at": datetime.now(timezone.utc).isoformat(), "files": [],
        "contents": {}, "file_sha256": {}, "artifact_sha256": {}, "errors": [],
        "provenance": "Hub immutable-revision read",
    }
    api = HfApi()
    try:
        info = api.repo_info(entry["repo"], repo_type=entry["type"],
                             revision=entry.get("revision", "main"), files_metadata=True)
        revision = info.sha
        if not revision or not HEX_REVISION.fullmatch(revision):
            raise ValueError("Hub did not return an immutable commit revision")
        snap["revision"] = revision
        # Read the listing again at the resolved commit to bind it unambiguously.
        info = api.repo_info(entry["repo"], repo_type=entry["type"],
                             revision=revision, files_metadata=True)
        if info.sha != revision:
            raise ValueError("Hub revision readback did not match resolved commit")
        sizes = {}
        for sibling in info.siblings or []:
            name = safe_relative(sibling.rfilename)
            snap["files"].append(name)
            sizes[name] = sibling.size
            lfs = getattr(sibling, "lfs", None)
            digest = getattr(lfs, "sha256", None) if lfs is not None else None
            if isinstance(lfs, dict):
                digest = lfs.get("sha256")
            if digest:
                snap["artifact_sha256"][name] = digest
        requested = [{"path": entry.get("readme", "README.md")}, *evidence_specs(entry)]
        for spec in requested:
            path = safe_relative(spec["path"])
            if path not in snap["files"]:
                if not spec.get("optional", False):
                    snap["errors"].append({"code": "MISSING_SOURCE_FILE", "path": path,
                                            "message": "Required source file is absent at the resolved revision"})
                continue
            size = sizes.get(path)
            if size is not None and size > max_bytes:
                snap["errors"].append({"code": "SOURCE_SIZE_LIMIT", "path": path,
                                        "message": f"Source exceeds configured {max_bytes} byte text limit"})
                continue
            try:
                # Keep tool-created cache files in the workspace/snapshot area.
                download_cache = (cache_root / "_download_cache") if cache_root else (Path.cwd() / "work" / "hf-audit-cache")
                downloaded = hf_hub_download(entry["repo"], path, repo_type=entry["type"], revision=revision,
                                             cache_dir=str(download_cache))
                raw = Path(downloaded).read_bytes()
                if len(raw) > max_bytes:
                    raise ValueError("Source exceeds configured text size limit")
                content = raw.decode("utf-8")
                snap["contents"][path] = content
                snap["file_sha256"][path] = hashlib.sha256(raw).hexdigest()
            except Exception as exc:
                snap["errors"].append({"code": "SOURCE_READ_FAILED", "path": path,
                                        "message": safe_error(exc)})
    except Exception as exc:
        snap["errors"].append({"code": "REPOSITORY_READ_FAILED", "message": safe_error(exc)})
    snap["external_sources"] = {}
    for spec in entry.get("external_evidence", []):
        path = spec["path"]
        try:
            request = Request(spec["url"], headers={"User-Agent": "SZL-read-only-evidence-audit/1"})
            with urlopen(request, timeout=30) as response:
                raw = response.read(max_bytes + 1)
            if len(raw) > max_bytes:
                raise ValueError("External evidence exceeds configured text size limit")
            digest = hashlib.sha256(raw).hexdigest()
            if spec.get("sha256") and digest != spec["sha256"]:
                raise ValueError("External evidence SHA-256 differs from inventory expectation")
            snap["contents"][path] = raw.decode("utf-8")
            snap["file_sha256"][path] = digest
            snap["external_sources"][path] = {"url": spec["url"], "revision": spec["revision"]}
        except Exception as exc:
            snap["errors"].append({"code": "EXTERNAL_EVIDENCE_READ_FAILED", "path": path, "message": safe_error(exc)})
    if cache_root:
        write_snapshot(entry, snap, cache_root)
    return snap


def safe_error(exc: Exception) -> str:
    # Avoid dumping request headers, tokens, or full credential-bearing URLs.
    text = str(exc).splitlines()[0][:400]
    text = re.sub(r"hf_[A-Za-z0-9]+", "[REDACTED]", text)
    text = re.sub(r"(?i)(token|authorization|access_token)=\S+", r"\1=[REDACTED]", text)
    return f"{type(exc).__name__}: {text}"


def write_snapshot(entry: dict[str, Any], snap: dict[str, Any], cache_root: Path) -> None:
    directory = safe_destination(cache_root, snapshot_key(entry))
    directory.mkdir(parents=True, exist_ok=True)
    # Contents stay embedded in metadata to preserve original source bytes' digest.
    (directory / "snapshot.json").write_text(json.dumps(snap, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for name, content in snap.get("contents", {}).items():
        destination = safe_destination(directory / "files", name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8", newline="")


def load_snapshot(entry: dict[str, Any], cache_root: Path) -> dict[str, Any]:
    path = safe_destination(cache_root, snapshot_key(entry)) / "snapshot.json"
    snap = json.loads(path.read_text(encoding="utf-8-sig"))
    if snap.get("repo") != entry["repo"] or snap.get("type") != entry["type"]:
        raise ValueError("Snapshot repository identity does not match inventory")
    requested = entry.get("revision", "main")
    if HEX_REVISION.fullmatch(requested) and snap.get("revision") != requested:
        raise ValueError("Snapshot immutable revision does not match inventory")
    if not HEX_REVISION.fullmatch(requested) and snap.get("requested_revision") != requested:
        raise ValueError("Snapshot requested branch/tag does not match inventory")
    # The recorded contents are authoritative; files/ is a human inspection aid.
    return snap


def walk_values(value: Any, path: str = ""):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from walk_values(child, f"{path}.{key}" if path else key)
    elif isinstance(value, list):
        for i, child in enumerate(value):
            yield from walk_values(child, f"{path}[{i}]")
    else:
        yield path, value


def measured_record(value: Any) -> bool:
    for key, item in walk_values(value):
        leaf = key.rsplit(".", 1)[-1].lower()
        if isinstance(item, str) and leaf in {"label", "maturity", "status", "evals", "evaluation_status"}:
            if item.upper() in {"MEASURED", "MEASURED_RESEARCH_ONLY", "MEASURED_RESEARCH_CANDIDATE"}:
                return True
        if leaf == "gate_ran" and item is True:
            return True
    return False


def field_value(data: Any, field: str) -> Any:
    return dict(walk_values(data)).get(field)


def evaluation_record(path: str, data: Any, explicit: list[str]) -> bool:
    """Identify evaluation sources by configuration or concrete source structure."""
    if path in explicit or re.search(r"(?:^|[/_.-])(?:eval|evaluation|evals|gate|benchmark)(?:[/_.-]|$)", path, re.I):
        return True
    if not isinstance(data, dict):
        return False
    if {"candidate_metrics", "baseline_metrics", "comparison", "decision"}.issubset(data):
        return True
    if {"candidate_results", "baseline_results", "receipt"}.issubset(data):
        return True
    return any(key.rsplit(".", 1)[-1] == "gate_ran" and value is True for key, value in walk_values(data))


def current_publication_denial(readme: str) -> bool:
    """Distinguish an admission draft's current status from a stale-banner quote."""
    paragraphs = re.sub(r"(?<!\n)\n(?!\n)", " ", readme)
    for sentence in re.split(r"(?<=[.!?])\s+|\n{2,}", paragraphs):
        if not re.search(r"not\s+yet\s+published", sentence, re.I):
            continue
        historical_before = re.search(r"\b(?:earlier|older|previous|historical|former|stale)\b[^.!?]{0,140}not\s+yet\s+published", sentence, re.I)
        stale_after = re.search(r"not\s+yet\s+published[^.!?]{0,100}\b(?:was|is)\s+stale\b", sentence, re.I)
        if not historical_before and not stale_after:
            return True
    return False


def current_prose(text: str) -> list[str]:
    """Keep current factual assertions; skip disclosed history and negative examples."""
    result = []
    historical_section = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            historical_section = bool(re.search(r"\bhistorical\b|\bprevious\b|\bdeprecated\b", stripped, re.I))
        if historical_section:
            continue
        # Treat historical qualification as sentence-local. A later historical
        # sentence must never suppress a current claim on the same source line.
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z])", line):
            if re.match(r"^[\s>*_`|-]*(?:(?:the|an?)\s+)?(?:older|historical|previous|legacy|formerly|stale)\b", sentence, re.I):
                continue
            result.append(sentence)
    return result


def read_frontmatter(text: str) -> tuple[dict[str, Any] | None, str | None]:
    lines = text.lstrip("\ufeff").splitlines()
    if not lines or lines[0].strip() != "---":
        return None, "YAML front matter is absent"
    end = next((i for i, line in enumerate(lines[1:], 1) if line.strip() == "---"), None)
    if end is None:
        return None, "YAML front matter is unclosed"
    try:
        value = yaml.safe_load("\n".join(lines[1:end]))
        if not isinstance(value, dict):
            return None, "YAML front matter must be a mapping"
        return value, None
    except yaml.YAMLError as exc:
        return None, f"Invalid YAML front matter: {str(exc).splitlines()[0]}"


def local_markdown_links(text: str, readme: str) -> list[tuple[str, str | None]]:
    links = []
    for match in re.finditer(r"!?\[[^\]]*\]\(\s*<?([^\s)>]+)>?(?:\s+[\"'][^\"']*[\"'])?\s*\)", text):
        target = match.group(1)
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc or not parsed.path or parsed.path.startswith("/"):
            continue
        decoded = unquote(parsed.path)
        # Hub card routes are repository-root routes, even when written ./tree/.
        route = decoded.removeprefix("./")
        if re.match(r"^(?:tree|blob|resolve)/[^/]+/", route):
            decoded = route.split("/", 2)[2]
            base = ""
        else:
            base = posixpath.dirname(readme)
        resolved = posixpath.normpath(posixpath.join(base, decoded))
        try:
            links.append((target, safe_relative(resolved)))
        except ValueError:
            links.append((target, None))
    return links


def classify_repository(entry: dict[str, Any], snapshot: dict[str, Any]) -> dict[str, Any]:
    """Pure classification: no network, no disk access, no remote modifications."""
    findings: list[dict[str, Any]] = []

    def add(code: str, status: str, message: str, path: str | None = None):
        item = {"code": code, "status": status, "message": message}
        if path:
            item["path"] = path
        if item not in findings:
            findings.append(item)

    files = set(snapshot.get("files", []))
    contents = snapshot.get("contents", {})
    readme_path = entry.get("readme", "README.md")
    readme = contents.get(readme_path, "")
    revision = snapshot.get("revision")
    checks = entry.get("checks", {})
    if not entry.get("class") or entry.get("class") == "unclassified":
        add("EVIDENCE_CLASS_UNSPECIFIED", "HOLD", "Inventory has no explicit evidence class")
    if snapshot.get("repo") != entry["repo"] or snapshot.get("type") != entry.get("type", "model"):
        add("SNAPSHOT_IDENTITY_MISMATCH", "HOLD", "Snapshot identity does not match inventory")
    if not revision or not HEX_REVISION.fullmatch(str(revision)):
        add("IMMUTABLE_REVISION_MISSING", "HOLD", "No verified immutable repository revision is recorded")
    for error in snapshot.get("errors", []):
        if isinstance(error, dict):
            add(error.get("code", "FETCH_ERROR"), "HOLD", error.get("message", "Source read failed"), error.get("path"))
        else:
            add("FETCH_ERROR", "HOLD", str(error))
    for path in files:
        try:
            safe_relative(path)
        except ValueError as exc:
            add("UNSAFE_SOURCE_PATH", "HOLD", str(exc))
    for path, content in contents.items():
        if path not in files and path not in snapshot.get("external_sources", {}):
            add("SOURCE_NOT_IN_LISTING", "HOLD", "Retained in-repository source is absent from the captured file listing", path)
        expected_digest = snapshot.get("file_sha256", {}).get(path)
        if not expected_digest:
            add("SOURCE_HASH_UNAVAILABLE", "HOLD", "Fetched source has no recorded SHA-256", path)
        elif hashlib.sha256(content.encode("utf-8")).hexdigest() != expected_digest:
            add("SOURCE_HASH_MISMATCH", "HOLD", "Snapshot source content differs from its recorded SHA-256", path)
    if not readme:
        add("README_UNAVAILABLE", "HOLD", "README was absent, empty, or could not be read", readme_path)
    if readme_path not in files:
        add("README_NOT_IN_LISTING", "HOLD", "README is absent from captured repository file listing", readme_path)
    frontmatter, yaml_error = read_frontmatter(readme)
    if yaml_error and readme and checks.get("yaml_frontmatter", True):
        add("YAML_FRONTMATTER_INVALID", "CONFLICT", yaml_error, readme_path)
    for requirement in checks.get("frontmatter_equals", []):
        if (frontmatter or {}).get(requirement["field"]) != requirement["value"]:
            add("FRONTMATTER_VALUE_MISMATCH", "CONFLICT", requirement.get(
                "message", f"Structured card metadata differs at {requirement['field']}"), readme_path)
    evidence = {}
    for spec in [*evidence_specs(entry), *entry.get("external_evidence", [])]:
        path = spec["path"]
        external = "url" in spec
        if external and path not in snapshot.get("external_sources", {}):
            add("EXTERNAL_EVIDENCE_UNVERIFIED", "HOLD", "Configured immutable external evidence has not been fetched", path)
            continue
        if external:
            source = snapshot["external_sources"][path]
            if source.get("url") != spec["url"] or source.get("revision") != spec.get("revision"):
                add("EXTERNAL_SOURCE_IDENTITY_MISMATCH", "HOLD", "External source snapshot identity differs from configured immutable evidence", path)
                continue
        if not external and path not in files:
            if not spec.get("optional", False):
                add("EVIDENCE_ABSENT", "HOLD", "Configured evidence file is absent from this revision", path)
            continue
        if path not in contents:
            add("EVIDENCE_UNREAD", "HOLD", "Evidence file exists but was not read", path)
            continue
        expected = spec.get("sha256")
        actual = snapshot.get("file_sha256", {}).get(path)
        if expected and expected.lower() != str(actual).lower():
            add("EVIDENCE_HASH_MISMATCH", "CONFLICT", "Evidence file SHA-256 differs from inventory expectation", path)
        if path.endswith(".json"):
            try:
                evidence[path] = json.loads(contents[path].lstrip("\ufeff"))
            except (TypeError, ValueError):
                add("EVIDENCE_JSON_INVALID", "HOLD", "Evidence JSON cannot be parsed", path)
        elif path.endswith((".yaml", ".yml")):
            try:
                evidence[path] = yaml.safe_load(contents[path])
            except yaml.YAMLError:
                add("EVIDENCE_YAML_INVALID", "HOLD", "Evidence YAML cannot be parsed", path)
    for path, data in evidence.items():
        trained_weights = field_value(data, "model.trained_weights_present")
        trained_claim = field_value(data, "claims.trained_model")
        if trained_weights is False and isinstance(trained_claim, str) and re.search(r"(?:SURROGATE|WEIGHTS|MODEL)_PRESENT", trained_claim):
            add("SOURCE_EVIDENCE_CONFLICT", "HOLD", "Source evidence sets model.trained_weights_present=false while claims.trained_model asserts a present artifact; README disclosure cannot repair the source record", path)
    for path in entry.get("required_files", []):
        if path not in files:
            add("CLAIMED_FILE_ABSENT", "CONFLICT", "Inventory-declared shipped file is absent", path)
    for target, path in local_markdown_links(readme, readme_path):
        if path is None:
            add("UNSAFE_LOCAL_LINK", "CONFLICT", f"Local README link escapes repository: {target}", readme_path)
        elif path not in files and not any(name.startswith(path.rstrip("/") + "/") for name in files):
            add("BROKEN_LOCAL_LINK", "CONFLICT", f"Local README link does not exist at this revision: {target}", readme_path)
    explicit_eval_records = entry.get("evaluation_records", checks.get("evaluation_records", []))
    eval_sources = [path for path, data in evidence.items() if evaluation_record(path, data, explicit_eval_records)]
    measured = [path for path in eval_sources if measured_record(evidence[path])]
    prose = current_prose(readme)
    no_eval = re.compile(r"none[-_ ]this[-_ ]run|\bno\s+(?:held[- ]out\s+)?eval(?:uation)?(?:s|\s+(?:record|receipt))?\s*(?:[.|:]|exists|available|has\s+been\s+run)|\bnot\s+evaluated\b", re.I)
    no_eval_assertions = [line for line in prose if no_eval.search(line)
                          and not re.search(r"\btraining\s+receipt\b.{0,100}\b(?:retains|records|lists|reports)\b|\bfor\s+(?:its\s+|the\s+)?training\s+step\b", line, re.I)]
    if measured and no_eval_assertions:
        add("MEASURED_RECORD_CONTRADICTS_NO_EVAL", "CONFLICT", "Current README no-evaluation claim conflicts with configured measured evidence: " + ", ".join(measured), readme_path)
    # A positive k/n held-out claim needs a configured, readable evaluation record.
    eval_claims = [line for line in prose if re.search(r"held[- ]out|\b(?:evaluation|eval|generate|grounding|abstain)\b", line, re.I)
                   and re.search(r"\b\d+\s*/\s*\d+\b", line)
                   and not re.search(r"\b(?:unsupported|removed|not\s+supported|does\s+not\s+establish|no\s+committed)\b", line, re.I)]
    if eval_claims and not eval_sources:
        add("EVALUATION_RECEIPT_MISSING", "HOLD", "README contains an evaluation fraction but no readable configured evaluation receipt", readme_path)
    for line in prose:
        for match in re.finditer(r"(?:joblib|torch)\.load\(\s*['\"]([^'\"]+)['\"]", line):
            path = match.group(1)
            if path not in files and not re.search(r"do\s+not|cannot|must\s+not", line, re.I):
                add("LOAD_TARGET_ABSENT", "CONFLICT", f"Current loading instructions reference absent artifact: {path}", readme_path)
    if frontmatter:
        tags = frontmatter.get("tags", [])
        tags = tags if isinstance(tags, list) else [str(tags)]
        loadable = frontmatter.get("library_name") in {"transformers", "peft", "diffusers", "sentence-transformers"} or any(tag in {"gguf", "safetensors"} for tag in tags)
        if loadable and not any(path.endswith(ARTIFACT_SUFFIXES) for path in files):
            if not re.search(r"source\s+pointer|historical\s+stub|roadmap|no\s+(?:model\s+)?weights|weights?\s+(?:are\s+)?(?:absent|not\s+(?:present|published))|empty\s+(?:hub\s+)?mirror", readme, re.I):
                add("LOADABLE_METADATA_WITHOUT_ARTIFACT", "CONFLICT", "Model metadata implies loadable weights but no supported artifact is listed", readme_path)
    for item in checks.get("required_patterns", []):
        item = {"pattern": item} if isinstance(item, str) else item
        if not re.search(item["pattern"], "\n".join(prose), re.I | re.M):
            add(item.get("code", "REQUIRED_DISCLOSURE_MISSING"), "HOLD", item.get("message", f"Required disclosure is absent: {item['pattern']}"), readme_path)
    for item in checks.get("prohibited_patterns", []):
        item = {"pattern": item} if isinstance(item, str) else item
        if re.search(item["pattern"], "\n".join(prose), re.I | re.M):
            add(item.get("code", "PROHIBITED_CLAIM_PRESENT"), "CONFLICT", item.get("message", f"Unsupported current claim matches: {item['pattern']}"), readme_path)
    for spec in checks.get("numeric_claims", []):
        data = evidence.get(spec["evidence_file"])
        numerator = field_value(data, spec["numerator_field"])
        denominator = field_value(data, spec["denominator_field"])
        if not isinstance(numerator, (int, float)) or not isinstance(denominator, (int, float)):
            add("NUMERIC_EVIDENCE_UNAVAILABLE", "HOLD", "Configured numeric evidence fields are absent or are not numeric", spec["evidence_file"])
            continue
        matches = list(re.finditer(spec["pattern"], "\n".join(prose), re.I | re.M))
        for match in matches:
            actual = (float(match.group(1)), float(match.group(2)))
            if actual != (numerator, denominator):
                add("NUMERIC_CLAIM_CONTRADICTION", "CONFLICT", spec.get("message", f"README reports {actual[0]:g}/{actual[1]:g}; source records {numerator:g}/{denominator:g}"), spec["evidence_file"])
        if spec.get("required", False) and not matches:
            add("NUMERIC_DISCLOSURE_MISSING", "HOLD", "Configured measured result is not visible in README", readme_path)
    for spec in checks.get("source_disclosures", []):
        value = field_value(evidence.get(spec["evidence_file"]), spec["field"])
        if value is None:
            add("SOURCE_FIELD_UNAVAILABLE", "HOLD", "Configured canonical evidence field is unavailable", spec["evidence_file"])
        elif value == spec["value"] and not re.search(spec["readme_pattern"], readme, re.I | re.M):
            add("CANONICAL_STATE_NOT_DISCLOSED", "CONFLICT", spec.get("message", f"Canonical source field {spec['field']}={value!r} lacks required README disclosure"), spec["evidence_file"])
    boundary_patterns = {
        "publication_eligible_false": r"publication[_ -]eligible\s*[`\"']?\s*[:=]\s*[`\"']?false|not\s+promotable|no\s+(?:publication\s+)?promotion",
        "no_autonomy": r"(?:not|non)[- ]autonomous|no\s+autonomy|autonomy[_ -]eligible\s*[`\"']?\s*[:=]\s*[`\"']?false|autonomy\s*(?:[:=]\s*)?(?:forbidden|none|false)",
        "no_deployment": r"no\s+(?:production\s+)?deployment|not\s+(?:for\s+)?(?:production|deploy)|deployment\s+(?:is\s+)?(?:forbidden|not\s+authoriz)|not\s+authoriz\w*\s+(?:for\s+)?(?:production|deployment)",
        "hold": r"\bHOLD\b", "unsigned": r"\bUNSIGNED(?:_HONEST)?\b",
        "proposal_only": r"proposal[_ -]only",
    }
    for boundary in entry.get("required_boundaries", []):
        pattern = boundary_patterns.get(boundary)
        if pattern is None:
            add("UNKNOWN_BOUNDARY_CHECK", "HOLD", f"Unknown required boundary: {boundary}")
        elif not re.search(pattern, "\n".join(prose), re.I):
            add("REQUIRED_BOUNDARY_MISSING", "HOLD", f"Required explicit boundary is not visible: {boundary}", readme_path)
    binding = checks.get("artifact_binding")
    if binding:
        mappings = binding.get("bindings", [])
        artifacts = binding.get("artifact_paths", [spec["artifact_path"] for spec in mappings])
        bound = set()
        for spec in mappings:
            path = spec["artifact_path"]
            data = evidence.get(spec["evidence_file"])
            digest = snapshot.get("artifact_sha256", {}).get(path) or snapshot.get("file_sha256", {}).get(path)
            asserted_path = field_value(data, spec["artifact_field"])
            asserted_digest = field_value(data, spec["sha256_field"])
            denied = any(key.rsplit(".", 1)[-1] in {"artifactBinding", "artifact_binding"} and value is False
                         for key, value in walk_values(data))
            if (path in files and path in artifacts and not denied and asserted_path == path
                    and isinstance(asserted_digest, str) and isinstance(digest, str)
                    and HEX_SHA256.fullmatch(asserted_digest) and asserted_digest.lower() == digest.lower()):
                bound.add(path)
        if not artifacts or not set(artifacts).issubset(bound):
            add("ARTIFACT_BINDING_INCOMPLETE", "MISSING_BINDING", "Configured evaluation records do not bind every configured artifact to its exact published SHA-256")
    publication = checks.get("publication_record")
    if publication:
        data = evidence.get(publication.get("path"))
        field = publication.get("field", "published")
        observed = field_value(data, field)
        if observed is None:
            add("PUBLICATION_FIELD_UNAVAILABLE", "HOLD", "Configured canonical publication field is unavailable", publication.get("path"))
        elif observed == publication.get("value", True) and current_publication_denial(readme):
            add("PUBLICATION_RECORD_CONTRADICTION", "CONFLICT", "Current README says not yet published, but configured publication record records Hub publication", readme_path)
    derived = checks.get("derived_artifact_binding")
    derivative_claim_lines = [line for line in prose if not re.search(r"\b(?:not|never|doesn.t)\b|\bdoes\s+not\b", line, re.I)]
    if derived and re.search(derived.get("unsupported_claim_pattern", r"GGUF.{0,80}(?:covered\s+by|bound\s+by).{0,40}signed"), "\n".join(derivative_claim_lines), re.I):
        add("DERIVED_ARTIFACT_BINDING_OVERCLAIM", "CONFLICT", "README extends source artifact binding to a derived runtime without configured exact-artifact evidence", readme_path)
    status = max((finding["status"] for finding in findings), key=SEVERITY.get, default="PASS")
    return {
        "repo": entry["repo"], "type": entry.get("type", "model"), "evidence_class": entry.get("class", "unclassified"),
        "status": status, "revision": revision, "requested_revision": snapshot.get("requested_revision", entry.get("revision", "main")),
        "fetched_at": snapshot.get("fetched_at"), "source_url": source_url(entry, revision, readme_path),
        "file_count": len(files), "source_sha256": snapshot.get("file_sha256", {}),
        "artifact_sha256_metadata": snapshot.get("artifact_sha256", {}), "measured_records": measured,
        "findings": findings, "canonical_source": entry.get("canonical_source"),
        "permission_status": "NOT_PROBED_READ_ONLY", "promotion_verdict": "NOT_ASSESSED",
        "source_links": {path: snapshot.get("external_sources", {}).get(path, {}).get("url") or source_url(entry, revision, path) for path in contents},
    }


def write_reports(results: list[dict[str, Any]], output: Path, offline: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "offline_snapshot" if offline else "live_immutable_revision_read", "read_only": True,
        "scope": "Configured documentation/evidence consistency only; no model qualification, safety, deployment, or promotion verdict.",
        "hash_scope": "Source SHA-256 hashes cover downloaded source bytes; artifact SHA-256 values are Hub LFS metadata unless an artifact was explicitly fetched.",
        "permission_status": "Write permission was not probed; a successful read does not establish write authority.",
        "detection_limits": "Pattern checks are conservative heuristics combined with explicit inventory facts. Required evidence read failures HOLD. This is not a complete natural-language truth verifier; external service status is checked only when represented by configured evidence.",
        "repositories": results,
    }
    json_path = output.with_suffix(".json")
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# Hugging Face estate evidence audit", "", f"Generated: {payload['generated_at']}", "",
             f"Mode: `{payload['mode']}`. Read-only; remote state was not modified.", "",
             "**PASS is a documentation consistency result only. It is never a promotion, qualification, deployment, autonomy, or safety verdict.**", "",
             payload["hash_scope"], "", payload["permission_status"], "", payload["detection_limits"], "",
             "| Repository | Evidence class | State | Immutable revision |", "| --- | --- | --- | --- |"]
    for row in results:
        lines.append(f"| [{row['repo']}]({row['source_url']}) | {row['evidence_class']} | **{row['status']}** | `{row['revision'] or 'UNRESOLVED'}` |")
    for row in results:
        lines.extend(["", f"## {row['repo']}", "", f"State: **{row['status']}**. Listed files: {row['file_count']}. Write permission: not probed.", ""])
        if row["findings"]:
            for finding in row["findings"]:
                path = f" (`{finding['path']}`)" if finding.get("path") else ""
                lines.append(f"- **{finding['status']} / {finding['code']}**: {finding['message']}{path}")
        else:
            lines.append("Configured consistency checks found no contradiction. Unconfigured claims and external evidence are outside this result.")
        lines.extend(["", "Source evidence and downloaded-byte SHA-256:", ""])
        for path, digest in row["source_sha256"].items():
            lines.append(f"- [{path}]({row['source_links'].get(path, row['source_url'])}): `{digest}`")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="Markdown output; JSON is written beside it")
    parser.add_argument("--snapshot-dir", type=Path, help="Save/read immutable source snapshots here")
    parser.add_argument("--offline", action="store_true", help="Classify saved snapshots without network access")
    parser.add_argument("--max-text-bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--fail-on-findings", action="store_true", help="Exit 1 when any repository is not PASS")
    args = parser.parse_args(argv)
    if args.output.suffix.lower() != ".md":
        parser.error("--output must be a .md file so Markdown and JSON paths are distinct")
    if args.offline and not args.snapshot_dir:
        parser.error("--offline requires --snapshot-dir")
    if args.max_text_bytes <= 0:
        parser.error("--max-text-bytes must be positive")
    try:
        inventory = read_inventory(args.inventory)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.error(safe_error(exc))
    results = []
    for entry in inventory:
        try:
            snap = load_snapshot(entry, args.snapshot_dir) if args.offline else fetch_snapshot(entry, args.snapshot_dir, args.max_text_bytes)
            result = classify_repository(entry, snap)
        except Exception as exc:
            snap = {"repo": entry["repo"], "type": entry["type"], "errors": [{"code": "AUDIT_FAILED", "message": safe_error(exc)}]}
            result = classify_repository({**entry, "evidence_files": []}, snap)
        results.append(result)
        print(f"{result['status']:15} {entry['repo']} {result['revision'] or 'UNRESOLVED'}", flush=True)
    write_reports(results, args.output, args.offline)
    print(f"Reports: {args.output} and {args.output.with_suffix('.json')}")
    return 1 if args.fail_on_findings and any(r["status"] != "PASS" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
