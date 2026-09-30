#!/usr/bin/env python3
"""Prepare a hash-bound local card diff. No network, auth, or publication path."""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = "publishing/curated-model-card-profiles.json"
MAX_BYTES = 128_000
START = b"<!-- SZL-CURATED-METADATA:v1:START -->"
END = b"<!-- SZL-CURATED-METADATA:v1:END -->"


class PreparationError(ValueError):
    """The reviewed input cannot be prepared without changing its contract."""


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def duplicate_guard(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise PreparationError("duplicate JSON key")
        result[key] = value
    return result


def reject_constant(value):
    raise PreparationError("non-finite JSON constant")


def read_local(root: Path, relative: str) -> bytes:
    if not isinstance(relative, str) or "\\" in relative:
        raise PreparationError("invalid local input path")
    path = PurePosixPath(relative)
    if path.is_absolute() or not relative or any(p in {"", ".", ".."} for p in relative.split("/")):
        raise PreparationError("unsafe local input path")
    resolved_root = root.resolve()
    candidate = resolved_root
    for part in path.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise PreparationError("symlinked local input")
    if not candidate.resolve().is_relative_to(resolved_root) or not candidate.is_file():
        raise PreparationError("local input unavailable")
    if not 0 < candidate.stat().st_size <= MAX_BYTES:
        raise PreparationError("local input byte limit")
    raw = candidate.read_bytes()
    if not 0 < len(raw) <= MAX_BYTES:
        raise PreparationError("local input byte limit")
    return raw


def load_json(raw: bytes):
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=duplicate_guard,
                           parse_constant=reject_constant)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PreparationError("invalid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise PreparationError("JSON input must be an object")
    return value


def require_hex(value, length: int):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{" + str(length) + r"}", value):
        raise PreparationError("missing or invalid immutable digest")


def display(value):
    if not isinstance(value, str) or not value or len(value) > 300:
        raise PreparationError("invalid metadata display value")
    if any(ord(c) < 32 or c in "`|<>" for c in value):
        raise PreparationError("unsafe metadata display value")
    return "`" + value + "`"


def frontmatter(raw: bytes) -> bytes:
    try:
        raw.decode("utf-8")
    except UnicodeError as exc:
        raise PreparationError("card is not UTF-8") from exc
    lines = raw.splitlines(keepends=True)
    if not lines or lines[0].rstrip(b"\r\n") != b"---":
        raise PreparationError("existing YAML frontmatter required")
    for i, line in enumerate(lines[1:], 1):
        if line.rstrip(b"\r\n") == b"---":
            value = b"".join(lines[:i + 1])
            if START in value or END in value:
                raise PreparationError("managed marker in YAML")
            return value
    raise PreparationError("incomplete YAML frontmatter")


def unmanaged(raw: bytes) -> bytes:
    yaml = frontmatter(raw)
    counts = raw.count(START), raw.count(END)
    if counts == (0, 0):
        return raw
    if counts != (1, 1):
        raise PreparationError("duplicate or incomplete managed block")
    begin, end = raw.index(START), raw.index(END)
    if begin <= len(yaml) or end <= begin or raw[begin - 1:begin] != b"\n":
        raise PreparationError("misplaced managed block")
    stop = end + len(END)
    if raw[stop:stop + 1] != b"\n":
        raise PreparationError("incomplete managed block terminator")
    return raw[:begin - 1] + raw[stop + 1:]


def validate_observation(value, profile):
    if value.get("schema") != "szl.curated-card-observation/v1":
        raise PreparationError("unknown observation schema")
    if value.get("repo_type") != "model" or value.get("repo_id") != profile["repo_id"]:
        raise PreparationError("target type or identity mismatch")
    if value.get("revision") != profile["observed_hub_revision"]:
        raise PreparationError("mixed or stale target revision")
    require_hex(value["revision"], 40)
    facts = value.get("classification", {})
    if facts.get("revision") != value["revision"]:
        raise PreparationError("classification revision mismatch")
    if facts.get("artifact_validity") != "NOT_EVALUATED" or facts.get("runtime_validity") != "NOT_EVALUATED":
        raise PreparationError("metadata input cannot grant runtime validity")
    if facts.get("status") not in {"COMPLETE_STRUCTURE", "UNKNOWN_STRUCTURE", "INCOMPLETE_STRUCTURE"}:
        raise PreparationError("unsupported metadata structure state")
    files = value.get("files")
    if not isinstance(files, list) or not 1 <= len(files) <= 200:
        raise PreparationError("file inventory outside bounds")
    by_path = {}
    for row in files:
        if row.get("revision") != value["revision"] or row.get("repo_type") != "model":
            raise PreparationError("mixed file revision or type")
        name = row.get("path")
        display(name)
        if name in by_path:
            raise PreparationError("duplicate inventory path")
        size = row.get("size")
        if size is not None and (type(size) is not int or size < 0):
            raise PreparationError("invalid observed size")
        by_path[name] = row
    if facts["status"] == "COMPLETE_STRUCTURE":
        if value.get("inventory_complete") is not True or facts.get("inventory_complete") is not True:
            raise PreparationError("complete structure needs complete inventory")
        packages = facts.get("packages")
        if not isinstance(packages, list) or not packages:
            raise PreparationError("complete structure needs recorded packages")
        for package in packages:
            if package.get("kind") not in {"ADAPTER", "MODEL"}:
                raise PreparationError("unsupported recorded package kind")
            names = [package.get("config")] + package.get("payloads", []) + package.get("indexes", []) + package.get("referenced_shards", [])
            if not package.get("payloads") or package.get("status") != "COMPLETE_STRUCTURE":
                raise PreparationError("package has no complete payload closure")
            if any(name not in by_path or by_path[name].get("size") in {None, 0} for name in names):
                raise PreparationError("package references unavailable config or payload")
    groups = facts.get("adapter_groups", [])
    if not isinstance(groups, list) or len(groups) > 20:
        raise PreparationError("adapter groups outside bounds")
    for group in groups:
        display(group.get("config"))
        if group.get("config") not in by_path:
            raise PreparationError("adapter config missing from inventory")
        base = group.get("base_model_name_or_path")
        if base is not None:
            display(base)
        revision = group.get("base_revision")
        if revision is not None:
            require_hex(revision, 40)
        if group.get("lineage_status") != "UNKNOWN" or group.get("compatibility") != "NOT_EVALUATED":
            raise PreparationError("reviewed metadata cannot establish adapter lineage")
    evidence = value.get("evidence", {})
    for key in ("source_pass_sha256", "classification_receipt_sha256", "classifier_sha256"):
        require_hex(evidence.get(key), 64)
    return facts


def block(value, facts, observation_sha256):
    lines = ["", START.decode(), "## Pinned packaging observation", "",
             "Metadata-only observation for " + display(value["repo_id"]) + ": " + display(value["revision"]) + ".",
             "Recorded package structure: " + display(facts["status"]) + ". Weight bodies were not fetched or validated.",
             "This records configuration and payload closure, not an independently verified full or merged model.",
             "Artifact and runtime validity: `NOT_EVALUATED`. Lineage: `UNKNOWN`.", "",
             "| Adapter configuration | Recorded base reference | Immutable base revision |",
             "|---|---|---|"]
    for group in facts.get("adapter_groups", []):
        base = display(group["base_model_name_or_path"]) if group.get("base_model_name_or_path") else "`UNKNOWN`"
        revision = display(group["base_revision"]) if group.get("base_revision") else "`UNKNOWN`"
        lines.append("| " + display(group["config"]) + " | " + base + " | " + revision + " |")
    lines.extend(["", "Each adapter retains its own reference; compatibility was not evaluated and no base aliases are inferred.",
                  "The existing license declaration, dated evaluation results and release/promotion holds are unchanged.",
                  "Reviewed observation SHA-256: " + display(observation_sha256) + ".",
                  "Some supplemental JSON raw bodies were not retained; their recorded digest fidelity is inherited from the reviewed reader.",
                  "This local preparation performs no publication and grants no release or rights clearance.", END.decode(), ""])
    return "\n".join(lines).encode("utf-8")


def prepare(root: Path = ROOT, profile_name: str = "willay"):
    registry_raw = read_local(root, REGISTRY)
    registry = load_json(registry_raw)
    if registry.get("schema") != "szl.curated-card-profiles/v1" or registry.get("mode") != "OFFLINE_ONLY":
        raise PreparationError("registry is not an offline input contract")
    if registry.get("source_repository") != "szl-holdings/szl-forge":
        raise PreparationError("source repository mismatch")
    profiles = registry.get("profiles")
    if not isinstance(profiles, dict) or set(profiles) != {"willay"} or profile_name not in profiles:
        raise PreparationError("profile is not admitted")
    profile = profiles[profile_name]
    if profile.get("repo_id") != "SZLHOLDINGS/WILLAY" or profile.get("source_path") != "willay/card/README.md":
        raise PreparationError("closed WILLAY input identity drifted")
    for key in ("source_sha256", "observation_sha256"):
        require_hex(profile.get(key), 64)
    for key in ("reviewed_source_revision", "observed_hub_revision"):
        require_hex(profile.get(key), 40)
    source = read_local(root, profile["source_path"])
    baseline = unmanaged(source)
    if digest(baseline) != profile["source_sha256"]:
        raise PreparationError("unmanaged source changed; review a fresh input first")
    observation_raw = read_local(root, profile["observation_path"])
    if digest(observation_raw) != profile["observation_sha256"]:
        raise PreparationError("reviewed observation hash mismatch")
    observation = load_json(observation_raw)
    facts = validate_observation(observation, profile)
    candidate = baseline + block(observation, facts, digest(observation_raw))
    if len(candidate) > MAX_BYTES or unmanaged(candidate) != baseline or frontmatter(candidate) != frontmatter(source):
        raise PreparationError("card preservation failed")
    diff = "".join(difflib.unified_diff(source.decode().splitlines(keepends=True), candidate.decode().splitlines(keepends=True),
                                       fromfile=profile["source_path"], tofile="candidate/README.md"))
    receipt = {"schema": "szl.curated-card-preparation/v1", "state": "PREPARED_LOCAL_ONLY",
               "profile": profile_name, "source_repository": registry["source_repository"],
               "reviewed_source_revision": profile["reviewed_source_revision"],
               "source_sha256": digest(baseline), "input_sha256": digest(source),
               "frontmatter_sha256": digest(frontmatter(source)), "registry_sha256": digest(registry_raw),
               "observation_sha256": digest(observation_raw), "candidate_sha256": digest(candidate),
               "diff_sha256": digest(diff.encode()), "observed_hub_revision": observation["revision"],
               "metadata_structure": facts["status"], "lineage": "UNKNOWN",
               "artifact_validity": "NOT_EVALUATED", "runtime_validity": "NOT_EVALUATED",
               "release_admission": "NOT_CHECKED", "publication": False, "promotion": False,
               "weight_bodies_read": False, "network_calls": 0, "auth_reads": 0,
               "unmanaged_content_preserved": True, "yaml_bytes_preserved": True}
    return candidate, diff, receipt


def write_review(output: Path, candidate: bytes, diff: str, receipt: dict):
    # Exclusive directory creation avoids overwriting a previous review or source.
    output.mkdir(parents=False, exist_ok=False)
    (output / "README.md").write_bytes(candidate)
    (output / "review.diff").write_bytes(diff.encode("utf-8"))
    (output / "receipt.json").write_bytes((json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="willay", choices=["willay"])
    parser.add_argument("--output-dir", type=Path, help="create a new local review directory; default writes nothing")
    args = parser.parse_args()
    try:
        candidate, diff, receipt = prepare(profile_name=args.profile)
        if args.output_dir is not None:
            write_review(args.output_dir, candidate, diff, receipt)
    except (PreparationError, OSError, KeyError, TypeError) as exc:
        print(json.dumps({"state": "BLOCKED", "error_type": type(exc).__name__}))
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
