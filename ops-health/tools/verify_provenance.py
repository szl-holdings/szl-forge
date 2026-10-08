"""Verify every bound hash of the landed v2 component against landed bytes or a declared citation.

SYNTHETIC.  ``ops-health/PROVENANCE.json`` lists every file that came from the research repository
(with the research commit and blob it came from) and one binding per distinct SHA-256 value that
any landed file under ``v2/`` or ``runs/`` names (receipts, manifests, results, reports, code;
the ``.jsonl`` row files are bound through their manifests).  This tool checks, from the landed
bytes alone:

1. every listed file exists with the listed sha256 and size, and carries no CR byte;
2. no unlisted file sits under ops-health/ (generated files that are never committed, the
   materialized ``hub/`` and ``scratch/`` excepted);
3. every SHA-256 value found in the scanned files has exactly one binding, and every binding is
   still used;
4. every binding holds under its rule:

   equal                  the landed file(s) hash to the value
   crlf_roundtrip_equal   the landed LF file, with every LF turned into CRLF, hashes to the value
                          (the analysis-time script was committed with CRLF line endings)
   regenerated            the value is the record of a split that tools/regenerate_data.py
                          rebuilds byte-exact (that tool checks the bytes; this one checks that the
                          value is the committed record)
   v1_hub                 one of v1's published files, staged elsewhere in this repository
                          (``*/huggingface/model/oac-system-health-v1/``) or materialized under
                          ``hub/oac-v1/``
   forge_blob             a git blob of this repository (the v1 dataset receipt the legacy copies
                          cite)
   in_cited_blob          a field of such a blob
   recomputed             a digest that the landed code recomputes (artifact and kernel mutants,
                          the fail-open kernel fixture)
   receipt_chain          a digest inside the governed search receipts, whose chain, envelopes and
                          payload digests verify, or that directory's digest
   cited_blob             a file of the unpublished research repository (research commit and blob
                          recorded); declared, not checkable outside that repository
   cited_scratch          a git-ignored scratch file of the research repository; declared only

The receipt's ``source.commit`` (92872f88) names the unpublished research repository; the origin
layer that resolves it (``freeze.verify_origin``) is UNAVAILABLE outside that repository.

Usage (from the szl-forge repository root):
    python -B ops-health/tools/verify_provenance.py [--report PATH]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

OPS_HEALTH = Path(__file__).resolve().parents[1]
SCHEMA = "szl-oac/ops-health-provenance/v1"
HEX64 = re.compile(r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
SCAN_DIRS = ("v2", "runs")
UNTRACKED_DIRS = ("hub", "scratch")
V1_STAGED_GLOB = "*/huggingface/model/oac-system-health-v1"
RULES = ("equal", "crlf_roundtrip_equal", "regenerated", "v1_hub", "forge_blob", "in_cited_blob",
         "recomputed", "receipt_chain", "cited_blob", "cited_scratch")
DECLARED_ONLY = ("cited_blob", "cited_scratch")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_provenance(root: Path) -> dict:
    prov = json.loads((root / "PROVENANCE.json").read_text(encoding="utf-8"))
    if prov.get("schema") != SCHEMA:
        raise SystemExit(f"PROVENANCE.json schema is not {SCHEMA}; refusing")
    return prov


def scan_occurrences(root: Path, files) -> dict[str, list[str]]:
    """sha256 value -> sorted landed paths that name it (v2/** and runs/**, not .jsonl)."""
    found: dict[str, set[str]] = defaultdict(set)
    for rel in sorted(files):
        if rel.split("/", 1)[0] not in SCAN_DIRS or rel.endswith(".jsonl"):
            continue
        text = (root / rel).read_bytes().decode("utf-8")
        for value in HEX64.findall(text):
            found[value].add(rel)
    return {value: sorted(paths) for value, paths in sorted(found.items())}


def _import_v2(root: Path):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from v2.research import registry  # noqa: PLC0415

    if registry.ROOT.resolve() != root.resolve():
        raise SystemExit(f"imported v2 from {registry.ROOT}, expected {root}; refusing")
    return registry


def repo_root(root: Path) -> Path:
    return root.parent


def v1_staged_dir(root: Path) -> Path | None:
    hits = sorted(p for p in repo_root(root).glob(V1_STAGED_GLOB) if p.is_dir())
    if len(hits) > 1:
        raise SystemExit(f"more than one staged v1 model directory: {hits}; refusing")
    return hits[0] if hits else None


def git_blob(root: Path, blob: str) -> bytes | None:
    try:
        done = subprocess.run(["git", "-C", str(repo_root(root)), "cat-file", "blob", blob],
                              capture_output=True, check=False)
    except OSError:
        return None
    return done.stdout if done.returncode == 0 else None


class Context:
    """Lazily computed reference sets that several bindings share."""

    def __init__(self, root: Path):
        self.root = root
        self._regenerated = None
        self._recomputed = None
        self._chain = None

    def regenerated_records(self) -> dict[str, str]:
        if self._regenerated is None:
            out = {}
            manifest = json.loads((self.root / "v2/data/MANIFEST.json").read_text(encoding="utf-8"))
            for name, info in manifest["splits"].items():
                out[f"split:{name}"] = info["sha256"]
            index = json.loads((self.root / "v2/results/known_good/INDEX.json").read_text(encoding="utf-8"))
            for key, info in index["retrains"].items():
                out[f"known_good:{key}"] = info["train"]["sha256"]
            contenders = json.loads((self.root / "v2/results/contenders/INDEX.json").read_text(encoding="utf-8"))
            for key, info in contenders["contenders"].items():
                if "train_sha256" in info:
                    out[f"contender_train_prefix:{key}"] = info["train_sha256"]
            self._regenerated = out
        return self._regenerated

    def recomputed(self) -> dict[str, set[str]]:
        if self._recomputed is None:
            registry = _import_v2(self.root)
            from v2.research import freeze, mutation  # noqa: PLC0415

            artifact = json.loads((registry.V2 / "ops-health" / freeze.MODEL_NAME).read_text(encoding="utf-8"))
            kernel_text = registry.KERNEL_PATH.read_bytes().decode("utf-8")
            self._recomputed = {
                "artifact_mutant": {sha256_bytes(freeze.canonical_model_bytes(m["artifact"]))
                                    for m in mutation.artifact_mutants(artifact)},
                "kernel_mutant": {sha256_bytes(mutation.apply_kernel_edits(kernel_text, edits).encode("utf-8"))
                                  for _kid, _d, _s, edits in mutation.KERNEL_MUTANTS},
                "fail_open_kernel": {sha256_bytes(mutation.fail_open_source().encode("utf-8"))},
            }
        return self._recomputed

    def receipt_chain(self) -> tuple[set[str], list[str]]:
        """Digests inside the verified governed-receipt directory, plus its directory digest."""
        if self._chain is None:
            _import_v2(self.root)
            from v2.research import search  # noqa: PLC0415

            trace = json.loads((self.root / "v2/results/search_trace.json").read_text(encoding="utf-8"))
            receipts_dir = self.root / trace["receipts_dir_relpath"]
            problems = [f"receipts: {e}" for e in search.ouroboros().verify_run_dir(str(receipts_dir))]
            digest = search.directory_digest(receipts_dir)["sha256"]
            if digest != trace["receipts"]["dir_digest"]["sha256"]:
                problems.append("receipts directory digest differs from search_trace.json")
            values = {digest}
            for path in sorted(receipts_dir.iterdir()):
                values.update(HEX64.findall(path.read_bytes().decode("utf-8")))
            self._chain = (values, problems)
        return self._chain


def check_binding(root: Path, prov: dict, value: str, b: dict, ctx: Context) -> list[str]:
    rule = b.get("rule")
    files = prov["files"]
    if rule not in RULES:
        return [f"{value[:16]}: unknown rule {rule!r}"]
    if rule == "equal":
        problems = []
        for rel in b["paths"]:
            if rel not in files:
                problems.append(f"{value[:16]}: equal target {rel} is not a landed file")
            elif sha256_bytes((root / rel).read_bytes()) != value:
                problems.append(f"{value[:16]}: {rel} does not hash to the bound value")
        return problems
    if rule == "crlf_roundtrip_equal":
        data = (root / b["path"]).read_bytes()
        if b"\r" in data:
            return [f"{value[:16]}: {b['path']} is not LF-only"]
        if sha256_bytes(data.replace(b"\n", b"\r\n")) != value:
            return [f"{value[:16]}: CRLF form of {b['path']} does not hash to the bound value"]
        return _check_cited(value, b["cited_blob"])
    if rule == "regenerated":
        record = ctx.regenerated_records().get(b["record"])
        return [] if record == value else [f"{value[:16]}: not the committed record {b['record']}"]
    if rule == "v1_hub":
        staged = v1_staged_dir(root)
        candidates = [d / b["name"] for d in (staged, root / "hub" / "oac-v1") if d is not None]
        present = [p for p in candidates if p.is_file()]
        if not present:
            return [f"{value[:16]}: v1 file {b['name']} is neither staged nor materialized here"]
        return [f"{value[:16]}: {p} does not hash to the bound value"
                for p in present if sha256_bytes(p.read_bytes()) != value]
    if rule == "forge_blob":
        data = git_blob(root, b["blob"])
        if data is None:
            return [f"{value[:16]}: git blob {b['blob']} is not in this repository"]
        return [] if sha256_bytes(data) == value else [f"{value[:16]}: blob {b['blob']} differs"]
    if rule == "in_cited_blob":
        data = git_blob(root, b["blob"])
        if data is None:
            return [f"{value[:16]}: git blob {b['blob']} is not in this repository"]
        field = json.loads(data.decode("utf-8")).get(b["field"])
        return [] if field == value else [f"{value[:16]}: {b['field']} of blob {b['blob']} differs"]
    if rule == "recomputed":
        values = ctx.recomputed().get(b["recipe"], set())
        return [] if value in values else [f"{value[:16]}: not recomputed by recipe {b['recipe']}"]
    if rule == "receipt_chain":
        values, problems = ctx.receipt_chain()
        return problems if value in values else [*problems, f"{value[:16]}: not in the verified receipts"]
    return _check_cited(value, b)


def _check_cited(value: str, b: dict) -> list[str]:
    problems = []
    if not b.get("research_path") or not b.get("reason"):
        problems.append(f"{value[:16]}: citation lacks research_path or reason")
    commit = b.get("research_commit")
    if commit is not None and not HEX40.fullmatch(commit):
        problems.append(f"{value[:16]}: research_commit is not a full sha")
    blob = b.get("research_blob")
    if blob is not None and not HEX40.fullmatch(blob):
        problems.append(f"{value[:16]}: research_blob is not a full blob id")
    return problems


def check_files(root: Path, prov: dict) -> list[str]:
    problems = []
    for rel, info in sorted(prov["files"].items()):
        path = root / rel
        if not path.is_file():
            problems.append(f"missing landed file {rel}")
            continue
        data = path.read_bytes()
        if sha256_bytes(data) != info["sha256"] or len(data) != info["bytes"]:
            problems.append(f"{rel}: bytes differ from PROVENANCE.json")
        if b"\r" in data:
            problems.append(f"{rel}: carries CR bytes")
        for key in ("research_commit", "research_blob"):
            if not HEX40.fullmatch(info.get(key, "")):
                problems.append(f"{rel}: {key} is not a full sha")
    for rel in prov["overlay"]:
        if not (root / rel).is_file():
            problems.append(f"missing overlay file {rel}")
    known = set(prov["files"]) | set(prov["overlay"]) | set(prov["regenerated_not_committed"])
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if path.is_dir() or "__pycache__" in path.parts or rel.split("/", 1)[0] in UNTRACKED_DIRS:
            continue
        if rel not in known:
            problems.append(f"unexpected file {rel}")
    return problems


def verify(root: Path) -> dict:
    prov = load_provenance(root)
    problems = check_files(root, prov)
    occurrences = scan_occurrences(root, prov["files"])
    bindings = prov["bindings"]
    for value in occurrences:
        if value not in bindings:
            problems.append(f"{value}: bound in {occurrences[value]} but has no binding")
    for value in bindings:
        if value not in occurrences:
            problems.append(f"{value}: binding is not used by any scanned file")
    ctx = Context(root)
    by_rule: dict[str, int] = defaultdict(int)
    declared = []
    for value, b in sorted(bindings.items()):
        problems.extend(check_binding(root, prov, value, b, ctx))
        by_rule[b.get("rule")] += 1
        if b.get("rule") in DECLARED_ONLY:
            declared.append({"value": value, "rule": b["rule"], "research_path": b.get("research_path")})
    return {
        "schema": "szl-oac/ops-health-provenance-check/v1",
        "synthetic": True,
        "ok": not problems,
        "problems": problems,
        "landed_files": len(prov["files"]),
        "bound_values": len(occurrences),
        "bindings_by_rule": dict(sorted(by_rule.items())),
        "declared_not_checkable_here": declared,
        "receipt_source_commit": prov["receipt_source_commit"]["commit"],
        "origin_layer": "UNAVAILABLE outside the research repository",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=OPS_HEALTH,
                        help="the landed ops-health/ directory (default: this tool's)")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    report = verify(args.root.resolve())
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_bytes(text.encode("utf-8"))
    sys.stdout.write(text)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
