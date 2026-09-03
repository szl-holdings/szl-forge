#!/usr/bin/env python3
"""Controlled candidate-corpus builder (cross-platform).

Port of the operator's local `build_controlled_v3.py` into the repo so the
build is reproducible anywhere. Reads only domain-canonical SZL sources,
refuses files containing secret-shaped material, and emits:

  operational/out/chakana-pairs.jsonl          szl.chakana-pairs/v1 rows
  operational/out/chakana-triples.jsonl        query/positive/negative (ops only)
  operational/out/tinku-admitted_triples.jsonl sentence1/sentence2/label rows
  operational/out/manifest.json                provenance manifest
  operational/out/review-sample.json           first rows for human review

Everything emitted is CANDIDATE_REQUIRES_REVIEW. The corpus is small by
construction; it proves the pipelines end-to-end and is NOT a production
training set. Threshold is 12 passages (lowered from the original 20 for
the smoke lane); the manifest records the count honestly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "out"

SOURCES = [
    "README.md",
    "RUNBOOK-EVAL.md",
    "RUNBOOK-NEMO.md",
    "RUNBOOK-CONJECTURE.md",
    "receiptagent/RUNBOOK-RECEIPTAGENT.md",
    "khipu/RUNBOOK-KHIPU.md",
    "portfolio/model_portfolio.json",
]

SECRET_PATTERNS = [
    re.compile(r"hf_[A-Za-z0-9_-]{20,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
]

QUESTION_RULES = [
    (r"doctrine|fabricat|honest|unknown", "What evidence-handling rule should an SZL model follow?"),
    (r"receipt|provenance|sign", "How does SZL record and verify model evidence?"),
    (r"evaluation|benchmark|measured|score", "How should an SZL model be evaluated without inventing results?"),
    (r"khipu|retrieval|navigator|abstain", "When should the Khipu retrieval model navigate or abstain?"),
    (r"nemo|nemotron|wrapper", "What is SZL-Nemo and what must not be claimed about it?"),
    (r"ollama|gguf|quant", "How should an SZL model be packaged for reliable Ollama inference?"),
    (r"publication|publish|hub", "What evidence is required before publishing an SZL model?"),
    (r"fine-tun|training|qlora|lora", "What kind of training does SZL Forge perform?"),
    (r"autonomy|proposal.only|authoriz", "What authority do SZL model weights have?"),
    (r"lambda|conjecture|theorem", "How must SZL describe the Lambda claim?"),
]

MIN_PASSAGES = 12  # smoke threshold; production promotion needs a larger reviewed corpus


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip()
    except Exception:
        return "UNKNOWN"


def clean(text: str) -> str:
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[|]{2,}", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def question_for(text: str) -> str | None:
    lower = text.lower()
    for pattern, question in QUESTION_RULES:
        if re.search(pattern, lower):
            return question
    return None


def split_for(identity: str) -> str:
    bucket = int(identity[:8], 16) % 10
    return "train" if bucket < 8 else ("validation" if bucket == 8 else "test")


def collect_passages(commit: str) -> list[dict]:
    passages: list[dict] = []
    for source in SOURCES:
        path = ROOT / source
        if not path.is_file():
            continue
        raw = path.read_text(encoding="utf-8", errors="ignore")
        if any(p.search(raw) for p in SECRET_PATTERNS):
            raise SystemExit(f"[build] REFUSE: potential secret in {source}")
        if path.suffix.lower() == ".json":
            try:
                raw = json.dumps(json.loads(raw), indent=2)
            except Exception:
                continue
        for index, section in enumerate(re.split(r"\n(?=#{1,4}\s)|\n{2,}", raw)):
            body = clean(section)
            if not 220 <= len(body) <= 1800:
                continue
            query = question_for(body)
            if query is None:
                continue
            digest = hashlib.sha256(body.encode()).hexdigest()
            passages.append({
                "id": digest, "query": query, "text": body,
                "source_path": source, "source_commit": commit,
                "section_index": index,
            })
    # one strongest passage per (question, source)
    selected: dict[tuple[str, str], dict] = {}
    for passage in passages:
        key = (passage["query"], passage["source_path"])
        current = selected.get(key)
        if current is None or len(passage["text"]) > len(current["text"]):
            selected[key] = passage
    return sorted(selected.values(), key=lambda p: (p["query"], p["source_path"]))


def negative_for(index: int, positive: dict, passages: list[dict]) -> dict:
    for offset in range(1, len(passages) + 1):
        candidate = passages[(index + offset) % len(passages)]
        if candidate["id"] == positive["id"]:
            continue
        if candidate["query"] == positive["query"]:
            continue
        if candidate["source_path"] == positive["source_path"]:
            continue
        return candidate
    raise RuntimeError("no valid negative")


def write_jsonl(path: Path, rows: list[dict]) -> dict:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"rows": len(rows), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "path": path.name}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    commit = git_commit()
    passages = collect_passages(commit)
    if len(passages) < MIN_PASSAGES:
        raise SystemExit(
            f"[build] only {len(passages)} controlled passages found; refusing weak corpus"
        )

    chakana_pairs, chakana_triples, tinku_triples = [], [], []
    for index, positive in enumerate(passages):
        negative = negative_for(index, positive, passages)
        split = split_for(positive["id"])
        chakana_pairs.append({"query": positive["query"], "positive": positive["text"]})
        chakana_triples.append({
            "id": positive["id"], "query": positive["query"],
            "positive": positive["text"], "negative": negative["text"],
            "positive_source": positive["source_path"],
            "negative_source": negative["source_path"],
            "source_commit": commit, "split": split,
        })
        tinku_triples.append({"sentence1": positive["query"], "sentence2": positive["text"], "label": 1, "split": split})
        tinku_triples.append({"sentence1": positive["query"], "sentence2": negative["text"], "label": 0, "split": split})

    manifest = {
        "schema": "szl.admission-candidate/v3",
        "source_repo": "szl-holdings/szl-forge",
        "source_commit": commit,
        "source_text_synthetic": False,
        "labels_generated": True,
        "admission_status": "CANDIDATE_REQUIRES_REVIEW",
        "min_passages": MIN_PASSAGES,
        "passage_count": len(passages),
        "corpus_class": "experimental-smoke" if len(passages) < 50 else "reviewed-candidate",
        "sources": SOURCES,
        "chakana_pairs": write_jsonl(out / "chakana-pairs.jsonl", chakana_pairs),
        "chakana_triples": write_jsonl(out / "chakana-triples.jsonl", chakana_triples),
        "tinku_triples": write_jsonl(out / "tinku-admitted_triples.jsonl", tinku_triples),
        "hub_put": "refused",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "review-sample.json").write_text(
        json.dumps({"manifest": manifest, "chakana": chakana_triples[:20],
                    "tinku": tinku_triples[:20]}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))
    print("CONTROLLED DATASET BUILD PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
