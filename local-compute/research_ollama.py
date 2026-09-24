#!/usr/bin/env python3
"""Run a read-only research cycle using an already installed local model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference.research_cycle import Corpus, GenerationIncomplete, run_cycle, sha256, strict_object
from benchmark_ollama import LocalClient, admit_model


def run(client: LocalClient, model: str, corpus_path: Path, question: str, output: Path,
        max_turns: int = 6) -> dict:
    if output.exists():
        raise ValueError("output already exists")
    if client.call("/api/ps").get("models"):
        raise ValueError("loaded Ollama work exists; refusing to disturb it")
    entries = {m["name"]: m for m in client.call("/api/tags").get("models", [])}
    if model not in entries:
        raise ValueError("select an already installed model; no download is performed")
    admit_model(entries[model], client.call("/api/show", {"model": model}))
    with corpus_path.open("rb") as stream:
        raw = stream.read(8_000_001)
    if len(raw) > 8_000_000:
        raise ValueError("corpus file exceeds bound")
    corpus = Corpus([strict_object(line) for line in raw.decode("utf-8").splitlines() if line.strip()])
    started = datetime.now(timezone.utc).isoformat()
    model_digest = entries[model]["digest"]
    options = {"temperature": 0, "seed": 907, "num_ctx": 8192, "num_predict": 768}

    def generate(messages: list[dict]) -> str:
        if sum(len(message["content"].encode()) for message in messages) > 6000:
            raise ValueError("local prompt byte budget exceeded; reduce source chunks")
        response = client.call("/api/chat", {"model": model, "messages": messages,
            "stream": False, "think": False, "format": "json", "keep_alive": 0, "options": options})
        if response.get("done") is not True or response.get("done_reason") != "stop":
            raise GenerationIncomplete("incomplete generation")
        content = response.get("message", {}).get("content")
        if not isinstance(content, str):
            raise ValueError("generation text missing")
        return content

    try:
        report = run_cycle(question, corpus, generate, execution_place="local", max_turns=max_turns)
    finally:
        corpus.close()
    report.update(model={"id": model, "digest": model_digest}, started_at=started,
                  finished_at=datetime.now(timezone.utc).isoformat(), generation=options,
                  corpus_file_sha256=sha256(raw), provider_cost_usd=0, trained=False,
                  electricity_cost="NOT_MEASURED", runner_sha256=sha256(Path(__file__).read_bytes()),
                  engine_sha256=sha256((Path(__file__).resolve().parents[1] / "inference/research_cycle.py").read_bytes()))
    try:
        latest = {m["name"]: m for m in client.call("/api/tags").get("models", [])}
        report["model_digest_stable"] = latest.get(model, {}).get("digest") == model_digest
    except Exception:
        report["model_digest_stable"] = False
    if not report["model_digest_stable"]:
        report.update(state="MODEL_BINDING_LOST", proposal=None)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin", default="http://127.0.0.1:11434")
    parser.add_argument("--model", required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-turns", type=int, default=6)
    args = parser.parse_args()
    report = run(LocalClient(args.origin), args.model, args.corpus, args.question, args.output, args.max_turns)
    print(json.dumps({"output": str(args.output), "state": report["state"], "turns": len(report["trace"])}))
    return 0 if report["state"] == "PROPOSAL_REQUIRES_REVIEW" else 1


if __name__ == "__main__":
    raise SystemExit(main())
