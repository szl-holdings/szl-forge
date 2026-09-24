#!/usr/bin/env python3
"""Explicit, bounded HF-router research run over public, reviewed inputs only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from urllib.request import Request, ProxyHandler, build_opener

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference.research_investigator import Corpus, GenerationIncomplete, canonical, run_cycle, sha256, strict_object
from benchmark_ollama import NoRedirects

ROUTER = "https://router.huggingface.co/v1/chat/completions"


class HFGenerator:
    def __init__(self, model: str, token: str, max_output_tokens: int = 1536):
        if not re.fullmatch(r"[\w.-]+/[\w.-]+:[\w-]+", model) or not token:
            raise ValueError("explicit HF model:provider and authenticated token required")
        self.model, self._token = model, token
        if type(max_output_tokens) is not int or not 512 <= max_output_tokens <= 4096:
            raise ValueError("output token budget must be 512..4096")
        self.max_output_tokens = max_output_tokens
        self.opener = build_opener(ProxyHandler({}), NoRedirects())
        self.usage: list[dict] = []
        self.completion_status: list[str] = []

    def __call__(self, messages: list[dict]) -> str:
        if len(canonical(messages)) > 32_000:
            raise ValueError("hosted prompt byte budget exceeded")
        request = Request(ROUTER, data=canonical({"model": self.model, "messages": messages,
            "max_tokens": self.max_output_tokens, "temperature": 0, "stream": False,
            "response_format": {"type": "json_object"}}),
            headers={"Authorization": "Bearer " + self._token, "Content-Type": "application/json"})
        with self.opener.open(request, timeout=90) as response:
            raw = response.read(24_001)
        body = strict_object(raw.decode("utf-8"))
        usage = body.get("usage", {})
        self.usage.append({key: usage[key] for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                           if type(usage.get(key)) is int and usage[key] >= 0})
        choices = body.get("choices")
        reason = choices[0].get("finish_reason") if isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict) else None
        self.completion_status.append(reason if reason in ("stop", "length", "content_filter", "tool_calls") else "UNKNOWN")
        if reason != "stop":
            raise GenerationIncomplete("hosted completion incomplete")
        content = choices[0].get("message", {}).get("content")
        if not isinstance(content, str):
            raise ValueError("hosted text missing")
        return content


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF repository:provider; no automatic fallback")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-turns", type=int, default=6, choices=range(1, 7))
    parser.add_argument("--max-output-tokens", type=int, default=1536, choices=(512, 1024, 1536, 2048, 4096))
    parser.add_argument("--public-inputs-reviewed", action="store_true", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output already exists")
    with args.corpus.open("rb") as stream:
        raw = stream.read(8_000_001)
    if len(raw) > 8_000_000:
        raise ValueError("corpus file exceeds bound")
    corpus = Corpus([strict_object(line) for line in raw.decode("utf-8").splitlines() if line.strip()])
    from huggingface_hub import get_token
    generator = HFGenerator(args.model, get_token() or "", args.max_output_tokens)
    started = datetime.now(timezone.utc).isoformat()
    try:
        report = run_cycle(args.question, corpus, generator, execution_place="remote", max_turns=args.max_turns)
    finally:
        corpus.close()
    report.update(model={"id": args.model, "serving_revision": "UNVERIFIED"}, usage=generator.usage,
                  started_at=started, finished_at=datetime.now(timezone.utc).isoformat(),
                  corpus_file_sha256=sha256(raw), trained=False, provider_cost_usd="NOT_MEASURED",
                  runner_sha256=sha256(Path(__file__).read_bytes()),
                  engine_sha256=sha256((Path(__file__).resolve().parents[1] / "inference/research_investigator.py").read_bytes()),
                  max_turns=args.max_turns, max_output_tokens_per_turn=args.max_output_tokens,
                  completion_status=generator.completion_status)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "state": report["state"], "turns": len(report["trace"])}))
    return 0 if report["state"] == "PROPOSAL_REQUIRES_REVIEW" else 1


if __name__ == "__main__":
    raise SystemExit(main())
