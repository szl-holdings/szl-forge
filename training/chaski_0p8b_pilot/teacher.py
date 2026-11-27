"""Phase 1: Nemotron teacher completions for admitted Chaski prompts.

The Nemotron guardrail, NOTICE, and reasoning helpers come from the
transformers-5 trainer lane (``training/trainer_compat.py`` on
``kallpa/nemotron-h-trainer``). This module does not reimplement them.

``--execute`` is refused. Generation runs only when a caller passes an
already-constructed model, which the CPU dry-run does with a tiny random
Nemotron-H. No Hub client is imported.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

from training.chaski_0p8b_pilot.constants import (
    TEACHER_ID,
    TEACHER_REVISION,
    refuse_forbidden_repo,
)


def trainer_compat():
    path = Path(__file__).resolve().parents[1] / "trainer_compat.py"
    spec = importlib.util.spec_from_file_location("szl_trainer_compat", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"transformers-5 trainer lane is missing: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = (
        "nemotron_base",
        "require_nemotron_guardrails",
        "write_nvidia_notice",
        "reasoning_settings",
        "render_chat_text",
        "require_padding_token",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise RuntimeError("transformers-5 trainer lane is missing " + ", ".join(missing))
    return module


def _user_text(messages: list[dict[str, str]]) -> str:
    return "\n".join(message["content"] for message in messages if message["role"] == "user")


def generate_completions(
    model: Any,
    tokenizer: Any,
    prompts: list[dict[str, Any]],
    *,
    compat: Any,
    base: dict[str, Any],
    guardrails: dict[str, Any],
    reasoning: dict[str, Any],
    max_new_tokens: int,
    out_dir: Path,
    heldout_users: set[str],
    dataset_repo: str,
    require_text: bool = True,
) -> dict[str, Any]:
    """Write teacher JSONL and NOTICE under ``out_dir``. No Hub write."""

    refuse_forbidden_repo(dataset_repo)
    if base.get("id") != TEACHER_ID or base.get("revision") != TEACHER_REVISION:
        raise RuntimeError("teacher base is not the pinned Nano-4B revision")
    if not prompts:
        raise RuntimeError("teacher generation has no admitted prompts")
    if max_new_tokens < 1:
        raise RuntimeError("max_new_tokens must be positive")
    if not heldout_users:
        raise RuntimeError("held-out user set is empty; refusing to generate")

    for prompt in prompts:
        user = _user_text(prompt["messages"])
        if user in heldout_users:
            raise RuntimeError("held-out user prompt was passed to teacher generation")

    guard = compat.require_nemotron_guardrails(base, tokenizer, guardrails)
    compat.require_padding_token(tokenizer)
    out_dir.mkdir(parents=True, exist_ok=True)
    notice_path = compat.write_nvidia_notice(out_dir, base)

    import torch

    was_training = model.training
    model.eval()
    rows: list[dict[str, Any]] = []
    try:
        for prompt in prompts:
            rendered = compat.render_chat_text(
                tokenizer,
                prompt["messages"],
                reasoning=reasoning,
                add_generation_prompt=True,
            )
            batch = tokenizer(rendered, return_tensors="pt")
            batch.pop("token_type_ids", None)
            device = next(model.parameters()).device
            batch = {key: value.to(device) for key, value in batch.items()}
            prompt_len = int(batch["input_ids"].shape[-1])
            with torch.no_grad():
                generated = model.generate(
                    **batch,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
            new_ids = generated[0][prompt_len:]
            if int(new_ids.shape[-1]) < 1:
                raise RuntimeError("teacher generate returned no new tokens")
            text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            if not text:
                if require_text:
                    raise RuntimeError("teacher completion is empty")
                text = tokenizer.decode(new_ids, skip_special_tokens=False).strip()
            if not text:
                text = "tokens:" + ",".join(str(int(token)) for token in new_ids)
            messages = list(prompt["messages"]) + [{"role": "assistant", "content": text}]
            rows.append(
                {
                    "id": prompt["id"],
                    "source": prompt["source"],
                    "license": prompt["license"],
                    "teacher": {"id": base["id"], "revision": base["revision"]},
                    "messages": messages,
                }
            )
    finally:
        model.train(was_training)

    corpus = out_dir / "teacher.jsonl"
    corpus.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    notice = Path(notice_path).read_text(encoding="utf-8")
    if base["notice"] not in notice:
        raise RuntimeError("NOTICE readback failed")
    return {
        "rows": len(rows),
        "corpus": str(corpus),
        "notice": notice_path,
        "guardrails": guard,
        "reasoning": reasoning,
        "dataset_repo": dataset_repo,
        "hub_write": False,
        "launch_executed": False,
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    requested = "execute" if args.execute else "prepare"
    raise SystemExit(
        "Chaski 0.8B teacher generation "
        f"({requested}) is not authorized to launch. "
        "Use the CPU dry-run. No model was loaded and no Hub write ran."
    )


if __name__ == "__main__":
    raise SystemExit(main())
