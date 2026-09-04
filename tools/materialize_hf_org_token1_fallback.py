#!/usr/bin/env python3
"""Materialize the established HF_ORG_TOKEN1 fallback into SZL Forge."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELECTOR = ROOT / "tools" / "acquire_hf_publisher_token.py"
WORKFLOW = ROOT / ".github" / "workflows" / "publish-model-inference-lab.yml"
TEST_PATH = '      - "tests/test_hf_org_token1_fallback.py"'
CANDIDATE_LINE = '          HF_ORG_TOKEN1_CANDIDATE: ${{ secrets.HF_ORG_TOKEN1 }}'


def patch_selector() -> None:
    source = SELECTOR.read_text(encoding="utf-8")
    if '("HF_ORG_TOKEN1", "HF_ORG_TOKEN1_CANDIDATE")' in source:
        return
    anchor = '    ("HF_ORG_TOKEN", "HF_ORG_TOKEN_CANDIDATE"),\n'
    if source.count(anchor) != 1:
        raise SystemExit("primary organization credential tuple drifted")
    replacement = anchor + '    ("HF_ORG_TOKEN1", "HF_ORG_TOKEN1_CANDIDATE"),\n'
    SELECTOR.write_text(source.replace(anchor, replacement, 1), encoding="utf-8")


def patch_workflow() -> None:
    source = WORKFLOW.read_text(encoding="utf-8")
    lines = source.splitlines()

    path_anchor = '      - "tests/test_acquire_hf_publisher_token.py"'
    if TEST_PATH not in lines:
        indices = [index for index, line in enumerate(lines) if line == path_anchor]
        if len(indices) != 1:
            raise SystemExit("publisher path trigger drifted")
        lines.insert(indices[0] + 1, TEST_PATH)

    env_anchor = '          HF_ORG_TOKEN_CANDIDATE: ${{ secrets.HF_ORG_TOKEN }}'
    if CANDIDATE_LINE not in lines:
        indices = [index for index, line in enumerate(lines) if line == env_anchor]
        if len(indices) != 2:
            raise SystemExit(
                f"expected two primary credential environments, found {len(indices)}"
            )
        for index in reversed(indices):
            lines.insert(index + 1, CANDIDATE_LINE)

    pytest_anchor = "          tests/test_acquire_hf_publisher_token.py"
    pytest_new = "          tests/test_hf_org_token1_fallback.py"
    if pytest_new not in lines:
        indices = [index for index, line in enumerate(lines) if line == pytest_anchor]
        if len(indices) != 1:
            raise SystemExit("publisher pytest invocation drifted")
        lines.insert(indices[0] + 1, pytest_new)

    unittest_anchor = (
        "          python -m unittest -q tests/test_acquire_hf_publisher_token.py"
    )
    unittest_new = (
        "          python -m unittest -q tests/test_hf_org_token1_fallback.py"
    )
    if unittest_new not in lines:
        indices = [index for index, line in enumerate(lines) if line == unittest_anchor]
        if len(indices) != 1:
            raise SystemExit("binding-job unittest invocation drifted")
        lines.insert(indices[0] + 1, unittest_new)

    rendered = "\n".join(lines) + "\n"
    if rendered.count("HF_ORG_TOKEN1_CANDIDATE") != 2:
        raise SystemExit("alternate organization candidate is not wired exactly twice")
    if rendered.count("tests/test_hf_org_token1_fallback.py") != 3:
        raise SystemExit("alternate credential test is not wired in all three locations")
    if rendered != source:
        WORKFLOW.write_text(rendered, encoding="utf-8")


def main() -> int:
    patch_selector()
    patch_workflow()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
