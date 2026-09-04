#!/usr/bin/env python3
"""Idempotently materialize HF_ORG_TOKEN1 and update attempt-order proofs."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELECTOR = ROOT / "tools" / "acquire_hf_publisher_token.py"
WORKFLOW = ROOT / ".github" / "workflows" / "publish-model-inference-lab.yml"
EXISTING_TEST = ROOT / "tests" / "test_acquire_hf_publisher_token.py"
NEW_TEST_PATH = '      - "tests/test_hf_org_token1_fallback.py"'
CANDIDATE_LINE = '          HF_ORG_TOKEN1_CANDIDATE: ${{ secrets.HF_ORG_TOKEN1 }}'


def patch_selector() -> None:
    source = SELECTOR.read_text(encoding="utf-8")
    new_tuple = '("HF_ORG_TOKEN1", "HF_ORG_TOKEN1_CANDIDATE")'
    if new_tuple in source:
        return
    anchor = '    ("HF_ORG_TOKEN", "HF_ORG_TOKEN_CANDIDATE"),\n'
    if source.count(anchor) != 1:
        raise SystemExit("primary organization credential tuple drifted")
    SELECTOR.write_text(
        source.replace(anchor, anchor + f"    {new_tuple},\n", 1),
        encoding="utf-8",
    )


def patch_existing_test() -> None:
    source = EXISTING_TEST.read_text(encoding="utf-8")
    replacement = (
        '        self.assertEqual("HF_ORG_TOKEN1", attempts[1].source)\n'
        '        self.assertFalse(attempts[1].present)\n'
        '        self.assertTrue(attempts[2].valid)\n'
    )
    if replacement in source:
        return
    anchor = '        self.assertTrue(attempts[1].valid)\n'
    if source.count(anchor) != 1:
        raise SystemExit("existing fallback-attempt assertion drifted")
    EXISTING_TEST.write_text(source.replace(anchor, replacement, 1), encoding="utf-8")


def insert_after_once(lines: list[str], anchor: str, value: str, label: str) -> None:
    if value in lines:
        return
    indices = [index for index, line in enumerate(lines) if line == anchor]
    if len(indices) != 1:
        raise SystemExit(f"{label} drifted: expected one anchor, found {len(indices)}")
    lines.insert(indices[0] + 1, value)


def patch_workflow() -> None:
    source = WORKFLOW.read_text(encoding="utf-8")
    lines = source.splitlines()
    insert_after_once(
        lines,
        '      - "tests/test_acquire_hf_publisher_token.py"',
        NEW_TEST_PATH,
        "publisher path trigger",
    )

    primary = '          HF_ORG_TOKEN_CANDIDATE: ${{ secrets.HF_ORG_TOKEN }}'
    if CANDIDATE_LINE not in lines:
        indices = [index for index, line in enumerate(lines) if line == primary]
        if len(indices) != 2:
            raise SystemExit(
                f"expected two primary credential environments, found {len(indices)}"
            )
        for index in reversed(indices):
            lines.insert(index + 1, CANDIDATE_LINE)

    insert_after_once(
        lines,
        "          tests/test_acquire_hf_publisher_token.py",
        "          tests/test_hf_org_token1_fallback.py",
        "publisher pytest invocation",
    )
    insert_after_once(
        lines,
        "          python -m unittest -q tests/test_acquire_hf_publisher_token.py",
        "          python -m unittest -q tests/test_hf_org_token1_fallback.py",
        "binding-job unittest invocation",
    )

    rendered = "\n".join(lines) + "\n"
    if rendered.count("HF_ORG_TOKEN1_CANDIDATE") != 2:
        raise SystemExit("alternate organization candidate is not wired exactly twice")
    if rendered.count("tests/test_hf_org_token1_fallback.py") != 3:
        raise SystemExit("alternate credential test is not wired in all three locations")
    if rendered != source:
        WORKFLOW.write_text(rendered, encoding="utf-8")


def main() -> int:
    patch_selector()
    patch_existing_test()
    patch_workflow()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
