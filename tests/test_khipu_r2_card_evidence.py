# SPDX-License-Identifier: Apache-2.0
"""Cross-source honesty contract for the KHIPU-R2 public model card."""
import ast
import builtins
import copy
import json
import re
from pathlib import Path

from tools import publish_khipu_card as publisher


ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "khipu-r2" / "card" / "README.md"
CANONICAL = ROOT / "khipu_r2" / "README.md"


def _read(path: Path) -> str:
    assert path.is_file() and not path.is_symlink()
    return path.read_text(encoding="utf-8")


def _metric_rows(card: str) -> dict[str, tuple[str, str]]:
    names = {"plan-valid", "grounding (`eval.jsonl` navigate)",
             "abstain (`adversarial.jsonl`)", "hallucinated citations"}
    rows = {}
    for line in card.splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = [" ".join(cell.replace("**", "").split())
                 for cell in line.strip().strip("|").split("|")]
        if cells[0] in names:
            assert len(cells) == 3 and cells[0] not in rows
            rows[cells[0]] = (cells[1], cells[2])
    assert set(rows) == names
    return rows


def test_card_and_canonical_kit_agree_on_bounded_observations() -> None:
    card = _read(CARD)
    canonical = _read(CANONICAL)

    shared_boundaries = (
        "SZLHOLDINGS/KHIPU-R2",
        "SZLHOLDINGS/SZL-Khipu-1.5B",
        "6a91bf11984507d9db4ea104",
        "publication_eligible: false",
    )
    for boundary in shared_boundaries:
        assert boundary in card
        assert boundary in canonical

    assert "adapter_model.safetensors" in card
    assert "adapter **147.8MB AVAILABLE**" in canonical
    assert "Abstain is MEASURED 3/6, not a pass." in card
    assert "abstain **MEASURED 3/6** (not a pass" in canonical
    rows = _metric_rows(card)
    for name, expected in (
        ("plan-valid", (11, 11)),
        ("grounding (`eval.jsonl` navigate)", (5, 5)),
        ("abstain (`adversarial.jsonl`)", (3, 6)),
    ):
        fraction = re.fullmatch(r"(\d+)\s*/\s*(\d+)", rows[name][0])
        assert fraction and tuple(map(int, fraction.groups())) == expected
    assert rows["plan-valid"][1] == "Small owner-run synthetic protocol"
    assert rows["grounding (`eval.jsonl` navigate)"][1] == "Five navigation cases"
    assert rows["abstain (`adversarial.jsonl`)"][1] == "Visible release blocker; not a pass"
    assert rows["hallucinated citations"] == ("0", "These cases only")
    assert "grounding **5/5**" in canonical
    assert "plan **11/11**" in canonical


def test_card_keeps_training_and_promotion_claims_separate() -> None:
    card = _read(CARD)
    flattened = " ".join(card.split())
    lowered = card.casefold()

    required = (
        "publication_eligible: false",
        "autonomy_eligible: false",
        "No signed R2 eval receipt in this atelier.",
        "The corrected runs are separate owner records and do not promote the model.",
        "held_out_in_gradients: false",
        "are not an independent evaluation or public leaderboard.",
        "inherits no KHIPU-R2 score or qualification.",
        "Outputs are proposals requiring external validation and approval.",
        "All prior fp16 scores produced by that buggy harness are void",
        "is a training metric, not evaluation.",
        "Current artifact bytes and their equality to historical evaluated bytes were not independently rehashed.",
        "`main@merge-time`, which is not an immutable base identity.",
        "Lab load remains forbidden.",
        "e44d53f29f2d443598e06d6c0441557fd3a5010888c7aa97b56ec3c0e050d349",
    )
    for boundary in required:
        assert boundary.casefold() in flattened.casefold()
    assert "2026-08-28 owner-run job record" in card
    assert "2026-09-13 corrected-harness CPU re-verification" in card

    forbidden = (
        "nobody else ships this combination",
        "one-of-one",
        "world-class",
        "best in the world",
        "publication_eligible: true",
        "autonomy_eligible: true",
        "status: production ready",
    )
    for claim in forbidden:
        assert claim not in lowered

    # Preserve the old documentation-only publication boundary against the
    # actual pure binding contract rather than a retired presentation sentence.
    binding = json.loads(publisher.build_source_binding(
        profile="khipu-r2", source_revision="a" * 40, source_assets={},
    ))
    assert binding["target"]["repo_id"] == "SZLHOLDINGS/KHIPU-R2"
    assert binding["target"]["controlled_paths"] == [
        "README.md", "holo-banner.svg", "szl-source-binding.json",
    ]
    assert binding["qualification"]["publication_eligible"] is False
    assert binding["qualification"]["abstention_result"] == "3/6"
    assert binding["qualification"]["release_blocker_preserved"] is True
    assert binding["authority"] == {
        "weights": False, "adapter": False, "configs": False, "evals": False,
        "visibility": False, "hardware": False, "runtime": False,
    }


def test_card_example_requires_immutable_base_before_model_clients() -> None:
    snippets = re.findall(r"```python\n(.*?)\n```", _read(CARD), re.S)
    assert len(snippets) == 1
    tree = ast.parse(snippets[0])
    prefix = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module in {"peft", "transformers"}:
            break
        prefix.append(node)
    assert len(prefix) < len(tree.body), "The guarded adapter example must remain explicit"
    assignment = next(node for node in prefix if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "BASE_REVISION"
                              for target in node.targets))
    assert isinstance(assignment.value, ast.Constant) and assignment.value.value == ""

    invalid = ("", "main", "a" * 39, "a" * 41, "A" * 40, "g" * 40)
    for revision in (*invalid, "a" * 40):
        candidate = copy.deepcopy(prefix)
        selected = next(node for node in candidate if isinstance(node, ast.Assign)
                        and any(isinstance(target, ast.Name) and target.id == "BASE_REVISION"
                                for target in node.targets))
        selected.value = ast.Constant(revision)
        imported = []

        def only_re(name, *args, **kwargs):
            imported.append(name)
            assert name == "re", f"Unexpected model/provider import: {name}"
            return re

        namespace = {"__builtins__": dict(vars(builtins), __import__=only_re)}
        program = ast.fix_missing_locations(ast.Module(body=candidate, type_ignores=[]))
        try:
            exec(compile(program, "KHIPU-R2-card-base-preflight", "exec"), namespace)
        except ValueError as error:
            assert revision in invalid
            assert "owner-verified immutable base revision is required" in str(error)
        else:
            assert revision not in invalid, f"Unrecorded/mutable base revision was accepted: {revision!r}"
            assert namespace["BASE_REVISION"] == revision
        assert imported == ["re"]


def test_canonical_kit_still_denies_implicit_provider_mutation() -> None:
    canonical = _read(CANONICAL)
    assert "No Hub PUT." in canonical
    assert "This checkout does not fire a job" in canonical
    assert "--run-job` on the launcher exits 2" in canonical
    assert "Not an autonomous agent" in canonical
