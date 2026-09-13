"""Cross-source identity checks, not a live Hub inventory or admission claim."""
import ast
from pathlib import Path

from model_candidates.specs import candidate_specs


def test_research_identities_do_not_shadow_model_lab():
    path = Path(__file__).resolve().parents[1] / "model-lab/src/szl_model_lab/catalog.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    declared = {node.value for node in ast.walk(tree)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)
                and node.value.startswith("SZLHOLDINGS/")}
    # Fail if the source format changes instead of treating an unread registry as empty.
    assert {"SZLHOLDINGS/A11OY-Router", "SZLHOLDINGS/A11OY-Invariant", "SZLHOLDINGS/YARQA-1"} <= declared
    research = {row["proposed_hf_id"] for row in candidate_specs()}
    assert len(research) == len(candidate_specs())
    assert not research.intersection(declared)
    assert research == {"SZLHOLDINGS/A11OY-RouteRank-Research-v1",
                        "SZLHOLDINGS/A11OY-InvariantRisk-Research-v1",
                        "SZLHOLDINGS/YARQA-Causal-Reference-v1"}


def test_workflow_watches_both_catalog_change_triggers():
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github/workflows/model-candidates.yml").read_text(encoding="utf-8")
    assert workflow.count("      - 'model-lab/src/szl_model_lab/catalog.py'") == 2
