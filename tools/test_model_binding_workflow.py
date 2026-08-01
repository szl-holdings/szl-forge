from __future__ import annotations

import unittest
from pathlib import Path


class ModelBindingWorkflowTests(unittest.TestCase):
    def test_binding_publication_follows_verified_space_publication(self) -> None:
        repository_root = Path(__file__).parents[1]
        workflow = (
            repository_root
            / ".github/workflows/publish-model-source-bindings.yml"
        ).read_text(encoding="utf-8")

        self.assertIn('workflows: ["Publish model inference lab"]', workflow)
        self.assertIn("types: [completed]", workflow)
        self.assertIn("branches: [main]", workflow)
        self.assertIn("workflow_run.conclusion == 'success'", workflow)
        self.assertIn("workflow_run.head_branch == 'main'", workflow)
        self.assertIn(
            "workflow_run.head_repository.full_name == github.repository",
            workflow,
        )
        self.assertIn("workflow_run.head_sha || github.sha", workflow)
        self.assertIn("ref: ${{ env.SOURCE_REVISION }}", workflow)
        self.assertIn('--source-revision "${SOURCE_REVISION}"', workflow)

    def test_sequence_contract_does_not_directly_trigger_the_publisher(self) -> None:
        repository_root = Path(__file__).parents[1]
        workflow = (
            repository_root
            / ".github/workflows/publish-model-source-bindings.yml"
        ).read_text(encoding="utf-8")

        self.assertNotIn(
            '- ".github/workflows/publish-model-source-bindings.yml"',
            workflow,
        )
        self.assertNotIn('- "tools/test_model_binding_workflow.py"', workflow)


if __name__ == "__main__":
    unittest.main()
