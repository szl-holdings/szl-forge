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
        self.assertIn(
            "SOURCE_REVISION: ${{ github.event.workflow_run.head_sha }}",
            workflow,
        )
        self.assertIn("ref: ${{ env.SOURCE_REVISION }}", workflow)
        self.assertIn('--source-revision "${SOURCE_REVISION}"', workflow)
        self.assertIn(
            '--expected-runtime-source-revision "${SOURCE_REVISION}"',
            workflow,
        )

    def test_publishers_share_one_protected_sequence(self) -> None:
        repository_root = Path(__file__).parents[1]
        binding_workflow = (
            repository_root
            / ".github/workflows/publish-model-source-bindings.yml"
        ).read_text(encoding="utf-8")
        space_workflow = (
            repository_root
            / ".github/workflows/publish-model-inference-lab.yml"
        ).read_text(encoding="utf-8")

        self.assertNotIn("workflow_dispatch", binding_workflow)
        self.assertNotIn("\n  push:", binding_workflow)
        self.assertNotIn("workflow_dispatch", space_workflow)
        self.assertEqual(
            binding_workflow.count(
                "group: publish-model-inference-lab-and-source-bindings"
            ),
            1,
        )
        self.assertEqual(
            space_workflow.count(
                "group: publish-model-inference-lab-and-source-bindings"
            ),
            1,
        )
        for path in (
            "publishing/model-source-bindings.json",
            "tools/publish_model_source_bindings.py",
            "tools/test_publish_model_source_bindings.py",
            ".github/workflows/publish-model-source-bindings.yml",
        ):
            self.assertIn(f'- "{path}"', space_workflow)


if __name__ == "__main__":
    unittest.main()
