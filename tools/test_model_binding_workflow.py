from __future__ import annotations

import unittest
from pathlib import Path


class ModelBindingWorkflowTests(unittest.TestCase):
    def test_binding_publication_is_a_dependent_job_in_the_space_run(self) -> None:
        repository_root = Path(__file__).parents[1]
        workflow = (
            repository_root
            / ".github/workflows/publish-model-inference-lab.yml"
        ).read_text(encoding="utf-8")

        self.assertIn("  deploy:", workflow)
        self.assertIn("  publish-bindings:", workflow)
        self.assertIn("    needs: deploy", workflow)
        self.assertIn(
            "SOURCE_REVISION: ${{ github.sha }}",
            workflow,
        )
        self.assertIn("ref: ${{ env.SOURCE_REVISION }}", workflow)
        self.assertIn('--source-revision "${SOURCE_REVISION}"', workflow)
        self.assertIn(
            '--expected-runtime-source-revision "${SOURCE_REVISION}"',
            workflow,
        )

    def test_publication_chain_has_one_protected_concurrency_scope(self) -> None:
        repository_root = Path(__file__).parents[1]
        space_workflow = (
            repository_root
            / ".github/workflows/publish-model-inference-lab.yml"
        ).read_text(encoding="utf-8")
        retired_workflow = (
            repository_root
            / ".github/workflows/publish-model-source-bindings.yml"
        )
        readme = (repository_root / "README.md").read_text(encoding="utf-8")

        self.assertFalse(retired_workflow.exists())
        self.assertNotIn("publish-model-source-bindings.yml", readme)
        self.assertIn("dependent\n`publish-bindings` job", readme)
        self.assertNotIn("workflow_dispatch", space_workflow)
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
        ):
            self.assertIn(f'- "{path}"', space_workflow)


if __name__ == "__main__":
    unittest.main()
