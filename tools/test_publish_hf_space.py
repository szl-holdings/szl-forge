from __future__ import annotations

import subprocess
import unittest
from unittest.mock import patch

import publish_hf_space as publisher


REVISION = "a" * 40


class SpacePublicationPlanTests(unittest.TestCase):
    def _tracked_files(self, *names: str) -> bytes:
        return b"\0".join(name.encode("utf-8") for name in names) + b"\0"

    def test_docker_plan_requires_dockerfile(self) -> None:
        source = publisher.ROOT / "spaces" / "szl-model-inference-lab"
        output = self._tracked_files(
            "spaces/szl-model-inference-lab/README.md",
        )
        result = subprocess.CompletedProcess([], 0, stdout=output, stderr=b"")
        with patch("publish_hf_space.subprocess.run", return_value=result):
            with self.assertRaisesRegex(
                publisher.PublishError, "Dockerfile"
            ):
                publisher.build_plan(source, "owner/space", REVISION)

    def test_static_plan_requires_index_not_dockerfile(self) -> None:
        source = publisher.ROOT / "spaces" / "szl-forge-lab"
        output = self._tracked_files(
            "spaces/szl-forge-lab/README.md",
            "spaces/szl-forge-lab/index.html",
        )
        result = subprocess.CompletedProcess([], 0, stdout=output, stderr=b"")
        with patch("publish_hf_space.subprocess.run", return_value=result):
            plan = publisher.build_plan(
                source,
                "owner/space",
                REVISION,
                static=True,
            )
        self.assertEqual(
            {"README.md", "index.html"},
            set(plan["files"]),
        )

    def test_static_plan_rejects_missing_index(self) -> None:
        source = publisher.ROOT / "spaces" / "szl-forge-lab"
        output = self._tracked_files(
            "spaces/szl-forge-lab/README.md",
        )
        result = subprocess.CompletedProcess([], 0, stdout=output, stderr=b"")
        with patch("publish_hf_space.subprocess.run", return_value=result):
            with self.assertRaisesRegex(publisher.PublishError, "index.html"):
                publisher.build_plan(
                    source,
                    "owner/space",
                    REVISION,
                    static=True,
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
