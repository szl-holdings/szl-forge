from __future__ import annotations

import subprocess
import unittest

import httpx
from huggingface_hub.utils import RepositoryNotFoundError
from types import SimpleNamespace
from unittest.mock import Mock, patch

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

    @staticmethod
    def _missing_space() -> RepositoryNotFoundError:
        request = httpx.Request(
            "GET", "https://huggingface.co/api/spaces/owner/space"
        )
        return RepositoryNotFoundError(
            "missing Space",
            response=httpx.Response(404, request=request),
        )

    def test_existing_space_is_write_confirmed_without_recreation(self) -> None:
        api = Mock()
        api.space_info.return_value = SimpleNamespace(
            id="owner/space",
            runtime=SimpleNamespace(volumes=[]),
        )

        result = publisher.ensure_space_repository(api, "owner/space")

        self.assertEqual(
            "EXISTING_AND_WRITE_CONFIRMED", result["state"]
        )
        self.assertFalse(result["created"])
        self.assertEqual("docker", result["sdk"])
        api.create_repo.assert_not_called()
        api.auth_check.assert_called_once_with(
            repo_id="owner/space",
            repo_type="space",
            write=True,
        )

    def test_missing_docker_space_is_created_and_read_back(self) -> None:
        api = Mock()
        api.space_info.side_effect = [
            self._missing_space(),
            SimpleNamespace(
                id="owner/space",
                runtime=SimpleNamespace(volumes=[]),
            ),
        ]

        result = publisher.ensure_space_repository(api, "owner/space")

        self.assertEqual(
            "CREATED_AND_WRITE_CONFIRMED", result["state"]
        )
        self.assertTrue(result["created"])
        self.assertEqual("public", result["visibility_on_create"])
        api.create_repo.assert_called_once_with(
            repo_id="owner/space",
            repo_type="space",
            space_sdk="docker",
            private=False,
            exist_ok=True,
        )
        api.auth_check.assert_called_once_with(
            repo_id="owner/space",
            repo_type="space",
            write=True,
        )

    def test_missing_static_space_uses_static_sdk(self) -> None:
        api = Mock()
        api.space_info.side_effect = [
            self._missing_space(),
            SimpleNamespace(
                id="owner/space",
                runtime=SimpleNamespace(volumes=[]),
            ),
        ]

        result = publisher.ensure_space_repository(
            api,
            "owner/space",
            static=True,
        )

        self.assertEqual("static", result["sdk"])
        api.create_repo.assert_called_once_with(
            repo_id="owner/space",
            repo_type="space",
            space_sdk="static",
            private=False,
            exist_ok=True,
        )

    def test_legacy_volume_removal_is_observed(self) -> None:
        api = Mock()
        api.space_info.side_effect = [
            SimpleNamespace(
                runtime=SimpleNamespace(
                    volumes=[SimpleNamespace(source="owner/model")]
                )
            ),
            SimpleNamespace(runtime=SimpleNamespace(volumes=[])),
        ]
        result = publisher.clear_legacy_space_volumes(
            api,
            "owner/space",
            wait_seconds=1,
        )
        self.assertEqual("CLEARED_AND_OBSERVED", result["state"])
        self.assertEqual(1, result["before_count"])
        self.assertEqual(0, result["after_count"])
        api.delete_space_volumes.assert_called_once_with(repo_id="owner/space")
        api.get_space_runtime.assert_not_called()

    def test_exact_runtime_wait_can_require_final_zero_volumes(self) -> None:
        api = Mock()
        info = SimpleNamespace(
            sha="b" * 40,
            runtime=SimpleNamespace(stage="RUNNING", volumes=[]),
        )
        api.space_info.return_value = info
        with patch("publish_hf_space.time.sleep") as sleep:
            observed = publisher.wait_for_exact_running_space(
                api,
                "owner/space",
                "b" * 40,
                wait_seconds=1,
                require_zero_volumes=True,
            )
        self.assertIs(info, observed)
        self.assertEqual(2, api.space_info.call_count)
        sleep.assert_called_once_with(10)

    def test_final_volume_reconciliation_restarts_changed_runtime(self) -> None:
        api = Mock()
        info = SimpleNamespace(
            sha="b" * 40,
            runtime=SimpleNamespace(stage="RUNNING", volumes=[]),
        )
        api.space_info.side_effect = [
            SimpleNamespace(
                runtime=SimpleNamespace(
                    volumes=[SimpleNamespace(source="owner/model")]
                )
            ),
            SimpleNamespace(runtime=SimpleNamespace(volumes=[])),
            info,
            info,
        ]
        api.get_space_runtime.side_effect = [
            SimpleNamespace(
                stage="BUILDING",
                raw={"domains": [{"stage": "BUILDING"}]},
            )
        ]
        with patch("publish_hf_space.time.sleep"):
            evidence, observed = publisher.reconcile_final_space_volumes(
                api,
                "owner/space",
                "b" * 40,
                wait_seconds=1,
            )
        self.assertIs(info, observed)
        self.assertTrue(evidence["restart_requested"])
        self.assertTrue(evidence["restart_transition"]["observed"])
        self.assertEqual("BUILDING", evidence["restart_transition"]["runtime_stage"])
        self.assertEqual(0, evidence["final_count"])
        api.delete_space_volumes.assert_called_once_with(repo_id="owner/space")
        api.restart_space.assert_called_once_with(repo_id="owner/space")


if __name__ == "__main__":
    unittest.main(verbosity=2)
