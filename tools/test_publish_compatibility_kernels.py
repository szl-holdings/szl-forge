from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import publish_compatibility_kernels as publisher


class CompatibilityKernelTests(unittest.TestCase):
    def manifest(self, root: Path) -> tuple[Path, dict[str, bytes]]:
        files = {"LICENSE": b"license", "build/pkg/__init__.py": b"def selfcheck(): return {'ok': True}"}
        payload = {
            "schema": "szl.compatibility-kernel-binding/v1",
            "artifacts": [{
                "repo_id": "SZLHOLDINGS/example",
                "artifact_class": "RETAINED_COMPATIBILITY_KERNEL",
                "promotion_state": "OPERATIONAL_REFERENCE_NOT_PERFORMANCE_PROMOTED",
                "source_repository": "szl-holdings/example",
                "source_revision": "a" * 40,
                "replacement_repository": "szl-holdings/replacement",
                "replacement_revision": "b" * 40,
                "expected_hub_revision": "c" * 40,
                "expected_files": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
                "license": {"spdx": "Apache-2.0", "evidence_path": "LICENSE", "evidence_sha256": hashlib.sha256(files["LICENSE"]).hexdigest()},
                "runtime": {"module_path": "build", "module": "pkg", "selfcheck": "selfcheck", "autonomy_boundary": "none", "performance_boundary": "none"}
            }]
        }
        path = root / "manifest.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path, files

    def test_dry_run_binds_hashes_and_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, files = self.manifest(root)
            def download(_repo: str, filename: str, **_: object) -> str:
                path = root / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(files[filename])
                return str(path)
            api = mock.Mock()
            api.model_info.return_value = SimpleNamespace(sha="c" * 40)
            result = publisher.run(manifest_path=manifest, report_path=root / "report.json", publish=False, token=None, api=api, download_fn=download)
            self.assertEqual(result["records"][0]["status"], "VERIFIED_DRY_RUN")
            self.assertTrue(result["records"][0]["runtime_receipt"]["selfcheck"]["ok"])

    def test_hash_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, files = self.manifest(root)
            def download(_repo: str, filename: str, **_: object) -> str:
                path = root / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"drift" if filename == "LICENSE" else files[filename])
                return str(path)
            api = mock.Mock()
            api.model_info.return_value = SimpleNamespace(sha="c" * 40)
            with self.assertRaisesRegex(publisher.QualificationError, "hash mismatch"):
                publisher.run(manifest_path=manifest, report_path=root / "r.json", publish=False, token=None, api=api, download_fn=download)

    def test_historical_model_repo_is_reclassified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest, _ = self.manifest(Path(temporary))
            loaded = publisher.load_manifest(manifest)
            self.assertEqual(loaded["artifacts"][0]["artifact_class"], "RETAINED_COMPATIBILITY_KERNEL")


if __name__ == "__main__":
    unittest.main()
