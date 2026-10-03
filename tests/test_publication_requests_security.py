"""Keep publication clients outside Requests' vulnerable extraction range."""
from pathlib import Path
import re
import tempfile
import tomllib
import unittest
from unittest.mock import patch
import zipfile

import requests
import yaml


ROOT = Path(__file__).resolve().parents[1]
PATCHED_VERSION = (2, 33, 0)
PUBLISHER_WORKFLOWS = (
    "foundation-runtime.yml",
    "model-kernel-frontier.yml",
    "oac-health-space.yml",
    "publish-forge-lab.yml",
    "publish-model-inference-lab.yml",
)


def version_tuple(value):
    return tuple(int(component) for component in value.split("."))


class PublicationRequestsSecurityTests(unittest.TestCase):
    def test_optional_test_client_uses_a_patched_exact_version(self):
        manifest = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))
        requirements = manifest["project"]["optional-dependencies"]["test"]
        request_pins = [item for item in requirements if item.startswith("requests")]
        self.assertEqual(len(request_pins), 1)
        match = re.fullmatch(r"requests==([0-9]+\.[0-9]+\.[0-9]+)", request_pins[0])
        self.assertIsNotNone(match, "The offline publisher client needs an exact pin")
        self.assertGreaterEqual(version_tuple(match[1]), PATCHED_VERSION)

    def test_each_publisher_installs_a_patched_exact_client(self):
        for name in PUBLISHER_WORKFLOWS:
            with self.subTest(workflow=name):
                workflow = yaml.safe_load((ROOT / ".github" / "workflows" / name).read_text("utf-8"))
                commands = [step.get("run", "") for job in workflow["jobs"].values() for step in job["steps"]]
                pins = re.findall(r"\brequests==([0-9]+\.[0-9]+\.[0-9]+)\b", "\n".join(commands))
                self.assertTrue(pins, "Publication must declare its exact Requests client")
                for pin in pins:
                    self.assertGreaterEqual(version_tuple(pin), PATCHED_VERSION)

    def test_estate_payload_excludes_the_vulnerable_range(self):
        source = (ROOT / "tools" / "estate_payload" / "SZL-PAYLOAD-V4.ps1").read_text("utf-8")
        minimums = re.findall(r"\brequests>=([0-9]+\.[0-9]+(?:\.[0-9]+)?)\b", source)
        self.assertEqual(len(minimums), 1)
        self.assertGreaterEqual(version_tuple(minimums[0]), PATCHED_VERSION)

    def test_archive_extraction_does_not_reuse_an_attacker_file(self):
        # CVE-2026-25645: old Requests reused tempdir/basename without
        # checking its contents. All paths and bytes here are controlled.
        with tempfile.TemporaryDirectory(prefix="szl-requests-control-") as directory:
            folder = Path(directory)
            archive = folder / "source.zip"
            member = "certs/szl-security-control.pem"
            legitimate = b"SZL controlled archive member\n"
            malicious = folder / "szl-security-control.pem"
            malicious.write_bytes(b"SZL controlled pre-existing incorrect member\n")
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr(member, legitimate)
            with patch.object(requests.utils.tempfile, "tempdir", directory):
                extracted = Path(requests.utils.extract_zipped_paths(str(archive / member)))
            self.assertTrue(extracted.resolve().is_relative_to(folder.resolve()))
            self.assertNotEqual(extracted, malicious)
            self.assertEqual(extracted.read_bytes(), legitimate)
            self.assertEqual(malicious.read_bytes(), b"SZL controlled pre-existing incorrect member\n")


if __name__ == "__main__":
    unittest.main(verbosity=2)
