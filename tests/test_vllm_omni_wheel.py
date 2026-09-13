"""Offline regressions; synthetic archives are never upstream execution evidence."""
from __future__ import annotations

import base64
import copy
import csv
import io
import json
import stat
import subprocess
import unittest
import zipfile
from unittest.mock import patch

from tools import evaluate_vllm_omni_wheel as subject

METADATA = (
    b"Metadata-Version: 2.4\nName: vllm-omni\nVersion: 0.29.0rc1\n"
    b"License-Expression: Apache-2.0\nRequires-Python: <3.14,>=3.10\n"
    b"Requires-Dist: transformers<5.15,>=5.13.0\n\nSynthetic fixture.\n"
)
SOURCE = {"vllm_omni/__init__.py": b'"""Synthetic, never imported."""\n'}


def wheel(extra=None, metadata=METADATA, rows_mutator=None, duplicate=None):
    members = {**SOURCE, subject.DIST + "METADATA": metadata}
    members.update(extra or {})
    rows = []
    for name, data in members.items():
        digest = base64.urlsafe_b64encode(bytes.fromhex(subject.sha256(data))).rstrip(b"=").decode()
        rows.append([name, "sha256=" + digest, str(len(data))])
    rows.append([subject.DIST + "RECORD", "", ""])
    if rows_mutator:
        rows_mutator(rows)
    stream = io.StringIO(newline="")
    csv.writer(stream).writerows(rows)
    members[subject.DIST + "RECORD"] = stream.getvalue().encode()
    result = io.BytesIO()
    with zipfile.ZipFile(result, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
        if duplicate:
            archive.writestr(duplicate, b"duplicate")
    return result.getvalue()


def tree(files):
    return b"".join(
        f"100644 blob {subject.git_blob(body)}\t{name}\0".encode()
        for name, body in files.items()
    )


def pypi():
    return {
        "info": {"name": "vllm-omni", "version": "0.29.0rc1", "yanked": False},
        "urls": [{
            "filename": subject.FILENAME, "url": subject.WHEEL_URL,
            "size": subject.WHEEL_SIZE, "yanked": False, "packagetype": "bdist_wheel",
            "digests": {"sha256": subject.WHEEL_SHA256},
        }],
    }


class FakeResponse(io.BytesIO):
    def __init__(self, data, *, status=200, url=subject.WHEEL_URL, headers=None):
        super().__init__(data)
        self.status = status
        self.url = url
        self.headers = headers or {}

    def geturl(self):
        return self.url


class AcquisitionTests(unittest.TestCase):
    def test_primary_pins(self):
        self.assertEqual(subject.UPSTREAM, "aff7d64948f6c0e81e9b3272234623044e215967")
        self.assertEqual(subject.ADMISSION, "fa16050c930a2f5a258619101a98bfc2013cb783")
        self.assertEqual(subject.WHEEL_SHA256, "5c5ab1d7e66b2289e81a5f01511bda41f734ea7f02bae1150534e8df16207d79")
        self.assertEqual(subject.WHEEL_SIZE, 7275029)

    def test_pypi_valid(self):
        raw = json.dumps(pypi()).encode()
        self.assertEqual(subject.validate_pypi(raw)["responseSha256"], subject.sha256(raw))

    def test_pypi_identity_drift(self):
        cases = [
            ("url", "https://example.invalid/wheel"), ("size", True), ("size", 1),
            ("yanked", True), ("yanked", None), ("packagetype", "sdist"),
            ("digests", {"sha256": "0" * 64}), ("digests", None),
        ]
        for key, value in cases:
            with self.subTest(key=key, value=value):
                data = pypi()
                data["urls"][0][key] = value
                with self.assertRaises(subject.AcquisitionError):
                    subject.validate_pypi(json.dumps(data).encode())

    def test_pypi_release_and_ambiguity(self):
        for mutator in (
            lambda d: d["info"].update(version="0.29.0"),
            lambda d: d["info"].update(yanked=True),
            lambda d: d["urls"].append(copy.deepcopy(d["urls"][0])),
            lambda d: d.update(urls=[]), lambda d: d.update(info=[]),
        ):
            data = pypi()
            mutator(data)
            with self.subTest(data=data), self.assertRaises(subject.AcquisitionError):
                subject.validate_pypi(json.dumps(data).encode())

    def test_json_duplicate_and_size(self):
        for raw in (b'{"info":{},"info":{}}', b" " * (1024 * 1024 + 1)):
            with self.assertRaises(subject.AcquisitionError):
                subject.validate_pypi(raw)

    def test_synthetic_record_and_metadata(self):
        result, blobs = subject.inspect_wheel(wheel())
        self.assertEqual(result["recordIntegrity"], "PASS")
        self.assertEqual(result["metadataSha256"], subject.sha256(METADATA))
        self.assertEqual(blobs, {name: subject.git_blob(body) for name, body in SOURCE.items()})
        self.assertEqual(result["requiresDist"], ["transformers<5.15,>=5.13.0"])

    def test_record_integrity_negatives(self):
        mutators = [
            lambda r: r[0].__setitem__(1, "sha256=" + "A" * 43),
            lambda r: r[0].__setitem__(2, "1"),
            lambda r: r.pop(0),
            lambda r: r.__setitem__(0, r[1]),
            lambda r: r[-1].__setitem__(1, "sha256=" + "A" * 43),
            lambda r: r[0].append("unexpected"),
            lambda r: r[0].__setitem__(0, "missing.py"),
        ]
        for index, mutate in enumerate(mutators):
            with self.subTest(index=index), self.assertRaises(subject.AcquisitionError):
                subject.inspect_wheel(wheel(rows_mutator=mutate))

    def test_path_traversal_and_aliases(self):
        for name in ("../escape", "/root", "a/../b", "a//b", "a/./b", "a\\b", "a:", "a/"):
            with self.subTest(name=name), self.assertRaises(subject.AcquisitionError):
                subject.inspect_wheel(wheel({name: b"bad"}))

    def test_duplicate_archive_path(self):
        with self.assertWarns(UserWarning):
            data = wheel(duplicate="vllm_omni/__init__.py")
        with self.assertRaises(subject.AcquisitionError):
            subject.inspect_wheel(data)

    def test_nonregular_member(self):
        result = io.BytesIO()
        with zipfile.ZipFile(result, "w") as archive:
            info = zipfile.ZipInfo("vllm_omni/link")
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, b"target")
        with self.assertRaises(subject.AcquisitionError):
            subject.inspect_wheel(result.getvalue())

    def test_bounds_before_member_reads(self):
        data = wheel()
        for key, limit in (("MAX_FILES", 1), ("MAX_MEMBER", 1), ("MAX_EXPANDED", 1)):
            with self.subTest(key=key), patch.object(subject, key, limit):
                with patch.object(zipfile.ZipFile, "open") as opened:
                    with self.assertRaises(subject.AcquisitionError):
                        subject.inspect_wheel(data)
                    opened.assert_not_called()

    def test_metadata_drift_and_duplicate_headers(self):
        for metadata in (
            METADATA.replace(b"0.29.0rc1", b"0.29.0"),
            METADATA.replace(b"Apache-2.0", b"unknown"),
            METADATA.replace(b"transformers<5.15,>=5.13.0", b"transformers>=5.17"),
            b"Name: vllm-omni\n" + METADATA,
        ):
            with self.subTest(metadata=metadata), self.assertRaises(subject.AcquisitionError):
                subject.inspect_wheel(wheel(metadata=metadata))

    def test_source_correspondence_and_generated_bound(self):
        _, packaged = subject.inspect_wheel(wheel({subject.GENERATED: b"generated"}))
        result = subject.compare_source(packaged, tree({**SOURCE, "vllm_omni/data.txt": b"data"}))
        self.assertEqual(result["state"], "PASS")
        self.assertEqual(result["comparedFiles"], 1)
        self.assertEqual(result["unpackagedNonPython"], ["vllm_omni/data.txt"])
        self.assertFalse(result["buildAttestationVerified"])

    def test_changed_extra_and_missing_sources(self):
        original = {name: subject.git_blob(body) for name, body in SOURCE.items()}
        for packaged, files, field in (
            ({"vllm_omni/__init__.py": "0" * 40}, SOURCE, "changed"),
            ({**original, "vllm_omni/new.py": "0" * 40}, SOURCE, "extra"),
            (original, {**SOURCE, "vllm_omni/missing.py": b"x"}, "missingPython"),
        ):
            with self.subTest(field=field):
                result = subject.compare_source(packaged, tree(files))
                self.assertEqual(result["state"], "FAIL")
                self.assertTrue(result[field])

    def test_empty_or_nonregular_source(self):
        original = {name: subject.git_blob(body) for name, body in SOURCE.items()}
        for data in (b"", tree(SOURCE).replace(b"100644", b"120000"), tree(SOURCE) * 2):
            with self.subTest(data=data), self.assertRaises(subject.AcquisitionError):
                subject.compare_source(original, data)
        with self.assertRaises(subject.AcquisitionError):
            subject.compare_source({}, tree(SOURCE))

    def test_unadmitted_url_never_opens(self):
        with patch.object(subject.urllib.request, "build_opener") as opener:
            with self.assertRaises(subject.AcquisitionError):
                subject.download("http://127.0.0.1/", 10)
            opener.assert_not_called()

    def test_download_actual_and_declared_bounds(self):
        responses = [
            FakeResponse(b"abc", headers={"Content-Length": "4"}),
            FakeResponse(b"abcd"), FakeResponse(b"ab"),
            FakeResponse(b"abc", status=201),
            FakeResponse(b"abc", url="https://example.invalid/"),
            FakeResponse(b"abc", headers={"Content-Encoding": "gzip"}),
        ]
        for response in responses:
            with self.subTest(response=response), patch.object(subject.urllib.request, "build_opener") as opener:
                opener.return_value.open.return_value = response
                with self.assertRaises(subject.AcquisitionError):
                    subject.download(subject.WHEEL_URL, 3, 3)

    def test_download_complete_and_redirect_rejected(self):
        with patch.object(subject.urllib.request, "build_opener") as opener:
            opener.return_value.open.return_value = FakeResponse(b"abc")
            self.assertEqual(subject.download(subject.WHEEL_URL, 3, 3), b"abc")
        with self.assertRaises(subject.AcquisitionError):
            subject.NoRedirect().redirect_request(None, None, 302, "", {}, subject.WHEEL_URL)

    def test_fixture_runner_keeps_runtime_and_build_authority_false(self):
        data = wheel()

        def fake_git(repo, *args):
            if args == ("rev-parse", "HEAD"):
                return (subject.UPSTREAM if repo.name == "upstream-vllm-omni" else "f" * 40).encode()
            if args[0] == "show":
                return (subject.ROOT / args[1].split(":", 1)[1]).read_bytes()
            return tree(SOURCE)

        with patch.object(subject, "WHEEL_SIZE", len(data)), patch.object(subject, "WHEEL_SHA256", subject.sha256(data)), patch.object(subject, "METADATA_SHA256", subject.sha256(METADATA)):
            with patch.object(subject, "_git", side_effect=fake_git), patch.object(subject, "download", side_effect=[json.dumps(pypi()).encode(), data]):
                report = subject.run()
        self.assertEqual(report["state"], "ACQUISITION_VERIFIED_RUNTIME_NOT_RUN")
        self.assertEqual(report["disposition"], "HOLD")
        self.assertEqual(len(report["sourceFiles"]), 4)
        for field in ("runtimeExecuted", "packageInstalled", "artifactSourceBound", "buildAttestationVerified", "dependencyClosureCaptured", "modelWeightsAcquired", "productionServingAuthorized", "hubPublicationAuthorized"):
            self.assertIs(report[field], False)

    def test_dirty_executing_source_fails_before_network(self):
        with patch.object(subject, "_git", side_effect=[b"f" * 40, b"not the source bytes"]), patch.object(subject, "download") as download:
            report = subject.run()
        download.assert_not_called()
        self.assertEqual(report["state"], "FAIL")
        self.assertIn("differs from Git", report["error"]["message"])

    def test_source_preflight_failure_has_no_network_or_pass(self):
        with patch.object(subject, "_git", side_effect=subprocess.CalledProcessError(1, ["git"])), patch.object(subject, "download") as download:
            report = subject.run()
        download.assert_not_called()
        self.assertEqual(report["state"], "FAIL")
        self.assertFalse(report["artifactBytesIndependentlyVerified"])
        self.assertFalse(report["productionServingAuthorized"])
        self.assertFalse(report["runtimeExecuted"])


if __name__ == "__main__":
    unittest.main()
