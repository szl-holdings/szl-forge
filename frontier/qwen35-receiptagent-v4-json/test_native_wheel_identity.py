# SPDX-License-Identifier: Apache-2.0
"""Synthetic ZIP/data fixtures only; no installation, real wheel or runtime proof."""
import hashlib
import json
from pathlib import Path
import random
import stat
import struct
import sys
import tempfile
import traceback
import unittest
from unittest import mock
import warnings
import zipfile
import zlib

HERE = Path(__file__).absolute().parent
sys.path.insert(0, str(HERE))
import native_identity
import native_wheel_identity as identity


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


class NativeWheelTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.parent = Path(self.temporary.name).absolute()
        self.wheels = self.parent / "wheels"
        self.installed = self.parent / "installed"
        self.wheels.mkdir()
        self.installed.mkdir()
        self.inventory = self.parent / "inventory.json"
        self.manifest = self.parent / "manifest.json"
        self.members = {"package/__init__.py": b"", "package/data.bin": b"opaque data",
                        "sample-1.0.dist-info/RECORD": b"opaque RECORD bytes\n"}
        self.wheel = self.wheels / "sample-1.0-py3-none-any.whl"
        self.write_wheel(self.wheel, list(self.members.items()))
        self.bind_tree(self.members)
        self.bind_inventory()

    def write_wheel(self, path, members, *, compression=zipfile.ZIP_STORED):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(path, "w", compression=compression, allowZip64=False) as archive:
                for name, data in members:
                    archive.writestr(name, data)

    def bind_tree(self, members):
        for name, raw in members.items():
            path = self.installed.joinpath(*name.split("/"))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        document = {"schema": native_identity.MANIFEST_SCHEMA, "files": [
            {"path": name, "size_bytes": len(raw), "sha256": sha(raw)}
            for name, raw in sorted(members.items())]}
        data = canonical(document)
        self.manifest.write_bytes(data)
        self.native_pin = sha(data)

    def bind_inventory(self):
        self.document = {"schema": identity.INVENTORY_SCHEMA,
                         "native_manifest_sha256": self.native_pin,
                         "wheels": [{"path": path.name, "size_bytes": path.stat().st_size,
                                     "sha256": sha(path.read_bytes())}
                                    for path in sorted(self.wheels.iterdir())]}
        self.write_inventory(self.document)

    def write_inventory(self, document=None, raw=None):
        raw = canonical(self.document if document is None else document) if raw is None else raw
        self.inventory.write_bytes(raw)
        self.inventory_pin = sha(raw)

    def verify(self):
        return identity.verify_native_wheel_inventory(
            self.wheels, self.inventory, self.installed, self.manifest,
            expected_inventory_sha256=self.inventory_pin,
            expected_native_manifest_sha256=self.native_pin)

    def deny(self, code=None):
        with self.assertRaises(identity.NativeWheelIdentityError) as caught:
            self.verify()
        if code:
            self.assertEqual(caught.exception.code, code)

    def mutate_zip(self, mutation):
        data = bytearray(self.wheel.read_bytes())
        mutation(data)
        self.wheel.write_bytes(data)
        self.bind_inventory()

    def replace_only_payload(self, payload, *, size=None, crc=None):
        """Test-only coherent ZIP header edits; never produce a real inventory."""
        raw = bytearray(self.wheel.read_bytes())
        central = struct.unpack_from("<I", raw, len(raw) - 22 + 16)[0]
        start = 30 + struct.unpack_from("<H", raw, 26)[0]
        raw[start:central] = payload
        central = start + len(payload)
        struct.pack_into("<I", raw, 18, len(payload))
        struct.pack_into("<I", raw, central + 20, len(payload))
        struct.pack_into("<I", raw, len(raw) - 22 + 16, central)
        if size is not None:
            struct.pack_into("<I", raw, 22, size)
            struct.pack_into("<I", raw, central + 24, size)
        if crc is not None:
            struct.pack_into("<I", raw, 14, crc)
            struct.pack_into("<I", raw, central + 16, crc)
        self.wheel.write_bytes(raw)
        self.bind_inventory()

    def test_exact_root_files_and_shared_namespace_directories(self):
        second_members = {"package/other.py": b"raise RuntimeError('never imported')\n",
                          "second-1.0.dist-info/RECORD": b"second opaque record"}
        self.write_wheel(self.wheels / "second-1.0-py3-none-any.whl", list(second_members.items()))
        self.bind_tree({**self.members, **second_members})
        self.bind_inventory()
        result = self.verify()
        self.assertEqual(result.wheel_count, 2)
        self.assertEqual(result.file_count, 5)
        self.assertEqual(result.total_bytes, sum(map(len, {**self.members, **second_members}.values())))
        self.assertEqual(result.inventory_sha256, self.inventory_pin)
        self.assertEqual(result.native_manifest_sha256, self.native_pin)
        with self.assertRaises(AttributeError):
            result.file_count = 99

    def test_external_inventory_and_native_pins_are_compulsory(self):
        for key in ("expected_inventory_sha256", "expected_native_manifest_sha256"):
            for value in (None, "", "A" * 64, True):
                arguments = {"expected_inventory_sha256": self.inventory_pin,
                             "expected_native_manifest_sha256": self.native_pin, key: value}
                with self.subTest(key=key, value=value), self.assertRaises(identity.NativeWheelIdentityError):
                    identity.verify_native_wheel_inventory(self.wheels, self.inventory, self.installed,
                                                           self.manifest, **arguments)
        self.inventory_pin = "a" * 64
        self.deny("INVENTORY_PIN_MISMATCH")

    def test_closed_inventory_canonical_names_and_sizes(self):
        baseline = canonical(self.document)
        mutations = [lambda d: d.update(extra=True),
                     lambda d: d.update(schema="szl.receiptagent.v4-training-admission/v1"),
                     lambda d: d["wheels"][0].update(path="../outside.whl"),
                     lambda d: d["wheels"][0].update(path="sub/nested.whl"),
                     lambda d: d["wheels"][0].update(size_bytes=True),
                     lambda d: d["wheels"].append(dict(d["wheels"][0])),
                     lambda d: d.update(native_manifest_sha256="f" * 64)]
        for mutation in mutations:
            document = json.loads(baseline)
            mutation(document)
            with self.subTest(mutation=mutation):
                self.write_inventory(document)
                self.deny()
        self.write_inventory(raw=baseline + b"\n")
        self.deny("NONCANONICAL_INVENTORY")
        self.write_inventory(raw=b'{"schema":"x","schema":"y"}')
        self.deny("DUPLICATE_JSON_FIELD")

    def test_wheel_pin_is_checked_before_zipfile_construction(self):
        self.wheel.write_bytes(b"different bytes")
        # Preserve size in the inventory so this specifically checks the digest.
        document = json.loads(canonical(self.document))
        document["wheels"][0]["size_bytes"] = self.wheel.stat().st_size
        self.write_inventory(document)
        with mock.patch.object(identity.zipfile, "ZipFile", side_effect=AssertionError("ZIP must not parse")):
            self.deny("WHEEL_DIGEST_MISMATCH")

    def test_resource_caps_apply_before_zipfile_construction(self):
        for name, value in (("MAX_INVENTORY_BYTES", 1), ("MAX_WHEEL_BYTES", 1),
                            ("MAX_TOTAL_WHEEL_BYTES", 1), ("MAX_CENTRAL_DIRECTORY_BYTES", 1),
                            ("MAX_MEMBERS", 1), ("MAX_TOTAL_MEMBERS", 1),
                            ("MAX_NAMESPACE_ENTRIES", 1), ("MAX_WHEELS", 0), ("MAX_MEMBER_BYTES", 1),
                            ("MAX_MEMBER_COMPRESSED_BYTES", 1), ("MAX_TOTAL_UNCOMPRESSED_BYTES", 1)):
            with self.subTest(cap=name), mock.patch.object(identity, name, value), mock.patch.object(
                    identity.zipfile, "ZipFile", side_effect=AssertionError("ZIP must not parse")):
                self.deny()

    def test_actual_central_records_must_match_eocd_count(self):
        self.mutate_zip(lambda raw: struct.pack_into("<HH", raw, len(raw) - 22 + 8, 1, 1))
        with mock.patch.object(identity.zipfile, "ZipFile", side_effect=AssertionError("ZIP must not parse")):
            self.deny()

    def test_eocd_multidisk_zip64_and_directory_caps_are_rejected(self):
        original = self.wheel.read_bytes()
        mutations = [lambda b: struct.pack_into("<H", b, len(b) - 22 + 4, 1),
                     lambda b: struct.pack_into("<I", b, len(b) - 22 + 12, 0xFFFFFFFF),
                     lambda b: struct.pack_into("<I", b, len(b) - 22 + 16, 0xFFFFFFFF),
                     lambda b: struct.pack_into("<HH", b, len(b) - 22 + 8, 65535, 65535)]
        for mutation in mutations:
            self.wheel.write_bytes(original)
            self.mutate_zip(mutation)
            with self.subTest(mutation=mutation), mock.patch.object(
                    identity.zipfile, "ZipFile", side_effect=AssertionError("ZIP must not parse")):
                self.deny()

    def test_preambles_trailing_and_concatenated_archives_rejected(self):
        original = self.wheel.read_bytes()
        for data in (b"stub" + original, original + b"overlay", original + original):
            with self.subTest(size=len(data)):
                self.wheel.write_bytes(data)
                self.bind_inventory()
                self.deny()

    def test_traversal_aliases_nul_and_data_relocation_rejected(self):
        for name in ("../escape", "/absolute", "C:/absolute", "p\\x", "p/./x", "p//x",
                     "package-1.0.data/purelib/a.py", "CON.txt"):
            with self.subTest(name=name):
                self.write_wheel(self.wheel, [(name, b"x")])
                self.bind_inventory()
                self.deny()
        self.write_wheel(self.wheel, [("p/xx", b"x")])
        self.wheel.write_bytes(self.wheel.read_bytes().replace(b"p/xx", b"p/\x00x"))
        self.bind_inventory()
        with mock.patch.object(identity.zipfile, "ZipFile", side_effect=AssertionError("ZIP must not parse")):
            self.deny()

    def test_duplicate_case_and_file_directory_collisions_rejected(self):
        for names in (("x", "x"), ("X", "x"), ("p", "p/x"), ("P/x", "p/y"), ("p/", "p")):
            with self.subTest(names=names):
                self.write_wheel(self.wheel, [(name, b"" if name.endswith("/") else b"x") for name in names])
                self.bind_inventory()
                self.deny()

    def test_cross_wheel_file_overlap_rejected_even_equal_bytes(self):
        self.write_wheel(self.wheels / "other-1.0-py3-none-any.whl", [("package/data.bin", b"opaque data")])
        self.bind_inventory()
        self.deny("WHEEL_FILE_OVERLAP")

    def test_explicit_shared_directories_and_empty_directory_rejected(self):
        self.write_wheel(self.wheel, [("package/", b""), *self.members.items()])
        second = {"package/other.py": b"opaque"}
        self.write_wheel(self.wheels / "second-1.0-py3-none-any.whl",
                         [("package/", b""), *second.items()])
        self.bind_tree({**self.members, **second})
        self.bind_inventory()
        self.assertEqual(self.verify().file_count, 4)
        self.write_wheel(self.wheel, [("empty/", b""), *self.members.items()])
        self.bind_inventory()
        self.deny("WHEEL_NATIVE_MANIFEST_MISMATCH")

    def test_cross_wheel_parent_case_collision_rejected(self):
        self.write_wheel(self.wheels / "second-1.0-py3-none-any.whl", [("Package/other.py", b"opaque")])
        self.bind_inventory()
        self.deny("ZIP_PATH_COLLISION")

    def test_symlink_special_entries_and_encryption_rejected(self):
        for mode in (stat.S_IFLNK, stat.S_IFIFO, stat.S_IFSOCK, stat.S_IFCHR):
            with self.subTest(mode=mode):
                item = zipfile.ZipInfo("p/file")
                item.create_system = 3
                item.external_attr = (mode | 0o600) << 16
                self.write_wheel(self.wheel, [(item, b"x")])
                self.bind_inventory()
                self.deny()
        self.write_wheel(self.wheel, list(self.members.items()))
        def encrypt(raw):
            central = raw.index(b"PK\x01\x02")
            struct.pack_into("<H", raw, 6, 1)
            struct.pack_into("<H", raw, central + 8, 1)
        self.mutate_zip(encrypt)
        self.deny()

    def test_extra_comment_and_data_descriptor_metadata_rejected(self):
        for option in ("extra", "comment"):
            item = zipfile.ZipInfo("p/file")
            setattr(item, option, b"\xfe\xca\x00\x00" if option == "extra" else b"comment")
            self.write_wheel(self.wheel, [(item, b"x")])
            self.bind_inventory()
            self.deny()
        self.write_wheel(self.wheel, list(self.members.items()))
        def descriptor(raw):
            central = raw.index(b"PK\x01\x02")
            struct.pack_into("<H", raw, 6, 8)
            struct.pack_into("<H", raw, central + 8, 8)
        self.mutate_zip(descriptor)
        self.deny()

    def test_archive_comment_unsupported_method_and_special_dos_attribute(self):
        with zipfile.ZipFile(self.wheel, "a") as archive:
            archive.comment = b"archive comment"
        self.bind_inventory()
        with mock.patch.object(identity.zipfile, "ZipFile", side_effect=AssertionError("ZIP must not parse")):
            self.deny()
        for mode in ("method", "attribute"):
            self.write_wheel(self.wheel, list(self.members.items()))
            def alter(raw):
                central = raw.index(b"PK\x01\x02")
                if mode == "method":
                    struct.pack_into("<H", raw, 8, 99)
                    struct.pack_into("<H", raw, central + 10, 99)
                else:
                    struct.pack_into("<I", raw, central + 38, 0x08)
            self.mutate_zip(alter)
            with self.subTest(mode=mode), mock.patch.object(
                    identity.zipfile, "ZipFile", side_effect=AssertionError("ZIP must not parse")):
                self.deny()

    def test_local_header_mismatch_rejected_before_zipfile(self):
        self.mutate_zip(lambda raw: struct.pack_into("<H", raw, 8, 8))
        with mock.patch.object(identity.zipfile, "ZipFile", side_effect=AssertionError("ZIP must not parse")):
            self.deny()

    def test_deflate_expansion_and_observed_crc_are_checked(self):
        self.write_wheel(self.wheel, [("p/huge", b"x" * 100000)], compression=zipfile.ZIP_DEFLATED)
        self.bind_inventory()
        with mock.patch.object(identity.zipfile, "ZipFile", side_effect=AssertionError("ZIP must not parse")):
            self.deny("ZIP_COMPRESSION_RATIO_CAP")
        self.write_wheel(self.wheel, list(self.members.items()))
        def corrupt_crc(raw):
            central = raw.index(b"PK\x01\x02")
            struct.pack_into("<I", raw, 14, 1)
            struct.pack_into("<I", raw, central + 16, 1)
        self.mutate_zip(corrupt_crc)
        self.deny()

    def test_complete_deflate_stream_rejects_hidden_suffix_and_declared_prefix(self):
        data = b"abcdef"
        compressor = zlib.compressobj(wbits=-15)
        payload = compressor.compress(data) + compressor.flush()
        variants = [(payload + b"hidden", None, None), (payload[:-1], None, None),
                    (payload, 3, zlib.crc32(data[:3])), (payload, 7, zlib.crc32(data))]
        for body, size, crc in variants:
            self.write_wheel(self.wheel, [("p/file", data)], compression=zipfile.ZIP_DEFLATED)
            self.replace_only_payload(body, size=size, crc=crc)
            with self.subTest(size=size, compressed=len(body)):
                self.deny()

    def test_multichunk_deflate_output_is_bounded(self):
        members = {**self.members, "package/large.so": random.Random(17).randbytes(3 * 1024**2 + 19)}
        self.write_wheel(self.wheel, list(members.items()), compression=zipfile.ZIP_DEFLATED)
        self.bind_tree(members)
        self.bind_inventory()
        limits = []
        original = identity.zlib.decompressobj
        class BoundedDecoder:
            def __init__(self, *args):
                self.decoder = original(*args)

            def decompress(self, data, maximum):
                limits.append(maximum)
                return self.decoder.decompress(data, maximum)

            def __getattr__(self, name):
                return getattr(self.decoder, name)
        with mock.patch.object(identity.zlib, "decompressobj", BoundedDecoder):
            self.assertEqual(self.verify().total_bytes, sum(map(len, members.values())))
        self.assertTrue(limits)
        self.assertTrue(all(0 < limit <= identity.HASH_CHUNK_BYTES for limit in limits))

    def test_exact_union_requires_unmodified_record_and_no_generated_extras(self):
        (self.installed / "sample-1.0.dist-info/RECORD").write_bytes(b"rewritten")
        self.deny()
        (self.installed / "sample-1.0.dist-info/RECORD").write_bytes(self.members["sample-1.0.dist-info/RECORD"])
        (self.installed / "generated-wrapper").write_bytes(b"generated")
        self.deny()
        (self.installed / "generated-wrapper").unlink()
        # An externally pinned native manifest can still not invent wheel bytes.
        self.bind_tree({**self.members, "generated.pyc": b"generated"})
        self.bind_inventory()
        self.deny("WHEEL_NATIVE_MANIFEST_MISMATCH")

    def test_changed_pinned_native_manifest_rejected(self):
        self.manifest.write_bytes(self.manifest.read_bytes() + b" ")
        self.deny("NATIVE_MANIFEST_PIN_MISMATCH")

    def test_archive_mutation_after_hash_rejected(self):
        original = identity._preflight_zip
        def mutate(stream, size):
            result = original(stream, size)
            with self.wheel.open("ab") as out:
                out.write(b"changed")
            return result
        with mock.patch.object(identity, "_preflight_zip", side_effect=mutate):
            self.deny()

    def test_opaque_startup_files_and_shared_libraries_never_execute(self):
        sentinel = self.parent / "must-not-exist"
        members = {**self.members,
                   "startup.pth": ("import pathlib; pathlib.Path(" + repr(str(sentinel)) + ").touch()\n").encode(),
                   "sitecustomize.py": b"raise RuntimeError('must never import')",
                   "package/native.so": b"opaque non-library data"}
        self.write_wheel(self.wheel, list(members.items()), compression=zipfile.ZIP_DEFLATED)
        self.bind_tree(members)
        self.bind_inventory()
        with mock.patch.object(identity.zipfile.ZipFile, "extract", side_effect=AssertionError("no extraction")), \
                mock.patch.object(identity.zipfile.ZipFile, "extractall", side_effect=AssertionError("no extraction")):
            self.assertEqual(self.verify().file_count, len(members))
        self.assertFalse(sentinel.exists())

    def test_member_reads_are_bounded_and_multimegabyte_bytes_match(self):
        members = {**self.members, "package/large.so": b"x" * (3 * 1024**2 + 17)}
        self.write_wheel(self.wheel, list(members.items()))
        self.bind_tree(members)
        self.bind_inventory()
        sizes = []
        original = identity.zipfile.ZipExtFile.read
        def observe(stream, maximum=-1):
            sizes.append(maximum)
            return original(stream, maximum)
        with mock.patch.object(identity.zipfile.ZipExtFile, "read", observe):
            self.assertEqual(self.verify().total_bytes, sum(map(len, members.values())))
        self.assertTrue(sizes)
        self.assertTrue(all(0 < size <= identity.HASH_CHUNK_BYTES for size in sizes))

    def test_normal_filesystem_traceback_is_redacted(self):
        self.wheel.unlink()
        try:
            self.verify()
        except identity.NativeWheelIdentityError as error:
            rendered = "".join(traceback.format_exception(error))
            self.assertEqual(error.code, "FILESYSTEM_UNAVAILABLE")
            self.assertNotIn(str(self.wheel), rendered)
            self.assertNotIn(self.wheel.name, rendered)
            self.assertTrue(error.__suppress_context__)
        else:
            self.fail("Missing wheel must not verify")


if __name__ == "__main__":
    unittest.main()
