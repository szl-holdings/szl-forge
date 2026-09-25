"""Protocol-level adapter sync and isolation. No GPU."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from frontier.asyncgrpo_lora_protocol import (
    ProtocolError,
    ProtocolServer,
    measure_sync_paths,
    run_negative_paths,
    validate_cache_at_use,
)


class ProtocolTests(unittest.TestCase):
    def test_adapter_only_transfers_fewer_bytes_than_merged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            server = ProtocolServer(
                Path(tmp),
                enable_lora=True,
                max_lora_rank=32,
                max_loras=6,
                max_staleness=4,
                runtime_updates=True,
                checkpoints_enabled=True,
            )
            measured = measure_sync_paths(server, name="policy", rank=32, version=1)
        self.assertLess(measured["adapterOnlyTransferredBytes"], measured["mergedTransferredBytes"])
        self.assertGreater(measured["adapterOnlyTransferredBytes"], 0)
        self.assertIs(measured["policyVersionCorrect"], True)
        self.assertIs(measured["gpuSyncExecuted"], False)
        self.assertEqual(measured["gpuGenerationThroughput"], "UNAVAILABLE")
        self.assertGreaterEqual(measured["adapterOnlySyncPauseSeconds"], 0.0)
        self.assertGreaterEqual(measured["mergedSyncPauseSeconds"], 0.0)

    def test_symlink_escape_is_rejected_at_use(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            server = ProtocolServer(
                root,
                enable_lora=True,
                max_lora_rank=32,
                max_loras=6,
                max_staleness=4,
                runtime_updates=True,
                checkpoints_enabled=True,
            )
            with self.assertRaises((ProtocolError, Exception)):
                server.cache_path(root / ".." / "outside")
            server.ensure_layout()
            outside = Path(tmp) / "outside"
            outside.write_bytes(b"SZL-LORA-V1\nnot-inside")
            link = server.cache_root / "escaped"
            link.symlink_to(outside)
            with self.assertRaises(ProtocolError) as caught:
                validate_cache_at_use(server.output_dir, link)
            self.assertEqual(caught.exception.code, "path_escape_symlink")

    def test_load_before_evict_allows_idle_victim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            server = ProtocolServer(
                Path(tmp),
                enable_lora=True,
                max_lora_rank=32,
                max_loras=2,
                max_staleness=0,
                runtime_updates=True,
                checkpoints_enabled=True,
            )
            a = server.write_adapter("p1", 8, 1)
            b = server.write_adapter("p2", 8, 2)
            c = server.write_adapter("p3", 8, 3)
            server.load_adapter("p1", a, rank=8, version=1)
            server.load_adapter("p2", b, rank=8, version=2)
            server.load_adapter("p3", c, rank=8, version=3)
            self.assertEqual(list(server.loaded), ["p2", "p3"])
            self.assertEqual(server.evicted, ["p1"])

    def test_all_negative_paths_observe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            observed = run_negative_paths(Path(tmp))
        failed = [key for key, ok in observed.items() if not ok]
        self.assertEqual(failed, [])


if __name__ == "__main__":
    unittest.main()
