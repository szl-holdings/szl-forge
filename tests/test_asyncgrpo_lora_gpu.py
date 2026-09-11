"""GPU backend stays UNAVAILABLE without CUDA. HTTP client is real, not a stub."""
from __future__ import annotations

import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from frontier.asyncgrpo_lora_contract import EvaluationContractError
from frontier.asyncgrpo_lora_gpu import (
    GPU_FIELDS,
    VllmEvalClient,
    VllmHandle,
    admit_gpu_measurements,
    gpu_blocker,
    measure_gpu_sync_paths,
    resolve_gpu_artifacts,
    try_gpu_negative_paths,
    try_gpu_sync_paths,
    unavailable_measurements,
    vllm_command,
    write_tiny_fixture,
)


READY = {
    "cudaAvailable": True,
    "vllmPresent": True,
    "exactTrlRevisionMatched": True,
    "peftPresent": True,
}
FAKE = {
    "adapterOnlyTransferredBytes": 128,
    "mergedTransferredBytes": 4096,
    "adapterOnlySyncPauseSeconds": 0.01,
    "mergedSyncPauseSeconds": 0.2,
    "adapterOnlyGenerationThroughput": 12.0,
    "mergedGenerationThroughput": 9.0,
    "policyVersionCorrect": True,
    "resourceUse": {"vramMiB": 1024},
}


class _MockState:
    def __init__(self, *, enable_lora: bool, runtime_updates: bool, max_loras: int) -> None:
        self.enable_lora = enable_lora
        self.runtime_updates = runtime_updates
        self.max_loras = max_loras
        self.loaded: list[str] = []
        self.served_model = "base"


class _MockHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args: object) -> None:
        return

    def _read_json(self) -> dict[str, object]:
        length = int(self.headers.get("Content-Length") or "0")
        raw = self.rfile.read(length) if length else b""
        if not raw:
            return {}
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _send(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        state: _MockState = self.server.state  # type: ignore[attr-defined]
        if self.path.startswith("/health"):
            self._send(200, {"status": "ok"})
            return
        if self.path.startswith("/v1/models"):
            self._send(200, {"data": [{"id": state.served_model, "max_model_len": 64}]})
            return
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        state: _MockState = self.server.state  # type: ignore[attr-defined]
        payload = self._read_json()
        if self.path.startswith("/pause") or self.path.startswith("/resume"):
            self._send(200, {"ok": True})
            return
        if self.path.startswith("/v1/load_lora_adapter"):
            if not state.runtime_updates:
                self._send(404, {"error": "runtime updates disabled"})
                return
            if not state.enable_lora:
                self._send(400, {"error": "lora disabled"})
                return
            if len(state.loaded) >= state.max_loras:
                self._send(400, {"error": "max loras exceeded"})
                return
            name = str(payload.get("lora_name") or "adapter")
            state.loaded.append(name)
            self._send(200, {"ok": True, "lora_name": name})
            return
        if self.path.startswith("/v1/unload_lora_adapter"):
            name = str(payload.get("lora_name") or "")
            state.loaded = [item for item in state.loaded if item != name]
            self._send(200, {"ok": True})
            return
        if self.path.startswith("/v1/completions"):
            model = str(payload.get("model") or "")
            if state.enable_lora and model not in state.loaded and model != state.served_model:
                self._send(404, {"error": "unknown model"})
                return
            tokens = payload.get("max_tokens")
            n_tokens = tokens if type(tokens) is int and tokens > 0 else 8
            self._send(
                200,
                {
                    "choices": [{"text": "x" * n_tokens, "logprobs": {"token_logprobs": [-0.1] * n_tokens}}],
                    "usage": {"completion_tokens": n_tokens},
                },
            )
            return
        self._send(404, {"error": "not found"})


class MockVllm:
    def __init__(self, *, enable_lora: bool = True, runtime_updates: bool = True, max_loras: int = 6) -> None:
        self.state = _MockState(enable_lora=enable_lora, runtime_updates=runtime_updates, max_loras=max_loras)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _MockHandler)
        self.server.state = self.state  # type: ignore[attr-defined]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def port(self) -> int:
        return int(self.server.server_address[1])

    @property
    def client(self) -> VllmEvalClient:
        return VllmEvalClient("127.0.0.1", self.port)

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


class _IdleProcess:
    def poll(self) -> int | None:
        return None

    def terminate(self) -> None:
        return None

    def wait(self, timeout: int = 5) -> int:
        del timeout
        return 0

    def kill(self) -> None:
        return None


def _write_artifacts(root: Path) -> dict[str, Path]:
    model = root / "model"
    adapter = root / "adapter"
    merged = root / "merged"
    model.mkdir(parents=True)
    adapter.mkdir()
    merged.mkdir()
    (model / "config.json").write_text('{"model_type": "gpt2"}\n', encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"base-weights-xxxxxxxx")
    (adapter / "adapter_config.json").write_text('{"r": 32, "peft_type": "LORA"}\n', encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter-xx")
    (merged / "config.json").write_text('{"model_type": "gpt2"}\n', encoding="utf-8")
    (merged / "model.safetensors").write_bytes(b"merged-weights-xxxxxxxxxxxxxxxxxxxx")
    return {"model": model, "adapter": adapter, "merged": merged}


class GpuBackendTests(unittest.TestCase):
    def test_blocker_on_this_host_is_cuda_or_packages(self) -> None:
        from frontier.asyncgrpo_lora_runner import probe_host

        probe = probe_host()
        self.assertIsNotNone(gpu_blocker(probe))
        self.assertEqual(gpu_blocker(READY), None)

    def test_try_gpu_sync_does_not_invent_numbers(self) -> None:
        from frontier.asyncgrpo_lora_runner import probe_host

        with tempfile.TemporaryDirectory() as tmp:
            result = try_gpu_sync_paths(probe_host(), Path(tmp))
        self.assertIs(result["executed"], False)
        self.assertEqual(result["status"], "UNAVAILABLE")
        for field in GPU_FIELDS:
            self.assertEqual(result[field], "UNAVAILABLE")
        self.assertEqual(result["gpuGenerationThroughput"], "UNAVAILABLE")

    def test_ready_probe_without_weights_is_still_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = try_gpu_sync_paths(READY, Path(tmp), execute=True)
        self.assertEqual(result["status"], "UNAVAILABLE")
        self.assertEqual(result["adapterOnlyGenerationThroughput"], "UNAVAILABLE")
        self.assertEqual(result["reason"], "gpu_model_weights_absent")

    def test_admit_rejects_fakes_without_hardware(self) -> None:
        cold = {
            "cudaAvailable": False,
            "vllmPresent": False,
            "exactTrlRevisionMatched": False,
            "peftPresent": False,
        }
        with self.assertRaises(EvaluationContractError):
            admit_gpu_measurements(cold, FAKE)

    def test_admit_accepts_complete_ready_measurements(self) -> None:
        admitted = admit_gpu_measurements(READY, FAKE)
        self.assertIs(admitted["executed"], True)
        self.assertEqual(admitted["adapterOnlyTransferredBytes"], 128)
        self.assertEqual(admitted["status"], "MEASURED")

    def test_gpu_negatives_unavailable_without_host(self) -> None:
        from frontier.asyncgrpo_lora_runner import probe_host

        with tempfile.TemporaryDirectory() as tmp:
            result = try_gpu_negative_paths(probe_host(), Path(tmp))
        self.assertTrue(result)
        for item in result.values():
            self.assertEqual(item["status"], "UNAVAILABLE")
            self.assertIs(item["observed"], False)

    def test_tiny_fixture_is_non_secret(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_tiny_fixture(Path(tmp))
            self.assertTrue((path / "config.json").is_file())
            text = (path / "config.json").read_text(encoding="utf-8")
            self.assertIn("szl_fixture", text)
            self.assertFalse(any(path.glob("*.safetensors")))

    def test_unavailable_helper_has_every_gpu_field(self) -> None:
        payload = unavailable_measurements("cuda_unavailable")
        for field in GPU_FIELDS:
            self.assertEqual(payload[field], "UNAVAILABLE")

    def test_resolve_requires_model_adapter_and_merged_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertIsNone(resolve_gpu_artifacts(root))
            _write_artifacts(root)
            found = resolve_gpu_artifacts(root)
            assert found is not None
            self.assertTrue(found["adapter"].joinpath("adapter_config.json").is_file())

    def test_vllm_command_matches_trl_runtime_flags(self) -> None:
        command = vllm_command(
            "python3", Path("/tmp/model"), 8010, enable_lora=True, max_lora_rank=32, max_loras=6
        )
        self.assertIn("--enable-lora", command)
        self.assertIn("32", command)
        self.assertIn("6", command)
        self.assertIn("vllm.entrypoints.openai.api_server", command)
        bare = vllm_command(
            "python3", Path("/tmp/model"), 8010, enable_lora=False, max_lora_rank=32, max_loras=6
        )
        self.assertNotIn("--enable-lora", bare)

    def test_http_client_measures_both_sync_paths(self) -> None:
        adapter_server = MockVllm(enable_lora=True, runtime_updates=True)
        merged_server = MockVllm(enable_lora=False, runtime_updates=False)
        merged_server.state.served_model = "merged"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                artifacts = _write_artifacts(Path(tmp))
                measured = measure_gpu_sync_paths(
                    adapter_client=adapter_server.client,
                    merged_client=merged_server.client,
                    adapter_dir=artifacts["adapter"],
                    merged_dir=artifacts["merged"],
                    cache_root=Path(tmp),
                    merged_ready_seconds=0.05,
                )
            self.assertLess(measured["adapterOnlyTransferredBytes"], measured["mergedTransferredBytes"])
            self.assertGreater(measured["adapterOnlyGenerationThroughput"], 0)
            self.assertGreater(measured["mergedGenerationThroughput"], 0)
            self.assertEqual(measured["adapterSyncMethod"], "v1/load_lora_adapter")
            self.assertEqual(measured["mergedSyncMethod"], "process-reload-merged-weights")
            self.assertIs(measured["gpuSyncExecuted"], True)
            self.assertIn("policy-v3", measured["loadedAdapterIdentities"])
        finally:
            adapter_server.close()
            merged_server.close()

    def test_load_lora_rejects_symlink_and_partial_before_http(self) -> None:
        client = VllmEvalClient("127.0.0.1", 1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = _write_artifacts(root)
            cache = root / "cache"
            cache.mkdir()
            link = cache / "escaped"
            link.symlink_to(artifacts["adapter"], target_is_directory=True)
            with self.assertRaises(Exception) as caught:
                client.load_lora("escaped", link, cache_root=cache)
            self.assertIn("path_escape_symlink", str(caught.exception))
            partial = root / "partial"
            partial.mkdir()
            (partial / "adapter_config.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaises(Exception) as caught_partial:
                client.load_lora("partial", partial, cache_root=root)
            self.assertIn("partially_written_adapter", str(caught_partial.exception))

    def test_injected_spawn_measures_without_inventing_cuda_on_evaluate_host(self) -> None:
        from frontier.asyncgrpo_lora_runner import evaluate_host, probe_host

        servers: list[MockVllm] = []

        def spawn(output_dir: Path, model: Path, python: str, **kwargs: object) -> VllmHandle:
            del output_dir, model, python
            enable = bool(kwargs.get("enable_lora", True))
            updates = bool(kwargs.get("runtime_updates", True))
            mock = MockVllm(enable_lora=enable, runtime_updates=updates)
            if not enable:
                mock.state.served_model = "merged"
            servers.append(mock)
            return VllmHandle(process=_IdleProcess(), client=mock.client, log_path=Path("/dev/null"))  # type: ignore[arg-type]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_artifacts(root)
            result = try_gpu_sync_paths(READY, root, execute=True, spawn=spawn)
            host = evaluate_host()
        for server in servers:
            server.close()
        self.assertEqual(result["status"], "MEASURED")
        self.assertGreater(result["mergedTransferredBytes"], result["adapterOnlyTransferredBytes"])
        self.assertEqual(host["gpuStatus"], "UNAVAILABLE")
        self.assertEqual(host["disposition"], "UNAVAILABLE")
        self.assertFalse(probe_host()["cudaAvailable"])

    def test_no_runtime_update_is_observed_against_mock(self) -> None:
        server = MockVllm(enable_lora=True, runtime_updates=False)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                artifacts = _write_artifacts(Path(tmp))
                with self.assertRaises(Exception) as caught:
                    server.client.load_lora("policy-v1", artifacts["adapter"], cache_root=Path(tmp))
            self.assertIn("runtime_adapter_update_unavailable", str(caught.exception))
        finally:
            server.close()

    def test_vllm_eval_client_rejects_non_int_port(self) -> None:
        with self.assertRaises(EvaluationContractError):
            VllmEvalClient("127.0.0.1", True)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
