import importlib.util
import json
from pathlib import Path
import sys

import pytest

from inference.research_investigator import sha256


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "local-compute"))
spec = importlib.util.spec_from_file_location("research_ollama", ROOT / "local-compute/research_ollama.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class FakeClient:
    def __init__(self, *, changed=False, incomplete=False, loaded=False):
        self.changed, self.incomplete, self.loaded = changed, incomplete, loaded
        self.calls = []

    def call(self, path, payload=None):
        self.calls.append(path)
        if path == "/api/ps":
            return {"models": ["busy"] if self.loaded else []}
        if path == "/api/tags":
            return {"models": [{"name": "test:latest", "size": 100,
                "digest": ("b" if self.changed and self.calls.count(path) > 1 else "a") * 64}]}
        if path == "/api/show":
            return {"details": {"format": "gguf"}}
        if path == "/api/chat":
            assert payload["keep_alive"] == 0
            assert payload["format"] == "json"
            return {"done": True, "done_reason": "length" if self.incomplete else "stop",
                    "message": {"content": '{"tool":"abstain","reason":"No relevant evidence"}'}}
        pytest.fail("unexpected endpoint")


@pytest.fixture
def paths(tmp_path):
    source = tmp_path / "corpus.jsonl"
    row = dict(id="test", text="Source evidence.", source="local-test", revision="a" * 40,
               visibility="public", content_sha256=sha256(b"Source evidence."))
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return source, tmp_path / "run.json"


def test_local_run_preserves_measured_identity_and_abstention(paths):
    source, output = paths
    result = runner.run(FakeClient(), "test:latest", source, "question", output)
    assert result["state"] == "ABSTAINED"
    assert result["model_digest_stable"] is True
    assert result["model"]["digest"] == "a" * 64
    assert not result["trained"]
    assert json.loads(output.read_text())["corpus_file_sha256"] == sha256(source.read_bytes())


@pytest.mark.parametrize("client,state", [(FakeClient(changed=True), "MODEL_BINDING_LOST"),
                                         (FakeClient(incomplete=True), "GENERATOR_ERROR")])
def test_runtime_failure_does_not_qualify(paths, client, state):
    source, output = paths
    assert runner.run(client, "test:latest", source, "question", output)["state"] == state


def test_existing_output_not_replaced(paths):
    source, output = paths
    output.write_text("existing", encoding="utf-8")
    with pytest.raises(ValueError, match="already exists"):
        runner.run(FakeClient(), "test:latest", source, "question", output)
    assert output.read_text() == "existing"


def test_loaded_work_not_disturbed(paths):
    source, output = paths
    client = FakeClient(loaded=True)
    with pytest.raises(ValueError, match="loaded"):
        runner.run(client, "test:latest", source, "question", output)
    assert client.calls == ["/api/ps"]


def test_missing_model_is_not_downloaded(paths):
    source, output = paths
    client = FakeClient()
    with pytest.raises(ValueError, match="already installed"):
        runner.run(client, "missing:latest", source, "question", output)
    assert "/api/chat" not in client.calls
