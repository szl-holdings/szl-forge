import importlib.util
import io
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "local-compute"))
spec = importlib.util.spec_from_file_location("research_hf", ROOT / "local-compute/research_hf.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class FakeOpener:
    def __init__(self, body):
        self.body = body
        self.calls = []

    def open(self, request, timeout):
        self.calls.append((request, timeout))
        return io.BytesIO(json.dumps(self.body).encode())


def adapter(body):
    value = runner.HFGenerator("owner/model:provider", "synthetic-test-token")
    value.opener = FakeOpener(body)
    return value


def test_hosted_call_is_bounded_and_records_only_usage():
    value = adapter({"choices": [{"finish_reason": "stop", "message": {"content": "{}"}}],
                     "usage": {"total_tokens": 5, "untrusted_extra": "ignore"}})
    assert value([{"role": "user", "content": "question"}]) == "{}"
    request, timeout = value.opener.calls[0]
    assert request.full_url == runner.ROUTER
    assert json.loads(request.data)["max_tokens"] == 1536
    assert timeout == 90
    assert value.usage == [{"total_tokens": 5}]


@pytest.mark.parametrize("body", [{"choices": []}, {"choices": [{"finish_reason": "length"}]},
                                  {"choices": [{"finish_reason": "stop", "message": {"content": None}}]}])
def test_incomplete_completion_rejected(body):
    with pytest.raises(ValueError):
        adapter(body)([{"role": "user", "content": "q"}])


def test_oversized_prompt_never_calls_provider():
    value = adapter({})
    with pytest.raises(ValueError, match="byte budget"):
        value([{"role": "user", "content": "x" * 32_001}])
    assert not value.opener.calls


@pytest.mark.parametrize("model", ["owner/model", "https://other.example/model", "model:provider"])
def test_no_implicit_provider_or_arbitrary_endpoint(model):
    with pytest.raises(ValueError):
        runner.HFGenerator(model, "synthetic")


def test_redirects_disabled():
    from benchmark_ollama import NoRedirects
    value = runner.HFGenerator("owner/model:provider", "synthetic")
    assert any(isinstance(handler, NoRedirects) for handler in value.opener.handlers)


@pytest.mark.parametrize("budget", [0, 4097, True])
def test_unbounded_token_settings_rejected(budget):
    with pytest.raises(ValueError, match="token budget"):
        runner.HFGenerator("owner/model:provider", "synthetic", budget)


def test_reasoning_model_budget_is_explicit_and_incomplete_status_retained():
    value = runner.HFGenerator("owner/model:provider", "synthetic", 4096)
    value.opener = FakeOpener({"choices": [{"finish_reason": "length"}]})
    with pytest.raises(runner.GenerationIncomplete):
        value([{"role": "user", "content": "q"}])
    assert json.loads(value.opener.calls[0][0].data)["max_tokens"] == 4096
    assert value.completion_status == ["length"]
