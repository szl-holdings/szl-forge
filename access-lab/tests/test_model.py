import json
from copy import deepcopy
from http.client import IncompleteRead
from unittest.mock import patch

import pytest

from szl_access import engine, model

NAME = "fixture:1"
MANIFEST = "a" * 64
BLOB = "b" * 64


def tags():
    return {"models": [{"name": NAME, "model": NAME, "digest": MANIFEST, "details": {"format": "gguf"}}]}


def details():
    return {"details": {"format": "gguf", "family": "fixture"},
            "model_info": {"general.architecture": "fixture", "general.parameter_count": 1000},
            "modelfile": f'# fixture only\nFROM C:\\PrivateUser\\.ollama\\models\\blobs\\sha256-{BLOB}\nTEMPLATE "{{{{ .Prompt }}}}"'}


def generated(bindings=None, **extra):
    if bindings is None:
        bindings = [{"candidate_id": analysis()["candidates"][0]["id"]}]
    return {"model": NAME, "done": True, "response": json.dumps({"bindings": bindings}), **extra}


def mock_sequence(opener, responses):
    opener.return_value.open.side_effect = [Reply(value) if not isinstance(value, Exception) else value for value in responses]


def called_paths(opener):
    return [call.args[0].full_url for call in opener.return_value.open.call_args_list]


def assert_no_prompt(opener):
    assert all(not path.endswith("/generate") for path in called_paths(opener))
    for call in opener.return_value.open.call_args_list:
        if call.args[0].data:
            assert "prompt" not in json.loads(call.args[0].data)


class Reply:
    def __init__(self, data):
        self.data = data if isinstance(data, bytes) else json.dumps(data).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self, length):
        return self.data[:length]


def analysis():
    return engine.analyze('<form><label>Email</label><input id="email" type="email"></form>')


@pytest.mark.parametrize("raw", [b'{', b'['*5000+b'0'+b']'*5000, b'x'*65537,
                                 b'{"done":false,"response":"{}"}',
                                 b'{"done":true,"response":"{\\"bindings\\":[],\\"bindings\\":[]}"}',
                                 b'{"done":true,"response":"{\\"bindings\\":[],\\"code\\":\\"evil\\"}"}'],
                         ids=["malformed", "deep", "oversize", "incomplete", "duplicate", "extra"])
def test_bad_model_responses_are_unavailable(raw):
    with patch.object(model, "build_opener") as opener:
        mock_sequence(opener, [tags(), details(), tags(), raw])
        with pytest.raises(model.ModelUnavailable):
            model.propose(analysis(), "fixture:1")


def test_valid_selection_is_bound_and_endpoint_is_local():
    observed = analysis()
    candidate = observed["candidates"][0]["id"]
    raw = json.dumps({"model": NAME, "done": True, "response": json.dumps({"bindings": [{"candidate_id": candidate}]})}).encode()
    with patch.object(model, "build_opener") as opener:
        mock_sequence(opener, [tags(), details(), tags(), raw, tags()])
        proposal, evidence = model.propose(observed, "fixture:1")
        req = opener.return_value.open.call_args_list[3].args[0]
        assert req.full_url == "http://127.0.0.1:11434/api/generate"
        assert json.loads(req.data)["options"]["num_ctx"] == 4096
        assert called_paths(opener) == ["http://127.0.0.1:11434/api/" + path for path in ("tags", "show", "tags", "generate", "tags")]
        handlers = opener.call_args.args
        assert isinstance(handlers[0], model.ProxyHandler) and handlers[0].proxies == {}
        assert isinstance(handlers[1], model.NoRedirect)
    assert proposal["source_sha256"] == observed["source_sha256"]
    assert evidence["trained_for_accessibility"] is False
    assert evidence["local_admission"]["manifest_sha256"] == MANIFEST
    assert evidence["local_admission"]["blob_sha256"] == BLOB
    assert evidence["local_admission"]["server_cloud_disable_state"] == "NOT_VERIFIED"
    assert "PrivateUser" not in json.dumps(evidence)
    assert engine.apply_proposal('<form><label>Email</label><input id="email" type="email"></form>', proposal)["state"] == "VERIFIED_SCOPED_REPAIR"


def test_redirects_are_not_followed():
    assert model.NoRedirect().redirect_request(None, None, 302, "", {}, "https://external.example") is None


@pytest.mark.parametrize("name", ["fixture:cloud", "qwen:80b-cloud", "Cloud/innocent:1", "https://remote.example/model", "a cloud"])
def test_cloud_style_names_rejected_before_network(name):
    with patch.object(model, "build_opener") as opener:
        with pytest.raises(ValueError):
            model.model_name(name)
        with pytest.raises(model.ModelUnavailable):
            model.propose(analysis(), name)
        opener.assert_not_called()


@pytest.mark.parametrize("where", ["tags", "show", "nested_show"])
@pytest.mark.parametrize("key", ["remote_host", "remote_model"])
def test_innocent_alias_remote_metadata_never_receives_prompt(where, key):
    listed, shown = tags(), details()
    if where == "tags":
        listed["models"][0][key] = "remote-value"
    elif where == "show":
        shown[key] = "remote-value"
    else:
        shown["details"][key] = "remote-value"
    with patch.object(model, "build_opener") as opener:
        mock_sequence(opener, [listed, shown])
        with pytest.raises(model.ModelUnavailable):
            model.propose(analysis(), NAME)
        assert_no_prompt(opener)


@pytest.mark.parametrize("change", ["missing_info", "missing_architecture", "missing_modelfile", "not_gguf", "remote_from", "tag_from", "relative_from", "unc_from", "two_from", "traversal_from"])
def test_missing_or_nonlocal_metadata_refused(change):
    shown = details()
    if change == "missing_info":
        shown.pop("model_info")
    elif change == "missing_architecture":
        shown["model_info"] = {}
    elif change == "missing_modelfile":
        shown.pop("modelfile")
    elif change == "not_gguf":
        shown["details"]["format"] = "remote"
    else:
        shown["modelfile"] = {
            "remote_from": f"FROM https://example.invalid/blobs/sha256-{BLOB}",
            "tag_from": "FROM innocent:latest",
            "relative_from": f"FROM models/blobs/sha256-{BLOB}",
            "unc_from": f"FROM \\\\server\\models\\blobs\\sha256-{BLOB}",
            "two_from": f"FROM /models/blobs/sha256-{BLOB}\nFROM innocent:latest",
            "traversal_from": f"FROM /models/../blobs/sha256-{BLOB}",
        }[change]
    with patch.object(model, "build_opener") as opener:
        mock_sequence(opener, [tags(), shown])
        with pytest.raises(model.ModelUnavailable):
            model.propose(analysis(), NAME)
        assert_no_prompt(opener)


@pytest.mark.parametrize("raw", [b'{', b'['*5000+b'0'+b']'*5000, b'x'*262145, b'{"x":NaN}', b'{"models":[],"models":[]}', b'{"x":1e9999}'],
                         ids=["malformed", "deep", "oversize", "nonfinite", "duplicate", "floatoverflow"])
@pytest.mark.parametrize("where", ["tags", "show"])
def test_malformed_metadata_fails_closed_without_prompt(raw, where):
    with patch.object(model, "build_opener") as opener:
        mock_sequence(opener, [raw] if where == "tags" else [tags(), raw])
        with pytest.raises(model.ModelUnavailable):
            model.propose(analysis(), NAME)
        assert_no_prompt(opener)


@pytest.mark.parametrize("change", ["missing", "duplicate", "digest", "alias", "format"])
def test_exact_installed_inventory_required(change):
    listed = tags()
    if change == "missing":
        listed["models"] = []
    elif change == "duplicate":
        listed["models"] *= 2
    elif change == "digest":
        listed["models"][0]["digest"] = "short"
    elif change == "alias":
        listed["models"][0]["model"] = "other:1"
    else:
        listed["models"][0]["details"]["format"] = ""
    with patch.object(model, "build_opener") as opener:
        mock_sequence(opener, [listed])
        with pytest.raises(model.ModelUnavailable):
            model.propose(analysis(), NAME)
        assert_no_prompt(opener)


@pytest.mark.parametrize("when", ["before", "after"])
def test_manifest_drift_refused(when):
    changed = deepcopy(tags())
    changed["models"][0]["digest"] = "c" * 64
    responses = [tags(), details(), changed] if when == "before" else [tags(), details(), tags(), generated(), changed]
    with patch.object(model, "build_opener") as opener:
        mock_sequence(opener, responses)
        with pytest.raises(model.ModelUnavailable):
            model.propose(analysis(), NAME)
        if when == "before":
            assert_no_prompt(opener)


@pytest.mark.parametrize("bindings", [[{"candidate_id": "invented"}], [{"candidate_id": "x", "code": "evil"}], "all", [1], "duplicate"])
def test_invalid_candidate_selections_refused(bindings):
    if bindings == "duplicate":
        candidate = analysis()["candidates"][0]["id"]
        bindings = [{"candidate_id": candidate}] * 2
    with patch.object(model, "build_opener") as opener:
        mock_sequence(opener, [tags(), details(), tags(), generated(bindings)])
        with pytest.raises(model.ModelUnavailable):
            model.propose(analysis(), NAME)


def test_generation_with_remote_metadata_refused():
    with patch.object(model, "build_opener") as opener:
        mock_sequence(opener, [tags(), details(), tags(), generated(remote_host="https://remote.invalid")])
        with pytest.raises(model.ModelUnavailable):
            model.propose(analysis(), NAME)


def test_timeout_fails_closed_before_prompt():
    with patch.object(model, "build_opener") as opener:
        mock_sequence(opener, [TimeoutError("fixture")])
        with pytest.raises(model.ModelUnavailable):
            model.propose(analysis(), NAME, timeout=0.1)
        assert_no_prompt(opener)


def test_incomplete_http_response_is_unavailable():
    with patch.object(model, "build_opener") as opener:
        mock_sequence(opener, [IncompleteRead(b"fixture")])
        with pytest.raises(model.ModelUnavailable):
            model.propose(analysis(), NAME)
        assert_no_prompt(opener)
