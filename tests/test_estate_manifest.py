"""Offline regressions: inventory observations never confer qualification."""
from __future__ import annotations

import importlib.util
import json
import io
from email.message import Message
from pathlib import Path
import urllib.error
import urllib.response

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("estate_manifest", ROOT / "scripts/generate_estate_manifest.py")
estate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(estate)
SHA = "a" * 40


def item(name="unknown", **fields):
    return {"id": f"SZLHOLDINGS/{name}", "sha": SHA, "private": False,
            "siblings": [], **fields}


def coverage():
    return {kind: {"state": "COMPLETE", "pages": 1, "observed_rows": 0} for kind in estate.KINDS}


@pytest.mark.parametrize("siblings", [None, {}, "", [{"wrong": "model.bin"}], [None]])
def test_missing_or_malformed_file_metadata_is_not_observed_absence(siblings):
    assert estate.classify_model("KHIPU-R2", {}, siblings)[0] == "UNAVAILABLE_FILE_METADATA"
    row = estate.build_row("model", item(siblings=siblings), {})
    assert row["weight_files_state"] == "UNAVAILABLE"
    assert row["metadata_state"] == "UNAVAILABLE_REQUIRED_FIELDS"


def test_observed_empty_listing_is_distinct_from_unavailable():
    assert estate.classify_model("unknown", {}, [])[0] == "NO_RECOGNIZED_WEIGHT_FILES_LISTED"
    row = estate.build_row("model", item(), {})
    assert row["weight_files_state"] == "ABSENT_FROM_LISTING"
    assert row["metadata_state"] == "AVAILABLE"


@pytest.mark.parametrize("filename", ["adapter_model.safetensors", "model.bin", "quant.gguf", "model.BIN"])
def test_weight_filenames_do_not_establish_training(filename):
    row = estate.build_row("model", item(siblings=[{"rfilename": filename}]), {})
    assert row["classification"] == "WEIGHTS_LISTED_UNBOUND_REQUIRES_BINDING"
    assert row["weight_files_state"] == "PRESENT"
    assert row["classification_basis"] == "LISTED_FILENAMES_ONLY"
    assert not row["classification"].startswith("TRAINED")


@pytest.mark.parametrize("name,expected", sorted(estate.MODEL_SOURCES.items()))
def test_existing_explicit_model_labels_are_not_promoted(name, expected):
    row = estate.build_row("model", item(name), {})
    assert row["classification"] == expected[1]
    assert row["classification_basis"] == "REPOSITORY_DECLARATION_NOT_REVERIFIED"
    assert "publication_eligible" not in row


@pytest.mark.parametrize("revision", [None, "", "main", "a" * 7, "z" * 40, 42])
def test_invalid_revision_is_explicit_unavailable(revision):
    row = estate.build_row("model", item(sha=revision), {})
    assert row["hub_revision"] is None
    assert row["revision_state"] == "UNAVAILABLE"
    assert "sha" in row["unavailable_fields"]


def test_omitted_visibility_is_not_public():
    value = item()
    value.pop("private")
    row = estate.build_row("dataset", value, {})
    assert row["private"] is None
    assert "private" in row["unavailable_fields"]


def test_kernel_registry_resolves_double_prefix_and_shared_source():
    sources = estate.load_kernel_sources()
    for name, expected in [("szl-kernels", "szl-holdings/szl-kernels"),
                           ("szl-governed-norm", "szl-holdings/szl-lambda-gate")]:
        for kind in ("kernel", "model"):
            row = estate.build_row(kind, item(name, files=[]), sources)
            assert row["github_source"] == expected
            assert row["binding_evidence"] == "publishing/kernel-hub-ownership.json"
            assert "szl-szl-" not in row["github_source"]


def test_undeclared_kernel_does_not_get_guessed_source():
    row = estate.build_row("kernel", item("governed-inference-meter", files=[]), estate.load_kernel_sources())
    assert row["github_source"] is None
    assert row["binding_state"].startswith("NOT_BOUND")
    assert row["next_inspection_owner"]
    assert row["next_inspection"]


def test_dataset_name_without_source_path_is_unbound():
    row = estate.build_row("dataset", item("unknown-dataset"), {})
    assert row["github_source"] is None
    assert row["binding_state"].startswith("NOT_BOUND")


@pytest.mark.parametrize("name", [
    "doctrine-corpus-v2", "szl-1-doctrine-sft", "free-energy-reasoning-sft-v1",
    "abstention-doctrine-sft-v1", "szl-frontier-sft-mix-v2", "rag-corpus-v1",
    "thesis-corpus-v18", "k-verify-benchmark-v1", "canonical-formulas-v1",
    "model-eval-receipts", "a11oy-verifiable-corpus", "szl-lake",
])
def test_legacy_dataset_generator_declarations_survive(name):
    row = estate.build_row("dataset", item(name), {})
    assert row["binding_state"] == "BOUND_GENERATOR_KNOWN"
    assert row["binding_basis"] == "REPOSITORY_DECLARATION_NOT_REVERIFIED"
    assert row["binding_evidence"] == "scripts/generate_estate_manifest.py"
    assert row["github_source"] is None
    assert row["next_inspection_owner"] == "szl-holdings"


def test_all_rows_count_once_and_model_kernel_twins_are_distinct():
    inventory = {"model": [item("szl-kernels"), item("new-weighted", siblings=[{"rfilename": "a.bin"}])],
                 "kernel": [item("szl-kernels", files=[])],
                 "dataset": [item("model-eval-receipts")],
                 "space": [item("unknown"), item("anatomy")]}
    result = estate.build_manifest(inventory, coverage(), estate.load_kernel_sources())
    assert result["counts"]["total"] == 6
    assert result["counts"]["bound"] == 4
    assert result["counts"]["not_bound"] == 2
    assert result["counts"]["source_path_missing"] == 3
    assert result["counts"]["bound"] + result["counts"]["not_bound"] == result["counts"]["total"]
    assert result["counts"]["by_type"]["kernel"] == 1
    assert all(row["binding_state"] for row in result["rows"])


def test_duplicate_typed_identity_is_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        estate.build_manifest({"model": [item(), item()]}, coverage(), {})


@pytest.mark.parametrize("full_flag", ["&full=true", ""])
def test_collector_requests_full_metadata_and_follows_pages(full_flag):
    calls = []
    def fetch(url, token):
        calls.append(url)
        assert token == "test-placeholder"
        assert "full=true" in url
        if url.startswith("/api/models"):
            if "cursor=" not in url:
                return [item("one")], f'<https://huggingface.co/api/models?author=SZLHOLDINGS{full_flag}&cursor=2>; rel="next"'
            return [item("two")], ""
        return [], ""
    inventory, observed = estate.collect_inventory("test-placeholder", fetch)
    assert [row["id"] for row in inventory["model"]] == ["SZLHOLDINGS/one", "SZLHOLDINGS/two"]
    assert observed["model"]["pages"] == 2
    assert len(calls) == 5
    assert all(state["state"] == "COMPLETE" for state in observed.values())
    assert any(url.startswith("/api/kernels?") for url in calls)


@pytest.mark.parametrize("link", [
    '<https://other.example/api/models?author=SZLHOLDINGS&full=true>; rel="next"',
    '<http://huggingface.co/api/models?author=SZLHOLDINGS&full=true>; rel="next"',
    '<https://huggingface.co/api/spaces?author=SZLHOLDINGS&full=true>; rel="next"',
    '<https://huggingface.co/api/models?author=OTHER&full=true>; rel="next"',
    '<https://huggingface.co/api/models?author=SZLHOLDINGS&full=false>; rel="next"',
    '<https://huggingface.co/api/models?author=SZLHOLDINGS&full=>; rel="next"',
    '<https://huggingface.co/api/models?author=SZLHOLDINGS&full=true&full=false>; rel="next"',
    '<https://huggingface.co/api/models?author=SZLHOLDINGS&author=OTHER>; rel="next"',
])
def test_untrusted_or_weakened_pagination_is_rejected(link):
    with pytest.raises(ValueError):
        estate.next_page(link, "models")


@pytest.mark.parametrize("parameters", ['rel=next', 'rel="NEXT"', 'type="application/json"; rel="next"',
                                       'rel="next"; title="next, page"'])
def test_valid_link_parameter_variations_are_followed(parameters):
    target = "/api/models?author=SZLHOLDINGS&full=true&cursor=2"
    assert estate.next_page(f'<https://huggingface.co{target}>; {parameters}', "models") == target


@pytest.mark.parametrize("header", ["not-a-link", '<https://huggingface.co/api/models>; title="next"',
                                   '<https://huggingface.co/api/models>; rel="next"; broken',
                                   '<https://huggingface.co/api/models>; rel="prev"; rel="next"'])
def test_malformed_link_cannot_silently_end_collection(header):
    with pytest.raises(ValueError):
        estate.next_page(header, "models")
    inventory, states = estate.collect_inventory("", lambda url, token: ([item()], header))
    assert all(state["state"] == "PARTIAL" for state in states.values())
    assert estate.build_manifest(inventory, states, {})["inventory_state"] == "INCOMPLETE"


def test_multiple_links_follow_only_next_and_duplicate_next_is_rejected():
    target = "/api/models?author=SZLHOLDINGS&full=true&cursor=2"
    previous = '<https://huggingface.co/api/models>; rel="prev"'
    following = f'<https://huggingface.co{target}>; rel="next"'
    assert estate.next_page(previous + ", " + following, "models") == target
    with pytest.raises(ValueError, match="multiple"):
        estate.next_page(following + ", " + following, "models")


def test_http_client_uses_redirect_blocker_and_only_hub_api(monkeypatch):
    installed = []
    class Response(io.BytesIO):
        headers = Message()
        headers.add_header("Link", '<https://huggingface.co/api/models>; rel="prev"')
        headers.add_header("Link", '<https://huggingface.co/api/models?author=SZLHOLDINGS&full=true&cursor=2>; rel="next"')
    class Opener:
        def open(self, request, timeout):
            assert request.full_url == "https://huggingface.co/api/models?full=true"
            assert request.headers["Authorization"] == "Bearer test-placeholder"
            return Response(b"[]")
    def build(*handlers):
        installed.extend(handlers)
        return Opener()
    monkeypatch.setattr(estate.urllib.request, "build_opener", build)
    payload, links = estate.hf("/api/models?full=true", "test-placeholder")
    assert payload == []
    assert estate.next_page(links, "models").endswith("&cursor=2")
    assert len(installed) == 1
    assert isinstance(installed[0], estate.NoRedirect)
    assert installed[0].redirect_request(None, None, 302, None, None, "https://other.example") is None
    with pytest.raises(ValueError):
        estate.hf("https://other.example/api/models", "test-placeholder")


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_redirect_transport_never_forwards_credentials(monkeypatch, status):
    calls = []
    class FakeHTTPS(estate.urllib.request.HTTPSHandler):
        def https_open(self, request):
            calls.append(request)
            headers = Message()
            headers.add_header("Location", "https://other.example/api/models")
            response = urllib.response.addinfourl(io.BytesIO(b""), headers, request.full_url, status)
            response.msg = "Redirect"
            return response
    real_build = estate.urllib.request.build_opener
    monkeypatch.setattr(estate.urllib.request, "build_opener", lambda *handlers: real_build(FakeHTTPS(), *handlers))
    with pytest.raises(urllib.error.HTTPError) as caught:
        estate.hf("/api/models?full=true", "test-placeholder")
    assert caught.value.code == status
    assert len(calls) == 1
    assert calls[0].full_url.startswith("https://huggingface.co/")


def test_pagination_loop_is_incomplete_not_success():
    def fetch(url, token):
        return [item()], f'<https://huggingface.co{url}>; rel="next"'
    inventory, observed = estate.collect_inventory("", fetch)
    result = estate.build_manifest(inventory, observed, {})
    assert result["inventory_state"] == "INCOMPLETE"
    assert all(state["state"] == "PARTIAL" for state in observed.values())


@pytest.mark.parametrize("empty", [False, True])
def test_fresh_cursors_cannot_exceed_page_budget(monkeypatch, empty):
    monkeypatch.setattr(estate, "MAX_PAGES", 2)
    calls = []
    def fetch(url, token):
        calls.append(url)
        api = url.split("?", 1)[0]
        rows = [] if empty else [item(f"asset-{len(calls)}")]
        return rows, f'<https://huggingface.co{api}?author=SZLHOLDINGS&cursor={len(calls)}>; rel="next"'
    inventory, states = estate.collect_inventory("", fetch)
    assert len(calls) == 2 * len(estate.KINDS)
    assert all(state["pages"] == 2 for state in states.values())
    assert all(state["state"] == ("UNAVAILABLE" if empty else "PARTIAL") for state in states.values())
    assert estate.build_manifest(inventory, states, {})["inventory_state"] == "INCOMPLETE"


@pytest.mark.parametrize("overflow", [False, True])
def test_item_budget_and_exact_boundary(monkeypatch, overflow):
    monkeypatch.setattr(estate, "MAX_ITEMS", 2)
    def fetch(url, token):
        api = url.split("?", 1)[0]
        if "cursor=" not in url:
            return [item("first")], f'<https://huggingface.co{api}?author=SZLHOLDINGS&cursor=2>; rel="next"'
        rows = [item("second"), item("third")] if overflow else [item("second")]
        return rows, ""
    inventory, states = estate.collect_inventory("", fetch)
    assert all(len(rows) == (1 if overflow else 2) for rows in inventory.values())
    assert all(state["state"] == ("PARTIAL" if overflow else "COMPLETE") for state in states.values())


@pytest.mark.parametrize("overflow", [False, True])
def test_http_response_byte_budget(monkeypatch, overflow):
    monkeypatch.setattr(estate, "MAX_PAGE_BYTES", 2)
    reads = []
    class Response(io.BytesIO):
        headers = Message()
        def read(self, size=-1):
            reads.append(size)
            return super().read(size)
    class Opener:
        def open(self, request, timeout):
            return Response(b"[] " if overflow else b"[]")
    monkeypatch.setattr(estate.urllib.request, "build_opener", lambda *handlers: Opener())
    if overflow:
        with pytest.raises(ValueError, match="byte budget"):
            estate.hf("/api/models", "")
    else:
        assert estate.hf("/api/models", "") == ([], "")
    assert reads == [3]


def test_http_failure_not_empty_estate_and_error_text_not_serialized():
    def fetch(url, token):
        if url.startswith("/api/kernels"):
            raise urllib.error.HTTPError(url, 401, "do not copy exception text", None, None)
        return [], ""
    inventory, observed = estate.collect_inventory("", fetch)
    result = estate.build_manifest(inventory, observed, {})
    assert observed["kernel"]["state"] == "UNAVAILABLE"
    assert observed["kernel"]["http_status"] == 401
    assert result["inventory_state"] == "INCOMPLETE"
    assert "do not copy" not in json.dumps(result)


@pytest.mark.parametrize("page", [{"error": "oops"}, [None], [{"id": "OTHER/asset"}], [item(), item()]])
def test_malformed_collection_cannot_pass(page):
    _, observed = estate.collect_inventory("", lambda url, token: (page, ""))
    assert all(state["state"] != "COMPLETE" for state in observed.values())


def test_absent_collection_status_is_incomplete():
    assert estate.build_manifest({}, {}, {})["inventory_state"] == "INCOMPLETE"


def test_authenticated_scope_and_private_visibility_retained():
    result = estate.build_manifest({"model": [item(private=True)]}, coverage(), {}, authenticated=True)
    assert result["collection_scope"] == "TOKEN_VISIBLE"
    assert result["rows"][0]["private"] is True
    assert result["policy"]["artifact_equivalence"] == "NOT_CLAIMED"
    assert result["policy"]["reproducible_build"] == "NOT_CLAIMED"


def test_existing_output_is_not_overwritten_or_fetched(tmp_path, monkeypatch):
    target = tmp_path / "prior.json"
    target.write_text("prior evidence")
    monkeypatch.setattr(estate, "collect_inventory", lambda token: pytest.fail("unexpected network collection"))
    with pytest.raises(SystemExit):
        estate.main(["--output", str(target)])
    assert target.read_text() == "prior evidence"


@pytest.mark.parametrize("complete,expected", [(True, 0), (False, 2)])
def test_cli_writes_only_explicit_review_output_and_returns_incomplete(tmp_path, monkeypatch, complete, expected):
    states = coverage()
    if not complete:
        states["kernel"]["state"] = "UNAVAILABLE"
    monkeypatch.setattr(estate, "collect_inventory", lambda token: ({"model": [item()]}, states))
    output = tmp_path / "review.json"
    assert estate.main(["--output", str(output), "--token-env", "UNSET_ESTATE_TEST_TOKEN"]) == expected
    result = json.loads(output.read_text())
    assert result["collection_scope"] == "ANONYMOUS_PUBLIC"
    assert result["counts"]["total"] == 1


def test_cli_missing_metadata_is_nonzero(tmp_path, monkeypatch):
    monkeypatch.setattr(estate, "collect_inventory", lambda token: ({"model": [item(sha=None)]}, coverage()))
    assert estate.main(["--output", str(tmp_path / "review.json")]) == 2
