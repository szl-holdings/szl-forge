# SPDX-License-Identifier: Apache-2.0
"""Offline provider races for the five closed card-publication profiles.

No provider package, token, model artifact, or network access is required. The
real asset validation and source-binding builder run on tiny synthetic assets;
the Hub boundary and immutable byte readback are replaced by explicit fakes.
"""
from __future__ import annotations

import builtins
import importlib
import json
import socket
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest


SOURCE_REVISION = "a" * 40
INITIAL_HEAD = "b" * 40
COMMITTED_HEAD = "c" * 40
INTERVENING_HEAD = "d" * 40
CONTROLLED_PATHS = {"README.md", "holo-banner.svg", "szl-source-binding.json"}
PROFILES = [
    ("publish_chaski_card", "chaski"),
    ("publish_chaski_card", "chaski-5050"),
    ("publish_chaski_card", "chaski-r2"),
    ("publish_khipu_card", "khipu"),
    ("publish_khipu_card", "khipu-r2"),
]


class ProviderFailure(RuntimeError):
    def __init__(self, status: int) -> None:
        super().__init__(f"synthetic provider status {status}")
        self.response = SimpleNamespace(
            status_code=status, headers={"Retry-After": "1"}
        )


class ReadbackFailure(RuntimeError):
    pass


class Operation:
    def __init__(self, *, path_in_repo: str, path_or_fileobj: bytes) -> None:
        self.path_in_repo = path_in_repo
        self.path_or_fileobj = path_or_fileobj
        self._is_committed = False


def _synthetic_assets(profile: Any) -> dict[str, bytes]:
    """Exercise the actual validation contracts without source downloads."""
    tags = getattr(profile, "required_card_tags", ())
    metadata = "---\nlicense: apache-2.0\n"
    if tags:
        metadata += "tags:\n" + "".join(f"- {tag}\n" for tag in tags)
    card = metadata + "---\n" + "\n".join(profile.required_card_boundaries) + "\n"
    svg_parts = []
    for boundary in profile.required_svg_boundaries:
        if boundary.startswith("viewBox="):
            viewbox = boundary
        elif boundary.startswith("id="):
            svg_parts.append(f"<g {boundary}/>")
        elif boundary.startswith("stop-color="):
            svg_parts.append(f"<linearGradient><stop {boundary}/></linearGradient>")
        else:
            raise AssertionError(f"unhandled fixture SVG contract: {boundary}")
    banner = (
        f'<svg xmlns="http://www.w3.org/2000/svg" {viewbox}>'
        + "".join(svg_parts)
        + "</svg>\n"
    )
    return {"README.md": card.encode("utf-8"), "holo-banner.svg": banner.encode("utf-8")}


@pytest.fixture(params=PROFILES, ids=[row[1] for row in PROFILES])
def publication(request: Any) -> SimpleNamespace:
    module_name, profile_name = request.param
    module = importlib.import_module(f"tools.{module_name}")
    profile = module.resolve_profile(profile_name)
    assets = _synthetic_assets(profile)
    evidence = module.validate_assets(assets, profile)
    targets = module.build_target_assets(
        profile=profile,
        source_revision=SOURCE_REVISION,
        source_assets=assets,
        source_evidence=evidence,
    )
    assert set(targets) == CONTROLLED_PATHS
    return SimpleNamespace(
        module=module, profile=profile, assets=assets, targets=targets
    )


@pytest.fixture(autouse=True)
def reject_network(monkeypatch: Any) -> None:
    def denied(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("offline concurrency fixture attempted network access")

    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket, "getaddrinfo", denied)


class ProviderHarness:
    """Model an immutable read, a concurrent branch tip, and server-side CAS."""

    def __init__(self, publication: Any) -> None:
        self.publication = publication
        self.initial_head: Any = INITIAL_HEAD
        self.actual_head = INITIAL_HEAD
        self.files = set(publication.targets) | {"model.safetensors"}
        self.observed = dict(publication.targets)
        self.observed["README.md"] += b"old provider card\n"
        self.repo_info_calls: list[dict[str, Any]] = []
        self.repo_info_heads: list[Any] = []
        self.list_calls: list[dict[str, Any]] = []
        self.create_calls: list[dict[str, Any]] = []
        self.readback_calls: list[dict[str, Any]] = []
        self.auth_calls: list[dict[str, Any]] = []
        self.guard_calls: list[str] = []
        self.slept: list[float] = []
        self.failure_statuses: list[int] = []
        self.race_on_initial_read = False
        self.race_after_rate_limit = False
        self.readback_failure: str | None = None
        self.initial_readback_failure = False
        self.client_noop = False
        self.post_commit_flags: str | None = None

    def repo_info(self, **kwargs: Any) -> Any:
        self.repo_info_calls.append(kwargs)
        if self.repo_info_heads:
            head = self.repo_info_heads.pop(0)
        else:
            head = self.initial_head if len(self.repo_info_calls) == 1 else self.actual_head
        return SimpleNamespace(sha=head)

    def list_repo_files(self, **kwargs: Any) -> list[str]:
        self.list_calls.append(kwargs)
        return sorted(self.files)

    def auth_check(self, **kwargs: Any) -> None:
        self.auth_calls.append(kwargs)

    def whoami(self) -> dict[str, str]:
        return {"name": "synthetic-offline-publisher"}

    def create_commit(self, **kwargs: Any) -> Any:
        self.create_calls.append(dict(kwargs))
        assert kwargs["parent_commit"] == INITIAL_HEAD
        assert kwargs["revision"] == "main"
        assert kwargs["create_pr"] is False
        assert kwargs["repo_id"] == self.publication.profile.repo_id
        assert kwargs["repo_type"] == "model"
        operations = kwargs["operations"]
        assert {operation.path_in_repo for operation in operations} == CONTROLLED_PATHS
        assert len(operations) == len(CONTROLLED_PATHS)
        assert {
            operation.path_in_repo: operation.path_or_fileobj
            for operation in operations
        } == self.publication.targets
        if self.failure_statuses:
            status = self.failure_statuses.pop(0)
            if status == 429 and self.race_after_rate_limit:
                self.actual_head = INTERVENING_HEAD
            raise ProviderFailure(status)
        if self.client_noop:
            # A second actor writes identical managed bytes. The pinned SDK may
            # optimize away all additions and return that head without CAS.
            self.actual_head = INTERVENING_HEAD
            return SimpleNamespace(oid=INTERVENING_HEAD)
        if self.actual_head != kwargs["parent_commit"]:
            raise ProviderFailure(412)
        self.actual_head = COMMITTED_HEAD
        for operation in operations:
            operation._is_committed = True
        if self.post_commit_flags == "partial":
            operations[0]._is_committed = False
        elif self.post_commit_flags == "missing":
            del operations[0]._is_committed
        return SimpleNamespace(oid=COMMITTED_HEAD)

    def readback(self, **kwargs: Any) -> dict[str, bytes]:
        self.readback_calls.append(dict(kwargs))
        assert kwargs["repo_id"] == self.publication.profile.repo_id
        assert set(kwargs["paths"]) == CONTROLLED_PATHS
        if kwargs["revision"] == INITIAL_HEAD:
            if self.initial_readback_failure:
                raise ReadbackFailure("synthetic initial read failure")
            if self.race_on_initial_read:
                self.actual_head = INTERVENING_HEAD
            return dict(self.observed)
        assert kwargs["revision"] == COMMITTED_HEAD
        if self.readback_failure == "exception":
            raise ReadbackFailure("synthetic committed read failure")
        observed = dict(self.publication.targets)
        if self.readback_failure == "missing":
            observed.pop("szl-source-binding.json")
        elif self.readback_failure == "mismatch":
            observed["README.md"] += b"corruption\n"
        return observed

    def install(self, monkeypatch: Any) -> None:
        module = self.publication.module
        provider = ModuleType("huggingface_hub")

        def api_factory(*, token: str) -> ProviderHarness:
            assert token == "synthetic-offline-token"
            return self

        provider.HfApi = api_factory
        provider.CommitOperationAdd = Operation
        provider.__version__ = "1.23.0"
        monkeypatch.setitem(sys.modules, "huggingface_hub", provider)
        monkeypatch.setattr(module, "_readback_bytes", self.readback)
        retry = module.publish_with_bounded_retry

        def retry_without_sleep(operation: Any, **kwargs: Any) -> Any:
            return retry(operation, **kwargs, sleeper=self.slept.append)

        monkeypatch.setattr(module, "publish_with_bounded_retry", retry_without_sleep)
        if module.__name__.endswith("publish_khipu_card"):
            monkeypatch.setattr(module, "assert_current_main", self.guard_calls.append)

    def publish(self) -> Any:
        return self.publication.module.publish(
            token="synthetic-offline-token",
            source_revision=SOURCE_REVISION,
            assets=self.publication.assets,
            profile=self.publication.profile,
        )

    def assert_single_pinned_observation(self, *, repo_info_count: int = 1) -> None:
        assert len(self.repo_info_calls) == repo_info_count
        assert all(call["revision"] == "main" for call in self.repo_info_calls)
        assert all(call["repo_type"] == "model" for call in self.repo_info_calls)
        assert all(call["repo_id"] == self.publication.profile.repo_id for call in self.repo_info_calls)
        assert len(self.list_calls) == 1
        assert self.list_calls[0]["revision"] == INITIAL_HEAD
        assert self.list_calls[0]["repo_type"] == "model"
        assert self.list_calls[0]["repo_id"] == self.publication.profile.repo_id


def test_changed_card_commits_only_allowed_paths_with_exact_parent(
    publication: Any, monkeypatch: Any
) -> None:
    provider = ProviderHarness(publication)
    provider.install(monkeypatch)
    revision, evidence, identity, changed = provider.publish()
    assert revision == COMMITTED_HEAD
    assert identity == "synthetic-offline-publisher"
    assert changed is True
    assert evidence == publication.module.evidence_for(publication.targets)
    assert len(provider.create_calls) == 1
    provider.assert_single_pinned_observation()
    assert [call["revision"] for call in provider.readback_calls] == [
        INITIAL_HEAD, COMMITTED_HEAD
    ]
    binding = json.loads(publication.targets["szl-source-binding.json"])
    assert binding["qualification"]["publication_eligible"] is False
    assert all(value is False for value in binding["authority"].values())
    if publication.module.__name__.endswith("publish_khipu_card"):
        assert provider.guard_calls == [SOURCE_REVISION, SOURCE_REVISION]


def test_absent_controlled_file_still_retains_observed_parent(
    publication: Any, monkeypatch: Any
) -> None:
    provider = ProviderHarness(publication)
    provider.files.remove("szl-source-binding.json")
    provider.install(monkeypatch)
    assert provider.publish()[3] is True
    assert len(provider.create_calls) == 1
    provider.assert_single_pinned_observation()
    assert [call["revision"] for call in provider.readback_calls] == [COMMITTED_HEAD]


def test_matching_card_is_noop_without_commit(
    publication: Any, monkeypatch: Any
) -> None:
    provider = ProviderHarness(publication)
    provider.observed = dict(publication.targets)
    provider.install(monkeypatch)
    revision, evidence, _identity, changed = provider.publish()
    assert revision == INITIAL_HEAD
    assert changed is False
    assert evidence == publication.module.evidence_for(publication.targets)
    assert provider.create_calls == []
    provider.assert_single_pinned_observation(repo_info_count=2)
    assert [call["revision"] for call in provider.readback_calls] == [INITIAL_HEAD]
    if publication.module.__name__.endswith("publish_khipu_card"):
        assert provider.guard_calls == [SOURCE_REVISION, SOURCE_REVISION]


@pytest.mark.parametrize("second_head", [INTERVENING_HEAD, None, ""])
def test_matching_card_with_changed_or_missing_rechecked_parent_fails_closed(
    publication: Any, monkeypatch: Any, second_head: Any
) -> None:
    provider = ProviderHarness(publication)
    provider.observed = dict(publication.targets)
    provider.repo_info_heads = [INITIAL_HEAD, second_head]
    provider.install(monkeypatch)
    with pytest.raises(
        publication.module.PublicationError,
        match="Hub parent changed during no-op verification",
    ):
        provider.publish()
    assert provider.create_calls == []
    assert provider.slept == []
    provider.assert_single_pinned_observation(repo_info_count=2)
    assert [call["revision"] for call in provider.readback_calls] == [INITIAL_HEAD]


@pytest.mark.parametrize("version", ["1.22.0", "2.0.0", ""])
def test_unreviewed_client_version_fails_before_provider_initialization(
    publication: Any, monkeypatch: Any, version: str
) -> None:
    provider = ProviderHarness(publication)
    provider.install(monkeypatch)
    monkeypatch.setattr(sys.modules["huggingface_hub"], "__version__", version)
    with pytest.raises(publication.module.PublicationError, match="reviewed.*1.23.0"):
        provider.publish()
    assert provider.auth_calls == []
    assert provider.repo_info_calls == []
    assert provider.create_calls == []


@pytest.mark.parametrize("initial_flag", [None, True])
def test_missing_or_used_operation_capability_fails_before_commit(
    publication: Any, monkeypatch: Any, initial_flag: Any
) -> None:
    provider = ProviderHarness(publication)
    provider.install(monkeypatch)

    def unsupported_operation(**kwargs: Any) -> Any:
        operation = SimpleNamespace(**kwargs)
        if initial_flag is not None:
            operation._is_committed = initial_flag
        return operation

    monkeypatch.setattr(
        sys.modules["huggingface_hub"], "CommitOperationAdd", unsupported_operation
    )
    with pytest.raises(publication.module.PublicationError, match="contract is unavailable"):
        provider.publish()
    assert provider.create_calls == []
    provider.assert_single_pinned_observation()


def test_same_bytes_client_shortcut_cannot_claim_server_cas_success(
    publication: Any, monkeypatch: Any
) -> None:
    provider = ProviderHarness(publication)
    provider.client_noop = True
    provider.install(monkeypatch)
    with pytest.raises(publication.module.PublicationError, match="expected-parent server commit"):
        provider.publish()
    assert provider.actual_head == INTERVENING_HEAD
    assert len(provider.create_calls) == 1
    assert all(
        operation._is_committed is False
        for operation in provider.create_calls[0]["operations"]
    )
    assert provider.slept == []
    provider.assert_single_pinned_observation()
    assert [call["revision"] for call in provider.readback_calls] == [INITIAL_HEAD]


@pytest.mark.parametrize("flag_mode", ["partial", "missing"])
def test_partial_or_missing_commit_origin_confirmation_cannot_report_success(
    publication: Any, monkeypatch: Any, flag_mode: str
) -> None:
    provider = ProviderHarness(publication)
    provider.post_commit_flags = flag_mode
    provider.install(monkeypatch)
    with pytest.raises(publication.module.PublicationError, match="expected-parent server commit"):
        provider.publish()
    assert len(provider.create_calls) == 1
    assert provider.actual_head == COMMITTED_HEAD
    assert [call["revision"] for call in provider.readback_calls] == [INITIAL_HEAD]
    provider.assert_single_pinned_observation()


@pytest.mark.parametrize("parent", [None, "", "main", "b" * 39, "g" * 40])
def test_missing_or_malformed_parent_fails_before_provider_commit(
    publication: Any, monkeypatch: Any, parent: Any
) -> None:
    provider = ProviderHarness(publication)
    provider.initial_head = parent
    provider.install(monkeypatch)
    with pytest.raises(publication.module.PublicationError, match="exact revision"):
        provider.publish()
    assert len(provider.repo_info_calls) == 1
    assert provider.list_calls == []
    assert provider.create_calls == []
    assert provider.readback_calls == []


def test_intervening_client_commit_is_not_overwritten_or_retried(
    publication: Any, monkeypatch: Any
) -> None:
    provider = ProviderHarness(publication)
    provider.race_on_initial_read = True
    provider.install(monkeypatch)
    with pytest.raises(ProviderFailure) as raised:
        provider.publish()
    assert raised.value.response.status_code == 412
    assert provider.actual_head == INTERVENING_HEAD
    assert len(provider.create_calls) == 1
    assert provider.slept == []
    provider.assert_single_pinned_observation()
    assert [call["revision"] for call in provider.readback_calls] == [INITIAL_HEAD]


def test_initial_readback_failure_aborts_before_commit(
    publication: Any, monkeypatch: Any
) -> None:
    provider = ProviderHarness(publication)
    provider.initial_readback_failure = True
    provider.install(monkeypatch)
    with pytest.raises(ReadbackFailure, match="initial read failure"):
        provider.publish()
    assert provider.create_calls == []
    provider.assert_single_pinned_observation()


@pytest.mark.parametrize("failure", ["missing", "mismatch", "exception"])
def test_committed_readback_failure_never_reports_success_or_retries_commit(
    publication: Any, monkeypatch: Any, failure: str
) -> None:
    provider = ProviderHarness(publication)
    provider.readback_failure = failure
    provider.install(monkeypatch)
    if failure == "exception":
        expected = ReadbackFailure
        message = "committed read failure"
    else:
        expected = publication.module.PublicationError
        message = "readback mismatch"
    with pytest.raises(expected, match=message):
        provider.publish()
    assert len(provider.create_calls) == 1
    assert provider.actual_head == COMMITTED_HEAD
    provider.assert_single_pinned_observation()
    assert provider.slept == []


def test_rate_limit_retries_keep_original_parent_without_refresh(
    publication: Any, monkeypatch: Any
) -> None:
    provider = ProviderHarness(publication)
    provider.failure_statuses = [429, 429]
    provider.install(monkeypatch)
    assert provider.publish()[3] is True
    assert len(provider.create_calls) == 3
    assert {call["parent_commit"] for call in provider.create_calls} == {INITIAL_HEAD}
    assert provider.slept == [1, 1]
    provider.assert_single_pinned_observation()
    if publication.module.__name__.endswith("publish_khipu_card"):
        assert provider.guard_calls == [SOURCE_REVISION] * 4


def test_race_during_rate_limit_wait_rejects_original_parent_without_refresh(
    publication: Any, monkeypatch: Any
) -> None:
    provider = ProviderHarness(publication)
    provider.failure_statuses = [429]
    provider.race_after_rate_limit = True
    provider.install(monkeypatch)
    with pytest.raises(ProviderFailure) as raised:
        provider.publish()
    assert raised.value.response.status_code == 412
    assert provider.actual_head == INTERVENING_HEAD
    assert len(provider.create_calls) == 2
    assert {call["parent_commit"] for call in provider.create_calls} == {INITIAL_HEAD}
    assert provider.slept == [1]
    provider.assert_single_pinned_observation()
    assert [call["revision"] for call in provider.readback_calls] == [INITIAL_HEAD]


def test_rate_limit_exhaustion_never_refreshes_or_reports_success(
    publication: Any, monkeypatch: Any
) -> None:
    provider = ProviderHarness(publication)
    provider.failure_statuses = [429, 429, 429]
    provider.install(monkeypatch)
    with pytest.raises(ProviderFailure) as raised:
        provider.publish()
    assert raised.value.response.status_code == 429
    assert len(provider.create_calls) == 3
    assert {call["parent_commit"] for call in provider.create_calls} == {INITIAL_HEAD}
    assert provider.actual_head == INITIAL_HEAD
    assert provider.slept == [1, 1]
    provider.assert_single_pinned_observation()


def _prepare_offline_main(publication: Any, monkeypatch: Any) -> None:
    original_import = builtins.__import__

    def no_provider_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "huggingface_hub" or name.startswith("huggingface_hub."):
            raise AssertionError("offline main imported the provider package")
        return original_import(name, *args, **kwargs)

    def no_git_guard(_revision: str) -> None:
        raise AssertionError("offline main performed a fresh-main query")

    monkeypatch.setattr(builtins, "__import__", no_provider_import)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(
        publication.module, "load_assets", lambda _profile: dict(publication.assets)
    )
    if publication.module.__name__.endswith("publish_khipu_card"):
        monkeypatch.setattr(publication.module, "assert_current_main", no_git_guard)


def test_dry_run_needs_no_token_provider_import_or_fresh_main_network(
    publication: Any, monkeypatch: Any, tmp_path: Path
) -> None:
    _prepare_offline_main(publication, monkeypatch)
    report = tmp_path / "dry-run.json"
    assert publication.module.main([
        "--profile", publication.profile.key,
        "--source-revision", SOURCE_REVISION,
        "--report", str(report),
    ]) == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["state"] == "DRY_RUN_VALIDATED"
    assert payload["profile"] == publication.profile.key
    assert payload["target"]["revision"] is None
    assert payload["target"]["commit_created"] is None
    assert payload["qualification"]["publication_eligible"] is False
    assert payload["secret_values_recorded"] is False


def test_missing_token_fails_before_provider_import_or_fresh_main_network(
    publication: Any, monkeypatch: Any, tmp_path: Path
) -> None:
    _prepare_offline_main(publication, monkeypatch)
    with pytest.raises(SystemExit, match="HF_TOKEN is required"):
        publication.module.main([
            "--profile", publication.profile.key,
            "--source-revision", SOURCE_REVISION,
            "--report", str(tmp_path / "not-published.json"),
            "--publish",
        ])


def test_khipu_stale_main_stops_before_provider_initialization(monkeypatch: Any) -> None:
    from tools import publish_khipu_card as publisher

    def stale(_revision: str) -> None:
        raise publisher.PublicationError("synthetic stale canonical main")

    original_import = builtins.__import__

    def no_provider_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "huggingface_hub" or name.startswith("huggingface_hub."):
            raise AssertionError("stale main reached provider initialization")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(publisher, "assert_current_main", stale)
    monkeypatch.setattr(builtins, "__import__", no_provider_import)
    with pytest.raises(publisher.PublicationError, match="stale canonical main"):
        publisher.publish(token="synthetic-offline-token", source_revision=SOURCE_REVISION,
                          assets={}, profile="khipu")


@pytest.mark.parametrize("guard_failure_at", [2, 3])
def test_khipu_main_change_during_preparation_or_retry_stops_write(
    publication: Any, monkeypatch: Any, guard_failure_at: int
) -> None:
    if not publication.module.__name__.endswith("publish_khipu_card"):
        pytest.skip("Khipu-specific main guard")
    provider = ProviderHarness(publication)
    provider.install(monkeypatch)
    if guard_failure_at == 3:
        provider.failure_statuses = [429]

    def advancing_main(revision: str) -> None:
        provider.guard_calls.append(revision)
        if len(provider.guard_calls) == guard_failure_at:
            raise publication.module.PublicationError("synthetic stale canonical main")

    monkeypatch.setattr(publication.module, "assert_current_main", advancing_main)
    with pytest.raises(publication.module.PublicationError, match="stale canonical main"):
        provider.publish()
    assert len(provider.create_calls) == (0 if guard_failure_at == 2 else 1)
    assert provider.slept == ([] if guard_failure_at == 2 else [1])
    assert provider.actual_head == INITIAL_HEAD
    provider.assert_single_pinned_observation()
