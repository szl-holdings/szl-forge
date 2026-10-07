# SPDX-License-Identifier: Apache-2.0
"""Offline contracts for the OAC Hub publisher and its workflow.

Every Hub interaction here is a synthetic fake; nothing touches the network.
Git reads use either an in-memory fake tree or the real checkout at HEAD.
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from tools import publish_oac_hub as publisher

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "publish-oac-hub.yml"
SOURCE = "a" * 40
PARENT = "b" * 40
NEW = "c" * 40
BASE = "d" * 40
CREATED = "e" * 40
MAIN = "refs/heads/main"
MODEL = publisher.PROFILES["oac-v1-model"]
DATASET = publisher.PROFILES["oac-v1-dataset"]
CREATE_ONLY = dataclasses.replace(MODEL, allow_create=True, replace_paths=frozenset())
CARD = b"---\nlicense: apache-2.0\n---\n\n# Synthetic card\n"
NEW_CARD = b"---\nlicense: apache-2.0\n---\n\n# Synthetic card, revised\n"
PUBLISH_IF = (
    "github.ref == 'refs/heads/main' && "
    "(github.event_name == 'push' || (github.event_name == 'workflow_dispatch' && inputs.publish))"
)
WATCHED_PATHS = [
    "clinical-gateway/huggingface/**",
    "ops-health/huggingface/**",
    "tools/publish_oac_hub.py",
    "tests/test_publish_oac_hub.py",
    "tools/acquire_hf_publisher_token.py",
    "tests/test_acquire_hf_publisher_token.py",
    "tests/test_hf_org_token1_fallback.py",
    ".github/workflows/publish-oac-hub.yml",
]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def steps_text(job: dict) -> str:
    return "\n".join(
        json.dumps(step.get("env", {}), sort_keys=True) + "\n" + str(step.get("run", ""))
        for step in job["steps"]
    )


def real_head() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, timeout=30, check=False,
        )
    except OSError:
        pytest.skip("git is unavailable")
    if result.returncode:
        pytest.skip("not a git checkout")
    return result.stdout.strip()


# --- synthetic Git and Hub --------------------------------------------------


def package(profile: publisher.HubProfile, **overrides: bytes) -> dict[str, bytes]:
    files = {path: f"synthetic {profile.key} {path}\n".encode() for path in profile.files}
    files["README.md"] = CARD
    files.update(overrides)
    return files


def tree(profile: publisher.HubProfile, files: dict[str, bytes]) -> dict[str, bytes]:
    return {f"{profile.staged_dir}/{path}": raw for path, raw in files.items()}


class FakeGit:
    """Serves rev-parse, ls-tree and cat-file from in-memory trees."""

    def __init__(self, trees: dict[str, dict[str, bytes]], *, head: str = SOURCE,
                 modes: dict[str, str] | None = None) -> None:
        self.trees = trees
        self.head = head
        self.modes = modes or {}
        self.blobs: dict[str, bytes] = {}
        self.calls: list[list[str]] = []

    def __call__(self, args: list[str]) -> bytes:
        self.calls.append(list(args))
        if args[:2] == ["rev-parse", "--verify"]:
            return (self.head + "\n").encode()
        if args[0] == "ls-tree":
            assert args[1:4] == ["-r", "-l", "-z"] and args[5] == "--"
            revision, prefix = args[4], args[6]
            out = b""
            for path, raw in sorted(self.trees.get(revision, {}).items()):
                if not path.startswith(prefix):
                    continue
                oid = hashlib.sha1(b"blob %d\0" % len(raw) + raw).hexdigest()
                self.blobs[oid] = raw
                mode = self.modes.get(path, "100644")
                out += f"{mode} blob {oid} {len(raw):>7}\t{path}".encode() + b"\0"
            return out
        if args[:2] == ["cat-file", "blob"]:
            return self.blobs[args[2]]
        raise AssertionError(f"unexpected git call {args}")


class Missing(Exception):
    """Synthetic stand-in for huggingface_hub's RepositoryNotFoundError."""


class Operation:
    def __init__(self, *, path_in_repo: str, path_or_fileobj: bytes) -> None:
        self.path_in_repo = path_in_repo
        self.path_or_fileobj = path_or_fileobj
        self._is_committed = False


class FakeApi:
    """A one-repository Hub with immutable revisions and a parent-commit CAS."""

    def __init__(self, profile: publisher.HubProfile, files: dict[str, bytes] | None, *,
                 resolved_id: str | None = None, private: bool | None = False,
                 create_error: Exception | None = None, commit_lands: bool = True) -> None:
        self.profile = profile
        self.revisions: dict[str, dict[str, bytes]] = {}
        self.head: str | None = None
        if files is not None:
            self.revisions[PARENT] = {".gitattributes": b"*.bin filter=lfs\n", **files}
            self.head = PARENT
        self.resolved_id = resolved_id or profile.repo_id
        self.private = private
        self.create_error = create_error
        self.commit_lands = commit_lands
        self.calls: list[tuple] = []
        self.commit_kwargs: dict | None = None
        self.after_commit = None

    def repo_info(self, repo_id, *, repo_type, revision):
        self.calls.append(("repo_info", repo_id, repo_type, revision))
        assert (repo_id, repo_type, revision) == (self.profile.repo_id, self.profile.repo_type, "main")
        if self.head is None:
            raise Missing(repo_id)
        return SimpleNamespace(id=self.resolved_id, private=self.private, sha=self.head)

    def list_repo_files(self, repo_id, *, repo_type, revision):
        self.calls.append(("list_repo_files", revision))
        return sorted(self.revisions[revision])

    def read_file(self, profile, path, revision):
        self.calls.append(("read", path, revision))
        return self.revisions[revision][path]

    def create_repo(self, repo_id, *, repo_type, private, exist_ok):
        self.calls.append(("create_repo", repo_id, repo_type, private, exist_ok))
        if self.create_error is not None:
            raise self.create_error
        self.revisions[CREATED] = {".gitattributes": b"*.bin filter=lfs\n"}
        self.head = CREATED

    def create_commit(self, **kwargs):
        self.calls.append(("create_commit",))
        self.commit_kwargs = kwargs
        if kwargs["parent_commit"] != self.head:
            raise RuntimeError("synthetic 412: parent moved")
        files = dict(self.revisions[self.head])
        for operation in kwargs["operations"]:
            files[operation.path_in_repo] = operation.path_or_fileobj
            operation._is_committed = self.commit_lands
        self.revisions[NEW] = files
        self.head = NEW
        if self.after_commit is not None:
            self.after_commit(self)
        return SimpleNamespace(oid=NEW)

    def names(self) -> list[str]:
        return [call[0] for call in self.calls]


def run(profile, api, *, publish, staged=None, fresh=None, git=None):
    staged = package(profile) if staged is None else staged
    git = git or FakeGit({SOURCE: tree(profile, staged)})
    fresh_calls: list[str] = []
    report = publisher.base_report(profile, SOURCE, publish)
    publisher.execute(
        profile, SOURCE, report,
        api=api, read_file=api.read_file, publish=publish, not_found=Missing,
        operation_factory=Operation, fresh_main=fresh or fresh_calls.append, git=git,
    )
    return report, fresh_calls


def refused(profile, api, match, **kwargs):
    with pytest.raises(publisher.PublicationRefused, match=match):
        run(profile, api, **kwargs)
    assert "create_commit" not in api.names()


# --- registry and workflow binding -------------------------------------------


def test_registry_is_closed_and_matches_the_alignment_verifier() -> None:
    alignment = load_module(
        "oac_verify_hub_alignment", ROOT / "clinical-gateway" / "tools" / "verify_hub_alignment.py"
    )
    assert set(publisher.PROFILES) == {"oac-v1-model", "oac-v1-dataset"}
    assert (MODEL.repo_id, MODEL.repo_type) == (alignment.MODEL_ID, "model")
    assert (DATASET.repo_id, DATASET.repo_type) == (alignment.DATASET_ID, "dataset")
    assert MODEL.files == alignment.MODEL_FILES
    assert DATASET.files == alignment.DATASET_FILES
    assert MODEL.staged_dir == "clinical-gateway/huggingface/model/oac-system-health-v1"
    assert DATASET.staged_dir == (
        "clinical-gateway/huggingface/dataset/oac-clinical-transport-observability-synthetic"
    )
    assert MODEL.replace_paths == {"README.md"}
    assert DATASET.replace_paths == frozenset()
    assert not MODEL.allow_create and not DATASET.allow_create
    assert MODEL.lock_group == "hf-write/model/SZLHOLDINGS/oac-system-health-v1"
    assert DATASET.lock_group == (
        "hf-write/dataset/SZLHOLDINGS/oac-clinical-transport-observability-synthetic"
    )
    with pytest.raises(publisher.PublicationRefused, match="unknown OAC Hub profile"):
        publisher.resolve_profile("operator-chosen-target")


@pytest.mark.parametrize(
    "change",
    [
        {"replace_paths": frozenset({"extra.bin"})},
        {"repo_id": "someone-else/oac-system-health-v1"},
        {"repo_type": "space"},
        {"files": MODEL.files | {".gitattributes"}},
        {"key": "renamed"},
        {"allow_create": True},
    ],
)
def test_registry_check_refuses_inconsistent_profiles(change: dict) -> None:
    broken = dataclasses.replace(MODEL, **change)
    with pytest.raises(publisher.PublicationRefused, match="registry is inconsistent"):
        publisher._check_registry({MODEL.key: broken})


def test_workflow_binds_one_locked_publish_job_per_profile() -> None:
    jobs = workflow()["jobs"]
    publish_jobs = {name for name in jobs if name.startswith("publish-")}
    assert publish_jobs == {profile.job_id for profile in publisher.PROFILES.values()}
    for profile in publisher.PROFILES.values():
        job = jobs[profile.job_id]
        assert job["concurrency"] == {"group": profile.lock_group, "cancel-in-progress": False}
        assert " ".join(job["if"].split()) == PUBLISH_IF
        text = steps_text(job)
        assert f"--profile {profile.key}" in text
        assert f"--target-repo {profile.repo_id}" in text
        assert f"--target-type {profile.repo_type}" in text
        assert f"--report reports/{profile.key}-publication.json" in text
        assert f'"PROFILE": "{profile.key}"' in text
        assert f'"TARGET_REPO": "{profile.repo_id}"' in text
        assert f'"TARGET_TYPE": "{profile.repo_type}"' in text
        assert "--publish" in text
        assert ("--allow-create" in text) is profile.allow_create
        assert "--oidc-resource" not in text
        assert "This publication run no longer owns current protected main." in text
        checkout = job["steps"][1]
        assert checkout["with"]["ref"] == "${{ github.sha }}"
        assert checkout["with"]["persist-credentials"] is False


def test_workflow_permissions_triggers_and_main_only_dispatch() -> None:
    data = workflow()
    triggers = data.get("on", data.get(True))
    assert data["permissions"] == {"contents": "read"}
    assert all("permissions" not in job for job in data["jobs"].values())
    assert "id-token" not in WORKFLOW.read_text(encoding="utf-8")
    assert triggers["pull_request"] == {"branches": ["main"], "paths": WATCHED_PATHS}
    assert triggers["push"] == {"branches": ["main"], "paths": WATCHED_PATHS}
    dispatch = triggers["workflow_dispatch"]["inputs"]
    assert set(dispatch) == {"publish"}
    assert dispatch["publish"]["type"] == "boolean"
    assert dispatch["publish"]["required"] is True
    assert dispatch["publish"]["default"] is False


def test_only_publish_jobs_see_credentials_or_publish() -> None:
    jobs = workflow()["jobs"]
    for name, job in jobs.items():
        dumped = yaml.safe_dump(job)
        if not name.startswith("publish-"):
            assert "secrets." not in dumped
            assert "--publish" not in steps_text(job)
    verify = jobs["verify"]
    assert verify["if"] == (
        "${{ github.event_name == 'pull_request' || "
        "(github.event_name == 'workflow_dispatch' && !inputs.publish) }}"
    )
    text = steps_text(verify)
    for profile in publisher.PROFILES.values():
        assert profile.key in text
    assert '--expect-delta-since "${BASE_REVISION}"' in text
    alignment = jobs["alignment"]
    assert set(alignment["needs"]) == {p.job_id for p in publisher.PROFILES.values()}
    assert " ".join(alignment["if"].split()) == PUBLISH_IF
    assert "clinical-gateway/tools/verify_hub_alignment.py" in steps_text(alignment)


def test_every_publish_job_holds_its_canonical_lock() -> None:
    # The LOCKED_WRITERS registration (HF upgrade plan D3) follows with a
    # Foundation-coordinated change, because tests/test_hf_write_locks.py is in
    # foundation-runtime.yml's push path filter. Until then the lock is asserted
    # here, and any registration that does exist must agree with it.
    jobs = workflow()["jobs"]
    locks = load_module("oac_hf_write_locks", ROOT / "tests" / "test_hf_write_locks.py")
    for profile in publisher.PROFILES.values():
        concurrency = jobs[profile.job_id]["concurrency"]
        assert concurrency["group"] == profile.lock_group
        assert concurrency["cancel-in-progress"] is False
        registered = locks.LOCKED_WRITERS.get(("publish-oac-hub.yml", profile.job_id))
        assert registered in (None, profile.lock_group)


# --- Git source ---------------------------------------------------------------


def test_staged_packages_at_head_are_the_closed_file_sets() -> None:
    head = real_head()
    for profile in publisher.PROFILES.values():
        files = publisher.staged_package(profile, head)
        assert set(files) == set(profile.files)
        assert all(files.values())
        assert b"\r" not in files["README.md"]


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda files: files.pop("model.json"), "file set drifted"),
        (lambda files: files.update({"extra.py": b"x\n"}), "file set drifted"),
        (lambda files: files.update({"README.md": CARD.replace(b"\n", b"\r\n")}), "LF line endings"),
        (lambda files: files.update({"README.md": b"# no front matter\n"}), "front matter is missing"),
        (lambda files: files.update({"README.md": b"---\nlicense: x\n"}), "not closed"),
        (lambda files: files.update({"README.md": b"---\n- a\n---\n"}), "must be a mapping"),
        (lambda files: files.update({"README.md": b"---\n\xff\n---\n"}), "UTF-8"),
        (lambda files: files.update({"model.json": b"x" * (publisher.MAX_FILE_BYTES + 1)}), "bounded"),
    ],
)
def test_staged_package_refuses_drift_and_unsafe_cards(mutate, match) -> None:
    files = package(MODEL)
    mutate(files)
    git = FakeGit({SOURCE: tree(MODEL, files)})
    with pytest.raises(publisher.PublicationRefused, match=match):
        publisher.staged_package(MODEL, SOURCE, git=git)


@pytest.mark.parametrize("mode", ["100755", "120000"])
def test_staged_package_refuses_executable_or_symlink_entries(mode: str) -> None:
    files = package(MODEL)
    path = f"{MODEL.staged_dir}/model.json"
    git = FakeGit({SOURCE: tree(MODEL, files)}, modes={path: mode})
    with pytest.raises(publisher.PublicationRefused, match="regular non-executable blob"):
        publisher.staged_package(MODEL, SOURCE, git=git)


def test_staged_change_since_names_exactly_the_changed_paths() -> None:
    old, new = package(MODEL), package(MODEL, **{"README.md": NEW_CARD})
    git = FakeGit({BASE: tree(MODEL, old), SOURCE: tree(MODEL, new)})
    assert publisher.staged_change_since(MODEL, BASE, SOURCE, git=git) == ["README.md"]
    git = FakeGit({BASE: tree(MODEL, new), SOURCE: tree(MODEL, new)})
    assert publisher.staged_change_since(MODEL, BASE, SOURCE, git=git) == []
    git = FakeGit({BASE: {}, SOURCE: tree(MODEL, new)})
    assert publisher.staged_change_since(MODEL, BASE, SOURCE, git=git) == sorted(MODEL.files)


# --- dry-run expectations against the pull request base ------------------------

HAND_EDIT = b"---\nlicense: apache-2.0\n---\n\n# Synthetic card, edited on the Hub\n"
WIDE = dataclasses.replace(MODEL, replace_paths=MODEL.files)


def dry_run_since(profile, *, base: dict, head: dict, hub: dict):
    git = FakeGit({BASE: tree(profile, base), SOURCE: tree(profile, head)})
    api = FakeApi(profile, hub)
    report, _ = run(profile, api, publish=False, staged=head, git=git)
    return report, git, api


def test_expected_delta_holds_when_the_hub_sits_at_the_base() -> None:
    base, head = package(MODEL), package(MODEL, **{"README.md": NEW_CARD})
    report, git, api = dry_run_since(MODEL, base=base, head=head, hub=base)
    publisher.assert_expected_delta(MODEL, BASE, SOURCE, report, git=git)
    assert report["expected_delta_since"] == {
        "base_revision": BASE,
        "parent_revision": PARENT,
        "hub_equals_base": True,
        "hub_drift": [],
        "delta": ["README.md"],
    }
    assert "create_commit" not in api.names()


def test_expected_delta_refuses_a_hub_that_drifted_from_the_base() -> None:
    # The path lists agree, so only the byte comparison can see the drift.
    base, head = package(MODEL), package(MODEL, **{"README.md": NEW_CARD})
    hub = package(MODEL, **{"README.md": HAND_EDIT})
    report, git, _ = dry_run_since(MODEL, base=base, head=head, hub=hub)
    assert report["delta"] == publisher.staged_change_since(MODEL, BASE, SOURCE, git=git)
    with pytest.raises(publisher.PublicationRefused, match="base revision's staged bytes: README.md"):
        publisher.assert_expected_delta(MODEL, BASE, SOURCE, report, git=git)
    assert report["expected_delta_since"]["hub_equals_base"] is False
    assert report["expected_delta_since"]["hub_drift"] == ["README.md"]


@pytest.mark.parametrize(
    ("base_drop", "hub_drop", "drift"),
    [
        ("example_input.json", None, "example_input.json"),  # on the Hub, not in the base
        (None, "LICENSE", "LICENSE"),  # in the base, not on the Hub
    ],
)
def test_expected_delta_refuses_a_path_present_on_one_side_only(base_drop, hub_drop, drift) -> None:
    base, head, hub = package(WIDE), package(WIDE), package(WIDE)
    base.pop(base_drop, None)
    hub.pop(hub_drop, None)
    report, git, _ = dry_run_since(WIDE, base=base, head=head, hub=hub)
    with pytest.raises(publisher.PublicationRefused, match=f"staged bytes: {drift}$"):
        publisher.assert_expected_delta(WIDE, BASE, SOURCE, report, git=git)
    assert report["expected_delta_since"]["hub_drift"] == [drift]


def test_expected_parent_pins_the_inspected_hub_head() -> None:
    report = {"parent_revision": PARENT}
    publisher.assert_expected_parent(PARENT, report)
    assert report["expected_parent"] == {"revision": PARENT, "observed": PARENT, "matches": True}
    with pytest.raises(publisher.PublicationRefused, match="expected parent"):
        publisher.assert_expected_parent(NEW, report)
    assert report["expected_parent"]["matches"] is False
    with pytest.raises(publisher.PublicationRefused, match="full lowercase"):
        publisher.assert_expected_parent("main", {"parent_revision": PARENT})


# --- lookup, create-or-adopt, delta ------------------------------------------


def test_dry_run_reports_the_card_delta_anonymously_and_writes_nothing() -> None:
    staged = package(MODEL, **{"README.md": NEW_CARD})
    api = FakeApi(MODEL, package(MODEL))
    report, fresh = run(MODEL, api, publish=False, staged=staged)
    assert report["state"] == "DELTA"
    assert report["delta"] == ["README.md"]
    assert report["parent_revision"] == PARENT
    assert report["artifacts_unchanged"] is True
    assert report["resolved_repo_id"] == MODEL.repo_id
    assert report["hub_files"] == sorted({".gitattributes", *MODEL.files})
    assert report["mode"] == "dry_run" and report["lookup"] == "anonymous"
    assert fresh == []
    assert "create_repo" not in api.names() and "create_commit" not in api.names()


@pytest.mark.parametrize("publish", [False, True])
def test_identical_hub_is_no_change_without_a_commit(publish: bool) -> None:
    api = FakeApi(DATASET, package(DATASET))
    report, fresh = run(DATASET, api, publish=publish)
    assert report["state"] == "NO_CHANGE"
    assert report["delta"] == []
    assert report["parent_revision"] == PARENT
    assert report["head_after"] == PARENT
    assert api.names().count("repo_info") == 2
    assert fresh == ([SOURCE] if publish else [])
    assert "create_commit" not in api.names()


@pytest.mark.parametrize("publish", [False, True])
def test_no_change_refuses_a_hub_head_that_moves_during_snapshot(publish: bool) -> None:
    api = FakeApi(DATASET, package(DATASET))
    original = api.read_file
    moved = False

    def concurrent_writer(profile, path, revision):
        nonlocal moved
        raw = original(profile, path, revision)
        if not moved:
            api.head = "f" * 40
            moved = True
        return raw

    api.read_file = concurrent_writer
    refused(DATASET, api, "Hub head moved before NO_CHANGE confirmation", publish=publish)
    assert api.names().count("repo_info") == 2


@pytest.mark.parametrize("publish", [False, True])
def test_absent_target_without_create_permission_is_refused(publish: bool) -> None:
    api = FakeApi(MODEL, None)
    refused(MODEL, api, "may not create", publish=publish)
    assert "create_repo" not in api.names()


def test_dry_run_reports_create_for_a_create_profile_without_creating() -> None:
    api = FakeApi(CREATE_ONLY, None)
    report, _ = run(CREATE_ONLY, api, publish=False)
    assert report["state"] == "CREATE"
    assert report["delta"] == sorted(MODEL.files)
    assert "create_repo" not in api.names()


def test_publish_creates_only_in_the_not_found_branch_then_commits() -> None:
    api = FakeApi(CREATE_ONLY, None)
    report, _ = run(CREATE_ONLY, api, publish=True)
    assert ("create_repo", MODEL.repo_id, "model", False, False) in api.calls
    assert report["repository_created"] is True
    assert report["parent_revision"] == CREATED
    assert api.commit_kwargs["parent_commit"] == CREATED
    assert report["delta"] == sorted(MODEL.files)
    assert report["artifacts_unchanged"] is False
    assert report["state"] == "PUBLISHED"


def test_concurrent_create_is_refused_by_exist_ok_false() -> None:
    api = FakeApi(CREATE_ONLY, None, create_error=RuntimeError("synthetic 409: repository exists"))
    with pytest.raises(RuntimeError, match="409"):
        run(CREATE_ONLY, api, publish=True)
    assert api.names().count("create_repo") == 1
    assert "create_commit" not in api.names()


def test_an_existing_empty_repository_is_adopted_not_recreated() -> None:
    api = FakeApi(CREATE_ONLY, {})
    report, _ = run(CREATE_ONLY, api, publish=True)
    assert "create_repo" not in api.names()
    assert report["repository_created"] is False
    assert report["delta"] == sorted(CREATE_ONLY.files)
    assert report["state"] == "PUBLISHED"


@pytest.mark.parametrize("publish", [False, True])
def test_an_existing_exact_create_target_is_no_change(publish: bool) -> None:
    api = FakeApi(CREATE_ONLY, package(CREATE_ONLY))
    report, _ = run(CREATE_ONLY, api, publish=publish)
    assert report["state"] == "NO_CHANGE" and report["delta"] == []
    assert report["head_after"] == PARENT
    assert "create_repo" not in api.names() and "create_commit" not in api.names()


@pytest.mark.parametrize("publish", [False, True])
def test_create_profile_refuses_a_partially_populated_target(publish: bool) -> None:
    api = FakeApi(CREATE_ONLY, {"README.md": CARD})
    refused(CREATE_ONLY, api, "partial staged package", publish=publish)
    assert "create_repo" not in api.names()


@pytest.mark.parametrize("publish", [False, True])
def test_create_profile_refuses_changed_existing_bytes(publish: bool) -> None:
    api = FakeApi(CREATE_ONLY, package(CREATE_ONLY, **{"model.json": b"synthetic drift\n"}))
    refused(CREATE_ONLY, api, "differing staged bytes: model.json", publish=publish)
    assert "create_repo" not in api.names()


def test_create_profile_never_overwrites_even_if_replace_paths_is_broadened() -> None:
    broad = dataclasses.replace(CREATE_ONLY, replace_paths=CREATE_ONLY.files)
    api = FakeApi(broad, package(broad, **{"README.md": NEW_CARD}))
    refused(broad, api, "differing staged bytes: README.md", publish=True)
    assert "create_repo" not in api.names()


def test_an_extra_hub_file_is_refused_because_nothing_is_deleted() -> None:
    api = FakeApi(MODEL, {**package(MODEL), "notes.txt": b"synthetic\n"})
    refused(MODEL, api, "outside the staged package", publish=True)


def test_a_differing_file_outside_replace_paths_is_refused() -> None:
    api = FakeApi(MODEL, package(MODEL, **{"model.json": b"synthetic drift\n"}))
    refused(MODEL, api, "outside replace_paths: model.json", publish=True)


def test_a_staged_file_missing_on_the_hub_counts_as_delta() -> None:
    hub = package(DATASET)
    hub.pop("data/test.jsonl")
    api = FakeApi(DATASET, hub)
    refused(DATASET, api, "outside replace_paths: data/test.jsonl", publish=False)


@pytest.mark.parametrize("private", [True, None])
def test_a_private_or_unknown_visibility_target_is_refused(private) -> None:
    api = FakeApi(MODEL, package(MODEL), private=private)
    refused(MODEL, api, "private", publish=True)


def test_a_renamed_or_redirected_target_is_refused() -> None:
    api = FakeApi(MODEL, package(MODEL), resolved_id="SZLHOLDINGS/oac-ops-health-v1-renamed")
    refused(MODEL, api, "renamed or redirected", publish=True)


def test_the_resolved_id_comparison_is_case_insensitive() -> None:
    api = FakeApi(MODEL, package(MODEL), resolved_id=MODEL.repo_id.lower())
    report, _ = run(MODEL, api, publish=False)
    assert report["state"] == "NO_CHANGE"


@pytest.mark.parametrize("bad", ["../x", "a\\b", "", "a//b", "x\ny"])
def test_unsafe_hub_paths_are_refused(bad: str) -> None:
    api = FakeApi(MODEL, package(MODEL))
    api.revisions[PARENT][bad] = b"x"
    refused(MODEL, api, "unsafe Hub file path", publish=False)


# --- the write -----------------------------------------------------------------


def test_publish_writes_one_parent_pinned_commit_and_reads_every_file_back() -> None:
    staged = package(MODEL, **{"README.md": NEW_CARD})
    api = FakeApi(MODEL, package(MODEL))
    report, fresh = run(MODEL, api, publish=True, staged=staged)
    kwargs = api.commit_kwargs
    assert kwargs["parent_commit"] == PARENT
    assert (kwargs["repo_id"], kwargs["repo_type"]) == (MODEL.repo_id, "model")
    assert kwargs["revision"] == "main" and kwargs["create_pr"] is False
    assert [op.path_in_repo for op in kwargs["operations"]] == ["README.md"]
    assert kwargs["commit_message"] == f"Publish README.md from szl-forge {SOURCE[:12]}"
    assert f"https://github.com/szl-holdings/szl-forge/commit/{SOURCE}" in kwargs["commit_description"]
    assert "Artifacts unchanged" in kwargs["commit_description"]
    assert report["state"] == "PUBLISHED"
    assert report["new_revision"] == NEW and report["head_after"] == NEW
    assert set(report["readback"]) == set(MODEL.files)
    assert all(item["matches"] for item in report["readback"].values())
    assert report["readback"]["README.md"]["sha256"] == hashlib.sha256(NEW_CARD).hexdigest()
    assert fresh == [SOURCE, SOURCE]
    assert api.calls.index(("create_commit",)) > max(
        i for i, call in enumerate(api.calls) if call[0] == "read" and call[2] == PARENT
    )


def test_a_lost_fresh_main_race_touches_no_hub_api() -> None:
    api = FakeApi(MODEL, package(MODEL))

    def stale(_revision: str) -> None:
        raise publisher.PublicationRefused("publication source no longer owns current main")

    with pytest.raises(publisher.PublicationRefused, match="no longer owns current main"):
        run(MODEL, api, publish=True, fresh=stale)
    assert api.calls == []


def test_main_moving_before_the_write_prevents_the_commit() -> None:
    api = FakeApi(MODEL, package(MODEL))
    calls: list[str] = []

    def second_call_fails(revision: str) -> None:
        calls.append(revision)
        if len(calls) == 2:
            raise publisher.PublicationRefused("publication source no longer owns current main")

    refused(MODEL, api, "no longer owns", publish=True,
            staged=package(MODEL, **{"README.md": NEW_CARD}), fresh=second_call_fails)


def test_a_moved_hub_parent_fails_the_compare_and_swap() -> None:
    api = FakeApi(MODEL, package(MODEL))
    original = api.read_file

    def concurrent_writer(profile, path, revision):
        api.head = "f" * 40
        return original(profile, path, revision)

    api.read_file = concurrent_writer
    with pytest.raises(RuntimeError, match="412"):
        run(MODEL, api, publish=True, staged=package(MODEL, **{"README.md": NEW_CARD}))
    assert NEW not in api.revisions


def test_a_client_without_the_commit_origin_contract_is_refused_before_writing() -> None:
    api = FakeApi(MODEL, package(MODEL))

    def operation(**kwargs):
        return SimpleNamespace(**kwargs)

    report = publisher.base_report(MODEL, SOURCE, True)
    staged = package(MODEL, **{"README.md": NEW_CARD})
    with pytest.raises(publisher.PublicationRefused, match="commit-origin contract"):
        publisher.execute(
            MODEL, SOURCE, report, api=api, read_file=api.read_file, publish=True,
            not_found=Missing, operation_factory=operation, fresh_main=lambda _: None,
            git=FakeGit({SOURCE: tree(MODEL, staged)}),
        )
    assert "create_commit" not in api.names()


def test_a_commit_that_never_reached_the_server_is_not_success() -> None:
    api = FakeApi(MODEL, package(MODEL), commit_lands=False)
    with pytest.raises(publisher.PublicationRefused, match="expected-parent server commit"):
        run(MODEL, api, publish=True, staged=package(MODEL, **{"README.md": NEW_CARD}))


def test_a_readback_mismatch_fails_the_publication() -> None:
    api = FakeApi(MODEL, package(MODEL))
    api.after_commit = lambda hub: hub.revisions[NEW].update({"README.md": b"corrupted\n"})
    with pytest.raises(publisher.PublicationRefused, match="readback mismatch: README.md"):
        run(MODEL, api, publish=True, staged=package(MODEL, **{"README.md": NEW_CARD}))


def test_a_head_that_moves_after_publication_fails_the_publication() -> None:
    api = FakeApi(MODEL, package(MODEL))

    def move(hub):
        hub.revisions["f" * 40] = dict(hub.revisions[NEW])
        hub.head = "f" * 40

    api.after_commit = move
    with pytest.raises(publisher.PublicationRefused, match="head moved after publication"):
        run(MODEL, api, publish=True, staged=package(MODEL, **{"README.md": NEW_CARD}))


def test_commit_text_names_changed_artifacts_when_more_than_the_card_changes() -> None:
    message, description = publisher.commit_text(MODEL, SOURCE, PARENT, ["README.md", "model.json"])
    assert message == f"Publish README.md, model.json from szl-forge {SOURCE[:12]}"
    assert "Changed paths: README.md, model.json" in description
    assert "Artifacts unchanged" not in description


# --- fresh main -----------------------------------------------------------------


def install_git(monkeypatch, *, local=SOURCE, remote=None):
    calls = []
    remote = f"{SOURCE}\t{MAIN}\n" if remote is None else remote

    def fake_run(command, **kwargs):
        calls.append(command)
        assert kwargs["cwd"] == publisher.ROOT and kwargs["check"] is False
        return SimpleNamespace(returncode=0, stdout=local + "\n" if len(calls) == 1 else remote,
                               stderr="")

    monkeypatch.setattr(publisher.subprocess, "run", fake_run)
    return calls


def test_fresh_main_queries_the_canonical_public_source(monkeypatch) -> None:
    calls = install_git(monkeypatch)
    publisher.assert_current_main(SOURCE)
    assert calls == [
        ["git", "rev-parse", "--verify", "HEAD"],
        ["git", "ls-remote", "--exit-code", "https://github.com/szl-holdings/szl-forge.git", MAIN],
    ]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"local": PARENT}, "checkout does not match"),
        ({"remote": f"{PARENT}\t{MAIN}\n"}, "no longer owns current main"),
        ({"remote": f"{SOURCE}\t{MAIN}\n{PARENT}\t{MAIN}\n"}, "one exact revision"),
        ({"remote": f"{SOURCE}\trefs/heads/dev\n"}, "one exact revision"),
    ],
)
def test_fresh_main_fails_closed(monkeypatch, kwargs, match) -> None:
    install_git(monkeypatch, **kwargs)
    with pytest.raises(publisher.PublicationRefused, match=match):
        publisher.assert_current_main(SOURCE)


def test_fresh_main_rejects_a_symbolic_revision_without_querying_git(monkeypatch) -> None:
    calls = install_git(monkeypatch)
    with pytest.raises(publisher.PublicationRefused, match="exact source revision"):
        publisher.assert_current_main("main")
    assert calls == []


# --- command line ----------------------------------------------------------------

TOKEN = "hf_" + "Q" * 34
CI = {
    "GITHUB_ACTIONS": "true",
    "GITHUB_REPOSITORY": "szl-holdings/szl-forge",
    "GITHUB_REF": MAIN,
    "HF_TOKEN": TOKEN,
}


def fake_clients(monkeypatch, api, *, version="1.23.0"):
    seen: dict = {}

    def clients(token, cache_dir):
        seen["token"] = token
        return {
            "version": version,
            "api": api,
            "read_file": api.read_file,
            "not_found": Missing,
            "operation_factory": Operation,
        }

    monkeypatch.setattr(publisher, "hub_clients", clients)
    return seen


def test_cli_dry_run_is_anonymous_and_records_the_real_staged_package(monkeypatch, tmp_path) -> None:
    head = real_head()
    api = FakeApi(MODEL, publisher.staged_package(MODEL, head))
    seen = fake_clients(monkeypatch, api)
    output = tmp_path / "dry-run.json"
    code = publisher.main(["--profile", "oac-v1-model", "--source-revision", head,
                           "--report", str(output)], environ={"HF_TOKEN": TOKEN})
    report = json.loads(output.read_text(encoding="utf-8"))
    assert code == 0 and seen["token"] is False
    assert report["state"] == "NO_CHANGE" and report["lookup"] == "anonymous"
    assert set(report["staged"]) == set(MODEL.files)
    assert TOKEN not in output.read_text(encoding="utf-8")


def test_cli_expected_delta_refuses_a_hub_that_is_not_at_the_base(monkeypatch, tmp_path) -> None:
    head = real_head()
    hub = publisher.staged_package(MODEL, head)
    hub["README.md"] = CARD
    fake_clients(monkeypatch, FakeApi(MODEL, hub))
    output = tmp_path / "dry-run.json"
    code = publisher.main(["--profile", "oac-v1-model", "--source-revision", head,
                           "--expect-delta-since", head, "--report", str(output)], environ={})
    report = json.loads(output.read_text(encoding="utf-8"))
    assert code == 1 and report["state"] == "REFUSED"
    assert report["delta"] == ["README.md"]
    assert report["expected_delta_since"] == {
        "base_revision": head,
        "parent_revision": PARENT,
        "hub_equals_base": False,
        "hub_drift": ["README.md"],
        "delta": [],
    }


def test_cli_dry_run_expectations_pin_the_base_bytes_and_the_parent(monkeypatch, tmp_path) -> None:
    base, head = package(MODEL), package(MODEL, **{"README.md": NEW_CARD})
    monkeypatch.setattr(publisher, "_git", FakeGit({BASE: tree(MODEL, base), SOURCE: tree(MODEL, head)}))
    args = ["--profile", "oac-v1-model", "--source-revision", SOURCE, "--expect-delta-since", BASE]

    fake_clients(monkeypatch, FakeApi(MODEL, base))
    output = tmp_path / "at-base.json"
    code = publisher.main([*args, "--expect-parent", PARENT, "--report", str(output)], environ={})
    report = json.loads(output.read_text(encoding="utf-8"))
    assert code == 0 and report["state"] == "DELTA" and report["delta"] == ["README.md"]
    assert report["expected_delta_since"]["hub_equals_base"] is True
    assert report["expected_parent"] == {"revision": PARENT, "observed": PARENT, "matches": True}

    output = tmp_path / "parent-moved.json"
    code = publisher.main([*args, "--expect-parent", NEW, "--report", str(output)], environ={})
    report = json.loads(output.read_text(encoding="utf-8"))
    assert code == 1 and report["state"] == "REFUSED" and "expected parent" in report["refusal"]

    # A hand edit on the Hub keeps the same delta path list but is refused on bytes.
    fake_clients(monkeypatch, FakeApi(MODEL, package(MODEL, **{"README.md": HAND_EDIT})))
    output = tmp_path / "hand-edit.json"
    code = publisher.main([*args, "--report", str(output)], environ={})
    report = json.loads(output.read_text(encoding="utf-8"))
    assert code == 1 and report["state"] == "REFUSED" and report["delta"] == ["README.md"]
    assert report["expected_delta_since"]["hub_drift"] == ["README.md"]


def test_cli_publish_is_refused_outside_the_canonical_main_workflow(monkeypatch, tmp_path) -> None:
    called = []
    monkeypatch.setattr(publisher, "hub_clients", lambda *args: called.append(args))
    environments = ({}, {**CI, "GITHUB_REF": "refs/heads/claude/x"},
                    {**CI, "GITHUB_REPOSITORY": "someone/szl-forge"}, {**CI, "GITHUB_ACTIONS": ""})
    for index, environ in enumerate(environments):
        output = tmp_path / f"publish-{index}.json"
        code = publisher.main(["--profile", "oac-v1-model", "--source-revision", SOURCE,
                               "--publish", "--report", str(output)], environ=environ)
        report = json.loads(output.read_text(encoding="utf-8"))
        assert code == 1 and report["state"] == "REFUSED"
        assert "canonical main workflow" in report["refusal"]
    assert called == []


def test_cli_publish_requires_a_token_and_the_reviewed_client(monkeypatch, tmp_path) -> None:
    output = tmp_path / "no-token.json"
    code = publisher.main(["--profile", "oac-v1-model", "--source-revision", SOURCE, "--publish",
                           "--report", str(output)], environ={**CI, "HF_TOKEN": ""})
    assert code == 1 and "HF_TOKEN is required" in json.loads(output.read_text(encoding="utf-8"))["refusal"]
    api = FakeApi(MODEL, package(MODEL))
    fake_clients(monkeypatch, api, version="2.1.1")
    output = tmp_path / "wrong-client.json"
    code = publisher.main(["--profile", "oac-v1-model", "--source-revision", SOURCE, "--publish",
                           "--report", str(output)], environ=CI)
    report = json.loads(output.read_text(encoding="utf-8"))
    assert code == 1 and "reviewed huggingface_hub 1.23.0" in report["refusal"]
    assert api.calls == []
    assert TOKEN not in output.read_text(encoding="utf-8")


def test_cli_publish_success_never_records_the_credential(monkeypatch, tmp_path) -> None:
    head = real_head()
    hub = publisher.staged_package(MODEL, head)
    hub["README.md"] = CARD
    api = FakeApi(MODEL, hub)
    seen = fake_clients(monkeypatch, api)
    fresh: list[str] = []
    monkeypatch.setattr(publisher, "assert_current_main", fresh.append)
    output = tmp_path / "publication.json"
    code = publisher.main(["--profile", "oac-v1-model", "--source-revision", head, "--publish",
                           "--report", str(output)], environ=CI)
    text = output.read_text(encoding="utf-8")
    report = json.loads(text)
    assert code == 0 and report["state"] == "PUBLISHED"
    assert report["parent_revision"] == PARENT and report["new_revision"] == NEW
    assert report["delta"] == ["README.md"] and report["lookup"] == "authenticated"
    assert seen["token"] == TOKEN and TOKEN not in text
    assert report["secret_values_recorded"] is False
    assert fresh == [head, head]


@pytest.mark.parametrize("flag", ["--expect-delta-since", "--expect-parent"])
def test_cli_refuses_expectation_flags_when_publishing(flag, tmp_path) -> None:
    output = tmp_path / "publish.json"
    code = publisher.main(["--profile", "oac-v1-model", "--source-revision", SOURCE, "--publish",
                           flag, BASE, "--report", str(output)], environ=CI)
    assert code == 1 and "dry-run assertions" in json.loads(output.read_text(encoding="utf-8"))["refusal"]


def test_cli_never_overwrites_an_existing_report(tmp_path) -> None:
    output = tmp_path / "existing.json"
    output.write_text("prior evidence\n", encoding="utf-8")
    code = publisher.main(["--profile", "oac-v1-model", "--source-revision", SOURCE,
                           "--report", str(output)], environ={})
    assert code == 1
    assert output.read_text(encoding="utf-8") == "prior evidence\n"


def test_cli_profile_choices_are_the_closed_registry() -> None:
    with pytest.raises(SystemExit):
        publisher.main(["--profile", "operator-chosen", "--source-revision", SOURCE,
                        "--report", "unused.json"], environ={})
