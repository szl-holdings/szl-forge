"""One-target, exact-source OAC Space publisher; never purchase compute."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime, timezone

SPACE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = SPACE_ROOT.parents[1]
TARGET = "SZLHOLDINGS/oac-system-health-lab"
SOURCE_VARIABLE = "SZL_GITHUB_SOURCE_REVISION"


class PublicationRefused(ValueError):
    """The exact-source, one-writer, or free-compute contract did not pass."""


def _enum_value(value):
    return getattr(value, "value", value)


def verify_entitlement(identity):
    """Use an existing organization plan, never create or upgrade a plan."""
    organizations = list(identity.get("orgs", []))
    if identity.get("name", "").casefold() == "szlholdings":
        organizations.append(identity)
    candidates = [
        o for o in organizations if o.get("name", "").casefold() == "szlholdings"
    ]
    if not any(o.get("plan") in {"team", "enterprise"} for o in candidates):
        raise PublicationRefused(
            "existing organization compute entitlement is unavailable"
        )
    return {"existing_plan_confirmed": True, "purchase_or_upgrade": False}


def verify_hardware(runtime, *, running=False):
    current = _enum_value(getattr(runtime, "hardware", None))
    requested = _enum_value(getattr(runtime, "requested_hardware", None))
    storage = _enum_value(getattr(runtime, "storage", None))
    volumes = getattr(runtime, "volumes", None)
    raw = getattr(runtime, "raw", {})
    raw_volumes = raw.get("volumes") if isinstance(raw, dict) else None
    if current not in {None, "cpu-basic"} or requested not in {None, "cpu-basic"}:
        raise PublicationRefused("only existing CPU Basic hardware is permitted")
    if storage is not None:
        raise PublicationRefused("persistent paid storage is outside this release")
    if volumes or raw_volumes:
        raise PublicationRefused("attached volumes are outside this release")
    if running and current != "cpu-basic":
        raise PublicationRefused("running CPU Basic hardware was not observed")
    return {
        "current": current,
        "requested": requested,
        "storage": storage,
        "volume_count": 0,
        "hardware_upgrade": False,
    }


def verify_head_and_bytes(revision, plan):
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise PublicationRefused("source revision must be immutable")
    observed = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
    ).strip()
    if observed != revision:
        raise PublicationRefused("checkout is not the declared source")
    subprocess.run(
        ["git", "diff", "--exit-code", "HEAD", "--", plan["source_dir"]],
        cwd=REPOSITORY_ROOT,
        check=True,
    )
    immutable_files = {}
    for target, item in plan["files"].items():
        size = int(
            subprocess.check_output(
                ["git", "cat-file", "-s", revision + ":" + item["source_path"]],
                cwd=REPOSITORY_ROOT,
                text=True,
            )
        )
        if not 0 <= size <= 2 * 1024 * 1024:
            raise PublicationRefused("publication blob exceeds bounded size")
        immutable = subprocess.check_output(
            ["git", "show", revision + ":" + item["source_path"]], cwd=REPOSITORY_ROOT
        )
        if (
            len(immutable) != size
            or hashlib.sha256(immutable).hexdigest() != item["sha256"]
        ):
            raise PublicationRefused(
                "tracked publication bytes differ from immutable source"
            )
        immutable_files[target] = immutable
    return immutable_files


def publish(api, revision, report, *, wait_seconds=900):
    # Reuse canonical Forge source planning/runtime helpers; the write itself
    # adds a parent-commit CAS and refuses unexpected files instead of deleting.
    sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))
    sys.path.insert(0, str(SPACE_ROOT))
    from publish_hf_space import build_plan, wait_for_exact_running_space
    from huggingface_hub import CommitOperationAdd
    from huggingface_hub.utils import RepositoryNotFoundError
    from verify_release import verify_release

    report["source_verification"] = verify_release(SPACE_ROOT, REPOSITORY_ROOT)
    plan = build_plan(SPACE_ROOT, TARGET, revision, static=False)
    immutable_files = verify_head_and_bytes(revision, plan)
    report["entitlement"] = verify_entitlement(api.whoami())
    try:
        info = api.space_info(TARGET)
        report["repository_created"] = False
    except RepositoryNotFoundError:
        # CPU Basic has no hourly charge; eligibility was verified above.
        # exist_ok=False refuses a concurrent creator rather than adopting it.
        api.create_repo(
            repo_id=TARGET,
            repo_type="space",
            space_sdk="docker",
            space_hardware="cpu-basic",
            private=False,
            exist_ok=False,
        )
        report["repository_created"] = True
        info = api.space_info(TARGET)
    if info.id.casefold() != TARGET.casefold() or info.sdk != "docker":
        raise PublicationRefused("target identity or SDK mismatch")
    api.auth_check(repo_id=TARGET, repo_type="space", write=True)
    report["hardware_before"] = verify_hardware(api.get_space_runtime(TARGET))
    expected_parent = info.sha
    if not isinstance(expected_parent, str) or not re.fullmatch(
        r"[0-9a-f]{40}", expected_parent
    ):
        raise PublicationRefused("target has no immutable parent revision")
    present = set(
        api.list_repo_files(repo_id=TARGET, repo_type="space", revision=expected_parent)
    )
    unexpected = present - set(plan["files"]) - {".gitattributes"}
    if unexpected:
        raise PublicationRefused("unexpected target files; no deletion is permitted")
    if api.space_info(TARGET).sha != expected_parent:
        raise PublicationRefused("target changed during preflight")
    report["expected_hub_parent"] = expected_parent
    operations = [
        CommitOperationAdd(path_in_repo=target, path_or_fileobj=immutable_files[target])
        for target in sorted(plan["files"])
    ]
    commit = api.create_commit(
        repo_id=TARGET,
        repo_type="space",
        operations=operations,
        parent_commit=expected_parent,
        commit_message="deploy: source-bound OAC preview " + revision[:12],
    )
    report["hub_commit"] = commit.oid
    api.add_space_variable(repo_id=TARGET, key=SOURCE_VARIABLE, value=revision)
    variables = api.get_space_variables(repo_id=TARGET)
    item = variables.get(SOURCE_VARIABLE)
    observed_variable = (
        item.get("value") if isinstance(item, dict) else getattr(item, "value", None)
    )
    if observed_variable != revision:
        raise PublicationRefused("source variable readback failed")
    live = wait_for_exact_running_space(
        api, TARGET, commit.oid, wait_seconds=wait_seconds
    )
    report["hardware_after"] = verify_hardware(
        api.get_space_runtime(TARGET), running=True
    )
    from huggingface_hub import hf_hub_download

    matches = []
    for target, expected in sorted(plan["files"].items()):
        downloaded = Path(
            hf_hub_download(
                repo_id=TARGET,
                repo_type="space",
                filename=target,
                revision=commit.oid,
                token=api.token,
                force_download=True,
            )
        )
        digest = hashlib.sha256(downloaded.read_bytes()).hexdigest()
        if digest != expected["sha256"]:
            raise PublicationRefused("published immutable file mismatch")
        matches.append({"path": target, "sha256": digest, "matches": True})
    if api.space_info(TARGET).sha != commit.oid or live.sha != commit.oid:
        raise PublicationRefused("Hub source drift after publication")
    report["files"] = matches
    report["complete"] = True
    report["status"] = "PUBLISHED_SOURCE_VERIFIED"
    # Actual GET/POST application checks are a separate mandatory workflow step.
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--wait-seconds", type=int, default=900)
    args = parser.parse_args(argv)
    report = {
        "schema": "szl.oac-health-space-publication/v1",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "repo_id": TARGET,
        "source_revision": args.source_revision,
        "complete": False,
        "status": "FAILED",
        "receipt_minted": False,
        "clinical_use_authorized": False,
        "production_promotion_allowed": False,
    }
    # Reserve evidence before any provider write. Existing receipts are never
    # overwritten, and an unwritable destination is a pre-publication refusal.
    try:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        handle = args.report.open("x", encoding="utf-8", newline="\n")
    except OSError as exc:
        print(
            json.dumps(
                {
                    "status": "REFUSED_BEFORE_PUBLICATION",
                    "complete": False,
                    "failure_type": type(exc).__name__,
                },
                sort_keys=True,
            )
        )
        return 1
    exit_code = 1
    try:
        if (
            os.environ.get("GITHUB_REF") != "refs/heads/main"
            or os.environ.get("GITHUB_REPOSITORY") != "szl-holdings/szl-forge"
        ):
            raise PublicationRefused(
                "publication is restricted to canonical main workflow"
            )
        if not 30 <= args.wait_seconds <= 1200:
            raise PublicationRefused("publication wait budget is invalid")
        from huggingface_hub import HfApi

        publish(
            HfApi(token=os.environ.get("HF_TOKEN")),
            args.source_revision,
            report,
            wait_seconds=args.wait_seconds,
        )
        exit_code = 0
    except Exception as exc:
        report["failure_type"] = type(exc).__name__
    with handle:
        json.dump(report, handle, sort_keys=True, indent=2)
        handle.write("\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "complete": report["complete"],
                "repo_id": TARGET,
                "failure_type": report.get("failure_type"),
            },
            sort_keys=True,
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
