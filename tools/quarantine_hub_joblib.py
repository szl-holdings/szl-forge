#!/usr/bin/env python3
"""Delete Hub model.joblib using an already-validated HF_TOKEN."""
from __future__ import annotations

import argparse
import os


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    args = parser.parse_args(argv)
    token = os.environ.get("HF_TOKEN") or ""
    if not token.strip():
        print("UNAVAILABLE: HF_TOKEN missing after credential acquire")
        return 2
    from huggingface_hub import CommitOperationDelete, HfApi

    api = HfApi(token=token)
    info = api.repo_info(repo_id=args.repo_id, repo_type="model")
    parent = info.sha
    siblings = {s.rfilename for s in (info.siblings or [])}
    if "model.joblib" not in siblings:
        print(f"VERIFIED_CURRENT: {args.repo_id}@{parent} has no model.joblib")
        return 0
    commit = api.create_commit(
        repo_id=args.repo_id,
        repo_type="model",
        operations=[CommitOperationDelete(path_in_repo="model.joblib")],
        commit_message="quarantine: remove model.joblib from approved path",
        create_pr=True,
        parent_commit=parent,
    )
    print(f"Hub PR opened for {args.repo_id} parent={parent} result={commit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
