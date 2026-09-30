#!/usr/bin/env python3
"""Run the HF public census with one short-lived, user-scoped OIDC grant.

The configured Hugging Face CI/CD identity is an account identity, not a repo
publisher.  Its exchanged token is read-only, cannot read private repositories,
and is used only by the fixed-origin HF census child.  Missing or mismatched
provider bindings fail closed with a non-secret receipt.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Callable, Mapping


REPOSITORY = "szl-holdings/szl-forge"
WORKFLOW_FILE = "public-estate-file-census.yml"
WORKFLOW_REF = (
    f"{REPOSITORY}/.github/workflows/{WORKFLOW_FILE}@refs/heads/main"
)
HF_USER_RESOURCE = "betterwithage"
HF_ORG = "SZLHOLDINGS"
OIDC_ISSUER = "https://token.actions.githubusercontent.com"
OIDC_AUDIENCE = "https://huggingface.co"
OBSERVER_TIMEOUT = 1100
SHA = re.compile(r"[0-9a-f]{40}\Z")
USER_TOKEN = re.compile(r"hf_oauth_[A-Za-z0-9._-]{8,16370}\Z")
HF_PREFIXES = ("HF_", "HUGGINGFACE_", "HUGGING_FACE_")
CRITICAL_PATHS = (
    "tools/hf_census_keyless_credentials.py",
    "tools/observe_public_estate_files.py",
    ".github/workflows/public-estate-file-census.yml",
)


class CensusCredentialError(RuntimeError):
    """A fixed, non-secret credential or execution failure code."""


@dataclass(frozen=True)
class ObserverResult:
    exit_code: int
    status: str


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def binding_receipt(state: str) -> dict[str, Any]:
    """Describe the exact external account binding; never infer wildcards."""
    return {
        "state": state,
        "configuration_surface": "HF_ACCOUNT_AUTHENTICATION_CI_CD_ACCESS",
        "resource": HF_USER_RESOURCE,
        "issuer": OIDC_ISSUER,
        "audience": OIDC_AUDIENCE,
        "claims": {
            "repository": REPOSITORY,
            "branch": "main",
            "workflow": WORKFLOW_FILE,
        },
        "allow_inference_providers": False,
    }


def credential_receipt() -> dict[str, Any]:
    return {
        "kind": "HF_ACCOUNT_CI_CD_IDENTITY",
        "provider_scope": "gated-repos",
        "read_only": True,
        "write": False,
        "private_repositories": False,
        "inference_providers": False,
        "maximum_lifetime_seconds": 3600,
        "shape": "hf_oauth_*",
        "logged": False,
        "persisted": False,
        "cached": False,
        "forwarded_only_to": "huggingface.co",
    }


def base_receipt(source_revision: str) -> dict[str, Any]:
    return {
        "schema": "szl.hf-public-census-keyless-credential/v1",
        "generated_at": now(),
        "status": "CHECKING",
        "source_revision": source_revision if SHA.fullmatch(source_revision) else None,
        "target": {
            "organization": HF_ORG,
            "scope": "PUBLIC_REPOSITORY_METADATA_ONLY",
        },
        "binding": binding_receipt("REQUIRED_UNVERIFIED"),
        "credential": credential_receipt(),
        "repository_mutation": "NOT_ATTEMPTED",
        "provider_mutation": "NOT_ATTEMPTED",
        "credential_exported_to_observer_only": False,
        "observation": None,
    }


def atomic_report(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = handle.name
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def require_workflow_context(
    environment: Mapping[str, str], source_revision: str
) -> None:
    if SHA.fullmatch(source_revision) is None:
        raise CensusCredentialError("INVALID_SOURCE_REVISION")
    expected = {
        "GITHUB_ACTIONS": "true",
        "GITHUB_REPOSITORY": REPOSITORY,
        "GITHUB_REF": "refs/heads/main",
        "GITHUB_WORKFLOW_REF": WORKFLOW_REF,
        "GITHUB_SHA": source_revision,
    }
    if any(environment.get(key) != value for key, value in expected.items()):
        raise CensusCredentialError("UNTRUSTED_WORKFLOW_CONTEXT")
    if environment.get("GITHUB_EVENT_NAME") not in {"push", "workflow_dispatch"}:
        raise CensusCredentialError("UNTRUSTED_WORKFLOW_EVENT")
    if not environment.get("ACTIONS_ID_TOKEN_REQUEST_URL") or not environment.get(
        "ACTIONS_ID_TOKEN_REQUEST_TOKEN"
    ):
        raise CensusCredentialError("OIDC_PERMISSION_UNAVAILABLE")


def verify_source(source_revision: str) -> None:
    """Bind credential-handling files to their canonical protected Git blobs."""
    root = Path(__file__).resolve().parents[1]
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            check=True,
            timeout=10,
        ).stdout.decode("ascii").strip()
        if head != source_revision:
            raise CensusCredentialError("SOURCE_HEAD_MISMATCH")
        for relative in CRITICAL_PATHS:
            committed = subprocess.run(
                ["git", "rev-parse", f"{source_revision}:{relative}"],
                cwd=root,
                capture_output=True,
                check=True,
                timeout=10,
            ).stdout.decode("ascii").strip()
            working = subprocess.run(
                ["git", "hash-object", "--", relative],
                cwd=root,
                capture_output=True,
                check=True,
                timeout=10,
            ).stdout.decode("ascii").strip()
            if SHA.fullmatch(committed) is None or committed != working:
                raise CensusCredentialError("CREDENTIAL_SOURCE_MISMATCH")
    except CensusCredentialError:
        raise
    except (OSError, subprocess.SubprocessError, UnicodeError):
        raise CensusCredentialError("SOURCE_VERIFICATION_UNAVAILABLE") from None


def _without_hf_credentials(environment: Mapping[str, str]) -> dict[str, str]:
    return {
        key: value
        for key, value in environment.items()
        if not key.startswith(HF_PREFIXES)
    }


def _user_token(value: str) -> str:
    token = str(value or "").strip()
    if len(token) > 16384 or USER_TOKEN.fullmatch(token) is None:
        raise CensusCredentialError("INVALID_USER_SCOPED_TOKEN_RESPONSE")
    return token


def exchange(environment: Mapping[str, str], source_revision: str) -> str:
    """Exchange exact GitHub claims in a temporary credential-free HF home."""
    require_workflow_context(environment, source_revision)
    child = _without_hf_credentials(environment)
    for key in ("GITHUB_TOKEN", "GH_TOKEN"):
        child.pop(key, None)
    with tempfile.TemporaryDirectory(prefix="hf-census-oidc-") as home:
        child.update(
            {
                "HF_HOME": home,
                "HF_OIDC_RESOURCE": HF_USER_RESOURCE,
                "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
            }
        )
        try:
            process = subprocess.run(
                ["hf", "auth", "token"],
                env=child,
                capture_output=True,
                text=True,
                check=False,
                timeout=60,
            )
        except Exception:
            raise CensusCredentialError("OIDC_EXCHANGE_UNAVAILABLE") from None
    if process.returncode != 0:
        raise CensusCredentialError("OIDC_EXCHANGE_REJECTED")
    # Accept the whole trimmed stdout only. Selecting one line from mixed output
    # could conceal a warning, fallback credential, or ambiguous exchange.
    return _user_token(process.stdout)


def run_observer(
    token: str,
    source_revision: str,
    output: Path,
    environment: Mapping[str, str],
) -> ObserverResult:
    """Give the grant only to the fixed-origin observer child, then discard it."""
    token = _user_token(token)
    if output.exists():
        raise CensusCredentialError("OBSERVATION_OUTPUT_ALREADY_EXISTS")
    root = Path(__file__).resolve().parents[1]
    child = _without_hf_credentials(environment)
    for key in (
        "GITHUB_TOKEN",
        "GH_TOKEN",
        "ACTIONS_ID_TOKEN_REQUEST_URL",
        "ACTIONS_ID_TOKEN_REQUEST_TOKEN",
    ):
        child.pop(key, None)
    child["HF_CENSUS_TOKEN"] = token
    child["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    try:
        process = subprocess.run(
            [
                sys.executable,
                "-I",
                "-B",
                str(root / "tools" / "observe_public_estate_files.py"),
                "--lane",
                "huggingface",
                "--source-revision",
                source_revision,
                "--output",
                str(output),
            ],
            cwd=root,
            env=child,
            capture_output=True,
            text=True,
            check=False,
            timeout=OBSERVER_TIMEOUT,
        )
    except Exception:
        raise CensusCredentialError("OBSERVER_UNAVAILABLE") from None
    try:
        summary = json.loads(process.stdout.strip())
    except (ValueError, UnicodeError):
        raise CensusCredentialError("INVALID_OBSERVER_SUMMARY") from None
    if not isinstance(summary, dict) or set(summary) != {"status", "populations"}:
        raise CensusCredentialError("INVALID_OBSERVER_SUMMARY")
    status = summary.get("status")
    expected = {
        0: "FILE_METADATA_OBSERVED_NOT_QUALIFIED",
        2: "PARTIAL_OR_UNAVAILABLE",
    }
    if process.returncode not in expected or status != expected[process.returncode]:
        raise CensusCredentialError("INVALID_OBSERVER_OUTCOME")
    if not output.is_file():
        raise CensusCredentialError("OBSERVATION_RECEIPT_MISSING")
    return ObserverResult(exit_code=process.returncode, status=status)


def execute(
    *,
    source_revision: str,
    output: Path,
    report_path: Path,
    environment: Mapping[str, str] | None = None,
    source_verifier: Callable[[str], None] = verify_source,
    supplier: Callable[[Mapping[str, str], str], str] = exchange,
    observer: Callable[[str, str, Path, Mapping[str, str]], ObserverResult] = run_observer,
) -> int:
    environment = os.environ if environment is None else environment
    report = base_receipt(source_revision)
    atomic_report(report_path, report)
    stage = "PREFLIGHT"
    try:
        require_workflow_context(environment, source_revision)
        source_verifier(source_revision)
        if output.exists():
            raise CensusCredentialError("OBSERVATION_OUTPUT_ALREADY_EXISTS")
        stage = "EXCHANGE"
        token = _user_token(supplier(environment, source_revision))
        report["binding"] = binding_receipt("EXCHANGE_VALIDATED")
        report["credential_exported_to_observer_only"] = True
        report["status"] = "USER_SCOPED_OIDC_CREDENTIAL_VALIDATED"
        report["generated_at"] = now()
        atomic_report(report_path, report)
        stage = "OBSERVATION"
        observed = observer(token, source_revision, output, environment)
        del token
        report["observation"] = {
            "status": observed.status,
            "exit_code": observed.exit_code,
            "receipt": str(output),
        }
        if observed.exit_code == 0:
            report["status"] = "CREDENTIAL_VALIDATED_OBSERVATION_COMPLETE"
        elif observed.exit_code == 2:
            report["status"] = "CREDENTIAL_VALIDATED_OBSERVATION_PARTIAL"
        else:
            raise CensusCredentialError("INVALID_OBSERVER_OUTCOME")
        code = observed.exit_code
    except Exception:
        if stage == "PREFLIGHT":
            report["status"] = "PREFLIGHT_UNAVAILABLE"
        elif stage == "EXCHANGE":
            report["status"] = "OIDC_IDENTITY_UNAVAILABLE"
        else:
            report["status"] = "CREDENTIAL_VALIDATED_OBSERVER_UNAVAILABLE"
        report["failure_stage"] = stage
        code = 2
    report["generated_at"] = now()
    atomic_report(report_path, report)
    return code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--credential-report", type=Path, required=True)
    args = parser.parse_args(argv)
    code = execute(
        source_revision=args.source_revision,
        output=args.output,
        report_path=args.credential_report,
    )
    try:
        report = json.loads(args.credential_report.read_text(encoding="utf-8"))
        status = report.get("status", "RECEIPT_UNAVAILABLE")
    except (OSError, ValueError, UnicodeError):
        status = "RECEIPT_UNAVAILABLE"
    print(json.dumps({"status": status, "exit_code": code}, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
