#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Bounded source qualification for IFM/K2-Horizon-7B.

This tool deliberately does not load model weights and never executes repository
Python. It pins one upstream revision, hashes bounded metadata/custom-code files,
performs a static AST inventory, and emits a fail-closed qualification receipt.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

MODEL_ID = "IFM/K2-Horizon-7B"
REVISION = "14985b2765262fc7850476f71a7d0ab08e86af69"
STALE_UPSTREAM_RECIPE_REVISION = "69ada542b68fe13d767479db2ab9421baff88681"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
MAX_REVIEW_FILE_BYTES = 4 * 1024 * 1024
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf", ".ckpt")
EXACT_METADATA_FILES = {
    "README.md",
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "special_tokens_map.json",
}
REVIEW_CODE_PREFIXES = (
    "configuration_",
    "modeling_",
    "processing_",
    "tokenization_",
)
HIGH_RISK_CALLS = {
    "eval",
    "exec",
    "compile",
    "__import__",
    "os.system",
    "os.popen",
    "subprocess.run",
    "subprocess.call",
    "subprocess.Popen",
    "subprocess.check_call",
    "subprocess.check_output",
}
HIGH_RISK_IMPORT_ROOTS = {"requests", "socket", "subprocess", "urllib"}


class QualificationError(RuntimeError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def is_weight_file(name: str) -> bool:
    lowered = name.lower()
    return lowered.endswith(WEIGHT_SUFFIXES) or (
        "model-" in lowered and lowered.endswith(".safetensors")
    )


def is_review_file(name: str) -> bool:
    base = Path(name).name
    if base in EXACT_METADATA_FILES:
        return True
    return base.endswith(".py") and base.startswith(REVIEW_CODE_PREFIXES)


def select_review_files(names: Iterable[str]) -> list[str]:
    return sorted(
        {name for name in names if is_review_file(name) and not is_weight_file(name)}
    )


def dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


def scan_python_source(text: str) -> dict[str, Any]:
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return {
            "syntax_ok": False,
            "syntax_error": f"{exc.msg}:{exc.lineno}:{exc.offset}",
            "imports": [],
            "high_risk_imports": [],
            "high_risk_calls": [],
        }

    imports: set[str] = set()
    high_risk_imports: set[str] = set()
    high_risk_calls: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                imports.add(alias.name)
                if root in HIGH_RISK_IMPORT_ROOTS:
                    high_risk_imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            root = module.split(".", 1)[0]
            imports.add(module)
            if root in HIGH_RISK_IMPORT_ROOTS:
                high_risk_imports.add(module)
        elif isinstance(node, ast.Call):
            name = dotted_name(node.func)
            if name in HIGH_RISK_CALLS:
                high_risk_calls.add(name)

    return {
        "syntax_ok": True,
        "syntax_error": None,
        "imports": sorted(imports),
        "high_risk_imports": sorted(high_risk_imports),
        "high_risk_calls": sorted(high_risk_calls),
    }


def sibling_size(sibling: Any) -> int | None:
    value = getattr(sibling, "size", None)
    if isinstance(value, int):
        return value
    lfs = getattr(sibling, "lfs", None)
    if lfs is not None:
        value = getattr(lfs, "size", None)
        if isinstance(value, int):
            return value
    return None


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def qualify(*, output: Path, cache_dir: Path) -> dict[str, Any]:
    from huggingface_hub import HfApi, hf_hub_download

    if not HEX40.fullmatch(REVISION):
        raise QualificationError("revision must be immutable 40-hex")

    api = HfApi()
    info = api.model_info(MODEL_ID, revision=REVISION, files_metadata=True)
    resolved = str(info.sha)
    if resolved != REVISION:
        raise QualificationError(f"resolved revision mismatch: {resolved}")

    siblings = list(info.siblings or [])
    names = [str(item.rfilename) for item in siblings]
    review_files = select_review_files(names)
    python_files = sorted(name for name in names if name.endswith(".py"))
    weight_files = sorted(name for name in names if is_weight_file(name))

    if not review_files:
        raise QualificationError("no bounded review files discovered")
    if not python_files:
        raise QualificationError(
            "K2 custom-code model unexpectedly exposed no Python files"
        )

    observed: dict[str, Any] = {}
    risk_findings: list[dict[str, Any]] = []
    cache_dir.mkdir(parents=True, exist_ok=True)
    sibling_map = {str(item.rfilename): item for item in siblings}

    for name in review_files:
        size = sibling_size(sibling_map[name])
        if size is not None and size > MAX_REVIEW_FILE_BYTES:
            raise QualificationError(
                f"review file exceeds {MAX_REVIEW_FILE_BYTES} bytes: {name} ({size})"
            )

        local = Path(
            hf_hub_download(
                MODEL_ID,
                filename=name,
                revision=REVISION,
                cache_dir=cache_dir,
            )
        )
        data = local.read_bytes()
        if len(data) > MAX_REVIEW_FILE_BYTES:
            raise QualificationError(f"downloaded review file exceeds bound: {name}")

        record: dict[str, Any] = {
            "declared_bytes": size,
            "bytes": len(data),
            "sha256": sha256_bytes(data),
            "kind": "python" if name.endswith(".py") else "metadata",
        }
        if name.endswith(".py"):
            scan = scan_python_source(data.decode("utf-8", errors="strict"))
            record["static_scan"] = scan
            if not scan["syntax_ok"]:
                risk_findings.append(
                    {
                        "file": name,
                        "kind": "syntax_error",
                        "details": scan["syntax_error"],
                    }
                )
            for import_name in scan["high_risk_imports"]:
                risk_findings.append(
                    {
                        "file": name,
                        "kind": "high_risk_import",
                        "details": import_name,
                    }
                )
            for call_name in scan["high_risk_calls"]:
                risk_findings.append(
                    {
                        "file": name,
                        "kind": "high_risk_call",
                        "details": call_name,
                    }
                )
        observed[name] = record

    card_data = getattr(info, "card_data", None)
    declared_license = str(getattr(card_data, "license", "") or "").lower()
    license_pass = declared_license == "apache-2.0"
    static_syntax_pass = all(
        record.get("kind") != "python"
        or record.get("static_scan", {}).get("syntax_ok") is True
        for record in observed.values()
    )

    receipt: dict[str, Any] = {
        "schema": "szl.frontier.k2-source-qualification.v1",
        "truth_label": "MEASURED",
        "candidate": {
            "candidate_id": "k2-horizon-7b",
            "model_id": MODEL_ID,
            "requested_revision": REVISION,
            "resolved_revision": resolved,
            "declared_license": declared_license,
        },
        "source_correction": {
            "stale_upstream_recipe_revision": STALE_UPSTREAM_RECIPE_REVISION,
            "state": "MEASURED_INVALID_404",
            "github_workflow_run_id": 34219351248,
            "github_workflow_job_id": 102038541050,
        },
        "source": {
            "files_reviewed": observed,
            "repository_python_files": python_files,
            "repository_weight_files_count": len(weight_files),
            "weight_files_downloaded": False,
            "repository_code_executed": False,
            "trust_remote_code_enabled": False,
        },
        "static_risk_inventory": risk_findings,
        "gates": {
            "exact_revision_pass": resolved == REVISION,
            "license_compatibility_for_bounded_evaluation": (
                "PASS" if license_pass else "REVIEW_REQUIRED"
            ),
            "static_python_syntax_pass": static_syntax_pass,
            "remote_code_manual_review": "REQUIRED",
            "runtime_compatibility": "UNAVAILABLE_NOT_EXECUTED",
            "hf_inference_provider_route": "UNAVAILABLE_UPSTREAM_NOT_DEPLOYED",
        },
        "evaluation_state": "SOURCE_ATTESTED_REMOTE_CODE_REVIEW_REQUIRED",
        "runtime_qualified": False,
        "publication_eligible": False,
        "production_disposition": "HOLD",
        "promotion_effect": "NONE",
        "model_output_authority": "PROPOSAL_ONLY",
        "consequential_action_admission_layer": "A11oy",
        "known_bounds": [
            "static_inventory_is_not_a_manual_security_review",
            "model_weights_not_downloaded",
            "model_not_executed",
            "runtime_hardware_unmeasured",
            "provider_route_unavailable_at_intake",
        ],
    }
    canonical = json.dumps(
        receipt, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    receipt["receipt_sha256"] = sha256_bytes(canonical)
    write_json(output, receipt)
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    receipt = qualify(output=args.output, cache_dir=args.cache_dir)
    print(
        json.dumps(
            {
                "ok": receipt["evaluation_state"]
                == "SOURCE_ATTESTED_REMOTE_CODE_REVIEW_REQUIRED",
                "model_id": MODEL_ID,
                "revision": REVISION,
                "receipt_sha256": receipt["receipt_sha256"],
                "production_disposition": "HOLD",
                "promotion_effect": "NONE",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
