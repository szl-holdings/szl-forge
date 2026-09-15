#!/usr/bin/env python3
"""Source-local and public-provider CPU observations in separate disposable containers.

Neither lane publishes anything. Source-local success is not Hub-loader success.
The provider lane compares every allowed executable before asking the real loader
with its default publisher-trust policy. A completed drift observation is not a
qualified runtime. Host validation rechecks the fixed numerical receipt contract.
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
import re
from pathlib import Path
import sys
import tempfile
from typing import Any

SOURCE_REPOSITORY = "szl-holdings/szl-kernels"
SOURCE_REVISION = "13668fd1b22890d6a3b92d3cd9ee516ec5a267c9"
HUB_REPOSITORY = "SZLHOLDINGS/szl-kernels"
HUB_REVISION = "09818b62d683c33d200fca32e2ebfd95c64c65c7"
CLIENT_VERSION = "0.16.1"
SOURCE_BLOBS = {
    "_kernel_api.py": "c575c6e767f5fca7d184447f82e6d100448304f8",
    "_chain.py": "87118ab32287dad4b64a6c0a92d61ac3bf58d05c",
    "_ops.py": "fa0c3bf41d64f9c332b0d3c47ba5b64f87db0942",
    "retrieval.py": "7588d75d18366619a06f6465eb0c0fc3fd0771ff",
}
SOURCE_ROOT = Path("/admitted-source")
MAX_BYTES = 262144
PREFIX = "build/torch-cpu/"
SCHEMA = "szl.kernel-0161-cpu-observation/v1"
BLOCKERS = {"SOURCE_PROVIDER_DRIFT", "PUBLISHER_UNTRUSTED", "PUBLISHER_TRUST_UNKNOWN"}


class QualificationError(ValueError):
    """A failed prerequisite cannot be changed into runtime qualification."""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def strict_json(data: str | bytes) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise QualificationError("duplicate JSON key")
            result[key] = value
        return result

    def reject(value: str) -> None:
        raise QualificationError("non-finite JSON")

    if len(data) > MAX_BYTES:
        raise QualificationError("oversized JSON")
    return json.loads(data, object_pairs_hook=pairs, parse_constant=reject)


def read_source(root: Path = SOURCE_ROOT) -> dict[str, bytes]:
    result = {}
    for name, expected in SOURCE_BLOBS.items():
        path = root / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_BYTES:
            raise QualificationError("invalid admitted source file")
        data = path.read_bytes()
        actual = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        if actual != expected:
            raise QualificationError("admitted Git blob mismatch")
        result[name] = data
    return result


def executable_map(source: dict[str, bytes]) -> dict[str, bytes]:
    if set(source) != set(SOURCE_BLOBS):
        raise QualificationError("incomplete source closure")
    result = {}
    for name, data in source.items():
        basename = "__init__.py" if name == "_kernel_api.py" else name
        for layout in ("", "szl_kernels/"):
            result[PREFIX + layout + basename] = data
    return result


def compare_bytes(expected: dict[str, bytes], observed: dict[str, bytes]) -> dict[str, Any]:
    if not expected or set(observed) != set(expected):
        raise QualificationError("incomplete executable observation")
    rows = [{"path": name, "source_sha256": sha256(expected[name]),
             "provider_sha256": sha256(observed[name]),
             "equal": expected[name] == observed[name]} for name in sorted(expected)]
    return {"all_equal": all(row["equal"] for row in rows), "files": rows}


def provider_files(api: Any) -> dict[str, int]:
    """Consume the bounded pinned tree; KernelInfo has no model-style siblings.

    Both RepoFile and RepoFolder are SDK types. Directory records are validated
    but never downloaded. An incomplete iterator raises, never returns an empty
    inventory. This is SDK-mediated metadata, not retained raw HTTP evidence.
    """
    from huggingface_hub import RepoFile, RepoFolder

    files: dict[str, int] = {}
    seen: set[str] = set()
    entries = api.list_repo_tree(HUB_REPOSITORY, repo_type="kernel",
                                revision=HUB_REVISION, recursive=True, token=False)
    for index, entry in enumerate(entries):
        if index >= 256:
            raise QualificationError("provider tree entry bound exceeded")
        if not isinstance(entry, (RepoFile, RepoFolder)):
            raise QualificationError("unexpected provider tree record type")
        path = entry.path
        if (not isinstance(path, str) or len(path) > 512
                or re.fullmatch(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*", path) is None
                or any(part in {".", ".."} for part in path.split("/"))):
            raise QualificationError("invalid provider tree path")
        if path in seen:
            raise QualificationError("duplicate provider tree path")
        seen.add(path)
        if isinstance(entry, RepoFile):
            if type(entry.size) is not int or entry.size < 0:
                raise QualificationError("invalid provider tree file size")
            files[path] = entry.size
    if not files:
        raise QualificationError("empty provider tree is not admitted")
    return files


def validate_control_bytes(source: dict[str, bytes], controls: dict[str, bytes]) -> None:
    """Check pinned provider control files before the loader consumes them.

    The embedded binding is a source claim, not independent build attestation.
    Its bytes and all executable bytes are included in the metadata digest map.
    """
    if set(controls) != {"metadata.json", "source-binding.json"}:
        raise QualificationError("missing provider control files")
    metadata = strict_json(controls["metadata.json"])
    binding = strict_json(controls["source-binding.json"])
    expected_fields = {"name", "id", "version", "license", "python-depends", "backend", "digest"}
    if (type(metadata) is not dict or set(metadata) != expected_fields
            or metadata.get("name") != "szl-kernels"
            or metadata.get("license") != "Apache-2.0"
            or type(metadata.get("version")) is not int or metadata["version"] != 1
            or metadata.get("backend") != {"type": "cpu"}
            or metadata.get("python-depends") != []
            or not isinstance(metadata.get("id"), str)
            or re.fullmatch(r"_szl_kernels_cpu_[0-9a-f]{8}", metadata["id"]) is None):
        raise QualificationError("provider control metadata mismatch")
    expected_digests = {
        key.removeprefix(PREFIX): base64.b64encode(hashlib.sha256(value).digest()).decode()
        for key, value in executable_map(source).items()}
    expected_digests["source-binding.json"] = base64.b64encode(
        hashlib.sha256(controls["source-binding.json"]).digest()).decode()
    if metadata["digest"] != {"algorithm": "sha256", "files": expected_digests}:
        raise QualificationError("provider control digest mismatch")
    if (type(binding) is not dict
            or binding.get("schema") != "szl.hf-first-class-kernel-binding/v1"
            or binding.get("source_repository") != SOURCE_REPOSITORY
            or binding.get("source_revision") != SOURCE_REVISION):
        raise QualificationError("provider source binding mismatch")
    artifact = binding.get("artifact")
    if (type(artifact) is not dict or artifact.get("repo_id") != HUB_REPOSITORY
            or artifact.get("repo_type") != "kernel" or artifact.get("backend") != "torch-cpu"
            or artifact.get("package") != "szl_kernels"
            or type(artifact.get("version")) is not int or artifact["version"] != 1):
        raise QualificationError("provider artifact binding mismatch")


def stage_local(root: Path, source: dict[str, bytes]) -> None:
    mapped = executable_map(source)
    for relative, data in mapped.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    metadata = {"name": "szl-kernels", "id": "_szl_kernels_cpu_" + sha256(b"".join(source[k] for k in sorted(source)))[:8], "version": 1,
                "license": "Apache-2.0", "python-depends": [], "backend": {"type": "cpu"},
                "digest": {"algorithm": "sha256", "files": {
                    name.removeprefix(PREFIX): base64.b64encode(hashlib.sha256(data).digest()).decode()
                    for name, data in mapped.items()}}}
    (root / PREFIX / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def smoke(module: Any) -> dict[str, Any]:
    import torch
    from publish_szl_kernels import validate_retrieval_runtime_evidence

    if getattr(module, "__version__", None) != "0.2.0":
        raise QualificationError("wrong kernel API package version")
    selfcheck = module.selfcheck()
    if selfcheck.get("ok") is not True or selfcheck.get("version") != "0.2.0":
        raise QualificationError("kernel selfcheck failed")
    x = torch.tensor([[1., 2., 3.], [-2., 0., 1.]], dtype=torch.float32, device="cpu")
    chain = module.UnifiedReceiptChain()
    result = module.governed_rms_norm(chain, x, eps=1e-6)
    reference = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(result, reference, rtol=1e-5, atol=1e-6)
    error = float((result - reference).abs().max())
    for threshold in (-0.01, 1.01, float("nan"), float("inf")):
        invalid = module.UnifiedReceiptChain()
        try:
            module.governed_lambda_gate(invalid, torch.tensor([0.5]), threshold=threshold)
        except ValueError:
            if invalid.verify() != (True, 0, -1):
                raise QualificationError("invalid gate emitted evidence")
        else:
            raise QualificationError("invalid threshold accepted")
    chain = module.UnifiedReceiptChain()
    retrieval = module.governed_cosine_topk(
        chain, torch.tensor([1., 0.]),
        torch.tensor([[0., 1.], [1., 0.], [1., 0.], [-1., 0.]]), k=4, block_rows=1)
    measured = {"indices": retrieval["indices"].tolist(), "scores": retrieval["scores"].tolist(),
                "receipt": retrieval["receipt"], "receipt_depth": 1,
                "chain_verified": chain.verify() == (True, 1, -1)}
    validate_retrieval_runtime_evidence(measured)
    return {"package_version": "0.2.0", "rms_max_abs_error": error,
            "invalid_thresholds_rejected": 4, "retrieval": measured}


def validate_report(report: Any, lane: str, source: dict[str, bytes] | None = None) -> None:
    """Host-side validation; this does not turn a container report into an attestation."""
    fixed = {"schema": SCHEMA, "lane": lane, "source_repository": SOURCE_REPOSITORY,
             "source_revision": SOURCE_REVISION, "hub_repository": HUB_REPOSITORY,
             "hub_revision": HUB_REVISION, "client_version": CLIENT_VERSION,
             "production_authorization": False, "gpu_qualified": False}
    if type(report) is not dict or any(type(report.get(k)) is not type(v) or report[k] != v
                                       for k, v in fixed.items()):
        raise QualificationError("invalid report identity or authority")
    if lane not in {"source", "provider"}:
        raise QualificationError("invalid report lane")
    if source is not None and report.get("source_sha256") != {k: sha256(v) for k, v in source.items()}:
        raise QualificationError("source fingerprint mismatch")
    packages = report.get("installed_packages")
    if not isinstance(packages, dict) or packages.get("kernels") != CLIENT_VERSION:
        raise QualificationError("missing installed-client observation")
    if lane == "provider" and report.get("state") != "SOURCE_PROVIDER_DRIFT":
        encoded = report.get("provider_control_bytes")
        if source is None or type(encoded) is not dict or set(encoded) != {"metadata.json", "source-binding.json"}:
            raise QualificationError("provider control byte evidence required")
        try:
            controls = {key: base64.b64decode(value, validate=True) for key, value in encoded.items()}
        except (ValueError, TypeError) as exc:
            raise QualificationError("invalid provider control encoding") from exc
        validate_control_bytes(source, controls)
    if report.get("state") in BLOCKERS and lane == "provider":
        if report.get("runtime_qualified") is not False or report.get("smoke") is not None:
            raise QualificationError("blocked provider cannot claim execution")
        state = report["state"]
        if state == "SOURCE_PROVIDER_DRIFT":
            comparison = report.get("byte_comparison", {})
            if not (report.get("missing_files") or report.get("unexpected_build_files") or comparison.get("all_equal") is False):
                raise QualificationError("drift requires observed differences")
        elif report.get("byte_comparison", {}).get("all_equal") is not True:
            raise QualificationError("trust observation requires preceding byte comparison")
        return
    if report.get("state") != "CPU_SMOKE_VERIFIED" or report.get("runtime_qualified") is not True:
        raise QualificationError("CPU execution not qualified")
    if report.get("loader") != ("get_local_kernel" if lane == "source" else "get_kernel"):
        raise QualificationError("loader-scope mismatch")
    if lane == "provider" and report.get("byte_comparison", {}).get("all_equal") is not True:
        raise QualificationError("provider qualification needs byte equivalence")
    measured = report.get("smoke")
    if type(measured) is not dict or measured.get("package_version") != "0.2.0":
        raise QualificationError("missing numerical evidence")
    error = measured.get("rms_max_abs_error")
    if type(error) is not float or not math.isfinite(error) or not 0 <= error <= 1e-5:
        raise QualificationError("invalid normalization error")
    if type(measured.get("invalid_thresholds_rejected")) is not int or measured["invalid_thresholds_rejected"] != 4:
        raise QualificationError("missing gate negatives")
    from publish_szl_kernels import validate_retrieval_runtime_evidence
    validate_retrieval_runtime_evidence(measured.get("retrieval"))


def observe(lane: str) -> dict[str, Any]:
    if lane not in {"source", "provider"}:
        raise QualificationError("unknown observation lane")
    if os.getenv("HF_HUB_DISABLE_IMPLICIT_TOKEN") != "1":
        raise QualificationError("implicit credentials forbidden")
    if any(os.getenv(key) for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN",
                                     "LOCAL_KERNELS", "HF_HUB_OFFLINE", "HF_ENDPOINT")):
        raise QualificationError("runtime environment override forbidden")
    if importlib.metadata.version("kernels") != CLIENT_VERSION:
        raise QualificationError("actual installed client mismatch")
    source = read_source()
    report: dict[str, Any] = {"schema": SCHEMA, "lane": lane, "source_repository": SOURCE_REPOSITORY,
        "source_revision": SOURCE_REVISION, "hub_repository": HUB_REPOSITORY, "hub_revision": HUB_REVISION,
        "client_version": CLIENT_VERSION, "runtime_qualified": False, "production_authorization": False,
        "gpu_qualified": False, "smoke": None,
        "source_sha256": {key: sha256(value) for key, value in source.items()},
        "installed_packages": {name: importlib.metadata.version(name)
                               for name in ("kernels", "torch", "huggingface-hub", "numpy")}}
    import kernels
    if lane == "source":
        with tempfile.TemporaryDirectory(prefix="szl-0161-") as directory:
            stage_local(Path(directory), source)
            module = kernels.get_local_kernel(Path(directory), backend="cpu")
            report["smoke"] = smoke(module)
            report["loader"] = "get_local_kernel"
    else:
        from huggingface_hub import HfApi, hf_hub_download
        api = HfApi(endpoint="https://huggingface.co", token=False)
        info = api.repo_info(HUB_REPOSITORY, repo_type="kernel", revision=HUB_REVISION, files_metadata=True, token=False)
        if info.sha != HUB_REVISION:
            raise QualificationError("provider revision mismatch")
        expected = executable_map(source)
        allowed = set(expected) | {PREFIX + "metadata.json", PREFIX + "source-binding.json"}
        files = provider_files(api)
        report["provider_tree_files"] = files
        actual_build = {name for name in files if name.startswith("build/")}
        report["missing_files"] = sorted(set(expected) - actual_build)
        report["unexpected_build_files"] = sorted(actual_build - allowed)
        if report["missing_files"] or report["unexpected_build_files"]:
            report["state"] = "SOURCE_PROVIDER_DRIFT"
            return report
        observed = {}
        for name in sorted(expected):
            size = files[name]
            if type(size) is not int or not 0 < size <= MAX_BYTES:
                raise QualificationError("provider file size unavailable or excessive")
            path = hf_hub_download(HUB_REPOSITORY, filename=name, repo_type="kernel", revision=HUB_REVISION, token=False)
            data = Path(path).read_bytes()
            if len(data) != size:
                raise QualificationError("provider file size changed")
            observed[name] = data
        report["byte_comparison"] = compare_bytes(expected, observed)
        if not report["byte_comparison"]["all_equal"]:
            report["state"] = "SOURCE_PROVIDER_DRIFT"
            return report
        controls = {}
        for name in ("metadata.json", "source-binding.json"):
            key = PREFIX + name
            size = files.get(key)
            if type(size) is not int or not 0 < size <= 65536:
                raise QualificationError("provider control file unavailable or excessive")
            path = hf_hub_download(HUB_REPOSITORY, filename=key, repo_type="kernel", revision=HUB_REVISION, token=False)
            data = Path(path).read_bytes()
            if len(data) != size:
                raise QualificationError("provider control file size changed")
            controls[name] = data
        validate_control_bytes(source, controls)
        report["provider_control_bytes"] = {key: base64.b64encode(value).decode()
                                            for key, value in controls.items()}
        trust = getattr(api.get_organization_overview("SZLHOLDINGS"), "trustedKernelPublisher", None)
        if trust is not True:
            report["state"] = "PUBLISHER_UNTRUSTED" if trust is False else "PUBLISHER_TRUST_UNKNOWN"
            return report
        module = kernels.get_kernel(HUB_REPOSITORY, revision=HUB_REVISION, backend="cpu", trust_remote_code=False)
        matching = [item for item in kernels.get_loaded_kernels()
                    if item.module is module and item.repo_info is not None
                    and item.repo_info.repo_id == HUB_REPOSITORY and item.repo_info.revision == HUB_REVISION]
        if len(matching) != 1:
            raise QualificationError("loader origin mismatch")
        report["smoke"] = smoke(module)
        report["loader"] = "get_kernel"
    report["runtime_qualified"] = True
    report["state"] = "CPU_SMOKE_VERIFIED"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", choices=("source", "provider"), required=True)
    parser.add_argument("--validate", type=Path)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    args = parser.parse_args()
    if args.validate is not None:
        validate_report(strict_json(args.validate.read_bytes()), args.lane, read_source(args.source_root))
        return 0
    try:
        # Keep machine-readable stdout separate from dependency/library logging.
        with contextlib.redirect_stdout(sys.stderr):
            report = observe(args.lane)
        print(json.dumps(report, sort_keys=True, allow_nan=False))
        return 0
    except Exception as exc:
        print(json.dumps({"schema": SCHEMA, "lane": args.lane, "state": "UNAVAILABLE_OR_FAILED",
                          "error_type": type(exc).__name__,
                          "detail": "".join(c if 32 <= ord(c) <= 126 else "?" for c in str(exc))[:512],
                          "runtime_qualified": False,
                          "production_authorization": False, "gpu_qualified": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
