"""Export an allowlisted source ZIP and one readable Markdown/Python payload.
SPDX-License-Identifier: Apache-2.0
No downloads, dependency installation, model execution, or remote publication.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import io
import json
from pathlib import Path, PurePosixPath
import uuid
import zipfile

ROOT = Path(__file__).resolve().parent
NAMES = (
    "lab.py", "worker.py", "probe_sandbox.py", "verify_model.py",
    "start-local.ps1", "recover-docker.ps1", "model.lock.json", "README.md",
    "RESULTS_2026-09-29.md", "test_lab.py", "test_run.py", "test_tasks.py",
    "test_release_guards.py", "build_release.py", "test_bundle.py",
    "tasks/canonical_report_digest.json", "tasks/evidence_admission.json",
    "tasks/stable_provider_routing.json",
)


def install_files(destination, files, hashes):
    """Validate every byte/path before creating a new, non-overwriting directory."""
    if not files or set(files) != set(hashes):
        raise ValueError("File set and integrity manifest differ")
    encoded = {}
    for name, text in files.items():
        path = PurePosixPath(name)
        reserved = {"con", "prn", "aux", "nul", *[f"com{i}" for i in range(1, 10)],
                    *[f"lpt{i}" for i in range(1, 10)]}
        if (path.is_absolute() or str(path) != name or "\\" in name or ":" in name
                or not path.parts or any(part in (".", "..") or part.endswith((" ", "."))
                    or part.split(".")[0].lower() in reserved for part in path.parts)
                or name == "SOURCE_MANIFEST.json"):
            raise ValueError("Unsafe payload path")
        data = text.encode("utf-8")
        if hashlib.sha256(data).hexdigest() != hashes[name]:
            raise ValueError("Payload SHA256 mismatch: " + name)
        encoded[name] = data
    if sum(map(len, encoded.values())) > 8 * 1024 * 1024:
        raise ValueError("Source payload exceeds size bound")
    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Refusing to overwrite existing destination: " + str(destination))
    if not destination.parent.is_dir():
        raise ValueError("Choose an existing parent directory")
    for parent in destination.parents:
        status = parent.lstat()
        if parent.is_symlink() or getattr(status, "st_file_attributes", 0) & 0x400:
            raise ValueError("Refusing a symlink/junction in destination parents")
    destination.mkdir()  # Exclusive creation. Partial failure never deletes prior work.
    for name, data in encoded.items():
        target = destination.joinpath(*PurePosixPath(name).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(data)
    with (destination / "SOURCE_MANIFEST.json").open("x", encoding="utf-8") as stream:
        json.dump({"schema": "szl.source-bundle/v1", "files": hashes}, stream, indent=2, sort_keys=True)
    return destination


def collect(root):
    files = {}
    for name in NAMES:
        path = Path(root).joinpath(*PurePosixPath(name).parts)
        if path.is_symlink() or getattr(path.lstat(), "st_file_attributes", 0) & 0x400:
            raise ValueError("Source file is a symlink/reparse point: " + name)
        data = path.read_bytes()
        if len(data) > 1024 * 1024:
            raise ValueError("Unexpected oversized source: " + name)
        files[name] = data.decode("utf-8")
    return dict(sorted(files.items()))


def hashes_for(files):
    return {name: hashlib.sha256(text.encode("utf-8")).hexdigest() for name, text in sorted(files.items())}


def zip_bytes(files):
    manifest = json.dumps({"schema": "szl.source-bundle/v1", "files": hashes_for(files)},
                          indent=2, sort_keys=True) + "\n"
    contents = {**files, "SOURCE_MANIFEST.json": manifest}
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, text in sorted(contents.items()):
            info = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, text.encode("utf-8"))
    return output.getvalue()


def payload(files):
    source = ("# SZL source installer: no downloads, services, inference, or overwrites.\n"
              "import argparse, hashlib, json\nfrom pathlib import Path, PurePosixPath\n\n"
              + "FILES = " + repr(files) + "\n\nHASHES = " + repr(hashes_for(files)) + "\n\n"
              + inspect.getsource(install_files) + "\n\n"
              + "if __name__ == '__main__':\n"
              + "    parser = argparse.ArgumentParser(description='Install SZL source into a NEW directory only')\n"
              + "    parser.add_argument('--directory', default='szl-local-model')\n"
              + "    args = parser.parse_args()\n"
              + "    print(install_files(args.directory, FILES, HASHES))\n")
    # Tildes keep embedded Markdown backticks in source strings from closing the fence.
    return ("# SZL Local Model Lab — one Python source-installation payload\n\n"
            "This installs the source package only. It is not proof of model task success, "
            "training, provider publication or production deployment. See RESULTS_2026-09-29.md.\n\n"
            "Save the Python block as install_szl.py, inspect it, then run "
            "`python -B install_szl.py --directory szl-local-model` in an existing parent directory. "
            "It refuses overwrites and checks all embedded source hashes before writing. "
            "No credentials, weights, logs, or unrelated repositories are embedded.\n\n"
            "~~~~python\n" + source + "~~~~\n")


def build_release(root=ROOT):
    files = collect(root)
    release = Path(root) / "releases" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                                         + "-" + uuid.uuid4().hex[:8])
    release.mkdir(parents=True)
    products = {"SZL_LOCAL_MODEL_SOURCE.zip": zip_bytes(files),
                "SZL_LOCAL_MODEL_PYTHON_PAYLOAD.md": payload(files).encode("utf-8")}
    for name, data in products.items():
        with (release / name).open("xb") as stream:
            stream.write(data)
    receipt = {"schema": "szl.source-release/v1", "files": hashes_for(files),
               "artifacts": {name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
                             for name, data in products.items()},
               "runtime_verified": False, "note": "Source packaging does not close the final model smoke gate."}
    with (release / "BUILD_MANIFEST.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
    return release


if __name__ == "__main__":
    print(build_release())
