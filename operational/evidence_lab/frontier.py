"""SZL local retrieval release: inspect, download, evaluate, serve and verify."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "operational"))
sys.path.insert(0, str(ROOT / "risk_kernel"))


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify_source():
    manifest = json.loads((ROOT / "source-manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        path = ROOT / name
        if path.is_symlink() or not path.is_file() or sha(path) != expected:
            raise ValueError("Source integrity mismatch: " + name)
    return manifest


def runtime_run(args):
    from external_retrieval_lab import paths
    return paths(args.lab_root, args.run_id)


def doctor(args):
    from external_retrieval_lab import load_frozen
    from external_retrieval_preview import verify_runtime
    from retrieval_risk_kernel import validate_completed_run
    root, run = runtime_run(args)
    packages = verify_runtime(run)
    _, _, _, payload = load_frozen(root, args.run_id)
    bound = validate_completed_run(run)
    return {"status": "LOCAL_ARTIFACTS_VERIFIED", "run_id": args.run_id,
            "documents": len(payload["documents"]), "packages": packages,
            "artifacts": bound["artifacts"], "models_loaded": False}


def client(args):
    from retrieval_risk_kernel import PreviewClient
    return PreviewClient(f"http://127.0.0.1:{args.port}", 120)


def verify_live(args):
    from retrieval_risk_kernel import artifact_hashes, verify_status, validate_response
    _, run = runtime_run(args)
    if args.output is None:
        raise ValueError("verify requires --output with a new receipt filename")
    if args.output.exists():
        raise FileExistsError(args.output)
    hashes = artifact_hashes(run)
    api = client(args)
    before = api.request("/api/status")
    verify_status(before, run, hashes, ROOT / "operational/external_retrieval_preview.py")
    source = before.get("runtime_source_sha256", {})
    required = {"preview", "external_retrieval_lab.py", "external_retrieval_core.py",
                "external_retrieval_reader.py", "external_retrieval_data.py",
                "unresolved_identifier_gate_v3.py", "retrieval_risk_kernel.py",
                "retrieval_risk_kernel_v2.py"}
    if set(source) != required or not before.get("capabilities", {}).get("identifier_guard"):
        raise ValueError("Live source set or guard capability is incomplete")
    for name, expected in source.items():
        directory = ROOT / ("risk_kernel" if name in {
            "unresolved_identifier_gate_v3.py", "retrieval_risk_kernel.py", "retrieval_risk_kernel_v2.py"}
            else "operational")
        path = directory / ("external_retrieval_preview.py" if name == "preview" else name)
        if sha(path) != expected:
            raise ValueError("Live runtime source mismatch: " + name)
    examples = before["examples"]
    positive, negative = examples[0], examples[1]
    observations = []
    for example, expected in ((positive, "ANSWER"), (negative, "ABSTAIN")):
        response = api.request("/api/answer-context", {
            "question": example["question"], "context": example["context"]})
        if response["status"] != expected or response["freeze_sha256"] != hashes["freeze.json"]:
            raise ValueError("Recorded passage replay failed")
        if expected == "ANSWER":
            evidence = response["evidence"]
            if example["context"][evidence["start"]:evidence["end"]] != response["answer"]:
                raise ValueError("Answer does not match the cited span")
        observations.append({"check": "recorded_passage_" + expected.lower(), "response": response})
    question = positive["question"]
    base = api.request("/api/query", {"question": question})
    guarded = api.request("/api/query-guarded", {"question": question})
    for response in (base, guarded):
        validate_response(response, before)
    if guarded["status"] != base["status"] or guarded["answer"] != base["answer"]:
        raise ValueError("Identifier guard changed the known positive replay")
    observations.extend([{"check": "corpus_baseline", "response": base},
                         {"check": "corpus_guarded", "response": guarded}])
    nonce = "Which city hosted SzlNqEvent0123456789abcdefAB for SzlNqSubjectfedcba9876543210AB?"
    blocked = api.request("/api/query-guarded", {"question": nonce})
    validate_response(blocked, before)
    if blocked["status"] != "ABSTAIN" or len(blocked["identifier_guard"]["unresolved_identifiers"]) != 2:
        raise ValueError("Guard failed its explicit unresolved-identifier contract")
    observations.append({"check": "unknown_identifiers", "response": blocked})
    searched = api.request("/api/search", {"question": question, "k": 5})
    if searched["status"] != "RETRIEVED" or len(searched["passages"]) != 5:
        raise ValueError("Search contract failed")
    observations.append({"check": "five_passages", "response": searched})
    after = api.request("/api/status")
    if after != before or artifact_hashes(run) != hashes:
        raise ValueError("Runtime or frozen artifacts changed during verification")
    result = {"status": "LIVE_LOCAL_CONTRACT_VERIFIED", "verified_at": datetime.now(timezone.utc).isoformat(),
              "scope": "wiring_and_recorded_replay_only", "checks": len(observations),
              "runtime": before, "artifacts": hashes, "observations": observations,
              "new_quality_evidence": False, "production_admitted": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    return {"status": result["status"], "checks": len(observations),
            "receipt": str(args.output.resolve()), "sha256": sha(args.output)}


def download(args):
    from external_retrieval_lab import CONFIG, QWEN_FILES, READER_FILES, verify_assets
    from huggingface_hub import snapshot_download
    # Public immutable downloads; exclude ambient credentials and executable model code.
    jobs = [
        ("Qwen/Qwen3-Embedding-0.6B", CONFIG["encoder_revision"], "model", "qwen", list(QWEN_FILES) + ["README.md"]),
        ("deepset/roberta-base-squad2", CONFIG["reader_revision"], "model", "reader", list(READER_FILES)),
        ("rajpurkar/squad_v2", "3ffb306f725f7d2ce8394bc1873b24868140c412", "dataset", "squad-v2", ["squad_v2/validation-00000-of-00001.parquet", "README.md"]),
        ("hotpotqa/hotpot_qa", "1908d6afbbead072334abe2965f91bd2709910ab", "dataset", "hotpot-qa", ["distractor/validation-00000-of-00001.parquet", "README.md"])]
    for repo, revision, kind, folder, files in jobs:
        snapshot_download(repo, revision=revision, repo_type=kind, token=False,
                          allow_patterns=files, local_dir=args.lab_root / "assets" / folder,
                          max_workers=2)
    verify_assets(args.lab_root.resolve(), args.lab_root.resolve() / "assets/qwen")
    return {"status": "PINNED_ASSETS_DOWNLOADED_AND_VERIFIED"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["doctor", "download", "test", "prepare", "evaluate", "preview", "query", "verify", "evaluate-risk"])
    parser.add_argument("--lab-root", type=Path, default=ROOT)
    parser.add_argument("--run-id", default="baseline-1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--question")
    parser.add_argument("--guarded", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("port must be 1024..65535")
    verify_source()
    if args.command == "doctor":
        result = doctor(args)
    elif args.command == "download":
        result = download(args)
    elif args.command == "verify":
        result = verify_live(args)
    elif args.command == "query":
        if not args.question:
            parser.error("query requires --question")
        result = client(args).request("/api/query-guarded" if args.guarded else "/api/query", {"question": args.question})
    elif args.command == "test":
        for folder, pattern in (("tests", "test_*.py"), ("operational/tests", "test_*.py"), ("risk_kernel/tests", "test_*.py")):
            subprocess.run([sys.executable, "-I", "-B", "-m", "unittest", "discover", "-s", str(ROOT / folder), "-p", pattern, "-v"], check=True, cwd=ROOT)
        result = {"status": "ALL_LOCAL_TEST_SUITES_PASSED"}
    else:
        if args.command == "evaluate-risk":
            if args.output is None:
                parser.error("evaluate-risk requires --output with a new run directory")
            _, run = runtime_run(args)
            kernel = "unresolved_identifier_gate_v3.py" if args.guarded else "retrieval_risk_kernel_v2.py"
            command = [sys.executable, "-I", "-B", str(ROOT / "risk_kernel" / kernel),
                "--run-dir", str(run),
                "--preview-source", str(ROOT / "operational/external_retrieval_preview.py"),
                "--output-dir", str(args.output.resolve()), "--base-url", f"http://127.0.0.1:{args.port}"]
            if args.guarded:
                command.append("--live-guard")
        elif args.command == "preview":
            command = [sys.executable, "-I", "-B", str(ROOT / "operational/external_retrieval_preview.py"),
                "--lab-root", str(args.lab_root.resolve()), "--run-id", args.run_id, "--port", str(args.port)]
        else:
            command = [sys.executable, "-I", "-B", str(ROOT / "operational/external_retrieval_lab.py"), args.command,
                "--lab-root", str(args.lab_root.resolve()), "--run-id", args.run_id]
            if args.command == "prepare":
                command += ["--encoder-path", str(args.lab_root.resolve() / "assets/qwen"), "--device", args.device]
        return subprocess.run(command, cwd=ROOT).returncode
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
