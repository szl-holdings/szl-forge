"""Opt-in live Docker probes using fixed trusted fixtures, never model code on host.
SPDX-License-Identifier: Apache-2.0
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

import lab


BOUNDARY = '''import errno
import os
import pathlib
import socket

def cannot_write(path):
    try:
        with open(path, "x") as stream:
            stream.write("probe")
    except OSError as exc:
        return exc.errno in (errno.EROFS, errno.EACCES, errno.EPERM)
    return False

def solve(value):
    status = pathlib.Path("/proc/self/status").read_text()
    fields = dict(line.split(":", 1) for line in status.splitlines() if ":" in line)
    memory = pathlib.Path("/sys/fs/cgroup/memory.max").read_text().strip()
    pids = pathlib.Path("/sys/fs/cgroup/pids.max").read_text().strip()
    cpu = pathlib.Path("/sys/fs/cgroup/cpu.max").read_text().split()
    return {
        "non_root": os.getuid() == 65534,
        "capabilities_dropped": int(fields["CapEff"].strip(), 16) == 0,
        "no_new_privileges": fields["NoNewPrivs"].strip() == "1",
        "only_loopback": [name for _, name in socket.if_nameindex()] == ["lo"],
        "root_readonly": cannot_write("/szl-readonly-probe"),
        "candidate_readonly": cannot_write("/candidate/szl-readonly-probe"),
        "no_docker_socket": not pathlib.Path("/var/run/docker.sock").exists(),
        "memory_limit": memory == "268435456",
        "pid_limit": pids == "32",
        "cpu_limit": int(cpu[0]) == int(cpu[1]),
        "proxy_values_empty": all(not os.environ.get(key) for key in (
            "HTTP_PROXY", "HTTPS_PROXY", "FTP_PROXY", "ALL_PROXY", "NO_PROXY",
            "http_proxy", "https_proxy", "ftp_proxy", "all_proxy", "no_proxy")),
    }
'''


def main():
    lab.preflight_docker()
    folder = lab.ROOT / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                                  + "-sandbox-" + uuid.uuid4().hex[:8])
    folder.mkdir(parents=True)
    report = {"schema": "szl.sandbox-live-probes/v1", "started_at": lab.now(),
              "harness_sha256": lab.sha(Path(lab.__file__).read_bytes()),
              "image": lab.IMAGE, "scope": "Fixed development probes, not a container-escape security proof.",
              "results": []}
    fixtures = [
        ("restrictions", BOUNDARY, 25),
        ("timeout", "def solve(value):\n    while True:\n        pass\n", 3),
        ("output_limit", "print('x' * 200000)\ndef solve(value):\n    return 42\n", 25),
        ("false_success", "import sys\nsys.exit(0)\ndef solve(value):\n    return 42\n", 25),
    ]
    for name, source, timeout in fixtures:
        candidate = folder / name
        candidate.mkdir()
        # Fixed trusted probe sources, not host execution of generated candidates.
        with (candidate / "solution.py").open("xb") as stream:
            stream.write(source.encode())
        with (candidate / "worker.py").open("xb") as stream:
            stream.write((lab.ROOT / "worker.py").read_bytes())
        try:
            result = lab.sandbox(candidate, [None], timeout=timeout)
            passed = False
            if name == "restrictions" and result["failure"] is None and result["exit_code"] == 0:
                observed = json.loads(result["stdout"])
                passed = (len(observed) == 1 and len(observed[0]) == 11
                          and all(value is True for value in observed[0].values()))
            elif name == "timeout":
                passed = result["failure"] == "TIMEOUT"
            elif name == "output_limit":
                passed = result["failure"] == "OUTPUT_LIMIT"
            elif name == "false_success":
                score = lab.evaluate(result, [{"input": None, "expected": 42}])
                passed = result["exit_code"] == 0 and score["reason"] == "INVALID_OUTPUT"
            passed = passed and result["cleanup_confirmed"]
            report["results"].append({"name": name, "passed": passed, "execution": result})
            print(json.dumps({"probe": name, "passed": passed}), flush=True)
            if not result["cleanup_confirmed"]:
                break
        except Exception as exc:
            report["results"].append({"name": name, "passed": False,
                                      "error": {"type": type(exc).__name__, "message": str(exc)[:500]}})
            break
    report["passed"] = len(report["results"]) == len(fixtures) and all(r["passed"] for r in report["results"])
    report["completed_at"] = lab.now()
    lab.save(folder / "sandbox-receipt.json", report)
    print(json.dumps({"receipt": str(folder / "sandbox-receipt.json"), "passed": report["passed"]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
