#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Step 1 - qualify the runtime before any compute. Writes out/runtime_qualification.json.

PASS means the environment is permitted to attempt a run; it is not evidence that anything trained.
The first PASS freezes `runtime_lock` in candidate.json; later runs must match it or fail with RUNTIME_DRIFT
(pass --relock to deliberately re-freeze after a documented environment change).
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import candidate_lib as lib  # noqa: E402

HEX40 = 40


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-cpu", action="store_true", help="permit a CPU-only contract smoke (tiny bases only)")
    ap.add_argument("--relock", action="store_true", help="re-freeze runtime_lock to the current environment")
    ap.add_argument("--offline", action="store_true", help="skip the Hub reachability probe")
    args = ap.parse_args()
    cand = lib.load_candidate()
    checks = []

    def check(name, ok, detail):
        checks.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})
        print(f"[qualify] {'PASS' if ok else 'FAIL'} {name}: {detail}")
        return ok

    ok = True
    v = lib.installed_versions()
    ok &= check("python", sys.version_info >= (3, 11), v["python"])
    ok &= check("torch_importable", v.get("torch") is not None, str(v.get("torch")))
    ok &= check("transformers_importable", v.get("transformers") is not None, str(v.get("transformers")))
    ok &= check("peft_importable", v.get("peft") is not None, str(v.get("peft")))
    gpu = lib.gpu_snapshot()
    need = float(cand["training_recipe"].get("minimum_free_gpu_gib", 4.0))
    if gpu["cuda_available"]:
        ok &= check("gpu_free_memory", float(gpu["free_gib"]) >= need, f"{gpu['device']} free {gpu['free_gib']} GiB (need {need})")
    else:
        ok &= check("cuda", args.allow_cpu, "CUDA unavailable" + (" (cpu allowed for contract smoke)" if args.allow_cpu else ""))
    free_gib = shutil.disk_usage(str(lib.OUT.parent)).free / 2**30
    ok &= check("disk_free", free_gib >= 5.0, f"{free_gib:.1f} GiB free under {lib.OUT.parent}")

    base = cand["actual_training_base"]
    rev = base.get("revision")
    ok &= check("base_revision_pinned", isinstance(rev, str) and len(rev) == HEX40,
                f"{base['repo_id']}@{rev}" if rev else f"{base['repo_id']} has no 40-hex revision pin")
    if not args.offline:
        try:
            from huggingface_hub import HfApi
            info = HfApi().model_info(base["repo_id"], revision=rev)
            ok &= check("base_reachable", True, f"sha {info.sha}")
        except Exception as e:
            ok &= check("base_reachable", False, f"{type(e).__name__}: {str(e)[:200]}")

    lock = {k: v.get(k) for k in lib.LOCK_PACKAGES}
    lock["python"] = v["python"]
    frozen = cand.get("runtime_lock") or {}
    if (not frozen or args.relock) and ok:
        cand["runtime_lock"] = lock
        lib.save_candidate(cand)
        check("runtime_lock", True, "frozen to the current environment" + (" (relock)" if args.relock else ""))
    elif not frozen:
        check("runtime_lock", False, "not frozen: qualification failed, nothing to lock")
    else:
        drift = {k: (frozen.get(k), lock.get(k)) for k in lock if frozen.get(k) != lock.get(k)}
        ok &= check("runtime_lock", not drift, "matches frozen lock" if not drift else f"RUNTIME_DRIFT {drift}")

    report = {"kind": "szl.frontier-runtime-qualification/v1", "generated_utc": lib.now_utc(),
              "candidate_id": cand["candidate_id"], "verdict": "PASS" if ok else "FAIL", "checks": checks,
              "gpu": gpu, "versions": v, "host": lib.host_info(),
              "meaning": "PASS permits an attempt. It is not evidence that training or evaluation happened."}
    lib.write_json(lib.OUT / "runtime_qualification.json", lib.stringify_floats(report))
    print(f"[qualify] verdict {report['verdict']} -> {lib.OUT / 'runtime_qualification.json'}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
