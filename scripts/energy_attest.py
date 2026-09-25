#!/usr/bin/env python3
"""Energy attestation for gate runs — v1.1 with plausibility clamp."""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "szl.energy-attest/v1"
SAMPLE_INTERVAL_S = 0.1
MAX_PLAUSIBLE_W = 250.0  # laptop GPU board-power ceiling; samples above are bad NVML reads


class EnergyAttest:
    def __init__(self, gate: str, model: str):
        self.gate = gate
        self.model = model
        self.samples = []
        self._nvml_ok = False

    def __enter__(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            pynvml.nvmlDeviceGetPowerUsage(self._handle)  # probe before claiming MEASURED
            self._nvml_ok = True
            self._stop = False
            import threading
            self._thread = threading.Thread(target=self._sample_loop, daemon=True)
            self._thread.start()
        except Exception:
            self._nvml_ok = False
        return self

    def _sample_loop(self):
        while not self._stop:
            try:
                mw = self._nvml.nvmlDeviceGetPowerUsage(self._handle)
                self.samples.append((time.monotonic(), mw / 1000.0))
            except Exception:
                self._nvml_ok = False
                return
            time.sleep(SAMPLE_INTERVAL_S)

    def __exit__(self, *exc):
        self._stop = True
        if self._nvml_ok:
            self._thread.join(timeout=2)
        try:
            self._nvml.nvmlShutdown()
        except Exception:
            pass
        return False

    def _joules(self, samples):
        if len(samples) < 2:
            return None
        total = 0.0
        for (t0, w0), (t1, w1) in zip(samples, samples[1:]):
            total += ((w0 + w1) / 2.0) * (t1 - t0)
        return round(total, 2)

    def receipt(self) -> dict:
        good = [(t, w) for (t, w) in self.samples if w <= MAX_PLAUSIBLE_W]
        rejected = len(self.samples) - len(good)
        j = self._joules(good)
        watts = [w for _, w in good]
        n = len(watts)
        return {
            "schema": SCHEMA,
            "gate": self.gate,
            "model": self.model,
            "energy_status": "MEASURED" if (j is not None and self._nvml_ok) else "UNAVAILABLE",
            "energy_j": j,
            "rejected_samples": rejected,
            "clamp_w": MAX_PLAUSIBLE_W,
            "sample_count": n,
            "mean_power_w": round(sum(watts) / n, 2) if n else None,
            "peak_power_w": round(max(watts), 2) if n else None,
            "sample_interval_s": SAMPLE_INTERVAL_S,
            "device": "gpu:0" if self._nvml_ok else None,
            "recorded_utc": datetime.now(timezone.utc).isoformat(),
            "signature_state": "UNSIGNED-honest",
            "doctrine": "v11; energy never fabricated; samples above clamp rejected as bad NVML reads",
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("command", nargs=argparse.REMAINDER, help="command after '--'")
    args = ap.parse_args()
    cmd = args.command[1:] if args.command and args.command[0] == "--" else args.command
    with EnergyAttest(gate=args.gate, model=args.model) as e:
        rc = subprocess.run(cmd).returncode if cmd else 0
    receipt = e.receipt()
    out = Path("frontier/evaluation/energy")
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out / f"{args.model.replace('/', '_')}_{args.gate}_{stamp}.json"
    path.write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))
    print(f"receipt: {path}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
