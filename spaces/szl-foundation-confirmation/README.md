---
title: Foundation Confirmation Workbench
emoji: 🧭
colorFrom: yellow
colorTo: gray
sdk: docker
app_port: 7860
license: apache-2.0
short_description: Real CPU model trials with exportable receipts.
suggested_hardware: cpu-basic
startup_duration_timeout: 30m
tags:
  - cpu
  - research
  - bounded-inference
  - provenance
---

# Foundation Confirmation Workbench

A Python service runs the three original trained selector models from the sealed Foundation Confirmation v0.4 research release. These are 26,792-parameter selectors for a synthetic experiment; this service is not a general language model or a demonstration of AGI.

The canonical source is [szl-holdings/szl-forge](https://github.com/szl-holdings/szl-forge/tree/main/spaces/szl-foundation-confirmation). Publication binds the exact protected GitHub revision to the Hub commit and the observed runtime variable. Health and trial receipts do not constitute a release signature or independent scientific approval.

At startup the service verifies the archive SHA-256, the complete immutable manifest, the recorded benchmark, and actual inference for each of model seeds 17, 23 and 41. The fixed benchmark result remains **FAILED**: the clean-family utility regression exceeds the registered threshold. New trials are exploratory and do not alter that registered result.

The public CPU service admits structured requests only, with a 4KiB body limit, a 10-second body deadline, one active trial, no waiting queue, and a shared 12-trials-per-minute limit with a burst of 2. Trial seed and world index are uint32 integers. Users cannot submit code, paths, arbitrary data, model locations or training jobs.

Receipts are bounded and ephemeral: at most 128, each at most 64KiB, and retained for at most 24 hours while this process survives. Restart or idle sleep can discard them. Export the JSON immediately to retain it. The original local workbench retains its own existing trial files separately.

CPU Basic is an on-demand service and may sleep when idle. It is not an always-on availability promise. No GPU, paid hardware upgrade or automatic training is enabled by this build.

[Read the frozen experiment and replay evidence](https://a11oy.net/experiments/confirmation/) · [A11oy research entry](https://a-11-oy.com/research/confirmation)

Archive SHA-256: `869e318dd5f328205dd181ee836ef267bd2ae278f6430a9e8ddc661fbc689d03`.

Dependencies are locked with package hashes, including the official PyTorch CPU wheels. The Linux container uses Python 3.12 and runs as UID 1000 with the admitted release and source owned by root and read-only. The adapter wraps the unchanged sealed runtime; it does not expose its loopback HTTP server.
