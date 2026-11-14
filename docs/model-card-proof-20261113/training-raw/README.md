# Immutable training-state inputs

These four public trainer-state records are retained byte-for-byte so the offline renderer can verify the plotted projection on every invocation. These are owner-reported training records, not a new evaluation, verified training execution or current-weight qualification.

| Model | Immutable input | Raw SHA-256 |
|---|---|---|
| SZLHOLDINGS/szl-triage-qwen3.5-0.8b-lora-study5 | [checkpoint-396/trainer_state.json](https://huggingface.co/SZLHOLDINGS/szl-triage-qwen3.5-0.8b-lora-study5/raw/1b52484634c767ae8a8fab40e484c3f8f35d0b00/checkpoint-396/trainer_state.json) | `dc3c265d6129fb03a64f9bd77ced1864af542db511a7b7b91ca47c4baaed2a78` |
| SZLHOLDINGS/abstention-v2-candidate | [checkpoint-200/trainer_state.json](https://huggingface.co/SZLHOLDINGS/abstention-v2-candidate/raw/1005f79ffc3e78831d6d5aa71f8b99808a6e560d/checkpoint-200/trainer_state.json) | `e87f0ca9b54e0c6ce4e3003466aec65c73d265f3c6bdf1e4357d4371c1407c4a` |
| SZLHOLDINGS/physics-master-v1-candidate | [checkpoint-200/trainer_state.json](https://huggingface.co/SZLHOLDINGS/physics-master-v1-candidate/raw/a6697a425f85304b6bfa5b27a1aa4d825acc156f/checkpoint-200/trainer_state.json) | `6cc62c7129fd4b4465052c096463d88d63c9e2ce5ea84f4b8b899dcbdcc44af7` |
| SZLHOLDINGS/quant-master-v1-candidate | [checkpoint-200/trainer_state.json](https://huggingface.co/SZLHOLDINGS/quant-master-v1-candidate/raw/abd79ed091abbf58cb2be896a9422d582152161e/checkpoint-200/trainer_state.json) | `49407815d8cd41eeddf1203e7edd81ef99827c7bae023ac9f5d3a843deba9011` |

The renderer uses digest-named files only, refuses links and non-regular or oversized files, verifies raw SHA-256, parses strict JSON, verifies checkpoint identity and derives every loss/step pair from log_history. Missing, changed, reordered or invented observations fail before plotting or creating output.

The expected digests are reviewed source inputs, not an independent signature trust root. Changing both the raw record and its reviewed manifest still requires normal source review.
