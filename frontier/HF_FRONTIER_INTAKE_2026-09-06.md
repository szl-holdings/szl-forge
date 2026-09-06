# Hugging Face Frontier Intake — 2026-09-06

Status: **REPORTED_UPSTREAM / NOT YET SZL-QUALIFIED**

This intake records material Hugging Face releases worth evaluating against SZL Forge, SZL Nemo, A11oy, retrieval, speech, multimodal, and deployment lanes. Upstream claims remain `REPORTED` until reproduced on SZL-controlled hardware and bound to receipts. Nothing in this file promotes a model or grants execution authority.

## P0 — Evaluate now

### 1. Z.ai GLM-5.3-Flash
Primary source: https://huggingface.co/zai-org/GLM-5.3-Flash

Why it matters:
- Natively multimodal image+text model.
- 320B total / 18B active parameters.
- Upstream model card documents vLLM, SGLang, Transformers, KTransformers, Unsloth and Docker deployment paths.
- Candidate for replacing or challenging current proposal-model lanes without changing Nemo/A11oy authority boundaries.

SZL gate:
1. Pin exact Hub revision, tokenizer, template, runtime and quantization.
2. Run proposal-only agent/coding bakeoff against current Forge baselines.
3. Add image-document and screenshot tasks.
4. Measure latency, VRAM, throughput, structured-output compliance and refusal behavior.
5. Persist only `MEASURED` results and do not inherit upstream benchmark labels as SZL evidence.

### 2. Funes — owned coding-agent memory
Primary source: https://huggingface.co/blog/funes

Why it matters:
- Treats agent memory as a user-owned Hugging Face dataset instead of a hosted black-box service.
- Uses local indexing plus Hub distribution/versioning and secret scanning.
- Directly relevant to Forge agent continuity and Second Brain handoff, while preserving the rule that memory is context/evidence rather than execution authority.

SZL gate:
1. Prototype against a private SZL memory dataset only.
2. Verify redaction behavior with synthetic secrets before any real trace ingestion.
3. Store retrieval provenance and content digests in Nemo envelopes.
4. Confirm no private chain-of-thought requirement; persist only allowed summaries/traces.
5. Keep A11oy as the sole consequential-action admission layer.

### 3. NeoMME multimodal retrieval
Primary source: https://huggingface.co/blog/Hcompany/neomme

Why it matters:
- 260M and 800M multilingual multimodal encoders.
- Designed for visual-document retrieval using page images and supports dense plus late-interaction embeddings.
- Strong fit for evidence-heavy PDF, chart, map, scan and mixed-layout retrieval in PRISM, Terra, Vessels and A11oy.

SZL gate:
1. Build a fixed visual-document retrieval corpus with answerable ground truth.
2. Compare NeoMME against the current text-only embedding lane and one established visual-document baseline.
3. Measure nDCG@10, recall@k, index size, page throughput and retrieval latency.
4. Bind retrieved page/image hashes into the evidence receipt.
5. No production replacement unless it wins a measured Pareto gate.

### 4. DeepSeek-V4-Flash-Vision-Exp
Primary source: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp

Why it matters:
- Experimental multimodal DeepSeek V4 derivative for vision+agent workflows.
- Candidate challenger for screenshots, OCR-heavy reasoning and multimodal agent tasks.

SZL gate:
1. Treat as experimental; no default-route promotion.
2. Pin exact revision and serving parameters.
3. Run the same multimodal bakeoff used for GLM-5.3-Flash.
4. Add OCR small-text, table, chart and UI grounding cases.
5. Require Nemo PRE/POST generation envelopes before any A11oy tool admission.

## P1 — Integrate into the evaluation matrix

### 5. Hugging Face `@huggingface/kernels` WebGPU collection
Primary source: https://huggingface.co/blog/webgpu-kernels

Why it matters:
- Hugging Face released a Hub-backed WebGPU kernel library with 207 versioned kernels plus correctness and benchmark artifacts.
- Relevant to browser-local inference and acceleration for lightweight SZL front ends.

SZL gate:
- Benchmark only deterministic, non-authoritative browser workloads first: embeddings, preprocessing, lightweight vision and local transforms.
- Record browser/GPU/kernel revision and correctness evidence.
- No browser kernel may bypass server-side authorization, Nemo witnessing or A11oy admission.

### 6. Granite Speech 5.0 Turbo CTC
Primary source: https://huggingface.co/blog/ibm-granite/granite-speech-5-0-470m-turboctc

Why it matters:
- Compact 470M English ASR models with an Apache-2.0 commercial lane.
- Upstream reports very high batched H200 transcription throughput while retaining competitive public-benchmark WER.
- Candidate for low-latency transcription before higher-level reasoning.

SZL gate:
- Evaluate the Apache-2.0 checkpoint only for commercial default consideration.
- Measure streaming latency, WER on SZL-style audio, punctuation/numbers, far-field robustness and CPU/GPU cost.
- Never treat transcript confidence as evidence truth without source-audio binding.

### 7. Open Yap 1K
Primary source: https://huggingface.co/blog/TheAgenticDataCompany/open-yap-1k

Why it matters:
- 1,000 hours of dual-channel full-duplex English conversation released for commercial and research use.
- Relevant to interruption, overlap, turn-taking and natural voice-agent training/evaluation.

SZL gate:
- Verify dataset license and consent documentation at exact revision.
- Use initially as evaluation/behavioral research data, not automatic training input.
- Add overlap/interruption/latency metrics before any speech post-training proposal.

## P2 — Research watch, do not absorb yet

### Puffin-World
Primary source: https://huggingface.co/blog/KangLiao/puffin-world

Potential value: physically grounded multimodal/3D world-state research for future spatial intelligence in Terra, Vessels, Aegis and robotics. Keep in research watch until reproducibility, compute fit and use-case gates are clearer.

### LoongForge TAOT
Primary source: https://huggingface.co/blog/nullnonenilNULL/loongforge-taot

Potential value: topology-aware MoE expert-replica planning and communication optimization. Upstream reports training-speed gains, but this is community-reported infrastructure and must not enter the production training path without independent reproduction on SZL hardware.

### VLM Run Gateway
Primary source: https://huggingface.co/blog/vlm-run/introducing-gateway

Potential value: unified serving API for OCR/VLM/vision models. Useful as a comparison point for model-routing ergonomics, but not a core dependency unless it beats SZL-controlled vLLM/SGLang paths on reproducibility, model identity, parameter transparency and evidence capture.

## Qualification order

1. **GLM-5.3-Flash** — capability ceiling and multimodal agent bakeoff.
2. **Funes** — owned agent-memory prototype with strict redaction/evidence boundaries.
3. **NeoMME** — visual-document retrieval benchmark.
4. **DeepSeek-V4-Flash-Vision-Exp** — experimental multimodal challenger.
5. **Granite Speech 5.0 Turbo CTC + Open Yap 1K** — speech runtime and full-duplex evaluation lane.
6. **WebGPU kernels** — browser-local acceleration experiments.
7. **Puffin-World / LoongForge / VLM Run Gateway** — research watch only.

## Non-negotiable SZL boundary

Every external release starts as `REPORTED_UPSTREAM`. Promotion requires exact-source pinning, license review, deterministic evaluation inputs, measured outputs, hardware/runtime identity, receipt binding, and explicit limitations. Model capability never grants tool authority; Nemo witnesses bounded inference envelopes and A11oy remains the consequential-action admission layer.
