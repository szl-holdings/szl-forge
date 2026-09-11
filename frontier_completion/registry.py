"""Exact identities and owning workstreams; new model revisions resolve on scan.

These are existing candidate IDs to recapture, not claims of newness or permission.
Missing runtime/model pins must be resolved and reviewed, never invented.
"""
from .scan import Candidate

CANDIDATES = [
    Candidate("deepseek-v41-flash", "deepseek-ai/DeepSeek-V4.1-Flash", "model", "large-moe-multimodal"),
    Candidate("minicpm5-target", "openbmb/MiniCPM5-2B", "model", "edge-agent"),
    Candidate("minicpm5-gguf", "openbmb/MiniCPM5-2B-GGUF", "model", "edge-agent-quantization"),
    Candidate("minicpm5-dspark", "openbmb/MiniCPM5-2B-DSpark", "model", "speculative-decoding"),
    Candidate("minicpm5-dspark-gguf", "openbmb/MiniCPM5-2B-DSpark-GGUF", "model", "speculative-decoding"),
    Candidate("minicpm5-layout-community", "bartowski/MiniCPM5-2B-GGUF", "model", "per-tensor-canary"),
    Candidate("k2-mova-36b", "IFM/K2-Horizon-MoVA-36B-A4B", "model", "sparse-agent-comparison"),
    Candidate("granite-patchtst-r2", "ibm-granite/granite-timeseries-patchtst-fm-r2", "model", "lyte-forecast"),
    Candidate("indic-transcribe", "bodhan-ai/indic-transcribe-core", "model", "multilingual-asr", "HOLD"),
    Candidate("indic-speak", "bodhan-ai/indic-speak", "model", "multilingual-tts", "HOLD"),
    Candidate("amx-architecture", "gdiamos/amx-reasoning-v1-instruct", "model", "cpu-architecture", "WATCH"),
    Candidate("ultradata-agent", "openbmb/UltraData-SFT-Agent-2609", "dataset", "rights-reviewed-post-training"),
    Candidate("ultradata-rl", "openbmb/UltraData-RL-2609", "dataset", "rights-reviewed-post-training"),
]

RELEASES = [
    ("huggingface/trl", "v1.13.0", "3d9261f1fec9f9a8140099c78a65c7da73dce79c"),
    ("huggingface/huggingface_hub", "v1.31.0", "495b17c8529614759ae0f1ccf1ebe9a61c148b7c"),
    ("huggingface/tau", "v0.4.2", "55df51608b8b2d172c4bbac2cd11e8345e307476"),
]

OWNERS = {
    "szl-forge": "evaluation runners, training and experimental evidence",
    "szl-frontier": "canonical release intake and materiality, existing HF projections",
    "szl-nemo": "independent qualification/policy witness; inspect existing seam",
    "szl-serve": "runtime recipes and serving boundaries; inspect existing seam",
    "szl-router": "policy-aware route selection; inspect existing seam",
    "lyte-services": "Forecast Loom, Granite adapter and risk windows",
    "a11oy": "product, consequential approval and canonical vertical publisher",
    "a11oy-net": "independent static proof-pointer origin",
    ".github": "org governance, reusable controllers and publisher closure",
}
