"""proofrun: proof-carrying training runs (Rikuy, SZL Frontier Benchmarking, 2026-11-19).

Every claim a training run makes (steps trained, final adapter, eval scores, image and
environment) must be backed by a signed, hash-chained receipt. `proofrun verify <dir>`
re-derives every hash it can and fails closed on anything missing, unsigned or inconsistent.
LOCAL PROTOTYPE: ephemeral test keys only; no network, no secrets.
"""
__version__ = "0.1.0"
SCHEMA = "proofrun/v0.1"
