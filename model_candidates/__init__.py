"""Training-ready research modules; no serving, scheduling or release authority."""

from .specs import candidate_spec, candidate_specs, encode_features

__all__ = ["candidate_spec", "candidate_specs", "encode_features"]
