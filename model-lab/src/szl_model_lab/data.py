"""Strict, bounded, split-aware tabular datasets; no text scraping or data synthesis.

Feature values must be explicitly normalized to [0,1] under a documented recipe.
Missing values are errors, never invented zero-cost or mean-quality observations.
Group and case identifiers are operator assertions, not independent provenance.
"""
from __future__ import annotations
from .safeio import strict_json
import hashlib
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from .catalog import track_for

MAX_DATA_BYTES = 10 * 1024 * 1024
MAX_ROWS = 10000

class Provenance(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    source_id: str = Field(min_length=1, max_length=128)
    rights_basis: Literal["owner-created", "licensed", "consented", "test-fixture"]
    synthetic: bool
    feature_time: str
    outcome_time: str
    normalization_id: str = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def timestamps(self) -> "Provenance":
        a = datetime.fromisoformat(self.feature_time.replace("Z", "+00:00"))
        b = datetime.fromisoformat(self.outcome_time.replace("Z", "+00:00"))
        if a.tzinfo is None or b.tzinfo is None or a >= b:
            raise ValueError("features_must_precede_outcome_with_timezone")
        return self

class Example(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    example_id: str = Field(min_length=1, max_length=128)
    group_id: str = Field(min_length=1, max_length=128)
    case_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    split: Literal["train", "validation", "test"]
    features: dict[str, float]
    label: bool
    provenance: Provenance

    @field_validator("features", mode="before")
    @classmethod
    def numeric_features(cls, value: object) -> object:
        import math
        if not isinstance(value, dict) or not value or len(value) > 32:
            raise ValueError("invalid_feature_mapping")
        if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1
               for v in value.values()):
            raise ValueError("features_require_finite_numeric_unit_interval")
        return value

class Dataset:
    def __init__(self, raw: bytes, track: str):
        if not raw or len(raw) > MAX_DATA_BYTES:
            raise ValueError("dataset_size_out_of_bounds")
        self.track = track_for(track)
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.rows: list[Example] = []
        examples: set[str] = set()
        groups: dict[str, str] = {}
        cases: dict[str, str] = {}
        normalized: set[str] = set()
        for line in raw.decode("utf-8").splitlines():
            if not line.strip():
                raise ValueError("blank_dataset_line")
            if len(self.rows) >= MAX_ROWS:
                raise ValueError("too_many_dataset_rows")
            row = Example.model_validate(strict_json(line))
            if set(row.features) != set(self.track.features):
                raise ValueError("feature_schema_mismatch")
            if row.example_id in examples:
                raise ValueError("duplicate_example_id")
            examples.add(row.example_id)
            for mapping, key in ((groups, row.group_id), (cases, row.case_sha256)):
                if key in mapping and mapping[key] != row.split:
                    raise ValueError("cross_split_leakage")
                mapping[key] = row.split
            normalized.add(row.provenance.normalization_id)
            self.rows.append(row)
        if len(normalized) != 1:
            raise ValueError("mixed_normalization_contracts")
        self.normalization_id = next(iter(normalized))
        self.counts = dict(Counter(r.split for r in self.rows))
        if set(self.counts) != {"train", "validation", "test"}:
            raise ValueError("three_nonempty_explicit_splits_required")
        for split in ("train", "validation", "test"):
            if {r.label for r in self.rows if r.split == split} != {False, True}:
                raise ValueError("each_split_requires_both_label_classes")

    @classmethod
    def read(cls, path: Path, track: str) -> "Dataset":
        from .safeio import read_regular
        return cls(read_regular(path, MAX_DATA_BYTES), track)

    def select(self, split: str) -> list[Example]:
        if split not in ("train", "validation", "test"):
            raise ValueError("invalid_split")
        return [r for r in self.rows if r.split == split]

    def summary(self) -> dict:
        return {"dataset_sha256": self.sha256, "split_counts": self.counts,
                "normalization_id": self.normalization_id,
                "all_synthetic": all(r.provenance.synthetic for r in self.rows),
                "rights_bases": sorted({r.provenance.rights_basis for r in self.rows}),
                "provenance": "OPERATOR_ASSERTED_NOT_INDEPENDENTLY_VERIFIED"}
