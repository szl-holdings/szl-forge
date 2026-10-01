import json
import pytest
from szl_model_lab.data import Dataset

def parse(rows):
    return Dataset(("\n".join(json.dumps(r) for r in rows) + "\n").encode(), "router")

def test_valid_counts(dataset):
    assert dataset.counts == {"train": 8, "validation": 8, "test": 8}
    assert dataset.summary()["all_synthetic"] is True

def test_groups_do_not_cross_splits(rows):
    rows[8]["group_id"] = rows[0]["group_id"]
    with pytest.raises(ValueError, match="cross_split_leakage"):
        parse(rows)

def test_case_hashes_do_not_cross_splits(rows):
    rows[8]["case_sha256"] = rows[0]["case_sha256"]
    with pytest.raises(ValueError, match="cross_split_leakage"):
        parse(rows)

def test_duplicate_ids_rejected(rows):
    rows[1]["example_id"] = rows[0]["example_id"]
    with pytest.raises(ValueError, match="duplicate_example_id"):
        parse(rows)

@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.01, 1.1, True, "0.5", None])
def test_bad_features(rows, value):
    rows[0]["features"]["input_load"] = value
    with pytest.raises(ValueError):
        parse(rows)

def test_unknown_feature(rows):
    rows[0]["features"]["implied_trust"] = 0.9
    with pytest.raises(ValueError, match="feature_schema_mismatch"):
        parse(rows)

def test_missing_feature(rows):
    del rows[0]["features"]["unit_cost"]
    with pytest.raises(ValueError, match="feature_schema_mismatch"):
        parse(rows)

def test_unknown_is_not_zero(rows):
    rows[0]["features"]["unit_cost"] = None
    with pytest.raises(ValueError):
        parse(rows)

def test_normalization_must_match(rows):
    rows[0]["provenance"]["normalization_id"] = "different"
    with pytest.raises(ValueError, match="mixed_normalization"):
        parse(rows)

@pytest.mark.parametrize("when", ["2026-09-02T00:00:00Z", "2026-09-01T00:00:00", "bad-date"])
def test_temporal_leakage(rows, when):
    rows[0]["provenance"]["feature_time"] = when
    with pytest.raises(ValueError):
        parse(rows)

def test_no_missing_split(rows):
    with pytest.raises(ValueError, match="three_nonempty"):
        parse(rows[:16])

def test_no_single_label_split(rows):
    for r in rows[:8]:
        r["label"] = False
    with pytest.raises(ValueError, match="both_label"):
        parse(rows)

def test_extra_fields_rejected(rows):
    rows[0]["execute"] = "anything"
    with pytest.raises(ValueError):
        parse(rows)

def test_blank_line_rejected(raw):
    with pytest.raises(ValueError, match="blank_dataset"):
        Dataset(raw + b"\n", "router")
