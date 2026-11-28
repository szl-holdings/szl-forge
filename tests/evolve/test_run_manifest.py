import json, os, copy
from szl_evolve import run_manifest as rm
F = os.path.join(os.path.dirname(__file__), "fixtures")
load = lambda n: json.load(open(os.path.join(F, n)))
codes = lambda r: {f["code"] for f in r["findings"] if f["severity"] == "ERROR"}

def test_good_manifest_valid_but_not_launched():
    r = rm.validate(load("manifest_good.json"))
    assert r["verdict"] == "VALID_NOT_LAUNCHED" and r["errors"] == 0

def test_w30_reconstructed_is_invalid():
    r = rm.validate(load("manifest_w30_reconstructed.json"))
    assert r["verdict"] == "INVALID"
    assert {"UNPINNED_REVISION", "NO_COST_CAP", "PUBLIC_OUTPUT", "AUTO_PUSH", "CONTAMINATION_GATE"} <= codes(r)
    assert any(f["code"] == "NOT_RECORDED_AT_LAUNCH" for f in r["findings"])

def test_salad_style_truncation_and_license_laundering_and_public():
    r = rm.validate(load("manifest_salad_style.json"))
    assert {"SILENT_TRUNCATION", "LICENSE_LAUNDERING", "PUBLIC_OUTPUT"} <= codes(r)

def test_truncation_allowed_when_reason_given():
    m = load("manifest_salad_style.json"); m["datasets"][0]["truncation_reason"] = "max_rows=2000 cap, approved"
    m["derived_license"] = "other"; m["output"]["visibility"] = "private"
    assert "SILENT_TRUNCATION" not in codes(rm.validate(m))

def test_short_sha_and_branch_rejected():
    m = load("manifest_good.json"); m["code"]["commit"] = "main"; m["base_model"]["revision"] = "abc1234"
    assert "UNPINNED_REVISION" in codes(rm.validate(m))

def test_zero_or_negative_budget_rejected():
    m = load("manifest_good.json")
    for v in (0, -5, True, None):
        m2 = copy.deepcopy(m); m2["budget"]["max_usd"] = v
        assert "NO_COST_CAP" in codes(rm.validate(m2))

def test_no_holdout_and_overwrite_flagged():
    m = load("manifest_good.json"); m["datasets"] = m["datasets"][:1]; m["output"]["overwrite_existing"] = True
    assert {"NO_HOLDOUT", "OVERWRITE"} <= codes(rm.validate(m))

def test_secret_in_manifest_flagged_without_echo():
    m = load("manifest_good.json"); m["hardware"]["env"] = "hf_" + "A" * 34
    r = rm.validate(m)
    assert "SECRET_IN_MANIFEST" in codes(r) and ("A" * 34) not in json.dumps(r)

def test_gate_verdict_must_be_clean():
    m = load("manifest_good.json"); m["contamination"]["verdict"] = "REJECT"
    assert "CONTAMINATION_GATE" in codes(rm.validate(m))

def test_wrong_schema_and_empty():
    assert rm.validate({})["verdict"] == "INVALID"
