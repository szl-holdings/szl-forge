import pytest
pytest.importorskip("yaml")
from szl_evolve import card_lint as cl

GOOD = """---
license: apache-2.0
base_model: Qwen/Qwen3.5-0.8B
datasets: [SZLHOLDINGS/x]
tags: [sft, lora]
---
# M
## Intended use
Internal. Data revision a1b2c3d4e5f6.
## Limitations
n=6 probes, 1/6 abstained.
## Bias and risk
None known beyond training data.
## Training data provenance
x@a1b2c3d4e5f6
"""
def codes(r, sev=None): return {f["code"] for f in r["findings"] if sev is None or f["severity"] == sev}

def test_good_card_no_errors():
    r = cl.lint(GOOD, "model", "x-sft"); assert r["errors"] == 0, r

def test_missing_front_matter_and_license():
    r = cl.lint("# no front matter\n", "model"); assert {"CL001", "CL002"} <= codes(r, "ERROR")

def test_license_other_without_name():
    assert "CL003" in codes(cl.lint("---\nlicense: other\n---\nx", "model"))

def test_adapter_without_base_model():
    assert "CL004" in codes(cl.lint("---\nlicense: apache-2.0\ntags: [lora]\n---\n# x\n", "model", "x-lora"))

def test_license_mismatch_with_qwen_research_base():
    t = GOOD.replace("Qwen/Qwen3.5-0.8B", "Qwen/Qwen2.5-Coder-3B")
    assert "CL005" in codes(cl.lint(t, "model", "code-master"), "ERROR")

def test_nvidia_base_without_notice_warns():
    t = GOOD.replace("Qwen/Qwen3.5-0.8B", "nvidia/Nemotron-Mini-4B-Instruct")
    assert "CL006" in codes(cl.lint(t, "model"))
    assert "CL006" not in codes(cl.lint(t + "\nNVIDIA Open Model License NOTICE", "model"))

def test_unsupported_claims_and_metric_without_n():
    t = GOOD.replace("n=6 probes, 1/6 abstained.", "Independently evaluated. every pin immutable. accuracy 0.93") 
    c = codes(cl.lint(t, "model")); assert "CL020" in c and "CL021" in c

def test_dataset_must_state_pii_status():
    r = cl.lint("---\nlicense: cc-by-4.0\n---\n# D\n## Limitations\nx\n", "dataset"); assert "CL011" in codes(r, "ERROR")
    ok = cl.lint("---\nlicense: cc-by-4.0\n---\n# D\n## Limitations\nx\nNo personal or sensitive information; scanned.\n", "dataset")
    assert "CL011" not in codes(ok)

def test_secret_in_card_never_echoed():
    r = cl.lint(GOOD + "\nhf_" + "Q" * 34, "model"); assert "CL040" in codes(r, "ERROR") and "QQQQ" not in str(r)

def test_placeholder():
    assert "CL030" in codes(cl.lint(GOOD + "\n[More Information Needed]", "model"))
