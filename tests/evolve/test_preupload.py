import os, pickle, io, zipfile, json
from szl_evolve import preupload_check as pc

class Evil:
    def __reduce__(self): return (__import__("os").system, ("echo never-run",))  # dumps only; never loaded

def write(p, b, mode="wb"):
    os.makedirs(os.path.dirname(p), exist_ok=True); open(p, mode).write(b)

def test_dangerous_pickle_detected_without_executing(tmp_path):
    write(tmp_path / "w.pkl", pickle.dumps(Evil()))
    r = pc.scan_pickle_file(str(tmp_path / "w.pkl"))
    assert r["verdict"] == "DANGEROUS" and any("system" in x for x in r["dangerous"])

def test_benign_pickle_unreviewed_not_dangerous(tmp_path):
    write(tmp_path / "ok.pkl", pickle.dumps({"a": [1, 2, 3]}))
    assert pc.scan_pickle_file(str(tmp_path / "ok.pkl"))["verdict"] == "ALLOWLISTED_ONLY"

def test_torch_zip_member_scanned(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z: z.writestr("archive/data.pkl", pickle.dumps(Evil()))
    write(tmp_path / "m.pt", buf.getvalue())
    assert pc.scan_pickle_file(str(tmp_path / "m.pt"))["verdict"] == "DANGEROUS"

def test_pickle_blocks_unless_allowed(tmp_path):
    write(tmp_path / "ok.pkl", pickle.dumps({"a": 1}))
    assert pc.check_dir(str(tmp_path))["verdict"] == "BLOCK"
    assert pc.check_dir(str(tmp_path), allow_pickle=True)["verdict"] == "PASS_NOT_UPLOADED"

def test_dangerous_pickle_blocks_even_if_allowed(tmp_path):
    write(tmp_path / "w.pkl", pickle.dumps(Evil()))
    assert pc.check_dir(str(tmp_path), allow_pickle=True)["verdict"] == "BLOCK"

def test_ssn_and_card_found_values_not_echoed(tmp_path):
    write(tmp_path / "d.jsonl", '{"t": "ssn 123-45-6789"}\n{"c": "card 4012 8888 8888 1881"}\n{"ok": "000-12-3456 and 9999999999999"}\n', "w")
    r = pc.check_dir(str(tmp_path)); kinds = [f["kind"] for f in r["findings"]]
    assert "PII:ssn_format" in kinds and "PII:card_luhn" in kinds and kinds.count("PII:ssn_format") == 1
    assert "123-45-6789" not in json.dumps(r) and "4012" not in json.dumps(r)
    assert r["verdict"] == "BLOCK"

def test_secret_patterns(tmp_path):
    write(tmp_path / "c.py", 'TOKEN = "hf_' + "Z" * 34 + '"\n-----BEGIN RSA PRIVATE KEY-----\n', "w")
    r = pc.check_dir(str(tmp_path)); kinds = {f["kind"] for f in r["findings"]}
    assert {"SECRET:hf_token", "SECRET:private_key_block"} <= kinds and "ZZZZ" not in json.dumps(r)

def test_public_target_needs_approval_ref(tmp_path):
    write(tmp_path / "a.md", "hello\n", "w")
    assert pc.check_dir(str(tmp_path), "public")["verdict"] == "BLOCK"
    assert pc.check_dir(str(tmp_path), "public", "APPROVAL-REF-X")["verdict"] == "PASS_NOT_UPLOADED"

def test_clean_dir_passes_and_safetensors_skipped(tmp_path):
    write(tmp_path / "README.md", "# card\n", "w"); write(tmp_path / "m.safetensors", b"\x00" * 16)
    r = pc.check_dir(str(tmp_path)); assert r["verdict"] == "PASS_NOT_UPLOADED" and r["files_scanned"] == 2

def test_luhn():
    assert pc.luhn("4012888888881881") and not pc.luhn("4012888888881882")

def test_example_email_allowed_other_warns(tmp_path):
    write(tmp_path / "e.txt", "a@example.com b@corp.io\n", "w")
    f = pc.check_dir(str(tmp_path))["findings"]; assert len(f) == 1 and f[0]["severity"] == "WARN"

def test_hash_digits_and_ids_not_flagged_as_cards(tmp_path):
    write(tmp_path / "h.txt", "sha 4012888888881881abc id 1700000000000000 ts 1234567890123456 x 9876543210987654\n", "w")
    kinds = [f["kind"] for f in pc.check_dir(str(tmp_path))["findings"]]
    assert "PII:card_luhn" not in kinds   # glued to letters / no issuer prefix / not Luhn

def test_brand_prefix_rules():
    assert pc.card_brand_ok("4012888888881881") and pc.card_brand_ok("378282246310005") and pc.card_brand_ok("5555555555554444")
    assert not pc.card_brand_ok("1234567890123452") and not pc.card_brand_ok("40128888888818")
