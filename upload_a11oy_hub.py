from huggingface_hub import HfApi
api = HfApi()
repo = "SZLHOLDINGS/A11OY-MINI"
existing = set(api.list_repo_files(repo_id=repo, repo_type="model"))
msg = "a11oy-mini: rebuilt from chaski-r2 winner (GGUF gate MEASURED 5/5 + 6/6); legacy failed-parent GGUFs deprecated; see gguf_gate_receipt.json and hub_put_receipt.json"
files = [
    ("chaski_r2/a11oy_mini_r2_gguf/a11oy-mini-r2-Q4_K_M.gguf", "a11oy-mini-r2-Q4_K_M.gguf"),
    ("chaski_r2/a11oy_mini_r2_gguf/a11oy-mini-r2-BF16-mmproj.gguf", "a11oy-mini-r2-BF16-mmproj.gguf"),
    ("chaski_r2/a11oy_mini_r2_gguf/gguf_gate_receipt.json", "gguf_gate_receipt.json"),
    ("chaski_r2/a11oy_mini_r2_gguf/hub_put_receipt.json", "hub_put_receipt.json"),
    ("chaski_r2/a11oy_mini_r2_gguf/Modelfile.a11oy-r2.gguf", "Modelfile.a11oy-r2.gguf"),
    ("a11oy-mini/card/README.md", "README.md"),
]
for src, dst in files:
    if dst in existing:
        print("already on Hub, skipping:", dst)
        continue
    print("uploading", dst, "...")
    api.upload_file(path_or_fileobj=src, path_in_repo=dst, repo_id=repo, repo_type="model", commit_message=msg)
print("[a11oy] Hub PUT complete:", repo)
