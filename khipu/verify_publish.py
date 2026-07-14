#!/usr/bin/env python3
"""Post-publish verifier for the Khipu Hub repo.

Called by publish-khipu.ps1 after uploads; can also be run by hand from the
khipu folder. Exit 0 only when what this publish run claims is actually live:

  - the three receipt files on the Hub are BYTE-IDENTICAL to the local ones
    (downloaded and sha256-compared, not just listed);
  - every uploadable file in khipu-model/ (and khipu-adapter/ -> adapter/)
    exists on the Hub with a matching size — pass --deep to also compare the
    Hub's LFS sha256 against a locally computed hash;
  - NO private-key pattern (*.pem, khipu_owner_ed25519*) exists on the Hub.

This is a courier check, not a judge: receipt SIGNATURE verification stays
server-side (Alloy /api/forge/family), and passing here never upgrades
trainingStatus/evalStatus.
"""
import argparse
import fnmatch
import hashlib
import os
import sys

RECEIPTS = ["owner_pubkey.json", "training_receipt.signed.json", "eval_receipt.signed.json"]
KEY_PATTERNS = ["*.pem", "khipu_owner_ed25519*"]
IGNORES = KEY_PATTERNS + ["*.gguf"]


def sha256_path(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True, help="Hub repo id, e.g. SZLHOLDINGS/SZL-Khipu-1.5B-BrainNavigator")
    ap.add_argument("--need-weights", action="store_true", help="require khipu-model/ contents on the Hub")
    ap.add_argument("--need-adapter", action="store_true", help="require khipu-adapter/ contents under adapter/")
    ap.add_argument("--deep", action="store_true", help="also sha256-compare LFS files (slow on big weights)")
    args = ap.parse_args()

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    problems = []

    tree = {}
    for e in api.list_repo_tree(args.repo, recursive=True):
        size = getattr(e, "size", None)
        lfs = getattr(e, "lfs", None)
        lfs_sha = None
        if lfs is not None:
            lfs_sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        tree[e.path] = (size, lfs_sha)

    # 1. No key material may exist remotely, ever.
    for p in tree:
        base = p.rsplit("/", 1)[-1]
        if any(fnmatch.fnmatch(base, pat) for pat in KEY_PATTERNS):
            problems.append("PRIVATE KEY PATTERN ON HUB: %s - rotate the owner key NOW and delete this file" % p)

    # 2. Receipts: byte identity between local and Hub.
    for f in RECEIPTS:
        if not os.path.exists(f):
            problems.append("local %s missing - run from the khipu folder" % f)
            continue
        if f not in tree:
            problems.append("%s not on Hub" % f)
            continue
        remote = hf_hub_download(repo_id=args.repo, filename=f, force_download=True)
        if sha256_path(remote) != sha256_path(f):
            problems.append("%s on Hub differs from local bytes (stale or wrong upload)" % f)

    # 3. Folder contents: deterministic name+size (sha256 with --deep).
    def check_folder(folder, prefix):
        if not os.path.isdir(folder):
            problems.append("local %s/ missing" % folder)
            return
        count = 0
        for root, _, files in os.walk(folder):
            for name in files:
                if any(fnmatch.fnmatch(name, pat) for pat in IGNORES):
                    continue
                lp = os.path.join(root, name)
                rel = prefix + os.path.relpath(lp, folder).replace(os.sep, "/")
                count += 1
                if rel not in tree:
                    problems.append("%s not on Hub" % rel)
                    continue
                size, lfs_sha = tree[rel]
                lsize = os.path.getsize(lp)
                if size is not None and size != lsize:
                    problems.append("%s: size mismatch local=%d hub=%d" % (rel, lsize, size))
                elif args.deep and lfs_sha:
                    print("[verify_publish] hashing %s (%.1f MB) ..." % (rel, lsize / 1e6))
                    if sha256_path(lp) != lfs_sha:
                        problems.append("%s: sha256 mismatch vs Hub LFS" % rel)
        if count == 0:
            problems.append("%s/ contains no uploadable files" % folder)

    if args.need_weights:
        check_folder("khipu-model", "")
        if not any(p.endswith(".safetensors") and "/" not in p for p in tree):
            problems.append("no top-level *.safetensors on Hub - weights upload did not land")
    if args.need_adapter:
        check_folder("khipu-adapter", "adapter/")

    if problems:
        print("[verify_publish] FAIL:")
        for p in problems:
            print("  - " + p)
        sys.exit(1)
    mode = "receipts byte-identical; folders " + ("sha256-verified" if args.deep else "name+size-verified")
    print("[verify_publish] PASS - %s; no key material on Hub." % mode)


if __name__ == "__main__":
    main()
