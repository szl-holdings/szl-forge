#!/usr/bin/env bash
set -euo pipefail

mode="${1:-}"
if [[ -n "${mode}" && "${mode}" != "--verify" ]]; then
  printf 'unsupported installer mode\n' >&2
  exit 2
fi
if [[ -n "${HF_TOKEN:-}" || -n "${HF_OIDC_ID_TOKEN:-}" ]]; then
  printf 'publisher credentials must be absent during builder installation\n' >&2
  exit 2
fi

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# The repo-scoped OIDC grant targets this existing kernel repo. The builder's
# unconditional repository-creation request was denied before any commit.
patch_path="${repo_root}/patches/hf-kernel-builder-existing-repo.patch"
upstream_revision="633246310320d85def0c67d62c7912fd444a842f"
patch_sha256="e30c7c5f4bb9833b3905984a3040690c857f5162f02c548f899360bda948c402"
builder_dir="$(mktemp -d "${RUNNER_TEMP:?}/szl-hf-kernel-builder.XXXXXX")"

test "$(sha256sum "${patch_path}" | cut -d ' ' -f1)" = "${patch_sha256}"
git init --quiet "${builder_dir}"
git -C "${builder_dir}" remote add origin https://github.com/huggingface/kernels.git
git -C "${builder_dir}" fetch --quiet --depth 1 origin "${upstream_revision}"
git -C "${builder_dir}" checkout --quiet --detach FETCH_HEAD
test "$(git -C "${builder_dir}" rev-parse HEAD)" = "${upstream_revision}"
git -C "${builder_dir}" apply --unidiff-zero --check "${patch_path}"
git -C "${builder_dir}" apply --unidiff-zero "${patch_path}"
test "$(git -C "${builder_dir}" diff --name-only)" = \
  $'kernel-builder/src/main.rs\nkernel-builder/src/upload.rs'

if [[ "${mode}" == "--verify" ]]; then
  cargo test --locked --manifest-path "${builder_dir}/Cargo.toml" \
    -p hf-kernel-builder --bin kernel-builder
fi
cargo install --locked --path "${builder_dir}/kernel-builder"
test "$(kernel-builder --version)" = "hf-kernel-builder 0.17.0-dev0"
kernel-builder upload --help | grep -Fq -- '--existing-repo'
