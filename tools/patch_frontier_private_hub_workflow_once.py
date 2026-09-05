#!/usr/bin/env python3
"""One-shot exact transformer for the model frontier workflow."""

from __future__ import annotations

from pathlib import Path


PATH = Path(".github/workflows/model-kernel-frontier.yml")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label} anchor count changed: expected 1, observed {count}")
    return text.replace(old, new, 1)


def main() -> int:
    text = PATH.read_text(encoding="utf-8")

    compile_anchor = "            tools/test_model_binding_workflow.py \\\n"
    text = replace_once(
        text,
        compile_anchor,
        compile_anchor + "            tools/test_model_kernel_frontier_workflow.py \\\n",
        "compile",
    )

    test_anchor = "          python tools/test_model_binding_workflow.py\n"
    text = replace_once(
        text,
        test_anchor,
        test_anchor + "          python tools/test_model_kernel_frontier_workflow.py\n",
        "test",
    )

    networked_dry_run = '''          python tools/publish_model_source_bindings.py \\
            --source-revision "${GITHUB_SHA}" \\
            --report reports/model-source-bindings-dry-run.json
'''
    text = replace_once(text, networked_dry_run, "", "networked dry run")

    old_live = '''      - name: Audit public Hugging Face artifacts
        run: |
          set -euo pipefail
          python tools/verify_model_portfolio.py \\
            --live \\
            --report reports/model-portfolio-live.json

      - name: Upload audit evidence
        if: always()
        uses: actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a # v7.0.1
        with:
          name: model-kernel-portfolio-${{ github.run_id }}
          path: reports/model-portfolio-live.json
          if-no-files-found: error
          retention-days: 30
'''
    new_live = '''      - name: Acquire validated private-Hub read credential
        if: github.event_name == 'push' && github.ref == 'refs/heads/main'
        env:
          HF_ORG_TOKEN_CANDIDATE: ${{ secrets.HF_ORG_TOKEN }}
          HF_ORG_TOKEN1_CANDIDATE: ${{ secrets.HF_ORG_TOKEN1 }}
          HF_WRITE_TOKEN_CANDIDATE: ${{ secrets.HF_WRITE_TOKEN }}
          HF_TOKEN_CANDIDATE: ${{ secrets.HF_TOKEN }}
          HUGGINGFACE_TOKEN_CANDIDATE: ${{ secrets.HUGGINGFACE_TOKEN }}
          HUGGING_FACE_HUB_TOKEN_CANDIDATE: ${{ secrets.HUGGING_FACE_HUB_TOKEN }}
        run: |
          set -euo pipefail
          python tools/acquire_hf_publisher_token.py \\
            --target-repo SZLHOLDINGS/SZL-Forge-1.5B-ReceiptAgent \\
            --target-type model \\
            --github-env "${GITHUB_ENV}" \\
            --report reports/hf-private-read-credential.json

      - name: Audit exact gated bindings and public portfolio
        if: github.event_name == 'push' && github.ref == 'refs/heads/main'
        run: |
          set -euo pipefail
          python tools/publish_model_source_bindings.py \\
            --source-revision "${GITHUB_SHA}" \\
            --report reports/model-source-bindings-dry-run.json
          python tools/verify_model_portfolio.py \\
            --live \\
            --report reports/model-portfolio-live.json

      - name: Upload audit evidence
        if: always()
        uses: actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a # v7.0.1
        with:
          name: model-kernel-portfolio-${{ github.run_id }}
          path: |
            reports/model-portfolio-live.json
            reports/model-source-bindings-dry-run.json
            reports/hf-private-read-credential.json
          if-no-files-found: warn
          retention-days: 30
'''
    text = replace_once(text, old_live, new_live, "live audit")

    PATH.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
