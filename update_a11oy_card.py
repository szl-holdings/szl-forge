import sys
p = "a11oy-mini/card/README.md"
s = open(p, encoding="utf-8").read()
banner_old = "> **GGUF of a failed parent (chaski).** Not flagship. Not a11oy production."
if banner_old not in s:
    sys.exit("card banner anchor not found - local card differs from Hub card; paste diff to forge")
s = s.replace(banner_old,
    "> **Rebuilt 2026-09-17 from the chaski-r2 winner (named-N gate champion).** "
    "Legacy GGUFs below are DEPRECATED failed-parent lineage. Not flagship. Not a11oy production.")
qanchor = "sha256 `06136ba385b2e052cf4cdb3dc8d333948e0b612bd15a541b314e170399c2faa6` **MEASURED** |"
if qanchor not in s:
    sys.exit("card Q4_K_M row anchor not found")
new_rows = (
    "\n| **R2 parent** | `chaski-r2` local named-N winner (GitHub receipts only; no Hub page) |"
    "\n| **R2 Q4_K_M** | `a11oy-mini-r2-Q4_K_M.gguf` 541903328 sha256 `6d42341c932a76e91b2c04a859a4248e7d2c77308f998a32771f43802f097b62` **MEASURED** - GGUF gate 5/5 + 6/6 (ollama, 2026-09-17) |"
    "\n| **R2 mmproj** | `a11oy-mini-r2-BF16-mmproj.gguf` 207346048 sha256 `4855efe034435b9b3b289b2c07d09b263c088749d1f85c43c1cf4672bc7fcbf2` **MEASURED** (vision projector pair) |"
)
s = s.replace(qanchor, qanchor + new_rows)
yanchor = "  new_train: false"
if yanchor not in s:
    sys.exit("card YAML anchor not found")
s = s.replace(yanchor, yanchor + "\n  r2_rebuild: 2026-09-17\n  r2_gate: \"5/5 + 6/6 MEASURED ollama\"")
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("card updated: banner + R2 rows + YAML keys")
