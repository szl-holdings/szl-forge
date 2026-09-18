import subprocess, json

def ledger(ref):
    out = subprocess.run(["git", "show", ref + ":chaski/family_leaderboard.json"],
                         capture_output=True, check=True).stdout
    return json.loads(out.decode("utf-8-sig"))

main = ledger("origin/main")
ours = ledger("HEAD")
ours_row = next((r for r in ours["rows"] if r["id"] == "SZLHOLDINGS/A11OY-MINI"), None)
if ours_row is None:
    raise SystemExit("our A11OY row not found - do not guess")
merged = False
for r in main["rows"]:
    if r["id"] == "SZLHOLDINGS/A11OY-MINI":
        r.update({"json_draft": ours_row.get("json_draft"),
                  "adversarial_refuse": ours_row.get("adversarial_refuse"),
                  "note": ours_row.get("note")})
        merged = True
if not merged:
    main["rows"].append(ours_row)
with open("chaski/family_leaderboard.json", "w", encoding="utf-8", newline="\n") as f:
    json.dump(main, f, indent=2)
    f.write("\n")
print("ledger resolved: main structure + branch A11OY row | rows =", len(main["rows"]))
