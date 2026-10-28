from __future__ import annotations

import hashlib
import html
import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_EVIDENCE = (
    "run_manifest.json",
    "eval_receipt.json",
    "training_summary.json",
    "thesis_formula_index.json",
    "science_source_ledger.json",
    "curriculum.json",
    "model_portfolio.json",
)
# Pins from the 2026-07-11 MEASURED run receipt (run_manifest.json).
RUN_MANIFEST_RECEIPT_SHA256 = (
    "0c0a23e292d09ced43830c8de80a387a40c988a4b62c53f52fda0e78809fa3c4"
)
RECEIPT_INDEX_SHA256 = (
    "5dbe4405b50672d150437fdcfa941213c6ed88f07979370f91fd60d3f65a13c2"
)
# GREEN entries whose cited lean_theorem was found declared, without sorry, in
# szl-holdings/lutar-lean main cd20ff9 by the 2026-10-14 static audit.
AUDITED_GREEN_ENTRIES = frozenset(
    {"TH1", "TH2", "TH2b", "TH11", "R1-G1", "TH-COMP", "TH-SHANNON", "TH-WHEELER"}
)


class StaticForgeContractTests(unittest.TestCase):
    def assert_curated_boundary(self, index: str) -> None:
        paragraphs = re.findall(
            r'<p\b[^>]*\bid="portfolio-evidence-boundary"[^>]*>(.*?)</p>',
            index,
            flags=re.DOTALL,
        )
        self.assertEqual(1, len(paragraphs))
        text = html.unescape(re.sub(r"<[^>]*>", " ", paragraphs[0]))
        text = " ".join(text.lower().split())
        for boundary in (
            "curated snapshot, not a complete hub inventory",
            "counts come from the packaged catalog",
            "a repository listing does not verify weight bytes",
            "declared-key integrity, not independent quality, safety, authorship",
            "or autonomous execution authority",
        ):
            self.assertIn(boundary, text)
        # Counts belong to the catalog-driven tiles, not a second hard-coded list.
        self.assertNotRegex(
            text,
            r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b",
        )

    def test_evidence_boundary_is_count_free_and_explicitly_curated(self) -> None:
        self.assert_curated_boundary((ROOT / "index.html").read_text(encoding="utf-8"))

    def test_evidence_boundary_rejects_in_memory_claim_regressions(self) -> None:
        index = (ROOT / "index.html").read_text(encoding="utf-8")
        # SIMULATED negative strings, never written into the source or a catalog.
        for old, replacement in (
            ("This is a curated snapshot", "One learned kernel contains weight bytes. This is a curated snapshot"),
            ("not a complete Hub inventory", "a complete Hub inventory"),
            ("does not verify weight bytes", "verifies weight bytes"),
            ("not independent quality", "independent quality"),
            ("or autonomous execution authority", "or unlimited authority"),
        ):
            with self.subTest(regression=old):
                self.assertIn(old, index)
                with self.assertRaises(AssertionError):
                    self.assert_curated_boundary(index.replace(old, replacement, 1))

    def test_portfolio_count_tiles_still_derive_from_catalog_entries(self) -> None:
        index = (ROOT / "index.html").read_text(encoding="utf-8")
        self.assertIn("const assets=portfolio.artifacts||[];", index)
        for kind in ("trained_model", "quantized_model", "learned_kernel", "software_kernel"):
            with self.subTest(kind=kind):
                self.assertIn(f'assets.filter(a=>a.kind==="{kind}").length', index)

    def test_space_is_an_explicit_static_evidence_console(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("sdk: static\n", readme)
        self.assertIn("app_file: index.html\n", readme)

    def test_mobile_console_loads_every_packaged_evidence_file(self) -> None:
        index = (ROOT / "index.html").read_text(encoding="utf-8")
        self.assertIn(
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            index,
        )
        self.assertIn(
            '<link rel="icon" type="image/svg+xml" '
            'href="szl/logos/szl_favicon.svg">',
            index,
        )
        self.assertTrue((ROOT / "szl" / "logos" / "szl_favicon.svg").is_file())
        self.assertIn("MODEL / KERNEL PORTFOLIO", index)
        for filename in REQUIRED_EVIDENCE:
            with self.subTest(filename=filename):
                self.assertTrue((ROOT / filename).is_file())
                self.assertIn(filename, index)
                json.loads((ROOT / filename).read_text(encoding="utf-8"))

    def test_szl_design_system_is_vendored_and_loaded_locally(self) -> None:
        index = (ROOT / "index.html").read_text(encoding="utf-8")
        szl = ROOT / "szl"
        self.assertIn('<link rel="stylesheet" href="szl/szl-design-system.css">', index)
        self.assertIn('<link rel="stylesheet" href="szl/szl-console.css">', index)
        source = json.loads((szl / "SOURCE.json").read_text(encoding="utf-8"))
        self.assertEqual("1.3.0", source["version"])
        vendored = sorted(
            path.relative_to(szl).as_posix()
            for path in szl.rglob("*")
            if path.is_file() and path.name != "SOURCE.json"
        )
        self.assertEqual(
            [
                "logos/szl_favicon.svg",
                "szl-console.css",
                "szl-design-system.css",
            ],
            vendored,
        )
        for relative in vendored:
            with self.subTest(file=relative):
                digest = hashlib.sha256((szl / relative).read_bytes()).hexdigest()
                self.assertEqual(source["sha256"][relative], digest)
        self.assertFalse((ROOT / "kanchay").exists())
        lowered = index.lower()
        for remote in (
            "fonts.googleapis",
            "fonts.gstatic",
            "cdnjs",
            "jsdelivr",
            "unpkg",
            "@font-face",
            ".woff2",
            "kanchay/",
        ):
            self.assertNotIn(remote, lowered)

    def test_model_portfolio_does_not_conflate_cards_with_weights(self) -> None:
        portfolio = json.loads(
            (ROOT / "model_portfolio.json").read_text(encoding="utf-8")
        )
        kinds = [item["kind"] for item in portfolio["artifacts"]]
        self.assertEqual(3, kinds.count("trained_model"))
        self.assertEqual(1, kinds.count("quantized_model"))
        self.assertEqual(0, kinds.count("learned_kernel"))
        self.assertEqual(12, kinds.count("software_kernel"))
        self.assertTrue(
            all(item["autonomy_eligible"] is False for item in portfolio["artifacts"])
        )

    def test_lambda_uniqueness_entry_is_not_declared_green(self) -> None:
        # Doctrine: Λ = Conjecture 1, never a theorem. TH10 is the packaged
        # Λ-uniqueness entry; its historical GREEN label was corrected 2026-10-06.
        # The repository-root snapshot (read by conjecture_machine.py) is checked
        # too when this test runs from a full checkout; the published Space has
        # no root copy.
        copies = [("space", ROOT / "thesis_formula_index.json")]
        root_copy = ROOT.parents[1] / "thesis_formula_index.json"
        if root_copy.is_file():
            copies.append(("repository root", root_copy))
        for label, path in copies:
            with self.subTest(copy=label):
                index = json.loads(path.read_text(encoding="utf-8"))
                th10 = [
                    entry for entry in index["entries"] if entry.get("id") == "TH10"
                ]
                self.assertEqual(1, len(th10))
                self.assertNotEqual("GREEN", th10[0]["status"])
                self.assertEqual("GREEN", th10[0]["correction"]["previous_status"])

    def test_run_manifest_receipt_is_not_reminted(self) -> None:
        # run_manifest.json is the 2026-07-11 MEASURED run receipt. A later
        # correction to a packaged input is shown by the console as a hash
        # MISMATCH against this receipt; it is not hidden by rewriting the pin
        # and recomputing receipt_sha256.
        manifest = json.loads(
            (ROOT / "run_manifest.json").read_text(encoding="utf-8")
        )
        unsigned = {k: v for k, v in manifest.items() if k != "receipt_sha256"}
        recomputed = hashlib.sha256(
            json.dumps(
                unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8")
        ).hexdigest()
        self.assertEqual(manifest["receipt_sha256"], recomputed)
        self.assertEqual(RUN_MANIFEST_RECEIPT_SHA256, manifest["receipt_sha256"])

    def test_post_receipt_index_correction_is_explained_not_hidden(self) -> None:
        # The 2026-10-06 TH10 correction (szl-forge#594) and the 2026-10-14
        # unresolved-Lean-pointer correction changed the packaged
        # thesis_formula_index.json after the July receipt. The console keeps
        # the computed MISMATCH and explains it under the integrity table; the
        # receipt and its pin stay unchanged.
        index = (ROOT / "index.html").read_text(encoding="utf-8")
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        table = index.index('<table class="table" id="integrity">')
        note = index.index('id="integrity-correction-note"')
        self.assertLess(table, note)
        self.assertLess(note, index.index("</section>", table))
        self.assertIn('<span class="st fail">MISMATCH</span>', index)
        note_text = index[note:index.index("</p>", note)]
        for surface, blob in (("index", note_text), ("readme", readme)):
            blob = " ".join(blob.split())
            with self.subTest(surface=surface):
                for phrase in (
                    "thesis_formula_index.json",
                    "2026-10-06",
                    "TH10 is no longer marked GREEN",
                    "meta.corrections",
                    "szl-holdings/szl-forge#594",
                    "2026-10-14",
                    "re-labelled from GREEN to TRACKED",
                    "kept byte-identical as a historical record",
                    "not re-minted",
                    "documented post-receipt corrections, not tampering",
                    "this note does not verify them",
                ):
                    self.assertIn(phrase, blob)
        manifest = json.loads(
            (ROOT / "run_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(RUN_MANIFEST_RECEIPT_SHA256, manifest["receipt_sha256"])
        pins = [
            artifact["sha256"]
            for artifact in manifest["payload"]["artifacts"]
            if artifact["path"] == "thesis_formula_index.json"
        ]
        self.assertEqual([RECEIPT_INDEX_SHA256], pins)
        corrected = (ROOT / "thesis_formula_index.json").read_bytes()
        corrected = corrected.replace(b"\r\n", b"\n")
        self.assertNotEqual(
            RECEIPT_INDEX_SHA256, hashlib.sha256(corrected).hexdigest()
        )
        formula_index = json.loads(corrected.decode("utf-8"))
        corrected_ids = [
            entry_id
            for correction in formula_index["meta"]["corrections"]
            for entry_id in correction["entries"]
        ]
        self.assertIn("TH10", corrected_ids)
        self.assertIn("TH-PUBLIC-CLAIMS", corrected_ids)

    def test_unresolved_lean_pointers_are_not_declared_green(self) -> None:
        # 2026-10-14 audit: 64 GREEN entries cited a lean_theorem that
        # szl-holdings/lutar-lean does not declare (main cd20ff9, and the index's
        # own meta.lean_commit c4d13795). Each was re-labelled TRACKED with a
        # dated correction field; only the eight audited entries whose cited
        # declaration exists may still be GREEN. Re-promoting any other entry
        # needs a new audit, and this list, to change together.
        copies = [("space", ROOT / "thesis_formula_index.json")]
        root_copy = ROOT.parents[1] / "thesis_formula_index.json"
        if root_copy.is_file():
            copies.append(("repository root", root_copy))
        for label, path in copies:
            with self.subTest(copy=label):
                index = json.loads(path.read_text(encoding="utf-8"))
                records = [
                    record
                    for record in index["meta"]["corrections"]
                    if record["date"] == "2026-10-14"
                ]
                self.assertEqual(1, len(records))
                listed = records[0]["entries"]
                self.assertEqual(64, len(listed))
                self.assertEqual(len(listed), len(set(listed)))
                dated = [
                    entry
                    for entry in index["entries"]
                    if entry.get("correction", {}).get("date") == "2026-10-14"
                ]
                self.assertEqual(listed, [entry["id"] for entry in dated])
                for entry in dated:
                    correction = entry["correction"]
                    self.assertEqual("TRACKED", entry["status"])
                    self.assertEqual("GREEN", correction["previous_status"])
                    self.assertNotEqual(
                        correction["previous_lean_theorem"], entry["lean_theorem"]
                    )
                    self.assertTrue(
                        correction["lean_ref"].startswith("szl-holdings/lutar-lean@")
                    )
                    self.assertEqual("NOT_RUN", correction["lean_build"])
                    self.assertEqual("NOT_RUN", correction["print_axioms"])
                green = {
                    entry["id"]
                    for entry in index["entries"]
                    if entry["status"] == "GREEN"
                }
                self.assertLessEqual(green, AUDITED_GREEN_ENTRIES)

    def test_atelier_lock_keeps_snapshot_from_becoming_a_studio(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        index = (ROOT / "index.html").read_text(encoding="utf-8")
        for blob in (readme, index):
            with self.subTest(surface="readme" if blob is readme else "index"):
                self.assertIn("BLUEPRINT_NOT_TRAINED", blob)
                self.assertIn("szl-model-inference-lab", blob)
                self.assertIn("SIMULATED", blob)
                self.assertIn("Ask", blob)
                self.assertIn("not a live control plane", blob.lower().replace("&amp;", "&"))
        lowered = (readme + "\n" + index).lower()
        self.assertIn("not unsloth studio", lowered)
        self.assertIn("not a jobs launcher", lowered)
        for forbidden in (
            "hf jobs uv run",
            "launch training",
            "start fine-tune",
            "train on this space",
        ):
            self.assertNotIn(forbidden, lowered)


if __name__ == "__main__":
    unittest.main(verbosity=2)
