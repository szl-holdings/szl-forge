import unittest

from publishing.collection_rebuild import dry_run


class CollectionRebuildTests(unittest.TestCase):
    def test_flagship_empty_no_write(self) -> None:
        report = dry_run()
        self.assertEqual(report["flagship"], [])
        self.assertFalse(report["ready"])
        self.assertIsNone(report["winner"])
        self.assertTrue(str(report["hub_write"]).startswith("DENIED"))
        self.assertIn("SZLHOLDINGS/chaski-5050", report["research"])


if __name__ == "__main__":
    unittest.main()
