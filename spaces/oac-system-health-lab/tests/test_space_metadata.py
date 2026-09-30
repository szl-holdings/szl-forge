"""Offline provider-card regressions for the source-bound OAC Space."""

from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
REJECTED_DESCRIPTION = "Fixed synthetic telemetry scoring with source-bound readiness."


def description_from_card(raw: str) -> str:
    if len(raw.encode("utf-8")) > 16_384 or not raw.startswith("---\n"):
        raise ValueError("invalid card boundary")
    parts = raw.split("---\n", 2)
    if len(parts) != 3:
        raise ValueError("missing card boundary")
    metadata = parts[1]
    for key, expected in (
        ("sdk", "docker"),
        ("app_port", "7860"),
        ("suggested_hardware", "cpu-basic"),
        ("license", "apache-2.0"),
    ):
        if re.findall(r"^" + key + r": ([^\n]+)$", metadata, re.MULTILINE) != [expected]:
            raise ValueError("runtime card contract")
    values = re.findall(r"^short_description: ([^\n]*)$", metadata, re.MULTILINE)
    if len(values) != 1 or not values[0] or not values[0].isascii() or len(values[0]) > 60:
        raise ValueError("provider description contract")
    return values[0]


class SpaceMetadataTests(unittest.TestCase):
    def setUp(self):
        self.card = (ROOT / "README.md").read_text(encoding="utf-8")
        self.description = description_from_card(self.card)

    def test_shipped_metadata_is_bounded_and_matches_runtime(self):
        self.assertGreater(len(self.description), 0)
        self.assertLessEqual(len(self.description), 60)

    def test_exact_failed_provider_card_is_refused(self):
        self.assertEqual(len(REJECTED_DESCRIPTION), 62)
        with self.assertRaises(ValueError):
            description_from_card(self.card.replace(self.description, REJECTED_DESCRIPTION))

    def test_description_limit_is_inclusive(self):
        for count in (1, 60):
            self.assertEqual(
                description_from_card(self.card.replace(self.description, "x" * count)),
                "x" * count,
            )
        with self.assertRaises(ValueError):
            description_from_card(self.card.replace(self.description, "x" * 61))

    def test_missing_empty_duplicate_and_paid_hardware_are_refused(self):
        for invalid in (
            self.card.replace("short_description: " + self.description + "\n", ""),
            self.card.replace(self.description, ""),
            self.card.replace("short_description: ", "short_description: duplicate\nshort_description: ", 1),
            self.card.replace("suggested_hardware: cpu-basic", "suggested_hardware: cpu-upgrade"),
        ):
            with self.subTest(), self.assertRaises(ValueError):
                description_from_card(invalid)


if __name__ == "__main__":
    unittest.main()
