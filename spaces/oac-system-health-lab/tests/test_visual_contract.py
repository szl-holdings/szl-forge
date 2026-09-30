"""Static design-token, boundary, touch-target and disclosure regressions.

These checks do not establish browser reflow, WCAG conformance or certification.
"""
from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]


def luminance(color: str) -> float:
    channels = [int(color[index:index + 2], 16) / 255 for index in (1, 3, 5)]
    linear = [value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4 for value in channels]
    return sum(value * weight for value, weight in zip(linear, (0.2126, 0.7152, 0.0722)))


def contrast(first: str, second: str) -> float:
    light, dark = sorted((luminance(first), luminance(second)), reverse=True)
    return (light + 0.05) / (dark + 0.05)


class LensParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.lenses = []

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if "signal-lens" in attrs.get("class", "").split():
            self.lenses.append((tag, attrs))


class VisualContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = (ROOT / "index.html").read_text(encoding="utf-8")

    def test_opaque_text_and_control_token_contrast(self):
        root = re.search(r":root\s*\{([^}]+)\}", self.html).group(1)
        tokens = dict(re.findall(r"--([a-z-]+)\s*:\s*(#[0-9a-f]{6})\s*;", root))
        pairs = []
        for foreground in ("ink", "muted", "green", "violet"):
            for background in ("paper", "surface", "surface-soft"):
                pairs.append((foreground, background, 4.5))
        pairs.extend([
            ("green", "green-soft", 4.5),
            ("amber", "amber-soft", 4.5),
            ("red", "red-soft", 4.5),
            ("paper", "green", 4.5),
            ("control", "paper", 3),
            ("control", "surface", 3),
            ("focus", "paper", 3),
            ("focus", "surface", 3),
            ("paper", "green", 3),
            ("ink", "line", 3),
        ])
        for foreground, background, minimum in pairs:
            with self.subTest(foreground=foreground, background=background):
                self.assertGreaterEqual(contrast(tokens[foreground], tokens[background]), minimum)
        self.assertRegex(self.html, r"\.numeric-field input\s*\{[^}]*border:1px solid var\(--control\)")
        self.assertRegex(self.html, r"\.numeric-field input\s*\{[^}]*background:var\(--paper\)")
        self.assertRegex(self.html, r"\.threshold-marker\s*\{[^}]*background:var\(--paper\); border:1px solid var\(--ink\)")

    def test_decorative_lens_is_hidden_from_assistive_technology(self):
        parser = LensParser()
        parser.feed(self.html)
        self.assertEqual(len(parser.lenses), 1)
        tag, attrs = parser.lenses[0]
        self.assertEqual(tag, "div")
        self.assertEqual(attrs.get("aria-hidden"), "true")
        self.assertNotIn("tabindex", attrs)
        self.assertIn('name="color-scheme" content="dark"', self.html)
        self.assertIn("does not simulate a result", self.html)
        self.assertIn("no clinical or device authority", self.html)
        self.assertIn("checks the response hashes’ format", self.html)

    def test_comfortable_action_targets_and_reduced_motion(self):
        for selector, minimum in [
            (".small-button", 44),
            (".primary-button", 48),
            (".numeric-field input", 48),
            (".toggle-row label", 44),
            (".masthead nav a", 44),
        ]:
            pattern = re.escape(selector) + r"\s*\{[^}]*min-height:" + str(minimum) + r"px"
            self.assertRegex(self.html, pattern)
        self.assertRegex(self.html, r"@media\(prefers-reduced-motion:reduce\)\s*\{\s*\.score-fill\s*\{\s*transition:none;")
        self.assertNotRegex(self.html, r"(?:animation\s*:|backdrop-filter\s*:|<img\b|<canvas\b)")

    def test_narrow_layout_and_long_identity_wrap_contract(self):
        self.assertIn("@media(max-width:620px)", self.html)
        self.assertRegex(self.html, r"\.workspace,\.numeric-grid,\.identity-grid\s*\{\s*grid-template-columns:minmax\(0,1fr\);")
        self.assertRegex(self.html, r"\.identity-grid dd\s*\{[^}]*overflow-wrap:anywhere")
        self.assertRegex(self.html, r"@media\(forced-colors:active\)\s*\{\s*\.signal-lens\s*\{\s*display:none;\s*\}\s*\.scope\s*\{\s*display:block;")
        self.assertNotRegex(self.html, r"overflow-x\s*:\s*hidden")
        self.assertIn('class="skip-link" href="#workspace"', self.html)
        self.assertIn('id="result-announcement" role="status" aria-live="polite"', self.html)


if __name__ == "__main__":
    unittest.main()
