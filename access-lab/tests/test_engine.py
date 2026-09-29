"""Synthetic tests only: no submitted HTML is rendered or executed."""

from copy import deepcopy
import hashlib
import unittest

from szl_access.engine import MAX_CANDIDATES, MAX_SOURCE_BYTES, analyze, apply_proposal, deterministic_proposal


class EngineTests(unittest.TestCase):
    SOURCE = '<form method="post"><div><label>Email address</label>\n<input id="email" type="email" required></div><button>Send</button></form>'

    def repair(self, source):
        return apply_proposal(source, deterministic_proposal(source))

    def test_adjacent_visible_label_is_bound(self):
        analysis = analyze(self.SOURCE)
        self.assertEqual(analysis["status"], "SUPPORTED")
        self.assertEqual(len(analysis["candidates"]), 1)
        self.assertEqual(analysis["candidates"][0]["label_text"], "Email address")
        self.assertEqual(analysis["candidates"][0]["control_id"], "email")
        result = self.repair(self.SOURCE)
        self.assertEqual(result["state"], "VERIFIED_SCOPED_REPAIR")
        self.assertEqual(result["output_html"], self.SOURCE.replace("<label>", '<label for="email">'))
        self.assertTrue(all(result["checks"].values()))
        self.assertEqual(result["output_sha256"], hashlib.sha256(result["output_html"].encode()).hexdigest())

    def test_source_bytes_case_entities_and_newlines_preserved(self):
        source = '<DIV>\r\n<LABEL title=\'Contact\'>Prénom &amp; nom</LABEL>\t<INPUT id="name" value="A&amp;B" />\r\n</DIV>'
        result = self.repair(source)
        self.assertEqual(result["state"], "VERIFIED_SCOPED_REPAIR")
        self.assertEqual(result["output_html"], source.replace("<LABEL", '<LABEL for="name"'))
        self.assertEqual(analyze(source)["candidates"][0]["label_text"], "Prénom & nom")

    def test_deterministic_candidate_identity_and_output(self):
        self.assertEqual(analyze(self.SOURCE), analyze(self.SOURCE))
        self.assertEqual(self.repair(self.SOURCE), self.repair(self.SOURCE))

    def test_applied_output_is_idempotent(self):
        first = self.repair(self.SOURCE)
        second = self.repair(first["output_html"])
        self.assertEqual(second["state"], "NO_CHANGE")
        self.assertEqual(second["output_html"], first["output_html"])

    def test_stale_or_replayed_proposal_refused_without_changes(self):
        proposal = deterministic_proposal(self.SOURCE)
        output = apply_proposal(self.SOURCE, proposal)["output_html"]
        for changed in (self.SOURCE + " ", output):
            with self.subTest(source=changed):
                result = apply_proposal(changed, proposal)
                self.assertEqual(result["state"], "REVIEW_REQUIRED")
                self.assertEqual(result["output_html"], changed)
                self.assertFalse(result["checks"]["source_binding"])

    def test_exact_schema_refuses_executable_or_extra_material(self):
        baseline = deterministic_proposal(self.SOURCE)
        proposals = []
        for extra in ("code", "output_html", "script", "approved"):
            changed = deepcopy(baseline)
            changed[extra] = "anything"
            proposals.append(changed)
        extra_binding = deepcopy(baseline)
        extra_binding["bindings"][0]["control_id"] = "other"
        proposals.append(extra_binding)
        proposals.extend([None, [], {}, {**baseline, "methodology": "execute"}, {**baseline, "bindings": "all"}, {**baseline, "source_sha256": True}])
        for proposal in proposals:
            with self.subTest(proposal=proposal):
                result = apply_proposal(self.SOURCE, proposal)
                self.assertEqual(result["state"], "REVIEW_REQUIRED")
                self.assertEqual(result["output_html"], self.SOURCE)

    def test_duplicate_and_forged_candidate_refused(self):
        for bindings in ([{"candidate_id": "invented"}], deterministic_proposal(self.SOURCE)["bindings"] * 2):
            proposal = {**deterministic_proposal(self.SOURCE), "bindings": bindings}
            self.assertEqual(apply_proposal(self.SOURCE, proposal)["state"], "REVIEW_REQUIRED")

    def test_model_may_only_select_candidates(self):
        proposal = {**deterministic_proposal(self.SOURCE), "methodology": "model"}
        self.assertEqual(apply_proposal(self.SOURCE, proposal)["state"], "VERIFIED_SCOPED_REPAIR")

    def test_partial_selection_preserves_remaining_candidates(self):
        source = '<label>First</label><input id="first"><label>Last</label><input id="last">'
        proposal = deterministic_proposal(source)
        proposal["bindings"] = proposal["bindings"][1:]
        result = apply_proposal(source, proposal)
        self.assertEqual(result["state"], "VERIFIED_SCOPED_REPAIR")
        self.assertEqual(len(analyze(result["output_html"])["candidates"]), 1)
        self.assertIn('<label>First</label>', result["output_html"])
        self.assertIn('<label for="last">Last</label>', result["output_html"])

    def test_empty_selection_makes_no_change(self):
        proposal = deterministic_proposal(self.SOURCE)
        proposal["bindings"] = []
        self.assertEqual(apply_proposal(self.SOURCE, proposal)["state"], "NO_CHANGE")

    def test_already_associated_is_no_change(self):
        source = '<label for="x">Name</label><input id="x">'
        self.assertEqual(analyze(source)["status"], "SUPPORTED")
        self.assertEqual(self.repair(source)["state"], "NO_CHANGE")

    def test_unsupported_sources_fail_closed_and_preserve_content(self):
        cases = {
            "duplicate_ids": '<label>A</label><input id="x"><label>B</label><input id="x">',
            "duplicate_noncontrol_id": '<div id="x"></div><label>A</label><input id="x">',
            "duplicate_attributes": '<label>A</label><input id="x" ID="y">',
            "missing_id": '<label>A</label><input>',
            "missing_label": '<input id="x">',
            "label_only": '<label>A</label>',
            "empty_label": '<label> \n </label><input id="x">',
            "zero_width_label": '<label>\u200b</label><input id="x">',
            "encoded_zero_width_label": '<label>&ZeroWidthSpace;</label><input id="x">',
            "bidi_control_label": '<label>\u202eeman</label><input id="x">',
            "two_labels": '<label>A</label><label>B</label><input id="x">',
            "multiple_bound_labels": '<label for="x">A</label><label for="x">B</label><input id="x">',
            "nested_label_content": '<label><span>A</span></label><input id="x">',
            "implicit_nested_control": '<label>A<input id="x"></label>',
            "intervening_text": '<label>A</label>or<input id="x">',
            "intervening_element": '<label>A</label><br><input id="x">',
            "different_parent": '<div><label>A</label></div><input id="x">',
            "hidden": '<label>A</label><input id="x" type="hidden">',
            "hidden_attr": '<label hidden>A</label><input id="x">',
            "css_visibility_unknown": '<label class="hidden">A</label><input id="x">',
            "style": '<label style="display:none">A</label><input id="x">',
            "aria_name": '<label>A</label><input id="x" aria-label="B">',
            "script": '<script>alert(1)</script><label>A</label><input id="x">',
            "event_handler": '<label onclick="alert(1)">A</label><input id="x">',
            "foreign_svg": '<svg><foreignObject><label>A</label><input id="x"></foreignObject></svg>',
            "external_action": '<form action="https://example.invalid"><label>A</label><input id="x"></form>',
            "external_image": '<img src="https://example.invalid/a.png">',
            "misnested": '<div><label>A</div></label><input id="x">',
            "unclosed": '<div><label>A</label><input id="x">',
            "nested_form": '<form><div><form></form></div></form>',
            "block_in_p": '<p><div><label>A</label><input id="x"></div></p>',
            "interactive_button": '<button><span><input id="x"></span></button>',
            "nonvoid_selfclose": '<div/><label>A</label><input id="x">',
            "unquoted_attr": '<label>A</label><input id=x>',
            "broken_quote": '<label>A</label><input id="x>',
            "partial_tag": '<label>A</label><input id="x"><',
            "end_tag_extra": '<label>A</label extra><input id="x">',
            "bound_to_missing": '<label for="other">A</label><input id="x">',
            "bound_to_div": '<label for="x">A</label><div id="x"></div>',
            "comment": '<label>A</label><!-- not explicit whitespace --><input id="x">',
            "doctype": '<!doctype html><label>A</label><input id="x">',
            "processing_instruction": '<?xml version="1.0"?><label>A</label><input id="x">',
            "control_char": '<label>A\x00</label><input id="x">',
            "encoded_control_attr": '<label>A</label><input id="x" value="&#10;">',
        }
        for name, source in cases.items():
            with self.subTest(name=name):
                analysis = analyze(source)
                self.assertEqual(analysis["status"], "UNSUPPORTED", analysis)
                self.assertEqual(analysis["candidates"], [])
                result = self.repair(source)
                self.assertEqual(result["state"], "REVIEW_REQUIRED")
                self.assertEqual(result["output_html"], source)
                self.assertEqual(result["applied_bindings"], [])

    def test_limits_and_invalid_types(self):
        sources = ["x" * (MAX_SOURCE_BYTES + 1), "é" * (MAX_SOURCE_BYTES // 2 + 1), "<div></div>" * 257, "\ud800", None, 42]
        sources.append("".join(f'<label>Name {i}</label><input id="x{i}">' for i in range(MAX_CANDIDATES + 1)))
        for source in sources:
            with self.subTest(type=type(source).__name__):
                self.assertEqual(analyze(source)["status"], "UNSUPPORTED")
                self.assertEqual(self.repair(source)["state"], "REVIEW_REQUIRED")

    def test_empty_fragment_does_not_claim_accessibility(self):
        result = self.repair("")
        self.assertEqual(result["state"], "NO_CHANGE")
        self.assertTrue(result["checks"]["human_evaluation_required"])
        self.assertTrue(any("not full accessibility" in item for item in result["limitations"]))

    def test_encoded_angle_brackets_remain_text(self):
        source = '<label>Use &lt;name&gt;</label><input id="name">'
        self.assertEqual(self.repair(source)["state"], "VERIFIED_SCOPED_REPAIR")
        self.assertEqual(analyze(source)["candidates"][0]["label_text"], "Use <name>")


if __name__ == "__main__":
    unittest.main()
