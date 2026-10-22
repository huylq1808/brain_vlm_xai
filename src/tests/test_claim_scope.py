"""Text-scope regressions; these fixtures do not supply medical image truth."""
import unittest
from pathlib import Path
import tempfile
from PIL import Image

from brain_vlm.baseline.claims import analyze, renderings, request, PREVIOUS_VERSION, RANGE_VERSION, VERSION


class ClaimScopeTests(unittest.TestCase):
    def test_plural_global_and_central_signals_are_distinct(self):
        text = "A mass displays isointense signals on T1W and T2W. Within the center of the lesion a small area exhibits high signal on T2W."
        a = analyze(text)
        self.assertEqual(a["selected"]["direction"], "intermediate")
        self.assertEqual([r["component_scope"] for r in a["mentions"]], ["whole_finding", "central_subregion"])
        req = request("case", text, [])
        self.assertEqual(len(req["claims"]),3)
        self.assertTrue(all(not s['editable'] for s in req['slots'][1:]))

    def test_subregion_alone_cannot_verify_whole_finding(self):
        a = analyze("Within the lesion center, there is high signal on T2W.")
        self.assertIsNone(a["selected"])
        self.assertEqual(a["mentions"][0]["component_scope"],"central_subregion")

    def test_unparsed_t2_mention_is_not_silently_ignored(self):
        a = analyze("The mass has low signal on T2W. Its center is bright in T2W images.")
        self.assertIsNone(a["selected"])
        self.assertEqual(len(a["unparsed_t2_mentions"]),1)

    def test_parenthetical_shared_sequence_edits_only_t2(self):
        text = "A mass shows slightly hyperintense signal on T2-weighted (T2W) and FLAIR sequences."
        a = analyze(text);r=a["selected"]
        self.assertEqual(r["sequences"],["t2","flair"])
        rendered=renderings(text,r)["low"]
        self.assertIn("slightly hypointense signal on T2-weighted (T2W)",rendered)
        self.assertIn("slightly hyperintense signal on FLAIR",rendered)
        self.assertIsNone(renderings(text,r)["intermediate"])
        self.assertEqual(renderings(text,r)["high"],r["source_span"]["text"])

    def test_qualified_other_sequence_does_not_contaminate_t2(self):
        a=analyze("A mass shows iso- to low signal on T1W, high signal on T2W and high signal on FLAIR.")
        self.assertEqual(a["selected"]["direction"],"high")

    def test_terminal_alias_belongs_to_the_same_assertion(self):
        a=analyze("The mass shows high signal on T2-weighted images (T2W).")
        self.assertEqual(a["selected"]["direction"],"high")
        self.assertEqual(a["unparsed_t2_mentions"],[])
        self.assertTrue(a["selected"]["source_span"]["text"].endswith("(T2W)"))

    def test_duplicate_or_compound_whole_directions_are_unknown(self):
        for text in ["A mass is high signal on T2W and low signal on T2W.","A mass has iso- to low signal on T2W."]:
            self.assertIsNone(analyze(text)["selected"])

    def test_source_spans_and_prenominal_phrase(self):
        text="The mass has high T2W signal."
        a=analyze(text);r=a["selected"]
        self.assertEqual(r["direction"],"high")
        s=r["source_span"];self.assertEqual(text[s["start"]:s["end"]],s["text"])

    def test_multiple_findings_remain_ambiguous(self):
        self.assertIsNone(analyze("Two masses exhibit high signal on T2W.")["selected"])
        self.assertIsNone(analyze("Multiple clustered abnormal signal shadows exhibit high signal on T2W.")["selected"])

    def test_negative_signal_is_not_positive_evidence(self):
        self.assertIsNone(analyze("There is no high signal on T2W.")["selected"])

    def test_full_signal_ranges_are_not_single_class_labels(self):
        for phrase in ["isointense to hypointense", "hypointense and hyperintense",
                       "low-to-high", "hypo- to hyperintense", "isointense or slightly hypointense",
                       "low/intermediate"]:
            with self.subTest(phrase=phrase):
                a=analyze(f"A mass exhibits {phrase} signal on T2W.")
                self.assertIsNone(a["selected"])
                self.assertIn("Compound direction is not one categorical value",a["issues"])

    def test_frozen_v3_is_explicit_and_reproducible(self):
        text="A mass exhibits isointense to hypointense signal on T2W."
        self.assertEqual(analyze(text,version=PREVIOUS_VERSION)["selected"]["direction"],"low")
        self.assertIsNone(analyze(text)["selected"])
        self.assertTrue(request("case",text,[],claim_protocol=PREVIOUS_VERSION)["preparation"]["mapped"])
        req=request("case",text,[])
        self.assertEqual(req["preparation"]["claim_protocol"],VERSION)
        self.assertFalse(req["preparation"]["mapped"])
        self.assertTrue(all(not s["editable"] for s in req["slots"]))

    def test_range_in_other_sequence_does_not_block_simple_t2(self):
        text="A mass has isointense to hypointense signal on T1W, hyperintense signal on T2W."
        self.assertEqual(analyze(text)["selected"]["direction"],"high")



    def test_perilesional_context_does_not_automatically_veto_core_proxy(self):
        req=request('fixture','A mass has low signal on T2W. Perilesional edema has high signal on T2W.',[])
        self.assertEqual(req['preparation']['observer_scope_alignment']['status'],'bounded_proxy')


if __name__ == "__main__":unittest.main()
