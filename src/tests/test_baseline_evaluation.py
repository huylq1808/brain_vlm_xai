"""Metric definitions and text compliance, not clinical accuracy tests."""
import unittest
from brain_vlm.baseline.evaluation import endpoint,totals,text_quality


class EvaluationTests(unittest.TestCase):
    def test_flagged_error_stays_an_error_and_zero_edit_precision_is_undefined(self):
        r={'study_id':'one','status':'completed','mapped':True,'accepted':False,'edits':0,
           **endpoint('low','low','high',False)}
        m=totals([r]);self.assertEqual(m['wrong_after'],1);self.assertEqual(m['correct_repair'],0)
        self.assertIsNone(m['repair_precision'])

    def test_all_edit_outcomes_are_distinct(self):
        self.assertTrue(endpoint('low','high','high',True)['correct_repair'])
        self.assertTrue(endpoint('high','low','high',True)['new_error'])
        self.assertTrue(endpoint('low','intermediate','high',True)['wrong_to_wrong_edit'])
        self.assertTrue(endpoint('high',None,'high',True)['new_error'])

    def test_unknown_reference_or_unmapped_before_is_not_scored(self):
        self.assertFalse(endpoint('high','high',None,False)['scored'])
        self.assertFalse(endpoint(None,'high','high',True)['scored'])
        self.assertIsNone(totals([{'study_id':'one','status':'failed'}])['wrong_after'])

    def test_text_tail_measurement_and_missing_diffusion_are_not_schema_success(self):
        q=text_quality('A mass is 2.5 cm. No restricted diffusion. The T1C',{'t1','t1c','t2','flair'})
        self.assertEqual(q['sentence_count'],3)
        self.assertIn('unfinished_sentence_tail',q['issues'])
        self.assertIn('measurement_despite_generator_prompt',q['issues'])
        self.assertIn('diffusion_statement_without_supplied_DWI_ADC',q['issues'])

    def test_complete_bounded_paragraph_and_diffusion_evidence(self):
        q=text_quality('A mass is seen. Diffusion restriction is uncertain.',{'dwi','adc'})
        self.assertEqual(q['issues'],[])
