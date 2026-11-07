import unittest
from finance_analytics import compute_financial_ratios
from financial_guardrails import verified_answer

class GuardrailTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            {'Year': 2026, 'Revenue': 478.834, 'Operating income': 158.063,
             'Net income': 132.175, 'Depreciation and amortization': None},
            {'Year': 2025, 'Revenue': 416.161, 'Operating income': 133.05,
             'Net income': 112.01, 'Depreciation and amortization': None},
        ]
        self.ratios = compute_financial_ratios(self.records)

    def test_missing_ebitda_does_not_invent(self):
        response = verified_answer('How did net EBITDA change?',self.records,self.ratios)
        self.assertIn('Insufficient verified data',response)
        self.assertIn('depreciation and amortization',response)
        self.assertNotIn('25.888',response)

    def test_ebitda_calculated_when_da_available(self):
        self.records[0]['Depreciation and amortization']=10.0
        self.records[1]['Depreciation and amortization']=9.0
        response=verified_answer('How did EBITDA change from 2025 to 2026?',self.records,self.ratios)
        self.assertIn('$168.06B',response)
        self.assertIn('$142.05B',response)
        self.assertIn('+26.01 USD billions',response)

    def test_margin_change_points(self):
        response=verified_answer('How did operating margin change 2025 to 2026?',self.records,self.ratios)
        self.assertIn('percentage points',response)
        self.assertNotIn('LaTeX',response)

    def test_unknown_question_can_use_qualitative_llm(self):
        self.assertIsNone(verified_answer('What business strategy is Apple pursuing?',self.records,self.ratios))

    def test_unsupported_ratio_blocked(self):
        self.assertIn('Not available',verified_answer('What is the gross margin?',self.records,self.ratios))

if __name__=='__main__':
    unittest.main()
