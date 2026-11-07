import tempfile
import unittest
from pathlib import Path
from finance_analytics import compute_financial_ratios, simple_cashflow_dcf
from financial_guardrails import interpret_previous_metric, resolve_followup_question
from history_store import save_session, list_sessions

class Day5Tests(unittest.TestCase):
    def setUp(self):
        self.records = [
            {'Year': 2024, 'Revenue': 100, 'Operating income': 20, 'Net income': 12, 'Total assets': 80, 'Stockholders equity': 40},
            {'Year': 2025, 'Revenue': 120, 'Operating income': 30, 'Net income': 17, 'Total assets': 90, 'Stockholders equity': 43},
            {'Year': 2026, 'Revenue': 140, 'Operating income': 42, 'Net income': 22, 'Total assets': 100, 'Stockholders equity': 46},
        ]
        self.ratios = compute_financial_ratios(self.records)

    def test_good_or_bad_follows_last_roa(self):
        chat = [{'role':'user', 'content':'How did ROA change from 2025 to 2026?'},
                {'role':'assistant', 'content':'ROA rose.'}]
        answer = interpret_previous_metric('was it good or bad?', chat, self.records, self.ratios)
        self.assertIn('ROA', answer)
        self.assertIn('FY2025', answer)
        self.assertIn('FY2026', answer)
        self.assertNotIn('revenue declined', answer.lower())

    def test_unknown_metric_requests_clarification(self):
        self.assertIn('Which financial metric', interpret_previous_metric('is this good?', [], self.records, self.ratios))

    def test_followup_formula(self):
        chat=[{'role':'user','content':'Operating margin in 2025 and 2026?'}]
        self.assertIn('Operating margin', resolve_followup_question('how did you calculate that?', chat))

    def test_average_asset_roa(self):
        row = self.ratios.loc[self.ratios['Year'] == 2026].iloc[0]
        self.assertAlmostEqual(row['ROA, average assets (%)'], 22/95*100)

    def test_dcf_rejects_invalid_terminal_growth(self):
        with self.assertRaises(ValueError):
            simple_cashflow_dcf(10, .02, .04, .03, 5)

    def test_opt_in_sqlite_save(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'history.sqlite3'
            self.assertEqual(list_sessions(path), [])
            save_session('sec','AAPL',[{'role':'user','content':'Margin?'}],path)
            rows=list_sessions(path)
            self.assertEqual(len(rows),1)
            self.assertEqual(rows[0]['label'],'AAPL')

if __name__ == '__main__':
    unittest.main()
