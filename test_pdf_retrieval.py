import unittest
from pdf_retrieval import (wants_new_findings, broad_findings_question, topic_queries,
                           select_diverse_passages, repeated_answer, passage_key)

class PDFRetrievalTests(unittest.TestCase):
    def test_vague_followups_trigger_novelty(self):
        for q in ('something more', 'apart from this', 'what else?',
                  'give me other important findings', 'three additional points'):
            with self.subTest(q=q):
                self.assertTrue(wants_new_findings(q))

    def test_specific_question_is_not_novelty(self):
        self.assertFalse(wants_new_findings('What was revenue in 2025?'))

    def test_summary_searches_multiple_topics(self):
        self.assertGreaterEqual(len(topic_queries('five important findings')), 5)
        self.assertEqual(topic_queries('What was net income?'), ['What was net income?'])

    def test_earlier_sources_excluded(self):
        old={'page':18,'text':'Revenue was $24 billion and AUM reached $14 trillion','score':0.9}
        new={'page':25,'text':'Acquired private markets business for $4 billion in March 2025','score':0.8}
        history=[{'role':'assistant','content':'Revenue was higher','sources':[old]}]
        self.assertEqual([p['page'] for p in select_diverse_passages([old,new],history,2,True)],[25])

    def test_lexical_near_duplicate_excluded(self):
        old={'page':18,'text':'Total managed assets of institutional clients increased in 2025 after significant positive net flows of cash','score':0.9}
        duplicated={'page':19,'text':'Total managed assets of institutional clients increased in 2025 after significant positive net flows of cash','score':0.85}
        fresh={'page':33,'text':'The acquisition of infrastructure investment operations closed during January','score':0.7}
        history=[{'role':'assistant','content':'Assets increased','sources':[old]}]
        self.assertEqual([p['page'] for p in select_diverse_passages([duplicated,fresh],history,2,True)],[33])

    def test_diversifies_topics(self):
        base='Specific financial results confirmed in audited annual data for fiscal year '
        inputs=[{'page':x,'text':base+str(x)+' involving separate operating units and results '+str(x),
                 'score':0.82, 'topic_id':0} for x in (10,11,12)]
        inputs += [{'page':40,'text':'Cash distributions and stock repurchases involved $3.1 billion for holders','score':0.78,'topic_id':2}]
        result=select_diverse_passages(inputs,[],2)
        self.assertEqual(len({p['topic_id'] for p in result}),2)

    def test_detects_wholesale_repetition(self):
        text='Revenue increased significantly in 2025 due to financial markets and improved profitability, while strategy and financial performance were strong in the fiscal period.'
        self.assertTrue(repeated_answer(text,[{'role':'assistant','content':text}]))

    def test_no_false_repetition_for_different_topics(self):
        a='The investment platform faces regulatory oversight and new industry challenges from artificial intelligence and cybersecurity risk programs.'
        b='Cash distributions included dividends and share repurchases while financing transactions and acquisitions expanded the product range.'
        self.assertFalse(repeated_answer(b,[{'role':'assistant','content':a}]))

if __name__=='__main__': unittest.main()
