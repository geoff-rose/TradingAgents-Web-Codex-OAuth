import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from tradingagents import ticker_memory as M, asx_signals as S, announcement_body as B


class MemoryTests(TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        p = patch.object(S,'DB_PATH',Path(self.tmp.name)/'signals.db')
        p.start();self.addCleanup(p.stop)
        p = patch.object(M,'seed_history',return_value=[])
        p.start();self.addCleanup(p.stop)

    def rec(self,fp='one',ticker='SRL',date='2026-09-08T01:00:00Z'):
        return dict(fingerprint=fp,ticker=ticker,released_at=date,headline='Update',url='https://example.test')

    def answer(self,prompt,**kwargs):
        text=prompt.split('Document section:\n')[1]
        quote=text.split('\n')[-2] if text.endswith('\n') else text.split('\n')[-1]
        return json.dumps({'facts':[dict(topic='financing',subject='Project facility',claim=quote,quote=quote)]})

    def test_duplicate_document_reuses_extraction_but_keeps_both_sources(self):
        with patch.object(S,'_ask_counted',side_effect=self.answer) as ask:
            first=M.index_document(self.rec(), '[PAGE 1]\nConditional funding is $400 million.',M.Budget())
            second=M.index_document(self.rec('two'), '[PAGE 1]\nConditional funding is $400 million.',M.Budget())
        self.assertEqual(ask.call_count,1)
        self.assertNotEqual(first['id'],second['id'])
        self.assertEqual(M.facts_for(first['id'])[0]['page'],1)

    def test_budget_resumes_without_repeating_completed_sections(self):
        body='[PAGE 1]\n'+('First section. '*2000)+'\n[PAGE 2]\nFunding is binding.'
        with patch.object(S,'_ask_counted',return_value='{"facts":[]}') as ask:
            first=M.index_document(self.rec(),body,M.Budget(1))
            self.assertFalse(first['complete'])
            second=M.index_document(self.rec(),body,M.Budget(10))
        self.assertTrue(second['complete'])
        self.assertEqual(ask.call_count,len(list(M.sections(body))))
        self.assertIn('Funding is binding.',list(M.sections(body))[-1][1])

    def test_future_equal_timestamp_and_other_ticker_facts_are_excluded(self):
        with patch.object(S,'_ask_counted',side_effect=self.answer):
            for fp,ticker,date in [('old','SRL','2026-09-01T00:00:00Z'),('future','SRL','2026-09-09T00:00:00Z'),('equal','SRL','2026-09-08T11:00:00+10:00'),('other','BHP','2026-09-01T00:00:00Z')]:
                M.index_document(self.rec(fp,ticker,date),'Funding is conditional.',M.Budget())
        facts,limited=M.prior_facts('SRL','2026-09-08T01:00:00Z',[{'topic':'financing','subject':'Project facility'}])
        self.assertEqual([f['fingerprint'] for f in facts],['old'])

    def test_invented_quote_cannot_enter_fact_store(self):
        answer=json.dumps({'facts':[dict(topic='financing',subject='facility',claim='funded',quote='invented')]})
        with patch.object(S,'_ask_counted',return_value=answer):
            with self.assertRaises(ValueError):
                M.index_document(self.rec(),'Funding remains conditional.',M.Budget())
        self.assertEqual(M.inspect('SRL')['documents'][0]['facts'],[])

    def test_pdf_line_wrapping_is_restored_without_accepting_changed_words(self):
        self.assertEqual(M.source_quote('Funding is\n  conditional.','Funding is conditional.'),'Funding is\n  conditional.')
        with self.assertRaises(ValueError):
            M.source_quote('Funding is conditional.','Funding is unconditional.')

    def test_missing_history_cannot_be_labelled_proven_new(self):
        with patch.object(S,'_ask_counted',side_effect=self.answer):
            prepared=M.prepare(self.rec(),'Funding is conditional.')
        fid=prepared['current'][0]['id']
        parsed={'changes':[dict(current_id=fid,status='new',prior_ids=[],reason='not in history')]}
        M.save_comparison(prepared,parsed,60)
        self.assertEqual(parsed['changes'][0]['status'],'uncertain')
        with self.assertRaises(ValueError):
            M.save_comparison(prepared,{'changes':[dict(current_id=fid,status='changed',prior_ids=[999])]},60)

    def test_contradictory_facts_are_preserved(self):
        with patch.object(S,'_ask_counted',side_effect=self.answer):
            old=M.index_document(self.rec('old',date='2026-09-01T00:00:00Z'),'Funding is conditional.',M.Budget())
            new=M.index_document(self.rec('new'),'Funding is binding.',M.Budget())
        self.assertEqual(M.facts_for(old['id'])[0]['claim'],'Funding is conditional.')
        self.assertEqual(M.facts_for(new['id'])[0]['claim'],'Funding is binding.')

    def test_all_pdf_pages_survive_extraction_and_visual_only_page_blocks(self):
        pages=[SimpleNamespace(extract_text=lambda:'A'*13000,images=[]),SimpleNamespace(extract_text=lambda:'Material news at the end.',images=[])]
        with patch('pypdf.PdfReader',return_value=SimpleNamespace(pages=pages)):
            text=B.extract_text(b'pdf')
        self.assertIn('[PAGE 2]\nMaterial news at the end.',text)
        pages.append(SimpleNamespace(extract_text=lambda:'',images=[object()]))
        with patch('pypdf.PdfReader',return_value=SimpleNamespace(pages=pages)):
            with self.assertRaisesRegex(ValueError,'Page 3'):
                B.extract_text(b'pdf')

    def test_incomplete_extraction_cannot_be_scored(self):
        body='Investment facts and source text. '*2000
        with patch.object(B,'fetch_body',return_value=body),patch.object(S,'_ask_counted',return_value='{"facts":[]}') as ask:
            self.assertIsNone(S.classify_one(self.rec(),memory_budget=M.Budget(1)))
        self.assertEqual([c.kwargs['kind'] for c in ask.call_args_list],['memory-extract'])
