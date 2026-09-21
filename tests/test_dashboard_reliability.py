import os
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, patch

import pandas as pd

from tradingagents import announcement_body as bodies, asx_signals as signals
from tradingagents import mover_log, scanner, swing, swing_db, swing_ibkr
from tradingagents.scan_volume import relative_volume


class StorageCase(TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        db = Path(self.tmp.name) / 'test.db'
        for module in (signals, mover_log, swing_db):
            p = patch.object(module, 'DB_PATH', db)
            p.start()
            self.addCleanup(p.stop)
        p = patch('tradingagents.asx_feed.DB_PATH', Path(self.tmp.name)/'missing-feed.db')
        p.start()
        self.addCleanup(p.stop)

    def proposal(self):
        return swing_db.create_proposal('BHP', '2026-09-08', 10, 12, 9, 100, 1000, 'test', strategy='range_model')


class DocumentTests(StorageCase):
    def test_net_is_recomputed_when_same_document_is_rescored(self):
        ann = dict(ticker='BHP', fingerprint='x', headline='Results', released_at='2026-09-08T01:00:00Z')
        with signals._ticker_connect() as c:
            c.execute("INSERT INTO signals(fingerprint,ticker,signal,score,classified_at,prompt_version,document_sha256) VALUES('x','BHP','Buy',80,'2026-09-08T02:00:00Z',?,'hash')", (signals.PROMPT_VERSION,))
            c.execute("INSERT INTO ticker_signals(ticker,session_date,signal,score,fingerprints,model,prompt_version,computed_at,n_announcements) VALUES('BHP','2026-09-08','Buy',70,'x',?,?,'2026-09-08T01:00:00Z',1)", (signals._MODEL_LABEL, signals.PROMPT_VERSION))
        with patch('tradingagents.asx_feed.recent_announcements', return_value=[ann]), patch.object(signals, 'combine_ticker_day', return_value=dict(signal='Buy',score=80,reason='results',n_announcements=1)) as combine:
            self.assertEqual(signals.refresh_ticker_signals('2026-09-08')['computed'], 1)
            combine.assert_called_once()

    def test_experimental_lane_requires_document(self):
        from tradingagents import prompt_ab, asx_feed
        feed = Path(self.tmp.name) / 'feed.db'
        with sqlite3.connect(feed) as c:
            c.executescript("CREATE TABLE universe(ticker,company); INSERT INTO universe VALUES('BHP','BHP'); CREATE TABLE announcements(fingerprint,ticker,company,headline,released_at,seen_at,is_halt,halt_kind,price_sensitive,url); INSERT INTO announcements VALUES('x','BHP','BHP','Record results','2026-09-08T01:00:00Z',NULL,0,NULL,1,'url');")
        with patch.object(asx_feed,'DB_PATH',feed), patch.object(bodies,'fetch_body',return_value=''), patch.object(signals,'_ask_counted') as ask:
            result = prompt_ab.score_under('v5-context','2026-09-08')
        self.assertEqual(result['scored'],0)
        ask.assert_not_called()

    def test_missing_body_never_calls_model_even_with_bullish_headline(self):
        rec = dict(ticker='BHP', headline='Record results', fingerprint='x', url='https://example.test')
        with patch.object(bodies, 'fetch_body', return_value=''), patch.object(signals, '_ask_counted') as ask:
            self.assertIsNone(signals.classify_one(rec))
            self.assertIsNone(signals.classify_one(rec, use_body=False))
            ask.assert_not_called()

    def test_document_score_records_exact_text_hash(self):
        text = 'Document details with real words and financial information. ' * 10
        bodies._store('x', text, None, False)
        rec = dict(ticker='BHP', headline='Results', fingerprint='x', url='https://example.test',released_at='2026-09-08T01:00:00Z')
        def answer(*args,**kw):
            if kw['kind']=='memory-extract':
                return '{"facts":[{"topic":"other","subject":"results","claim":"financial information","quote":"Document details with real words and financial information."}]}'
            return '{"score":75,"reason":"growth","changes":[{"current_id":1,"status":"uncertain","prior_ids":[],"reason":"No history"}]}'
        with patch('tradingagents.context_pack.build', return_value=''), patch.object(signals, '_ask_counted', side_effect=answer) as ask:
            result = signals.classify_one(rec)
        self.assertEqual(result['document_sha256'], bodies.evidence('x')['sha256'])
        self.assertEqual(result['document_truncated'], 0)
        self.assertIn('financial information', ask.call_args.args[0])

    def test_transient_failure_retries_and_exhaustion_is_bounded(self):
        with patch.object(bodies.time, 'time', return_value=1000):
            bodies._store('x', '', 'timeout')
            self.assertFalse(bodies.retry_due('x'))
        with patch.object(bodies.time, 'time', return_value=9999999):
            self.assertTrue(bodies.retry_due('x'))
            for _ in range(3):
                bodies._store('x', '', 'timeout')
        with patch.object(bodies.time, 'time', return_value=999999999):
            self.assertFalse(bodies.retry_due('x'))
            self.assertEqual(bodies.evidence('x')['status'], 'document_unavailable')

    def test_legacy_scores_are_preserved_but_not_served(self):
        with signals._connect() as c:
            c.execute("INSERT INTO signals(fingerprint,ticker,signal,score,classified_at) VALUES('x','BHP','Buy',90,'2026-09-08')")
        self.assertEqual(signals.get_signals_for(['x']), {})
        with signals._connect() as c:
            self.assertEqual(c.execute('SELECT score FROM signals').fetchone()[0], 90)

    def test_group_does_not_hide_unread_material_document(self):
        items = [dict(ticker='BHP', fingerprint='x', headline='Results', released_at='2026-09-08T01:00:00Z', score=80, signal='Buy'),
                 dict(ticker='BHP', fingerprint='y', headline='Capital raising', released_at='2026-09-08T01:01:00Z', score=None)]
        with patch.object(signals, 'get_ticker_signals', return_value={}):
            group = signals.group_announcements(items)[0]
        self.assertIsNone(group['score'])
        self.assertEqual(group['n_scored'], 1)


class ScannerTests(StorageCase):
    def test_summer_and_winter_open_use_sydney_zone(self):
        self.assertEqual(scanner.session_state(datetime(2026, 1, 6, 0, 0, tzinfo=timezone.utc))[0], 'open')
        self.assertEqual(scanner.session_state(datetime(2026, 7, 6, 0, 0, tzinfo=timezone.utc))[0], 'open')
        self.assertEqual(scanner.session_state(datetime(2026, 1, 6, 5, 0, tzinfo=timezone.utc))[0], 'closed')

    def test_partial_hour_is_excluded_from_volume(self):
        idx = pd.DatetimeIndex([d + pd.Timedelta(hours=h) for d in pd.bdate_range('2026-08-17', '2026-09-08', tz='Australia/Sydney') for h in (10, 11)])
        df = pd.DataFrame({'Volume': 100., 'Close': 10.}, index=idx)
        df.loc['2026-09-08 10:00', 'Volume'] = 300
        df.loc['2026-09-08 11:00', 'Volume'] = 90000
        value = relative_volume(df, datetime(2026, 9, 8, 1, 30, tzinfo=timezone.utc))
        self.assertEqual(value['ratio'], 3)
        self.assertEqual(value['sessions'], 16)
        self.assertIsNone(relative_volume(df, datetime(2026, 9, 8, 0, 30, tzinfo=timezone.utc)))

    def test_snapshot_preserves_initial_score_and_new_observations(self):
        row = dict(ticker='BHP', last=10, open=9, prev_close=9, ai_score=70)
        mover_log.log_movers([row], '2026-09-08')
        row.update(ai_score=20, last=8)
        # Explicit timestamps in the DB make the same-second unique key deterministic.
        with mover_log._connect() as c:
            c.execute("UPDATE mover_observations SET observed_at='2026-09-08T00:00:00+00:00'")
        mover_log.log_movers([row], '2026-09-08')
        with mover_log._connect() as c:
            self.assertEqual(c.execute('SELECT ai_score, close FROM mover_log').fetchone()[:], (70, 8))
            self.assertEqual(c.execute('SELECT count(*) FROM mover_observations').fetchone()[0], 2)

    def test_stale_bar_cannot_record_current_news_for_past_day(self):
        mover_log.log_movers([dict(ticker='BHP', last=10, open=9, prev_close=9, ai_score=80, is_stale=True)], '2026-09-07')
        with mover_log._connect() as c:
            self.assertEqual(c.execute('SELECT count(*) FROM mover_observations').fetchone()[0], 0)


class SwingTests(StorageCase):
    def test_completed_orders_are_read_and_other_clients_ignored(self):
        def order(oid, client):
            return SimpleNamespace(order=SimpleNamespace(orderId=oid,clientId=client,totalQuantity=100,ocaGroup='g'),
                orderStatus=SimpleNamespace(status='Filled',avgFillPrice=10,filled=100,remaining=0))
        ib=MagicMock()
        ib.reqCompletedOrders.return_value=[order(1,71),order(2,99)]
        ib.trades.return_value=[]
        ib.reqExecutions.return_value=[]
        with patch.object(swing_ibkr,'_connect',return_value=ib):
            result=swing_ibkr.fetch_order_statuses([1,2])
        self.assertEqual(result[1]['filled'],100)
        self.assertNotIn(2,result)

    def test_replaced_exit_fill_history_is_cumulative(self):
        tid=self.proposal()
        swing_db.update_trade(tid,ibkr_target_id=2)
        swing_db.record_order_fills(swing_db.get_trade(tid),{2:dict(filled=40,fill_price=12)})
        swing_db.update_trade(tid,ibkr_target_id=4)
        ledger=swing_db.record_order_fills(swing_db.get_trade(tid),{4:dict(filled=60,fill_price=13)})
        self.assertEqual(ledger['target'],(100,12.6))

    def test_approval_is_claimed_once_and_ids_persist_before_submission(self):
        tid = self.proposal()
        def submit(*args, **kw):
            kw['persist_ids'](dict(parent_id=1, target_id=2, stop_id=3))
            self.assertEqual(swing_db.get_trade(tid)['ibkr_stop_id'], 3)
            return dict(ok=True, parent_id=1, target_id=2, stop_id=3)
        with patch.object(swing_ibkr, 'place_bracket', side_effect=submit) as send:
            self.assertTrue(swing.approve_trade(tid)['ok'])
            self.assertFalse(swing.approve_trade(tid)['ok'])
            send.assert_called_once()

    def test_ambiguous_submission_remains_active(self):
        tid = self.proposal()
        with patch.object(swing_ibkr, 'place_bracket', side_effect=TimeoutError('unknown')):
            self.assertFalse(swing.approve_trade(tid)['ok'])
        self.assertEqual(swing_db.get_trade(tid)['status'], 'attention')
        self.assertTrue(swing_db.has_active_trade('BHP'))

    def test_partial_entry_is_not_expired_and_protection_failure_is_retried(self):
        tid = self.proposal()
        swing_db.update_trade(tid, status='submitted', ibkr_parent_id=1)
        states = {1: dict(status='Cancelled', filled=40, fill_price=10)}
        with patch.object(swing_ibkr, 'fetch_order_statuses', return_value=states), patch.object(swing_ibkr, 'ensure_exits', return_value=dict(ok=False, error='stop rejected')) as repair:
            swing.sync_all()
            swing.sync_all()
        self.assertEqual(repair.call_count, 2)
        self.assertEqual(swing_db.get_trade(tid)['filled_shares'], 40)
        self.assertEqual(swing_db.get_trade(tid)['status'], 'attention')

    def test_no_local_close_until_sibling_is_cancelled(self):
        tid = self.proposal()
        swing_db.update_trade(tid, status='open', ibkr_parent_id=1, ibkr_target_id=2, ibkr_stop_id=3)
        states = {1: dict(status='Filled', filled=100, fill_price=10), 2: dict(status='Filled', filled=100, fill_price=12), 3: dict(status='Submitted', filled=0)}
        with patch.object(swing_ibkr, 'fetch_order_statuses', return_value=states), patch.object(swing_ibkr, 'cancel_order', return_value=False):
            swing.sync_all()
        self.assertEqual(swing_db.get_trade(tid)['status'], 'attention')
        with patch.object(swing_ibkr, 'fetch_order_statuses', return_value=states), patch.object(swing_ibkr, 'cancel_order', return_value=True):
            swing.sync_all()
        self.assertEqual(swing_db.get_trade(tid)['status'], 'closed')
        self.assertEqual(swing_db.get_trade(tid)['pnl'], 200)

    def test_target_changes_in_place_without_cancelling_stop(self):
        order = SimpleNamespace(orderId=2, clientId=71, action='SELL', ocaGroup='swing', lmtPrice=12, totalQuantity=40)
        trade = SimpleNamespace(order=order, contract=SimpleNamespace(symbol='BHP'))
        ib = MagicMock()
        ib.openTrades.return_value = [trade]
        with patch.object(swing_ibkr, '_connect', return_value=ib), patch.object(swing_ibkr, '_acknowledge'):
            result = swing_ibkr.refresh_target_order('BHP', 100, 2, 13)
        self.assertTrue(result['ok'])
        self.assertEqual(order.totalQuantity, 40)
        self.assertEqual(order.lmtPrice, 13)
        ib.cancelOrder.assert_not_called()


class OutcomeTests(StorageCase):
    def test_forward_entry_uses_first_score_and_next_actual_open(self):
        from tradingagents import signal_outcomes, sectors
        from datetime import datetime as real_datetime
        class Clock(real_datetime):
            @classmethod
            def now(cls, tz=None):
                return cls(2026,9,10,0,0,tzinfo=timezone.utc).astimezone(tz)
        with signals._ticker_connect() as c:
            c.execute("INSERT INTO ticker_signals(ticker,session_date,signal,score,n_announcements,fingerprints,prompt_version,computed_at) VALUES('BHP','2026-09-04','Buy',80,1,'x',?,'2026-09-04T01:00:00Z')", (signals.PROMPT_VERSION,))
            c.execute('INSERT INTO ticker_signal_history SELECT * FROM ticker_signals')
            c.execute("UPDATE ticker_signals SET score=90,computed_at='2026-09-07T02:00:00Z'")
        with signal_outcomes._connect() as c:
            c.execute("INSERT INTO signal_outcomes(ticker,as_of,session_date,score,updated_at) VALUES('BHP','2026-09-04','2026-09-04',60,'2026-09-04')")
        idx=pd.to_datetime(['2026-09-04','2026-09-07','2026-09-08'])
        prices=pd.DataFrame({'Open':[10.,11.,12.],'Close':[11.,12.,13.]},index=idx)
        raw=pd.concat({sectors.MARKET_INDEX:prices},axis=1)
        with patch.object(signal_outcomes,'datetime',Clock), patch('tradingagents.screener.bulk_daily',return_value={'BHP':prices}), patch.object(sectors,'get_map',return_value={}), patch.object(sectors,'benchmark_index',return_value=sectors.MARKET_INDEX), patch('yfinance.download',return_value=raw), patch('tradingagents.forward_returns._session_opens',return_value={}):
            signal_outcomes.collect(days=100)
        with signal_outcomes._connect(prospective=True) as c:
            row=c.execute('SELECT * FROM document_signal_outcomes').fetchone()
            self.assertEqual(row['as_of'],'2026-09-07')
            self.assertEqual(row['score'],80)
            self.assertAlmostEqual(row['fwd_1d_pct'],(13/11-1)*100,places=3)
            old=c.execute('SELECT * FROM signal_outcomes').fetchone()
            self.assertEqual(old['score'],60)
            self.assertAlmostEqual(old['fwd_1d_pct'],(12/11-1)*100,places=3)


class AuthTests(TestCase):
    def test_protected_configuration_and_cookie_payload(self):
        from itsdangerous import URLSafeTimedSerializer
        with patch.dict(os.environ, {'TRADINGAGENTS_SESSION_SECRET': 'x' * 48, 'TRADINGAGENTS_WEB_PASSWORD': 'test-only'}):
            import web.server as server
        good = server._signer.dumps('authenticated')
        wrong = server._signer.dumps('other')
        self.assertTrue(server._is_authenticated(SimpleNamespace(cookies={server._SESSION_COOKIE: good})))
        self.assertFalse(server._is_authenticated(SimpleNamespace(cookies={server._SESSION_COOKIE: wrong})))
        self.assertFalse(server._is_authenticated(SimpleNamespace(cookies={server._SESSION_COOKIE: URLSafeTimedSerializer('old-secret').dumps('authenticated')})))
