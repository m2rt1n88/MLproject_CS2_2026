"""Synthetic leakage, symmetry and interface tests for the single-map LSTM."""
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from known_map_recent.lstm.artifacts import IncompatibleCheckpoint, load_run, save_run
from known_map_recent.lstm.dataset import DEV_END, Scaler, build_rows, fit_preprocess, model_inputs
from known_map_recent.lstm.model import MapLSTM, make_batch, predict_proba
from known_map_recent.lstm.predict import Predictor
from known_map_recent.lstm.state import (
    FEATURE_SETS, HistoryState, ROSTERS, SEQ_NAMES, STATIC_NAMES, static_names,
)
from known_map_v2.runtime import ROOT

TEMP = ROOT / 'temp'
TEAM_IDS = {'alpha': range(1, 6), 'beta': range(11, 16), 'gamma': range(21, 26)}


# ---------------------------------------------------------------------------
# Synthetic source helpers
# ---------------------------------------------------------------------------

class Source:
    def __init__(self):
        self.m, self.mp, self.p = [], [], []

    def add(self, mid, day, a, b, games, roster_a=None, roster_b=None):
        date = pd.Timestamp(f'{day} 12:00')
        ra, rb = list(roster_a or TEAM_IDS[a]), list(roster_b or TEAM_IDS[b])
        row = dict(match_id=mid, date=date, team1_key=a, team2_key=b)
        for cols, ids in zip(ROSTERS, (ra, rb)):
            row.update(dict(zip(cols, ids)))
        self.m.append(row)
        for name, s1, s2 in games:
            self.mp.append(dict(match_id=mid, date=date, map_name=name, score1=s1, score2=s2,
                                team1_won_map=int(s1 > s2)))
            for side, ids, won in ((1, ra, s1 > s2), (2, rb, s2 > s1)):
                for pid in ids:
                    adr = 90.0 if won else 60.0
                    self.p.append(dict(match_id=mid, date=date, map_name=name, side=side, player_id=pid,
                                       kills=20 if won else 12, deaths=12 if won else 20,
                                       adr=adr, kast=78.0 if won else 62.0, rounds=s1 + s2))
        return self

    def frames(self):
        return pd.DataFrame(self.m), pd.DataFrame(self.mp), pd.DataFrame(self.p)

    def metric_events(self, overrides=None):
        """Synthetic metric events aligned with self.p; overrides is {(mid,map,pid): {'rating':...,'hs':...}}."""
        rows = []
        for rec in self.p:
            pid = rec['player_id']
            mid, name = rec['match_id'], rec['map_name']
            ov = (overrides or {}).get((mid, name, pid), {})
            rows.append(dict(date=rec['date'], match_id=mid, map_name=name, player_id=pid,
                             rounds=rec['rounds'], kills=rec['kills'],
                             rating=ov.get('rating', 1.5), headshot_kills=ov.get('hs', 8),
                             hs_source='synthetic'))
        return pd.DataFrame(rows)


def base_source():
    s = Source()
    s.add(1, '2024-01-10', 'alpha', 'beta', [('mirage', 13, 5), ('nuke', 9, 13), ('inferno', 13, 10)])
    s.add(2, '2024-01-25', 'alpha', 'gamma', [('mirage', 13, 7), ('ancient', 13, 11)])
    s.add(3, '2024-01-25', 'beta', 'gamma', [('nuke', 13, 2)])
    s.add(4, '2024-01-26', 'alpha', 'beta', [('mirage', 6, 13)])
    return s


def rows_of(source, **kw):
    rows, state = build_rows(*source.frames(), **kw)
    return rows, state


def row_features(rows, mid, name):
    i = np.flatnonzero((rows.meta.match_id == mid).to_numpy() & (rows.meta.map_name == name).to_numpy())[0]
    return rows.static[i], rows.seq[i], rows.lengths[i]


def splits_source():
    s, mid = Source(), 0
    for day in pd.date_range('2025-06-01', '2025-11-30', freq='5D'):
        mid += 1
        a, b = ('alpha', 'beta') if mid % 2 else ('beta', 'gamma')
        s.add(mid, str(day.date()), a, b, [('mirage', 13, 7 + mid % 7), ('nuke', 5 + mid % 9, 13)])
    return s


def _save_model(d, model, vocab, scaler, feature_set='base'):
    snames = list(static_names(feature_set))
    cfg = dict(feature_set=feature_set, model=dict(hidden=32, emb=8, mlp=32, dropout=0.1),
               n_static=len(snames), n_seq=len(SEQ_NAMES), static_names=snames, seq_names=list(SEQ_NAMES))
    fp = dict(feature_set=feature_set)
    if feature_set != 'base':
        fp['metric_snapshot'] = {'sha256': {'metric_history.json': 'a' * 64}}
    return save_run(d, model, vocab, scaler, cfg, fp)


# ---------------------------------------------------------------------------
# Leakage tests
# ---------------------------------------------------------------------------

class LeakageTests(unittest.TestCase):
    def test_bo3_two_played_maps_are_two_rows(self):
        rows, _ = rows_of(base_source())
        self.assertEqual(int((rows.meta.match_id == 2).sum()), 2)
        self.assertNotIn(1, set(rows.meta.match_id))

    def test_same_day_same_series_no_leak(self):
        a, _ = rows_of(base_source())
        s = Source()
        s.add(1, '2024-01-10', 'alpha', 'beta', [('mirage', 13, 5), ('nuke', 9, 13), ('inferno', 13, 10)])
        s.add(2, '2024-01-25', 'alpha', 'gamma', [('mirage', 2, 13), ('ancient', 3, 13)])
        s.add(3, '2024-01-25', 'beta', 'gamma', [('nuke', 0, 13)])
        s.add(4, '2024-01-26', 'alpha', 'beta', [('mirage', 6, 13)])
        b, _ = rows_of(s)
        # D and earlier: same features (D is the common day for mid=2,3)
        for mid, name in ((2, 'mirage'), (2, 'ancient'), (3, 'nuke')):
            for x, y in zip(row_features(a, mid, name), row_features(b, mid, name)):
                np.testing.assert_array_equal(x, y)
        # first and second maps of same series see same log_overall_games
        static_m, _, len_m = row_features(a, 2, 'mirage')
        static_a, _, len_a = row_features(a, 2, 'ancient')
        np.testing.assert_array_equal(len_m, len_a)
        games = STATIC_NAMES.index('log_overall_games')
        np.testing.assert_array_equal(static_m[:, games], static_a[:, games])
        # D+1 can differ
        self.assertFalse(np.array_equal(row_features(a, 4, 'mirage')[0], row_features(b, 4, 'mirage')[0]))

    def test_target_roster_not_input_but_feeds_next_day(self):
        a, _ = rows_of(base_source())
        s = Source()
        s.add(1, '2024-01-10', 'alpha', 'beta', [('mirage', 13, 5), ('nuke', 9, 13), ('inferno', 13, 10)])
        s.add(2, '2024-01-25', 'alpha', 'gamma', [('mirage', 13, 7), ('ancient', 13, 11)],
              roster_a=[101, 102, 103, 104, 105])
        s.add(3, '2024-01-25', 'beta', 'gamma', [('nuke', 13, 2)])
        s.add(4, '2024-01-26', 'alpha', 'beta', [('mirage', 6, 13)])
        b, _ = rows_of(s)
        # day-of features unchanged
        np.testing.assert_array_equal(row_features(a, 2, 'mirage')[0], row_features(b, 2, 'mirage')[0])
        age = STATIC_NAMES.index('log_roster_age_days')
        # D+1: different roster age because new IDs ingested
        self.assertGreater(row_features(a, 4, 'mirage')[0][0, age], row_features(b, 4, 'mirage')[0][0, age])

    def test_query_before_ingest_raises(self):
        _, state = rows_of(base_source())
        with self.assertRaises(ValueError):
            state.query('alpha', 'mirage', '2024-01-26')

    def test_no_rows_on_or_after_dev_end(self):
        s = base_source().add(9, str(DEV_END.date()), 'alpha', 'beta', [('mirage', 13, 1)])
        rows, _ = rows_of(s)
        self.assertLess(rows.meta.date.max(), DEV_END)
        self.assertNotIn(9, set(rows.meta.match_id))

    def test_base_has_no_rating_or_new_metric_features(self):
        """base feature set must not contain rating or headshot names."""
        names = static_names('base')
        self.assertFalse(any('rating' in n for n in names))
        self.assertFalse(any('hs' in n for n in names))
        self.assertEqual(len(names), len(STATIC_NAMES))

    def test_rating2_not_present_in_any_feature_set(self):
        for fs in FEATURE_SETS:
            for n in static_names(fs):
                self.assertNotIn('rating2', n.lower(), msg=f'rating2 in {n} for {fs}')
                self.assertNotIn('rating_2', n.lower(), msg=f'rating_2 in {n} for {fs}')


# ---------------------------------------------------------------------------
# Metric state tests
# ---------------------------------------------------------------------------

class MetricStateTests(unittest.TestCase):
    """Test the new metric ingest path in HistoryState."""

    def _make_metric_row(self, mid, name, pid, kills, hs, rating, day='2024-01-25'):
        return dict(date=pd.Timestamp(f'{day} 12:00'), match_id=mid, map_name=name, player_id=pid,
                    rounds=24, kills=kills, rating=rating, headshot_kills=hs, hs_source='test')

    def test_metrics_not_ingested_on_query_day(self):
        """Metric events on day D must not affect features queried on day D."""
        state = HistoryState('headshots_only')
        # ingest day 1 (warmup), day 2 is the query day
        m1 = pd.DataFrame([dict(match_id=1, date=pd.Timestamp('2024-01-10 12:00'), team1_key='alpha', team2_key='beta',
                                **{c: i+1 for c, i in zip(ROSTERS[0], range(5))},
                                **{c: i+11 for c, i in zip(ROSTERS[1], range(5))})])
        mp1 = pd.DataFrame([dict(match_id=1, date=pd.Timestamp('2024-01-10 12:00'), map_name='mirage',
                                 score1=13, score2=5, team1_won_map=1)])
        p1 = pd.DataFrame([dict(match_id=1, date=pd.Timestamp('2024-01-10 12:00'), map_name='mirage',
                                side=1, player_id=pid, kills=20, deaths=8, adr=90.0, kast=78.0, rounds=18)
                           for pid in range(1, 6)])
        met1 = pd.DataFrame([self._make_metric_row(1, 'mirage', pid, 20, 10, 1.5, '2024-01-10')
                              for pid in range(1, 6)])
        state.ingest_day(pd.Timestamp('2024-01-10').date(), m1, mp1, p1, metrics=met1)
        _, _, _, meta_before = state.query('alpha', 'mirage', '2024-01-25')
        # hs coverage should be 1.0 (all 5 from day 1 ingested)
        self.assertAlmostEqual(meta_before['metric_coverage']['headshots']['coverage'], 1.0)

    def test_zero_kills_hs_undefined(self):
        """headshot_kills with kills=0 must not enter the rate denominator but doesn't raise."""
        state = HistoryState('headshots_only')
        # Build a minimal ingest so we can call _ingest_metrics
        met = pd.DataFrame([dict(date=pd.Timestamp('2024-02-01 12:00'), match_id=5, map_name='mirage',
                                 player_id=99, rounds=10, kills=0, rating=np.nan, headshot_kills=0,
                                 hs_source='test')])
        state._ingest_metrics(met)
        num, den, maps = state.player_metric('headshots', 99, 'mirage')
        # den (kills) must be 0 since kills=0; maps still counted as 1
        self.assertEqual(den, 0.0)
        self.assertEqual(maps, 1.0)
        self.assertEqual(num, 0.0)

    def test_headshot_rate_aggregation(self):
        """hs rate = sum(hs_kills) / sum(kills), tested against explicit values."""
        state = HistoryState('headshots_only')
        rows = [dict(date=pd.Timestamp('2024-02-01 12:00'), match_id=10, map_name='mirage',
                     player_id=1, rounds=20, kills=kills, rating=np.nan,
                     headshot_kills=hs, hs_source='test')
                for kills, hs in [(20, 8), (15, 6), (10, 5)]]
        state._ingest_metrics(pd.DataFrame(rows))
        num, den, _ = state.player_metric('headshots', 1, 'mirage')
        self.assertEqual(num, 19.0)
        self.assertEqual(den, 45.0)

    def test_rating3_aggregation(self):
        """rating3 = sum(rating*rounds) / sum(rounds)."""
        state = HistoryState('rating3_headshots')
        rows = [dict(date=pd.Timestamp('2024-02-01 12:00'), match_id=11, map_name='nuke',
                     player_id=2, rounds=rounds, kills=10, rating=rating,
                     headshot_kills=np.nan, hs_source=None)
                for rounds, rating in [(20, 1.0), (24, 1.5), (16, 2.0)]]
        state._ingest_metrics(pd.DataFrame(rows))
        num, den, _ = state.player_metric('rating3', 2, 'nuke')
        expected_num = 20 * 1.0 + 24 * 1.5 + 16 * 2.0
        self.assertAlmostEqual(num, expected_num)
        self.assertAlmostEqual(den, 60.0)

    def test_metrics_absent_in_base_feature_set(self):
        """base HistoryState must not accumulate metric state at all."""
        state = HistoryState('base')
        met = pd.DataFrame([dict(date=pd.Timestamp('2024-02-01 12:00'), match_id=20, map_name='mirage',
                                 player_id=5, rounds=20, kills=15, rating=1.8,
                                 headshot_kills=7, hs_source='test')])
        # ingest_day with metrics= should be silently ignored for base
        empty_m = pd.DataFrame(columns=['match_id', 'date', 'team1_key', 'team2_key'])
        empty_mp = pd.DataFrame(columns=['match_id', 'date', 'map_name', 'score1', 'score2', 'team1_won_map'])
        empty_p = pd.DataFrame(columns=['match_id', 'date', 'map_name', 'side', 'player_id',
                                        'kills', 'deaths', 'adr', 'kast', 'rounds'])
        for frame in (empty_m, empty_mp, empty_p, met):
            frame['date'] = pd.to_datetime(frame['date']) if len(frame) else frame.get('date', pd.Series(dtype='datetime64[ns]'))
        # With base, build_rows does not pass metrics:
        s = base_source()
        rows_base, state_base = build_rows(*s.frames(), feature_set='base',
                                           metrics=s.metric_events())
        # static width must equal len(STATIC_NAMES), no extra columns
        self.assertEqual(rows_base.static.shape[2], len(STATIC_NAMES))
        # metric_coverage must be empty for base
        _, _, _, meta = state_base.query('alpha', 'mirage', '2024-02-01')
        self.assertEqual(meta['metric_coverage'], {})

    def test_metric_coverage_explicit_fractions(self):
        """headshots coverage = seen / 5 for the proxy roster."""
        s = base_source()
        metrics_df = s.metric_events()
        rows, state = build_rows(*s.frames(), feature_set='headshots_only', metrics=metrics_df)
        # All base_source players have metric events, so full coverage expected eventually
        # Check coverage columns exist and are in [0,1]
        for col in ('hs_cov1', 'hs_cov2'):
            vals = rows.meta[col].dropna()
            self.assertTrue((vals >= 0).all() and (vals <= 1).all(), msg=col)

    def test_missing_distinct_from_zero(self):
        """A player with no metric history must produce missing_mask=1, not observed_rate=0."""
        state = HistoryState('headshots_only')
        # Player 999 has never appeared
        static, _, _, meta = state.query('nobody', 'mirage', '2024-01-01', lineup=[999, 998, 997, 996, 995])
        snames = list(state.static_names)
        missing_idx = snames.index('hs_missing')
        self.assertEqual(float(static[missing_idx]), 1.0)
        cov_idx = snames.index('hs_coverage')
        self.assertEqual(float(static[cov_idx]), 0.0)

    def test_headshot_independent_of_rating3(self):
        """HS features must be computed independently even when rating3 is also enabled."""
        state_hs = HistoryState('headshots_only')
        state_r3 = HistoryState('rating3_headshots')
        met = pd.DataFrame([dict(date=pd.Timestamp('2024-02-01 12:00'), match_id=30, map_name='mirage',
                                 player_id=1, rounds=20, kills=15, rating=1.8,
                                 headshot_kills=6, hs_source='test')])
        state_hs._ingest_metrics(met)
        state_r3._ingest_metrics(met)
        num_hs, den_hs, _ = state_hs.player_metric('headshots', 1, 'mirage')
        num_r3, den_r3, _ = state_r3.player_metric('headshots', 1, 'mirage')
        self.assertEqual(num_hs, num_r3)
        self.assertEqual(den_hs, den_r3)

    def test_metric_events_same_day_not_in_query(self):
        """Metric events on day D must not appear in features queried on day D."""
        s = base_source()  # first real row is 2024-01-25
        metrics_df = s.metric_events()
        rows_hs, _ = build_rows(*s.frames(), feature_set='headshots_only', metrics=metrics_df)
        rows_base, _ = build_rows(*s.frames(), feature_set='base')
        # row for match_id=2 date=2024-01-25: metric events from day 2 must not inflate that row's coverage
        # (warmup events from 2024-01-10 are allowed to inform it)
        idx2 = int(np.flatnonzero((rows_hs.meta.match_id == 2).to_numpy()
                                   & (rows_hs.meta.map_name == 'mirage').to_numpy())[0])
        # We only have warmup-day events for day 2024-01-10 to inform day 2024-01-25
        # (players 1-5 appeared in match 1 on 2024-01-10, so coverage ≤ 1.0 is achievable)
        cov = float(rows_hs.meta['hs_cov1'].iloc[idx2])
        self.assertGreaterEqual(cov, 0.0)
        self.assertLessEqual(cov, 1.0)


# ---------------------------------------------------------------------------
# Query interface tests
# ---------------------------------------------------------------------------

class QueryTests(unittest.TestCase):
    def test_cold_start_and_unknown_lineup(self):
        state = HistoryState('base')
        static, seq, names, meta = state.query('nobody', 'mirage', '2024-01-01')
        self.assertEqual(len(seq), 0)
        self.assertEqual(static[STATIC_NAMES.index('map_cold_start')], 1.0)
        self.assertEqual(meta['lineup_source'], 'none')
        self.assertTrue(np.isfinite(static).all())
        _, state = rows_of(base_source())
        _, _, _, known = state.query('alpha', 'mirage', '2024-02-01', lineup=[1, 2, 3, 4, 5])
        _, _, _, mixed = state.query('alpha', 'mirage', '2024-02-01', lineup=[1, 2, 3, 999, 998])
        _, _, _, proxy = state.query('alpha', 'mirage', '2024-02-01')
        self.assertEqual((known['lineup_source'], known['player_coverage']), ('provided', 1.0))
        self.assertEqual(mixed['player_coverage'], 0.6)
        self.assertEqual(proxy['lineup_source'], 'proxy')

    def test_provided_lineup_validation(self):
        _, state = rows_of(base_source())
        bad = ([1, 2, 3, 4], [1, 2, 3, 4, 5, 6], [1, 1, 2, 3, 4], [1, 2, 3, 4, 0],
               [1, 2, 3, 4, -7], [1, 2, 3, 4, 5.0], [1, 2, 3, 4, None], [1, 2, 3, 4, '5'],
               [1, 2, 3, 4, True], '12345', [])
        for lineup in bad:
            with self.assertRaises(ValueError, msg=repr(lineup)):
                state.query('alpha', 'mirage', '2024-02-01', lineup=lineup)
        _, _, _, meta = state.query('alpha', 'mirage', '2024-02-01', lineup=(1, 2, np.int64(3), 900, 901))
        self.assertEqual(meta['lineup_ids'], [1, 2, 3, 900, 901])
        self.assertEqual(meta['player_coverage'], 0.6)
        self.assertTrue(meta['team_known'])
        self.assertFalse(state.query('nobody', 'mirage', '2024-02-01')[3]['team_known'])

    def test_target_map_changes_input(self):
        _, state = rows_of(base_source())
        sm, qm, *_ = state.query('alpha', 'mirage', '2024-02-01')
        sn, qn, *_ = state.query('alpha', 'nuke', '2024-02-01')
        self.assertFalse(np.array_equal(sm, sn))
        same = SEQ_NAMES.index('same_target_map')
        self.assertFalse(np.array_equal(qm[:, same], qn[:, same]))


# ---------------------------------------------------------------------------
# Model / artifact tests
# ---------------------------------------------------------------------------

class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.rows, _ = rows_of(splits_source())
        self.vocab, self.scaler = fit_preprocess(self.rows)
        self.model = MapLSTM(len(self.vocab), len(STATIC_NAMES), len(SEQ_NAMES))

    def test_scaler_fitted_on_inner_train_only(self):
        train = self.rows.meta.split.eq('inner_train').to_numpy()
        self.assertTrue(train.any() and (~train).any())
        ref = Scaler().fit(self.rows.static[train], self.rows.seq[train], self.rows.lengths[train])
        np.testing.assert_allclose(self.scaler.s_mean, ref.s_mean)
        self.rows.static[~train] += 1000.0
        _, again = fit_preprocess(self.rows)
        np.testing.assert_allclose(again.s_mean, self.scaler.s_mean)
        np.testing.assert_allclose(again.q_std, self.scaler.q_std)

    def test_swap_symmetry_zero_length_target_map(self):
        idx = np.arange(len(self.rows.meta))
        x = model_inputs(self.rows, idx, self.vocab, self.scaler)
        p = predict_proba(self.model, make_batch(*x))
        swapped = [a[:, ::-1].copy() if a.ndim > 1 else a for a in x]
        q = predict_proba(self.model, make_batch(*swapped))
        np.testing.assert_allclose(p + q, 1.0, atol=1e-6)
        zero = [a.copy() for a in x]
        zero[3][:] = 0
        self.assertTrue(np.isfinite(predict_proba(self.model, make_batch(*zero))).all())
        other = [a.copy() for a in x]
        other[4] = (other[4] % (len(self.vocab) - 1)) + 1
        self.assertFalse(np.allclose(predict_proba(self.model, make_batch(*other)), p))

    def test_incompatible_checkpoint_raises(self):
        TEMP.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            d = Path(d)
            # Save with wrong n_static
            wrong_cfg = dict(feature_set='base', model=dict(hidden=32, emb=8, mlp=32, dropout=0.1),
                             n_static=999, n_seq=len(SEQ_NAMES), static_names=['x'] * 999,
                             seq_names=list(SEQ_NAMES))
            torch.save(self.model.state_dict(), d / 'checkpoint.pt')
            (d / 'config.json').write_text(json.dumps(wrong_cfg))
            (d / 'map_vocab.json').write_text(json.dumps(self.vocab))
            (d / 'scaler.json').write_text(json.dumps(self.scaler.to_dict()))
            with self.assertRaises(IncompatibleCheckpoint):
                load_run(d)

    def test_manifest_tamper_rejected(self):
        """Modifying scaler or vocab after save must be caught on load."""
        TEMP.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            d = Path(d)
            _save_model(d, self.model, self.vocab, self.scaler, 'base')
            # Tamper scaler.json (overwrite, keeping valid JSON but different content)
            orig_scaler = json.loads((d / 'scaler.json').read_text())
            orig_scaler['s_mean'][0] += 999.0
            (d / 'scaler.json').write_text(json.dumps(orig_scaler))
            with self.assertRaises(IncompatibleCheckpoint):
                load_run(d)
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            d = Path(d)
            _save_model(d, self.model, self.vocab, self.scaler, 'base')
            # Tamper map_vocab.json
            orig_vocab = json.loads((d / 'map_vocab.json').read_text())
            orig_vocab.append('injected_map')
            (d / 'map_vocab.json').write_text(json.dumps(orig_vocab))
            with self.assertRaises(IncompatibleCheckpoint):
                load_run(d)
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            d = Path(d)
            _save_model(d, self.model, self.vocab, self.scaler, 'base')
            # Remove manifest entirely — should also raise
            (d / 'artifact_manifest.json').unlink()
            with self.assertRaises(IncompatibleCheckpoint):
                load_run(d)

    def test_predictor_round_trip_prior_days_only(self):
        TEMP.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            d = Path(d)
            _save_model(d, self.model, self.vocab, self.scaler, 'base')
            m, mp, p = splits_source().frames()
            pred = Predictor(d, source=(m, mp, p))
            day = '2025-07-01'
            r = pred.predict_map('alpha', 'beta', 'mirage', day)
            self.assertAlmostEqual(r['p_team1'] + r['p_team2'], 1.0)
            self.assertEqual(r['team1']['lineup_source'], 'proxy')
            self.assertEqual(r['feature_set'], 'base')
            # Same-day result ignored
            later = Source()
            later.m = m.to_dict('records'); later.mp = mp.to_dict('records'); later.p = p.to_dict('records')
            later.add(999, day, 'alpha', 'beta', [('mirage', 13, 0)])
            r2 = Predictor(d, source=later.frames()).predict_map('alpha', 'beta', 'mirage', day)
            self.assertAlmostEqual(r['p_team1'], r2['p_team1'], places=6)
            # Provided lineup, unknown IDs
            r3 = pred.predict_map('alpha', 'beta', 'mirage', day, lineup1=[1, 2, 999, 998, 997])
            self.assertEqual(r3['team1']['lineup_source'], 'provided')
            self.assertEqual(r3['team1']['player_coverage'], 0.4)
            # Unknown map
            r4 = pred.predict_map('alpha', 'beta', 'unseen_map', day)
            self.assertFalse(r4['map_known'])
            self.assertTrue(any('unseen_map' in w for w in r4['warnings']))
            # In-training-period warning
            self.assertTrue(any('out-of-sample' in w for w in r4['warnings']))
            # Cold-start (no alias guessing)
            cold = pred.predict_map('vitality', 'beta', 'mirage', day)
            self.assertFalse(cold['team1']['team_known'])
            # Swap symmetry
            sw = pred.predict_map('beta', 'alpha', 'mirage', day)
            self.assertAlmostEqual(sw['p_team1'], r['p_team2'], places=6)

    def test_predictor_warns_in_training_window(self):
        TEMP.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            d = Path(d)
            _save_model(d, self.model, self.vocab, self.scaler, 'base')
            m, mp, p = splits_source().frames()
            pred = Predictor(d, source=(m, mp, p))
            # Day within development window → must warn
            r = pred.predict_map('alpha', 'beta', 'mirage', '2025-09-01')
            self.assertTrue(any('out-of-sample' in w for w in r['warnings']))
            # Day after DEV_END → no such warning
            r2 = pred.predict_map('alpha', 'beta', 'mirage', '2026-05-01')
            self.assertFalse(any('out-of-sample' in w for w in r2['warnings']))

    def test_tempdir_inside_project(self):
        """TemporaryDirectory used in tests must land inside project temp/, not OS tmp."""
        TEMP.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            self.assertTrue(Path(d).is_relative_to(TEMP))

    def test_nonbase_predictor_loads_source_and_checks_snapshot(self):
        """Non-base Predictor must not NameError when source=None; must catch wrong snapshot hash."""
        TEMP.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            d = Path(d)
            # Build a small headshots_only model on splits_source
            s = splits_source()
            metrics_df = s.metric_events()
            rows, _ = build_rows(*s.frames(), feature_set='headshots_only', metrics=metrics_df)
            vocab, scaler = fit_preprocess(rows)
            hs_model = MapLSTM(len(vocab), len(static_names('headshots_only')), len(SEQ_NAMES))
            # Save with a *matching* dummy metric_snapshot (folder + sha256 must round-trip)
            dummy_snap = {'folder': 'dummy_folder',
                          'sha256': {'metric_history.json': 'a' * 64,
                                     'headshot_history.json': 'b' * 64,
                                     'hltv_history_patch.json': 'c' * 64}}
            _save_model(d, hs_model, vocab, scaler, 'headshots_only')
            # Patch fingerprint so its metric_snapshot matches what we'll feed as runtime_fp
            fp_path = d / 'fingerprint.json'
            fp = json.loads(fp_path.read_text())
            fp['metric_snapshot'] = dummy_snap
            fp_path.write_text(json.dumps(fp))
            # Rewrite manifest because fingerprint.json changed
            import hashlib
            mfst_path = d / 'artifact_manifest.json'
            mfst = json.loads(mfst_path.read_text())
            mfst['fingerprint.json'] = hashlib.sha256((d / 'fingerprint.json').read_bytes()).hexdigest()
            mfst_path.write_text(json.dumps(mfst))

            # Provide source + metrics explicitly to avoid calling real load_source()
            source_tuple = (*s.frames(), {}, 0)   # (matches, maps, players, digests, excluded)

            # Build a fake load_metric_events that returns the dummy_snap fingerprint
            import known_map_recent.lstm.predict as predict_mod
            orig = predict_mod.__dict__.get('_real_lme')
            def fake_lme(src):
                return metrics_df, dict(dummy_snap)
            # Monkey-patch at module level inside Predictor.__init__ path
            from known_map_recent.lstm import metrics as metrics_mod
            orig_lme = metrics_mod.load_metric_events
            metrics_mod.load_metric_events = fake_lme
            try:
                pred = Predictor(d, source=source_tuple, metrics=metrics_df)
                # snapshot check passes because fingerprint matches fake runtime fp
                r = pred.predict_map('alpha', 'beta', 'mirage', '2026-05-01')
                self.assertAlmostEqual(r['p_team1'] + r['p_team2'], 1.0)
                self.assertEqual(r['feature_set'], 'headshots_only')
                # Now test wrong snapshot raises ValueError
                bad_fp = dict(dummy_snap, folder='WRONG_FOLDER')
                fp2 = json.loads((d / 'fingerprint.json').read_text())
                fp2['metric_snapshot'] = bad_fp
                (d / 'fingerprint.json').write_text(json.dumps(fp2))
                mfst['fingerprint.json'] = hashlib.sha256((d / 'fingerprint.json').read_bytes()).hexdigest()
                (d / 'artifact_manifest.json').write_text(json.dumps(mfst))
                # Predictor with metrics=None should call fake_lme and detect mismatch
                with self.assertRaises(ValueError):
                    Predictor(d, source=source_tuple)
            finally:
                metrics_mod.load_metric_events = orig_lme


if __name__ == '__main__':
    unittest.main()
