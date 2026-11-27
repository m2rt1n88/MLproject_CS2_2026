import unittest
from unittest.mock import patch

from known_map_v2.runtime import configure

configure()

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from known_map_v2.data_adapter import TARGET, load, pairs, swap
from known_map_v2.evaluation import assert_split, fit_stacking, inner_splits, outer_splits, score
from known_map_v2.models import BASES, build_elo, encode, fit, predict


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.manifest = load()

    def test_schema_identity_and_dates(self):
        self.assertEqual((len(self.rows), self.rows.matchID.nunique()), (1010, 577))
        self.assertEqual(len(pairs(self.rows)), 71)
        self.assertFalse(self.rows.duplicated(['matchID', 'map']).any())
        self.assertEqual(self.rows.groupby('matchID').day.nunique().max(), 1)
        self.assertEqual(self.manifest['raw_date_time_columns'], [])

    def test_outer_and_inner_folds_keep_whole_series(self):
        splits = list(outer_splits(self.rows))
        self.assertEqual(len(splits), 31)
        for name, tr, te in splits:
            chronological = name == 'date_holdout'
            assert_split(self.rows, tr, te, chronological)
            self.assertEqual(len(tr) + len(te), len(self.rows))
            training = self.rows.iloc[tr]
            for a, b in inner_splits(training, chronological):
                assert_split(training, a, b, chronological)
        self.assertEqual(self.rows.iloc[splits[-1][2]].day.min(), pd.Timestamp('2024-01-16'))

    def test_swap_moves_all_stats_and_flips_label(self):
        rows = self.rows.iloc[:5]
        reverse = swap(rows)
        self.assertTrue(reverse['map'].equals(rows['map']))
        self.assertTrue(reverse['T1_mapwinrate'].equals(rows['T2_mapwinrate']))
        self.assertTrue(reverse[TARGET].equals(1 - rows[TARGET]))
        pd.testing.assert_frame_equal(swap(reverse)[rows.columns], rows)

    def test_m2_features_exclude_metadata_and_use_map_interactions(self):
        rows = self.rows.iloc[:5]
        b = {'view': 'diff_interaction', 'stats': pairs(self.rows)}
        x = encode(b, rows)
        with_ids_changed = rows.assign(matchID=123, day=pd.Timestamp('2030-01-01'), team1='Other A', team2='Other B')
        np.testing.assert_array_equal(encode(b, with_ids_changed.drop(columns=[TARGET])), x)
        reversed_x = encode(b, swap(rows))
        np.testing.assert_array_equal(reversed_x, -x)
        self.assertEqual(x.shape[1], 85)
        with_new_map = rows.assign(map=np.where(rows['map'].eq('mirage'), 'nuke', 'mirage'))
        self.assertFalse(np.array_equal(encode(b, with_new_map), x))

    def test_train_then_predict_without_results_or_series_ids(self):
        r = self.rows.iloc[:120]
        with threadpool_limits(1):
            model = fit('two_feature_logistic', r, pairs(self.rows))
        x = r.iloc[:3][['team1', 'team2', 'map', 'T1_mapwinrate', 'T2_mapwinrate',
                        'T1_Rating_2.0_mean', 'T2_Rating_2.0_mean']]
        np.testing.assert_allclose(predict(model, x) + predict(model, swap(x)), 1)
        with self.assertRaisesRegex(ValueError, 'Unknown map'):
            predict(model, x.assign(map='none'))
        with self.assertRaisesRegex(ValueError, 'fraction'):
            predict(model, x.assign(T1_mapwinrate=50))

    def test_metric_precision_matches_saved_probability(self):
        y = np.array([0, 1, 0, 1])
        p = np.array([.123456, .523456, .537896, .678912], dtype=np.float32)
        self.assertEqual(score(y, p), score(y, p.astype(np.float64)))

    def test_stacking_out_of_fold_base_training(self):
        t, v = list(inner_splits(self.rows, temporal=False))[0]
        rows = self.rows.iloc[t].copy()
        outside_ids = set(self.rows.iloc[v].matchID)

        def fake_fit(n, training, stats):
            return set(training.matchID)

        def fake_predict(training_ids, cases):
            assert not (training_ids & set(cases.matchID))
            assert not (outside_ids & (training_ids | set(cases.matchID)))
            return np.full(len(cases), .51)

        with patch('known_map_v2.evaluation.fit', side_effect=fake_fit), \
                patch('known_map_v2.evaluation.predict', side_effect=fake_predict):
            b = fit_stacking(rows, [], {n: n for n in BASES}, temporal=False)
        self.assertEqual(b['n_meta'], len(rows))


class EloTests(unittest.TestCase):
    def test_only_strictly_earlier_results_apply(self):
        rows = pd.DataFrame({'team1': ['A', 'A'], 'team2': ['B', 'B'], 'map': ['mirage'] * 2,
                             'day': pd.to_datetime(['2024-01-01', '2024-01-03']), TARGET: [1, 0]})
        model = {'name': 'rolling_elo', 'history': build_elo(rows)}
        queries = rows.iloc[[0, 0, 1]].assign(day=pd.to_datetime(['2023-12-31', '2024-01-01', '2024-01-03']))
        before, same, later = predict(model, queries)
        self.assertEqual(before, .5)
        self.assertEqual(same, .5)
        self.assertGreater(later, .5)
        self.assertAlmostEqual(predict(model, swap(queries))[-1], 1 - later)


if __name__ == '__main__':
    unittest.main()
