import unittest

from known_map_v2.runtime import configure

configure()

import numpy as np
import pandas as pd

from known_map_v2.pre_veto import pre_veto_inputs


class PreVetoTests(unittest.TestCase):
    def test_only_earlier_days_and_same_probability_for_all_series_maps(self):
        history = pd.DataFrame({'day': pd.to_datetime(['2024-01-01', '2024-01-03']),
                                'team1': ['A', 'A'], 'team2': ['B', 'B'],
                                'map': ['nuke', 'mirage'], 'team1_win': [1, 0]})
        queries = pd.DataFrame({'day': pd.to_datetime(['2024-01-01', '2024-01-03', '2024-01-03', '2024-01-04']),
                                'team1': ['A', 'A', 'A', 'A'], 'team2': ['B', 'B', 'B', 'B'],
                                'map': ['nuke', 'nuke', 'mirage', 'ancient'], 'team1_win': [1, 1, 0, 0]})
        x, seen = pre_veto_inputs(history, queries)
        self.assertEqual(tuple(seen.sum(axis=1)), (0, 2, 2, 2))
        self.assertEqual(x[0, 0, 0], x[0, 1, 0])
        self.assertGreater(x[1, 0, 0], x[1, 1, 0])
        self.assertLess(x[3, 0, 0], x[1, 0, 0])
        np.testing.assert_array_equal(x[1], x[2])
        x2, _ = pre_veto_inputs(history.assign(map='vertigo'), queries.assign(map='inferno', team1_win=0))
        np.testing.assert_array_equal(x, x2)

    def test_reversing_teams_reverses_past_features(self):
        history = pd.DataFrame({'day': pd.to_datetime(['2024-01-01']), 'team1': ['A'],
                                'team2': ['B'], 'team1_win': [0]})
        q = history.assign(day=pd.to_datetime(['2024-01-02']))
        x, _ = pre_veto_inputs(history, q)
        rev, _ = pre_veto_inputs(history, q.rename(columns={'team1': 'team2', 'team2': 'team1'}))
        np.testing.assert_array_equal(rev, x[:, ::-1])


if __name__ == '__main__':
    unittest.main()
