"""A map result can only enter a later day's pre-match ratings."""
import unittest

import pandas as pd
from scripts.basic_team_features import build_features
from scripts.match_splits_and_metrics import split_matches


class MapEloTests(unittest.TestCase):
    def test_current_veto_and_result_affect_only_later_days(self):
        matches = split_matches(pd.DataFrame({
            'match_id': [11, 12, 13, 14],
            'date': ['2025-01-01 09:00', '2025-01-01 14:00', '2025-01-02 10:00', '2025-01-03 12:00'],
            'team1': ['A'] * 4, 'team2': ['B'] * 4, 'team1_won': [1, 0, 1, 0],
            'tier': ['tier1'] * 4, 'bo': [3] * 4,
        }))
        maps = pd.DataFrame({'match_id': [11, 13], 'map_name': ['Nuke', 'Nuke'], 'team1_won_map': [1, 0]})
        baseline = build_features(matches, maps)
        self.assertEqual(baseline.diff_map_elo.iloc[:2].tolist(), [0., 0.])
        self.assertGreater(baseline.diff_map_elo.iloc[2], 0.)
        maps.loc[1, ['map_name', 'team1_won_map']] = ['Train', 1]
        changed = build_features(matches, maps)
        pd.testing.assert_frame_equal(baseline.iloc[:3], changed.iloc[:3])
        self.assertGreater(changed.diff_map_elo.iloc[-1], baseline.diff_map_elo.iloc[-1])
