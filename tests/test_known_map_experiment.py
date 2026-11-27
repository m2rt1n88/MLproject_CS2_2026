"""Safety checks for the known-map experiment's chronological feature builder."""
import unittest

import numpy as np
import pandas as pd

from scripts.experiment_known_map import (
    PLAYER_MAP, ROSTERS, TEAM_MAP, fit_classifier, known_map_history, paired_delta, predict,
)


class KnownMapTests(unittest.TestCase):
    def sample(self):
        matches = pd.DataFrame({
            'match_id': [1, 2, 3, 4, 5],
            'date': pd.to_datetime(['2025-01-01 09:00', '2025-01-01 14:00', '2025-01-02 10:00',
                                    '2025-01-02 14:00', '2025-05-05 09:00']),
            'team1_key': 'a', 'team2_key': 'b',
        })
        for pid, name in enumerate(ROSTERS[0] + ROSTERS[1], 1):
            matches[name] = pid
        maps = pd.DataFrame({
            'match_id': [1, 2, 3, 4, 5],
            'map_name': ['mirage', 'mirage', 'mirage', 'nuke', 'mirage'],
            'team1_won_map': [1, 1, 0, 1, 0],
        })
        players = pd.DataFrame({
            'match_id': [1, 2], 'map_name': 'mirage', 'player_id': [1, 1],
            'kills': [20, 21], 'deaths': [10, 12], 'adr': [100., 90.], 'kast': [80., 75.],
        })
        return matches, maps, players

    def test_results_box_scores_and_rosters_enter_only_after_day(self):
        matches, maps, players = self.sample()
        original = known_map_history(matches, maps, players)
        maps.loc[0, 'team1_won_map'] = 0
        players.loc[0, ['adr', 'kast', 'kills']] = [30., 32., 5]
        changed = known_map_history(matches, maps, players)
        pd.testing.assert_frame_equal(original.iloc[:2], changed.iloc[:2])
        np.testing.assert_array_equal(original.loc[:1, TEAM_MAP + PLAYER_MAP].to_numpy(), 0.)
        self.assertGreater(original.loc[2, 'map_elo_diff'], changed.loc[2, 'map_elo_diff'])
        self.assertGreater(original.loc[2, 'map_player_adr_diff'], changed.loc[2, 'map_player_adr_diff'])
        # Even using a changed lineup in the target series must not affect its input.
        matches.loc[2, ROSTERS[0][0]] = 55
        lineup = known_map_history(matches, maps, players)
        pd.testing.assert_frame_equal(lineup.iloc[:4], changed.iloc[:4])

    def test_specific_map_and_recent_window(self):
        original = known_map_history(*self.sample())
        self.assertGreater(original.loc[2, 'map_win90_diff'], 0.)
        self.assertGreater(original.loc[2, 'team1_player_coverage'], 0.)
        np.testing.assert_array_equal(original.loc[3, TEAM_MAP + PLAYER_MAP].to_numpy(dtype=float), 0.)  # Nuke
        self.assertEqual(original.loc[4, 'map_win90_diff'], 0.)   # expired history
        self.assertEqual(original.loc[4, 'team1_player_coverage'], 0.)
        self.assertGreater(original.loc[4, 'map_elo_diff'], 0.)   # Elo decays rather than expires

    def test_swap_complements_model_and_paired_delta_is_zero_for_identical_scores(self):
        x = pd.DataFrame({'strength': [-4., -1., 1., 4.]})
        dates = pd.Series(pd.date_range('2025-01-01', periods=4))
        y = np.array([0, 0, 1, 1])
        fitted = fit_classifier(x, y, dates, .01)
        p = predict(fitted, x)
        np.testing.assert_allclose(p + predict(fitted, -x), 1.)
        comparisons = paired_delta(y, p, p, dates, n_boot=20)
        self.assertEqual(comparisons['acc']['ci95_by_day'], [0., 0.])
        self.assertEqual(comparisons['logloss']['ci95_by_day'], [0., 0.])
