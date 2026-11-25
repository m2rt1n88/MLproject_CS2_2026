"""Adversarial cleaning cases: quarantine ambiguity without inventing replacements."""
import unittest

import numpy as np
import pandas as pd

from scripts.build_clean_v3 import refine, STATS, SLOTS


def example(mid=1):
    base = dict(match_id=mid, game_id=-mid, datetime='2025-01-02 10:00:00',
                team1=' A  Team ', team2='B Team', team1_id=10, team2_id=20,
                tournament='Event', bestOf=1, score1_match=1, score2_match=0,
                is_total=True, map_name=None, score1_game=np.nan, score2_game=np.nan)
    for side in (1, 2):
        for slot in range(1, 6):
            p = f'team{side}_player{slot}'
            base[p+'_id'] = side * 10 + slot
            for stat, val in zip(STATS, (10, 10, 3, 80., 70., 0)):
                base[p+'_'+stat] = val
    game = {**base, 'game_id': mid, 'is_total': False, 'map_name': ' Mirage ',
            'score1_game': 13, 'score2_game': 7}
    return [base, game]


def run_rows(rows):
    tiers = pd.Series({r['match_id']: 'tier1' for r in rows})
    return refine(pd.DataFrame(rows), tiers)


class CleanV3Tests(unittest.TestCase):
    def test_normalization_and_unknown_format_without_source_mutation(self):
        source = pd.DataFrame(example())
        original = source.copy(deep=True)
        t, e = refine(source, pd.Series({1: 'tier1'}))
        pd.testing.assert_frame_equal(source, original)
        self.assertEqual(t['matches'].team1.iloc[0], 'a team')
        self.assertTrue(t['matches'].bo.isna().all())
        self.assertEqual(len(e[1]), 10)
        self.assertTrue(t['matches'].maps_complete.iloc[0])

    def test_deaths_over_rounds_excludes_map_statistics_not_match(self):
        rows = example(); rows[1]['team1_player2_deaths'] = 21
        t, e = run_rows(rows)
        self.assertEqual(len(t['matches']), 1)
        self.assertEqual(len(t['maps']), 1)
        self.assertEqual(e[1], [])
        self.assertTrue(t['maps'].player_stats_quarantined.iloc[0])
        self.assertIn('map_statistics_internal_contradiction', set(t['audit_log'].reason))

    def test_fractional_player_id_is_not_truncated(self):
        rows = example(); rows[1]['team1_player1_id'] = 11.5
        t, e = run_rows(rows)
        self.assertEqual(len(e[1]), 9)
        self.assertNotIn(11, [p for p, _, _ in e[1]])
        self.assertIn('invalid_player_id', set(t['audit_log'].reason))

    def test_duplicate_and_cross_team_rosters_do_not_supply_stats(self):
        for kind in ('repeat', 'cross'):
            with self.subTest(kind=kind):
                rows = example()
                if kind == 'repeat':
                    rows[1]['team1_player2_id'] = rows[1]['team1_player1_id']
                    expected = 5
                else:
                    rows[1]['team2_player1_id'] = rows[1]['team1_player1_id']
                    expected = 0
                t, e = run_rows(rows)
                self.assertEqual(len(e[1]), expected)
                self.assertEqual(len(t['matches']), 1)

    def test_conflicting_map_candidates_are_all_quarantined(self):
        rows = example(); rows.append({**rows[1], 'game_id': 100, 'score2_game': 8})
        t, e = run_rows(rows)
        self.assertEqual(len(t['maps']), 0)
        self.assertEqual(len(t['quarantined_rows']), 2)
        self.assertEqual(t['quarantined_rows'].team1.tolist(), [' A  Team '] * 2)
        self.assertTrue(t['match_quality'].conflicting_duplicate_map_score.iloc[0])
        self.assertTrue(t['matches'].ambiguous_map.iloc[0])
        self.assertFalse(t['matches'].maps_complete.iloc[0])

    def test_original_aggregate_roster_quarantine_is_preserved(self):
        rows = example(); rows[0]['team1_player2_id'] = rows[0]['team1_player1_id']
        t, e = run_rows(rows)
        self.assertTrue(t['matches'].team1_repeat.iloc[0])
        self.assertTrue(t['matches'][SLOTS[0]].isna().all().all())
        self.assertEqual(len(e[1]), 5)

    def test_aggregate_repeat_and_cross_clear_both_rosters(self):
        rows = example()
        rows[0]['team1_player2_id'] = 11
        rows[0]['team2_player1_id'] = 11
        t, e = run_rows(rows)
        self.assertTrue(t['matches'].roster_cross.iloc[0])
        self.assertTrue(t['matches'][SLOTS[0] + SLOTS[1]].isna().all().all())
        self.assertEqual(e[1], [])

    def test_duplicate_aggregate_can_fill_observed_missing_player(self):
        rows = example(); complete = dict(rows[0]); rows[0]['team1_player2_id'] = np.nan
        rows.append(complete)
        t, e = run_rows(rows)
        self.assertEqual(t['matches'].team1_player2_id.iloc[0], 12)
        self.assertIn('recovered_duplicate_aggregate_roster', set(t['audit_log'].reason))

    def test_reordered_roster_is_not_a_conflict_but_different_player_is(self):
        for other, conflict in [(12, False), (99, True)]:
            rows = example(); duplicate = dict(rows[0])
            duplicate['team1_player1_id'] = other
            duplicate['team1_player2_id'] = 11
            rows.append(duplicate)
            t, e = run_rows(rows)
            self.assertEqual(t['matches'][SLOTS[0]].isna().all(axis=1).iloc[0], conflict)

    def test_duplicate_map_stat_conflict_not_resolved_by_row_order(self):
        rows = example(); rows.append({**rows[1], 'game_id': 100, 'team1_player1_adr': 90})
        t, e = run_rows(rows)
        self.assertEqual(len(t['maps']), 1)
        self.assertEqual(e[1], [])

    def test_conflicting_aggregate_does_not_stop_other_matches(self):
        rows = example() + example(2)
        rows.append({**rows[0], 'team2': 'Other Team'})
        t, e = run_rows(rows)
        self.assertEqual(t['matches'].match_id.tolist(), [2])
        self.assertEqual(set(t['quarantined_rows'].match_id), {1})

    def test_map_date_and_orientation_conflicts_are_not_overwritten(self):
        for field, value in [('datetime', '2025-01-03'), ('team1', 'other team')]:
            with self.subTest(field=field):
                rows = example(); rows[1][field] = value
                t, e = run_rows(rows)
                self.assertEqual(len(t['maps']), 0)
                self.assertEqual(len(t['matches']), 1)

    def test_stat_range_and_kddiff_validation(self):
        rows = example(); rows[1]['team1_player1_kast'] = 101
        t, e = run_rows(rows)
        self.assertEqual(len(e[1]), 9)
        rows = example(); rows[1]['team1_player1_kddiff'] = 1
        t, e = run_rows(rows)
        self.assertEqual(e[1], [])

    def test_no_automatic_score_repair_or_outlier_trimming(self):
        rows = example(); rows[1]['team1_player1_adr'] = 250
        t, e = run_rows(rows)
        self.assertIn(250, [adr for _, adr, _ in e[1]])
        rows = example(); rows[1]['score1_game'] = 1; rows[1]['score2_game'] = 0
        t, e = run_rows(rows)
        self.assertTrue(t['maps'].short_score_needs_review.iloc[0])
        self.assertEqual(t['maps'].score1.iloc[0], 1)


if __name__ == '__main__':
    unittest.main()
