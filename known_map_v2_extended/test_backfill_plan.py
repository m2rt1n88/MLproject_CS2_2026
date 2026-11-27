"""Check the offline priority queue against the same cutoff and local evidence rules."""
import json
import unittest

from known_map_v2_extended.backfill_plan import build_plan
from known_map_v2_extended.build import HOME, reconstruct
from known_map_v2_extended.test_build import sample


class BackfillPlanTests(unittest.TestCase):
    def test_ranking_requires_past_same_map_and_excludes_conflicting_series(self):
        m, g, p = sample()
        g['date'], g['score1'], g['score2'] = m['date'], 13, 7
        p['date'] = p.match_id.map(m.set_index('match_id').date)
        jobs = {
            2: ({'hltv_id': 20, 'match_url': 'synthetic', 'evidence': 'archive_scores'}, {'mirage': [13, 7]}),
            3: ({'hltv_id': 30, 'match_url': 'synthetic', 'evidence': 'archive_scores'}, {'mirage': [13, 8]}),
        }
        # Only matches 4 and 5 are target maps; matches 1/6 are outside the relevant windows.
        g = g[g.match_id.ne(6)].copy()
        scores = {(3, 'mirage', pid): {'rating': 1.25} for pid in range(1, 11)}
        p = p[p.match_id.ne(6) & (p.match_id.ne(1) | p.player_id.ne(2))].copy()
        # Day 1 is one day too early. The two same-day targets cannot feed each other.
        out = build_plan(m, g, p, history=scores, jobs=jobs)
        self.assertEqual(out['target_maps'], 2)
        self.assertEqual(out['used_history_maps'], 2)
        self.assertEqual(out['score_conflict_series'], 1)
        self.assertEqual(out['current_complete_team_windows']['rating'], 0)
        first, other = out['rows']
        self.assertEqual(first['local_match_id'], 2)
        self.assertEqual(first['possible_completions_if_only_this_map_added']['rating'], 4)
        self.assertIsNone(other['candidate'])
        self.assertEqual(out['suggested_uncached_rating_pilot'][0]['hltv_id'], 20)
        for pid in range(1, 11):
            scores[2, 'mirage', pid] = {'rating': 1.05}
        out = build_plan(m, g, p, history=scores, jobs=jobs)
        self.assertEqual(out['current_complete_team_windows']['rating'], 4)
        dataset, _ = reconstruct(m, g, p, metric_history=scores)
        for score in dataset['T1_Rating_3.0_mean']:
            self.assertAlmostEqual(score, 1.15)
        self.assertNotIn('T1_Rating_2.0_mean', dataset)

    def test_no_vote_from_a_different_map_or_partial_roster(self):
        m, g, p = sample()
        g['date'], g['score1'], g['score2'] = m['date'], 13, 7
        p['date'] = p.match_id.map(m.set_index('match_id').date)
        g.loc[g.match_id.eq(2), 'map_name'] = 'nuke'
        p.loc[p.match_id.eq(2), 'map_name'] = 'nuke'
        p = p[p.player_id.ne(1)]
        out = build_plan(m, g, p)
        self.assertNotIn('nuke', [r['map'] for r in out['rows']])
        self.assertEqual(out['eligible_team_windows']['rating'], 3)

    def test_published_plan_matches_five_metric_snapshot(self):
        root = HOME/'data'
        plan = root/'backfill_plan_ranked.json'
        plan = json.loads(plan.read_text(encoding='utf-8'))
        base = json.loads((root/'five_metrics_v1/manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(plan['target_maps'], base['appended_maps'])
        self.assertEqual(plan['current_complete_team_windows']['rating'], 1)
        self.assertEqual(plan['linked_candidate_maps'], sum(bool(r['candidate']) for r in plan['rows']))
        self.assertEqual(len({(r['local_match_id'], r['map']) for r in plan['rows']}), plan['used_history_maps'])
        self.assertEqual(len({r['hltv_id'] for r in plan['suggested_uncached_rating_pilot']}), 12)
