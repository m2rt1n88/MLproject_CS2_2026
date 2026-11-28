"""Offline tests for the bulk HLTV backfill (no network)."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from known_map_v2_extended import bulk_backfill as bb
from known_map_v2_extended.offline_mapstats import Rejected

TMP = Path(__file__).resolve().parents[1] / 'temp'


def box(base, shift, map_name, scores, mapstats_id):
    players = [dict(hltv_player_id=base + 10 * (side - 1) + i, kills=10 + i + shift, deaths=10 + side,
                    adr=70.0 + i + shift, kast=70.0 + side, rating=1.0, rating_version='3.0', page_side=side)
               for side in (1, 2) for i in range(5)]
    return dict(mapstats_id=mapstats_id, map_name=map_name, scores=scores, players=players, rating_version='3.0')


def frames(team1='beta', extra_page_map=False):
    s = dict(match_id=99, day='2025-08-12', bo=3,
             teams=[dict(name='Alpha Esports', score=2), dict(name='Beta', score=1)],
             maps=[box(100, 0, 'ancient', [13, 10], 5), box(100, 1, 'train', [13, 11], 6),
                   box(100, 2, 'nuke', [9, 13], 7)])
    matches = pd.DataFrame([dict(match_id=1, date=pd.Timestamp('2025-08-12'), team1_key=team1, team2_key='alpha',
                                 score1=1.0, score2=2.0, bo_reported=3.0)])
    local = [('ancient', 10, 13), ('train', 11, 13)] + ([] if extra_page_map else [('nuke', 13, 9)])
    maps = pd.DataFrame([dict(match_id=1, map_name=n, score1=float(a), score2=float(b),
                              player_stats_quarantined=False) for n, a, b in local])
    rows = [dict(match_id=1, map_name=m['map_name'], side=3 - r['page_side'], player_id=r['hltv_player_id'] + 5000,
                 kills=float(r['kills']), deaths=float(r['deaths']), adr=r['adr'], kast=r['kast'])
            for m in s['maps'] for r in m['players'] if m['map_name'] in set(maps.map_name)]
    return s, matches, maps, pd.DataFrame(rows)


class AnchorTests(unittest.TestCase):
    def test_alias_needs_two_full_fingerprint_maps(self):
        s, matches, maps, players = frames()
        fps = bb.local_fp_index(players, {1})
        lid, flip, kept, dropped, method = bb.anchored_link(s, dict(local_match_ids=[1]), matches, maps, set(), fps)
        self.assertEqual((lid, flip, method, dropped), (1, True, 'plan_anchor_fp_alias', []))
        self.assertEqual(len(kept), 3)
        # Break fingerprints on two maps: only one full 5v5 box remains -> reject.
        broken = players.copy()
        broken.loc[broken.map_name.isin(['ancient', 'train']), 'adr'] += 5
        with self.assertRaises(Rejected):
            bb.anchored_link(s, dict(local_match_ids=[1]), matches, maps, set(), bb.local_fp_index(broken, {1}))

    def test_missing_local_map_dropped_not_relabelled(self):
        s, matches, maps, players = frames(extra_page_map=True)
        matches.loc[0, 'team1_key'] = 'beta'
        _, _, kept, dropped, _ = bb.anchored_link(s, dict(local_match_ids=[1]), matches, maps, set(),
                                                  bb.local_fp_index(players, {1}))
        self.assertEqual(sorted(m['map_name'] for m in kept), ['ancient', 'train'])
        self.assertEqual(dropped, [dict(map_name='nuke', reason='map_absent_from_local_snapshot')])

    def test_anchor_requires_day_bo_and_single_plan_id(self):
        s, matches, maps, players = frames()
        fps = bb.local_fp_index(players, {1})
        for plan in (dict(local_match_ids=[]), dict(local_match_ids=[1, 2])):
            with self.assertRaises(Rejected):
                bb.anchored_link(s, plan, matches, maps, set(), fps)
        with self.assertRaises(Rejected):
            bb.anchored_link(dict(s, day='2025-08-13'), dict(local_match_ids=[1]), matches, maps, set(), fps)
        with self.assertRaises(Rejected):  # series score disagrees
            bb.anchored_link(dict(s, teams=[dict(name='Alpha Esports', score=2), dict(name='Beta', score=0)]),
                             dict(local_match_ids=[1]), matches, maps, set(), fps)


class MergeAndQueueTests(unittest.TestCase):
    def test_conflict_isolated_not_fatal(self):
        r = dict(match_id=1, map_name='ancient', player_id=7, day='d', side=1, kills=1, deaths=1, adr=1.0, kast=1.0,
                 rounds=23, mapstats_id=5, hltv_player_id=3, metrics={'rating': 1.0}, rating_version='3.0',
                 metric_source='m', hltv_id=9)
        merged, added, dup, conflicts = bb.merge_isolated(dict(rows=[r]), [dict(r), dict(r, metrics={'rating': 2.0})])
        self.assertEqual((len(merged['rows']), added, dup, len(conflicts)), (1, [], 1, 1))

    def test_saved_pages_ignores_other_files_and_checks_duplicates(self):
        with TemporaryDirectory(dir=TMP) as a, TemporaryDirectory(dir=TMP) as b:
            for d in (a, b):
                (Path(d) / 'match_5.html').write_text('x', encoding='utf-8')
            (Path(a) / 'alltime_gadr.html').write_text('y', encoding='utf-8')
            pages, dupes = bb.saved_pages([a, b])
            self.assertEqual((list(pages), len(dupes)), ([('match', 5)], 1))
            (Path(b) / 'match_5.html').write_text('z', encoding='utf-8')
            with self.assertRaises(ValueError):
                bb.saved_pages([a, b])

    def test_queue_skips_saved_and_is_unique(self):
        series = {i: dict(hltv_id=i, match_url=f'https://www.hltv.org/matches/{i}/x', day='d', maps=['a'] * (i % 3),
                          rating_completions=i, rating_windows=0) for i in range(1, 8)}
        with TemporaryDirectory(dir=TMP) as d:
            pending, batch = bb.write_queue(series, {('match', 2): None}, d, 3)
            self.assertNotIn(2, [s['hltv_id'] for s in pending])
            self.assertEqual(len(batch), 3)
            self.assertEqual(batch[0]['hltv_id'], 5)  # the only multi-map series comes first
            self.assertEqual(batch, sorted(batch, key=bb.priority))
            lines = (Path(d) / 'next_browser_commands.txt').read_text(encoding='utf-8').splitlines()
            self.assertEqual(len(set(lines)), 3)


if __name__ == '__main__':
    unittest.main()
