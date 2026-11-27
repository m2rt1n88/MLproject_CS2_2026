"""Check the temporal splice, calculations, and published snapshot."""
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from known_map_v2 import data_adapter as v2
from known_map_v2_extended.build import HOME, reconstruct, ROSTERS, team_features, upgraded_columns
from known_map_v2_extended.opencs2_headshots import demo_counts, local_nickname


def sample():
    # With target 2024-01-21, The window starts 2023-10-23 (inclusive).
    dates = pd.to_datetime(['2023-10-22', '2023-10-23', '2024-01-20', '2024-01-21', '2024-01-21', '2024-01-22'])
    matches = pd.DataFrame(dict(date=dates, match_id=[1, 2, 3, 4, 5, 6], team1_key='a', team2_key='b'))
    for i, c in enumerate(ROSTERS[0] + ROSTERS[1], 1):
        matches[c] = i
    maps = matches[['match_id']].assign(map_name='mirage', team1_won_map=[1, 0, 1, 1, 0, 1])
    players = pd.DataFrame([dict(match_id=mid, map_name='mirage', player_id=p, kills=p, deaths=10, assists=5,
                                 adr=80, rounds=20) for mid in matches.match_id for p in range(1, 11)])
    return matches, maps, players


class ReconstructionTests(unittest.TestCase):
    def test_cutoff_and_aggregation(self):
        m, g, p = sample()
        p.loc[p.match_id.eq(2), ['adr', 'rounds']] = [100, 30]
        data, prov = reconstruct(m, g, p)
        r = data.set_index('matchID').loc[-4]
        self.assertEqual(r.T1_mapwinrate, .5)
        self.assertEqual(r.T1_Maps_played_mean, 2)
        self.assertEqual(r.T1_Rounds_played_mean, 50)
        self.assertAlmostEqual(r['T1_Damage_/_Round_mean'], 92.)
        self.assertAlmostEqual(r.T1_Total_kills_mean, 6.)
        self.assertAlmostEqual(r.T1_Total_kills_std, np.std([2, 4, 6, 8, 10], ddof=1))
        self.assertEqual(prov.iloc[0].team1_roster_day, '2024-01-20')
        self.assertEqual(prov.iloc[0].history_window_start, '2023-10-23')
        self.assertNotIn('T1_Rating_2.0_mean', data)

    def test_same_day_lineups_scores_and_stats_do_not_enter_features(self):
        m, g, p = sample()
        original, prov = reconstruct(m, g, p)
        m.loc[m.match_id.ge(4), ROSTERS[0][0]] = 42
        g.loc[g.match_id.ge(4), 'team1_won_map'] = 0
        p.loc[p.match_id.ge(4), 'adr'] = 110
        new, revised = reconstruct(m, g, p)
        early = original.day.eq('2024-01-21')
        pd.testing.assert_frame_equal(original[early].drop(columns='team1_win'), new[early].drop(columns='team1_win'))
        pd.testing.assert_frame_equal(prov[early], revised[early])
        self.assertNotEqual(original.iloc[-1].T1_mapwinrate, new.iloc[-1].T1_mapwinrate)
        self.assertNotEqual(original.iloc[-1]['T1_Damage_/_Round_mean'], new.iloc[-1]['T1_Damage_/_Round_mean'])

    def test_patch_history_only_map_leaves_targets_team_history_and_same_day_unchanged(self):
        m, g, p = sample()
        # A complete historic stat line not present in labels may be used for player history.
        g.loc[g.match_id.eq(4), 'map_name'] = 'nuke'
        p = p[~p.match_id.eq(4)]
        patch = pd.DataFrame([dict(match_id=4, map_name='mirage', player_id=1, kills=20,
                                   deaths=20, adr=100, rounds=25)])
        orig, _ = reconstruct(m, g, p)
        data, _ = reconstruct(m, g, p, extra_history=patch)
        early = data.day.eq('2024-01-21')
        pd.testing.assert_frame_equal(orig.loc[early], data.loc[early])
        self.assertEqual(len(orig), len(data))
        r, base = data.iloc[-1], orig.iloc[-1]
        self.assertEqual(r.T1_mapwinrate, base.T1_mapwinrate)
        self.assertEqual(r.T1_Total_kills_mean, base.T1_Total_kills_mean + 4)
        self.assertTrue(np.isnan(r['T1_Assists_/_round_mean']))
        self.assertEqual(r['T2_Assists_/_round_mean'], base['T2_Assists_/_round_mean'])

    def test_headshots_use_all_earlier_local_events_or_stay_missing(self):
        m, g, p = sample()
        shots = {(mid, 'mirage', pid): int(mid == 2) for mid in (2, 3, 4) for pid in range(1, 6)}
        data, _ = reconstruct(m, g, p, headshots=shots)
        r = data.set_index('matchID').loc[-4]
        self.assertAlmostEqual(r['T1_Headshot_%_mean'], np.mean([1 / (2*k) for k in range(1, 6)]))
        self.assertNotIn('T2_Headshot_%_mean', data)
        # Missing one relevant prior player-map does not count as zero headshots.
        shots.pop((3, 'mirage', 1))
        incomplete, _ = reconstruct(m, g, p, headshots=shots)
        self.assertNotIn('T1_Headshot_%_mean', incomplete)

    def test_no_partial_five_or_missing_metric_aggregate(self):
        m, g, p = sample()
        data, prov = reconstruct(m, g, p[~p.player_id.eq(5)])
        self.assertEqual(prov.iloc[0].team1_players_with_history, 4)
        self.assertNotIn('T1_Total_kills_mean', data)
        form = np.arange(10, dtype=float)
        missing = form.copy()
        missing[0] = np.nan
        out = team_features([form]*4 + [missing], 1)
        self.assertNotIn('T1_Assists_/_round_mean', out)
        self.assertIn('T1_Total_kills_mean', out)


class ReplayTests(unittest.TestCase):
    def test_nickname_removes_only_profile_suffix(self):
        self.assertEqual(local_nickname('b1t (Valerij Vakhovsjkyj), UA'), 'b1t')
        self.assertEqual(local_nickname('dex, SE'), 'dex')
        self.assertEqual(local_nickname('techno4k'), 'techno4k')

    def test_replay_aggregate_must_match_kill_log(self):
        teams = [{'players': [{'steamid': i} for i in ids]} for ids in (range(1, 6), range(6, 11))]
        players = [dict(steamid=i, name=f'p{i}', side=side, n_rounds=20,
                        kills=int(i == 1 and side == 'ct'), deaths=int(i == 6 and side == 't'),
                        headshots=int(i == 1 and side == 'ct'))
                   for i in range(1, 11) for side in ('ct', 't', 'all')]
        kill = dict(attacker_steamid=1, victim_steamid=6, attacker_side='ct', victim_side='t', headshot=True)
        row = dict(teams=teams, players=players, rounds_played=20, kills=[kill])
        _, result = demo_counts(row)
        self.assertEqual(result['p1'], (1, 1, 0, 1))
        kill['headshot'] = False
        self.assertIsNone(demo_counts(row))


class PublishedDatasetTests(unittest.TestCase):
    def test_published_rating_snapshot_keeps_original_ids_labels_and_rating2(self):
        base = HOME/'data'
        work = base/'five_metrics_v1'
        manifest = json.loads((work/'manifest.json').read_text(encoding='utf-8'))
        for name, expected in manifest['files_sha256'].items():
            self.assertEqual(v2.digest(work/name), expected)
        frozen, trial = (pd.read_csv(d/'maps.csv') for d in (base, work))
        self.assertEqual(list(trial), upgraded_columns(frozen.columns))
        self.assertEqual(trial.shape, (12615, 158))
        keys = ['day', 'matchID', 'team1', 'team2', 'map', 'team1_win']
        pd.testing.assert_frame_equal(frozen[keys], trial[keys])
        ratings2 = [c for c in frozen if '_Rating_2.0_' in c]
        pd.testing.assert_frame_equal(frozen[ratings2], trial[ratings2])
        old = trial.matchID.gt(0)
        pd.testing.assert_frame_equal(frozen.loc[old], trial.loc[old, frozen.columns])
        self.assertEqual(sum(c for c in (trial['T1_Rating_3.0_mean'].notna() | trial['T2_Rating_3.0_mean'].notna())), 1)

    def test_frozen_kaggle_prefix_and_extended_labels(self):
        directory = HOME / 'data'
        manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
        for name, hash_ in manifest['files_sha256'].items():
            self.assertEqual(v2.digest(directory/name), hash_)
        with (directory/'maps.csv').open('rb') as new:
            original = (v2.DATA/'maps.csv').read_bytes()
            self.assertEqual(new.read(len(original)), original)
        dataset = pd.read_csv(directory/'maps.csv')
        sidecar = pd.read_csv(directory/'row_provenance.csv')
        self.assertEqual(len(dataset), 12615)
        self.assertEqual(dataset.shape[1], 148)
        self.assertFalse(dataset.duplicated(v2.KEY).any())
        self.assertTrue(dataset.team1_win.isin([0, 1]).all())
        self.assertEqual(dataset.day.max(), '2026-10-31')
        pd.testing.assert_frame_equal(dataset[['day', *v2.KEY]], sidecar[['day', *v2.KEY]])
        late = dataset.matchID.lt(0)
        self.assertEqual(int(late.sum()), 11605)
        self.assertTrue(dataset.loc[late, manifest['appended_unavailable_columns']].isna().all().all())
        for side in (1, 2):
            label = f'team{side}_roster_day'
            known = sidecar.loc[sidecar[label].notna()]
            self.assertTrue(known[label].lt(known.day).all())
        self.assertEqual(int(dataset.loc[late, manifest['common_candidates']].notna().all(axis=1).sum()), 7375)


if __name__ == '__main__':
    unittest.main()
