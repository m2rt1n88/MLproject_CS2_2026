"""Offline tests for browser-saved HLTV mapstats import (no network)."""
import copy
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from known_map_recent.hltv_backfill.parse import Blocked, InvalidPage
from known_map_v2_extended import offline_mapstats as om

ROOT = Path(__file__).resolve().parents[1]
SAVED = ROOT / 'temp/hltv_verified'
TMP = ROOT / 'temp'
HAVE_SAVED = (SAVED / 'capture_manifest.json').is_file()


def row(pid, k, d, adr, kast, hs=1, eco_kast=None, rating='1.00', mobile_kast=None):
    eco = f'{eco_kast if eco_kast is not None else kast + 1}%'
    return f'''<tr><td class="st-player"><a href="/stats/players/{pid}/p{pid}">p{pid}</a></td>
      <td class="st-kast gtSmartphone-only traditional-data">{kast}%</td>
      <td class="st-kast gtSmartphone-only eco-adjusted-data hidden">{eco}</td>
      <td class="st-kills traditional-data">{k} <span>({hs})</span></td>
      <td class="st-kills eco-adjusted-data hidden">{k + 3} <span>({hs})</span></td>
      <td class="st-assists">2 <span>(1)</span></td>
      <td class="st-deaths traditional-data">{d} <span>(2)</span></td>
      <td class="st-deaths eco-adjusted-data hidden">{d + 1} <span>(2)</span></td>
      <td class="st-adr traditional-data">{adr}</td><td class="st-adr eco-adjusted-data hidden">{adr + 9}</td>
      <td class="st-kast smartphone-only traditional-data">{mobile_kast if mobile_kast is not None else kast}%</td>
      <td class="st-rating">{rating}</td></tr>'''


def table(cls, team, rows):
    return (f'<table class="stats-table {cls}"><thead><tr><th class="st-teamname">{team}</th>'
            f'<th class="st-rating">Rating <span class="ratingDesc">3.0</span></th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


def mapstats_html(base=10, **kw):
    a = [row(base + i, 10 + i, 10, 70.0 + i, 70.0, **kw) for i in range(5)]
    b = [row(base + 10 + i, 12 + i, 11, 60.0 + i, 65.0) for i in range(5)]
    # Split tables carry different numbers and come in the same document; they must be ignored.
    split = [row(base + i, 1, 1, 1.0, 1.0) for i in range(5)]
    return f'''<html><head><title>HLTV.org</title></head><body>
      <div class="match-info-box"><a href="/stats?event=1">Event</a>
      <div class="small-text"><span data-unix="1755018000000">x</span></div>Ancient
      <div class="team-left"><a href="/stats/teams/1/alpha">Alpha</a><div class="bold won">13</div></div>
      <div class="team-right"><a href="/stats/teams/2/beta">Beta</a><div class="bold lost">10</div></div></div>
      <a class="match-page-link button" href="/matches/99/alpha-vs-beta">More</a>
      {table("ctstats hidden", "Alpha", split)}{table("totalstats", "Alpha", a)}
      {table("totalstats", "Beta", b)}{table("tstats hidden", "Beta", split)}</body></html>'''


class ParseTests(unittest.TestCase):
    def test_only_total_traditional_columns(self):
        box = om.parse_mapstats(mapstats_html(), 5)
        p = next(r for r in box['players'] if r['hltv_player_id'] == 10)
        self.assertEqual((p['kills'], p['deaths'], p['adr'], p['kast'], p['headshot_kills']), (10, 10, 70.0, 70.0, 1))
        self.assertEqual((box['map_name'], box['match_id'], box['rating_version']), ('ancient', 99, '3.0'))
        self.assertEqual(len(box['players']), 10)

    def test_responsive_duplicate_kast_must_agree(self):
        with self.assertRaises(InvalidPage):
            om.parse_mapstats(mapstats_html(mobile_kast=55.0), 5)

    def test_challenge_page_rejected(self):
        with self.assertRaises(Blocked):
            om.parse_mapstats('<html><head><title>Just a moment...</title></head></html>', 5)
        with self.assertRaises(Blocked):
            om.soup_of('<html><script src="/cdn-cgi/challenge-platform/h/b/orchestrate"></script></html>')

    def test_wrong_mapstats_id_rejected(self):
        html = mapstats_html().replace('Event</a>', 'Event</a><a href="/stats/matches/mapstatsid/7/x">o</a>')
        with self.assertRaises(InvalidPage):
            om.parse_mapstats(html, 5)

    @unittest.skipUnless(HAVE_SAVED, 'saved HLTV pages not present')
    def test_real_page_ignores_eco_and_split(self):
        box = om.parse_mapstats((SAVED / 'mapstats_232687.html').read_text(encoding='utf-8'), 232687)
        mir = next(r for r in box['players'] if r['hltv_player_id'] == 8789)
        # Traditional 14(7) / 15(3) / 74.2 / 77.3%; eco-adjusted shows 16 kills, 78.4 ADR, 77.2%.
        self.assertEqual((mir['kills'], mir['headshot_kills'], mir['deaths'], mir['traded_deaths'],
                          mir['adr'], mir['kast'], mir['rating']), (14, 7, 15, 3, 74.2, 77.3, 1.21))
        ids = {r['hltv_player_id'] for r in box['players']}
        self.assertIn(24950, ids)      # AquaRS plays Mirage
        self.assertNotIn(20101, ids)   # fame does not
        dust = om.parse_mapstats((SAVED / 'mapstats_232691.html').read_text(encoding='utf-8'), 232691)
        self.assertIn(20101, {r['hltv_player_id'] for r in dust['players']})
        self.assertEqual(box['utc'], '2026-07-11T18:00:00+00:00')


def local_frames(dup=False, map_scores=((13, 10), (13, 11))):
    m = [dict(match_id=1, date=pd.Timestamp('2025-08-12'), team1_key='beta', team2_key='alpha',
              score1=0.0, score2=2.0, bo_reported=3.0)]
    if dup:
        m.append(dict(m[0], match_id=2))
    maps = pd.DataFrame([dict(match_id=1, map_name=n, score1=float(b), score2=float(a), player_stats_quarantined=False)
                         for n, (a, b) in zip(('ancient', 'train'), map_scores)])
    return pd.DataFrame(m), maps


def series_obj():
    def players(base, shift):
        out = []
        for side, team in ((1, 1), (2, 2)):
            for i in range(5):
                out.append(dict(hltv_player_id=base + 10 * (side - 1) + i, nickname='same', kills=10 + i + shift,
                                deaths=10 + side, adr=70.0 + i + shift, kast=70.0 + side, rating=1.0,
                                rating_version='3.0', page_side=side, headshot_kills=1))
        return out
    maps = [dict(mapstats_id=5, map_name='ancient', scores=[13, 10], players=players(100, 0),
                 sha256='a' * 64, source_url='u5', captured_utc='c'),
            dict(mapstats_id=6, map_name='train', scores=[13, 11], players=players(100, 1),
                 sha256='b' * 64, source_url='u6', captured_utc='c')]
    return dict(match_id=99, day='2025-08-12', bo=3, source_url='m', sha256='c' * 64,
                teams=[dict(name='Alpha', score=2, hltv_team_id=1), dict(name='Beta', score=0, hltv_team_id=2)],
                maps=maps)


def local_players(series, flip=True, drop=()):
    rows = []
    for m in series['maps']:
        for r in m['players']:
            if (m['map_name'], r['hltv_player_id']) in drop:
                continue
            side = 3 - r['page_side'] if flip else r['page_side']
            rows.append(dict(match_id=1, map_name=m['map_name'], side=side, player_id=r['hltv_player_id'] + 5000,
                             kills=float(r['kills']), deaths=float(r['deaths']), adr=r['adr'], kast=r['kast']))
    return pd.DataFrame(rows)


class LinkTests(unittest.TestCase):
    def test_unique_series_with_reversed_sides(self):
        matches, maps = local_frames()
        self.assertEqual(om.link_local(series_obj(), matches, maps, set()), (1, True))

    def test_ambiguous_or_score_mismatch_rejected(self):
        with self.assertRaises(om.Rejected):
            om.link_local(series_obj(), *local_frames(dup=True), set())
        with self.assertRaises(om.Rejected):
            om.link_local(series_obj(), *local_frames(map_scores=((13, 10), (13, 9))), set())
        with self.assertRaises(om.Rejected):
            om.link_local(series_obj(), *local_frames(), {(1, 'train')})

    def test_two_map_identity_and_side_reversal(self):
        s = series_obj()
        accepted, skipped = om.link_players([(s, 1, True)], local_players(s))
        self.assertEqual((len(accepted), skipped), (20, []))
        for _, _, _, m, r, side, local, links in accepted:
            self.assertEqual(local, r['hltv_player_id'] + 5000)
            self.assertEqual(side, 3 - r['page_side'])
            self.assertEqual(sorted(links), [5, 6])
        # Wrong orientation: sides no longer match any local fingerprint.
        accepted, skipped = om.link_players([(s, 1, False)], local_players(s))
        self.assertEqual(len(accepted), 0)
        self.assertEqual({x['reason'] for x in skipped}, {'no_local_fingerprint'})

    def test_single_map_player_skipped(self):
        s = series_obj()
        s['maps'][1]['players'][4] = dict(s['maps'][1]['players'][4], hltv_player_id=999)  # stand-in on train
        local = local_players(s, drop={('train', 999)})
        accepted, skipped = om.link_players([(s, 1, True)], local)
        reasons = {(x['map_name'], x['hltv_player_id']): x['reason'] for x in skipped}
        self.assertEqual(reasons, {('ancient', 104): 'single_map_identity_evidence',
                                   ('train', 999): 'no_local_fingerprint'})
        self.assertEqual(len(accepted), 18)

    def test_duplicate_fingerprint_and_prior_conflict(self):
        s = series_obj()
        twin = dict(s['maps'][0]['players'][0], hltv_player_id=777)
        s['maps'][0]['players'][1] = dict(twin, page_side=1)
        accepted, _ = om.link_players([(s, 1, True)], local_players(s))
        self.assertNotIn(100, {a[4]['hltv_player_id'] for a in accepted})
        s = series_obj()
        accepted, skipped = om.link_players([(s, 1, True)], local_players(s), prior={(100, 1)})
        self.assertNotIn(100, {a[4]['hltv_player_id'] for a in accepted})
        self.assertIn('conflicting_identity', {x['reason'] for x in skipped})

    def test_metric_row_metadata_and_merge(self):
        s = series_obj()
        a = om.link_players([(s, 1, True)], local_players(s))[0][0]
        r = om.metric_row(*a, '2026-11-28T00:00:00+00:00')
        self.assertEqual((r['rating_version'], r['metric_source'], set(r['metrics'])), ('3.0', om.SOURCE, {'rating'}))
        self.assertEqual((r['scores'], r['rounds'], r['identity_as_of'], r['source_url']), ([10, 13], 23, '2025-08-12', 'u5'))
        self.assertNotEqual(r['captured_utc'], r['imported_utc'])
        merged, added, dup = om.merge(dict(rows=[r]), [copy.deepcopy(r)])
        self.assertEqual((len(merged['rows']), added, dup), (1, [], 1))
        with self.assertRaises(ValueError):
            om.merge(dict(rows=[r]), [dict(r, metrics={'rating': 9.0})])

    def test_merge_rejects_duplicate_base_keys(self):
        s = series_obj()
        a = om.link_players([(s, 1, True)], local_players(s))[0][0]
        r = om.metric_row(*a, '2026-11-28T00:00:00+00:00')
        for twin in (copy.deepcopy(r), dict(r, metrics={'rating': 9.0})):  # identical and conflicting
            with self.assertRaises(ValueError):
                om.merge(dict(rows=[r, twin]), [])


class ManifestTests(unittest.TestCase):
    def test_unexpected_file_name_is_invalid_page(self):
        with TemporaryDirectory(dir=TMP) as tmp:
            for name in ('notes.html', 'match_0.html', None):
                path = Path(tmp) / 'capture_manifest.json'
                entry = dict(url='https://www.hltv.org/matches/1/a', captured_utc='2026-11-28T06:50:33Z',
                             sha256='0' * 64)
                if name is not None:
                    entry['file'] = name
                path.write_text(json.dumps(dict(files=[entry])), encoding='utf-8')
                with self.assertRaises(InvalidPage):
                    om.read_manifest(tmp, path)


@unittest.skipUnless(HAVE_SAVED, 'saved HLTV pages not present')
class SavedPilotTests(unittest.TestCase):
    def test_end_to_end_import(self):
        from known_map_recent.features import load_source
        from known_map_v2_extended.metric_history import read_metrics

        with TemporaryDirectory(dir=TMP) as tmp:
            out = Path(tmp) / 'run'
            rep = om.run(SAVED, SAVED / 'capture_manifest.json',
                         om.Path(om.__file__).parent / 'data/five_metrics_v1/metric_history.json', out,
                         om.Path(om.__file__).parent / 'data/five_metrics_v1/hltv_history_patch.json')
            self.assertEqual((rep['series_linked'], rep['rejected_series']), (2, []))
            self.assertEqual(rep['skipped_by_reason'], {'single_map_identity_evidence': 1, 'no_local_fingerprint': 1})
            self.assertEqual({(x['mapstats_id'], x['hltv_player_id']) for x in rep['skipped_player_maps']},
                             {(232687, 24950), (232691, 20101)})  # AquaRS (Mirage only), fame (Dust2 only)
            cap = json.loads((out / 'capture.json').read_text(encoding='utf-8'))
            vp = next(s for s in cap['series'] if s['match_id'] == 2395661)
            self.assertEqual((vp['day'], vp['bo'], vp['local_match_id']), ('2026-07-11', 3, 7303252))
            self.assertTrue(all(m['captured_utc'] != vp['imported_utc'] for m in vp['maps']))
            patch = json.loads((out / 'rating_patch.json').read_text(encoding='utf-8'))
            self.assertTrue(all(r['match_id'] == 3500687 for r in patch['rows']))
            self.assertEqual(len(patch['rows']), 20)
            with self.assertRaises(FileExistsError):
                om.run(SAVED, SAVED / 'capture_manifest.json', None, out)

    def test_tampered_capture_rejected(self):
        with TemporaryDirectory(dir=TMP) as tmp:
            for f in SAVED.iterdir():
                shutil.copy(f, Path(tmp) / f.name)
            p = Path(tmp) / 'match_2384281.html'
            p.write_bytes(p.read_bytes() + b' ')
            with self.assertRaises(InvalidPage):
                om.read_manifest(tmp, Path(tmp) / 'capture_manifest.json')


if __name__ == '__main__':
    unittest.main()
