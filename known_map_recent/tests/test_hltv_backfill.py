from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from known_map_recent.features import earlier_day_specialty, KEY, PLAYER_FIT, TEAM_FIT
from known_map_recent.hltv_backfill.history import build_patch, read_patch
from known_map_recent.hltv_backfill.parse import InvalidPage, parse_detail
from known_map_recent.hltv_backfill.store import import_cache, open_store, page_path, seed, status
from scripts.experiment_known_map import known_map_history, ROSTERS, TEAM_MAP


STATS = {'dust2': (13, 7), 'mirage': (9, 13), 'overpass': (13, 5)}
SIDES = {1: 81, 2: 82}
VETO = ('1. Alpha removed Ancient', '2. Beta removed Nuke', '3. Alpha picked Dust2', '4. Beta picked Mirage',
        '5. Alpha removed Anubis', '6. Beta removed Train', '7. Overpass was left over')


def table(side, count, offset=0):
    rows = [f'''<tr><td><a href="/player/{SIDES[side]*100+i+offset}/name">Player</a></td>
              <td class="kd traditional-data">{10+i} - {8+i}</td>
              <td class="adr traditional-data">{70+i}</td>
              <td class="kast traditional-data">80%</td><td class="rating">1.08</td></tr>'''
            for i in range(count)]
    return f'<table><tr><td><a href="/team/{SIDES[side]}/name">Team</a></td><td class="rating">Rating 3.0</td></tr>{"".join(rows)}</table>'


def page(ident, *, veto=True, partial=None, sweep=False, mismatched=False):
    teams = ''.join(f'''<div class="team{s}-gradient"><a href="/team/{t}/{name.lower()}">
                      <div class="teamName">{name}</div></a></div>'''
                    for (s, t), name in zip(SIDES.items(), ('Alpha', 'Beta')))
    steps = ''.join(f'<div>{v}</div>' for v in VETO) if veto else ''
    content = []
    for j, (name, (a, b)) in enumerate(STATS.items(), 1):
        played = not (sweep and j == 3)
        if sweep and j == 2:
            a, b = 13, 9
        m = ident*10 + j
        link = f'<a href="/stats/matches/mapstatsid/{m}/a-vs-b">Details</a>' if played else ''
        content.append(f'''<div class="mapholder"><div class="mapname">{name}</div>{link}
          <div class="results-left"><span class="results-team-score">{a if played else '-'}</span></div>
          <div class="results-right"><span class="results-team-score">{b if played else '-'}</span></div></div>''')
        if played:
            content.append(f'''<div id="{m}-content">{table(1, 4 if partial == j else 5)}
                           {table(1, 5, 20)}{table(2, 5)}{table(2, 5, 20)}</div>''')
    return f'''<head><link rel="canonical" href="https://www.hltv.org/matches/{ident+int(mismatched)}/a-vs-b"/></head>
            <div class="teamsBox"><img class="team1" alt="Europe"><img class="team2" alt="Sweden">{teams}</div><div class="veto-box"><div class="padding">{steps}</div></div>
            {''.join(content)}'''


def archive(path, rows):
    with closing(sqlite3.connect(path)) as db, db:
        db.executescript('''CREATE TABLE matches(match_id,start_date,slug,format,team_a,team_b,score_a,score_b,source);
                            CREATE TABLE maps(match_id,map_name,score_a,score_b);''')
        for ident, stamp, score in rows:
            db.execute('INSERT INTO matches VALUES (?,?,?,?,?,?,?,?,?)',
                       (f'hltv:{ident}', stamp, 'a-vs-b', 'bo3', 'Alpha', 'Beta', 2, score, 'hltv'))
            for name, (a, b) in STATS.items():
                if score == 0 and name == 'mirage':
                    a, b = 13, 9
                if score == 0 and name == 'overpass':
                    a = b = 0
                db.execute('INSERT INTO maps VALUES (?,?,?,?)', (f'hltv:{ident}', name, a, b))


class BackfillTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.archive = self.root / 'upstream.db'
        archive(self.archive, [(120, '2026-06-09', 0), (121, '2026-06-10', 1)])
        self.db = open_store(self.root / 'queue.db')
        self.addCleanup(self.db.close)
        self.assertEqual(seed(self.db, self.archive)['inserted'], 2)

    def write(self, ident, html):
        (self.root / f'{ident}.html').write_text(html, encoding='utf-8')

    def test_cache_progress_retains_decider_and_retries_only_missing_components(self):
        job = self.db.execute('SELECT * FROM jobs WHERE id=120').fetchone()
        sha = hashlib.sha1(page_path(job).encode()).hexdigest()[:24]
        (self.root / f'{sha}.html').write_text(page(120, sweep=True, veto=False, partial=2), encoding='utf-8')
        self.assertEqual(seed(self.db, self.archive)['inserted'], 0)
        self.assertEqual(import_cache(self.db, [self.root]), {'partial': 1})
        r = self.db.execute('SELECT * FROM progress WHERE id=120').fetchone()
        self.assertEqual((r['scores'], r['veto'], r['player_maps']), ('page', 'missing', 1))
        self.assertEqual(import_cache(self.db, [self.root]), {'cached': 1})
        self.write(120, page(120, sweep=True))
        self.assertEqual(import_cache(self.db, [self.root]), {'complete': 1})
        self.assertEqual(status(self.db)['complete'], 1)
        self.assertEqual(import_cache(self.db, [self.root]), {})
        veto = json.loads(self.db.execute('SELECT veto FROM accepted_veto').fetchone()[0])
        self.assertEqual((veto[-1]['action'], veto[-1]['name']), ('decider', 'overpass'))
        self.assertEqual(self.db.execute('SELECT count(*) FROM accepted_players').fetchone()[0], 2)
        self.assertEqual(self.db.execute('SELECT count(*) FROM observations').fetchone()[0], 2)
        payload = json.loads(self.db.execute('SELECT payload FROM accepted_players LIMIT 1').fetchone()[0])
        self.assertEqual({p['rating_version'] for p in payload['rows']}, {'3.0'})
        self.assertEqual(len(payload['rows']), 10)

    def test_block_survives_reopen_and_no_next_match_is_processed(self):
        self.write(120, '<title>Just a moment...</title>')
        self.write(121, page(121))
        self.assertEqual(import_cache(self.db, [self.root]), {'blocked': 1})
        with closing(open_store(self.root / 'queue.db')) as re_open:
            self.assertIn('stopped', import_cache(re_open, [self.root]))
            self.assertEqual(re_open.execute('SELECT id FROM observations').fetchone()[0], 120)
            self.write(120, page(120, sweep=True))
            self.assertEqual(import_cache(re_open, [self.root], resume=True, limit=1), {'complete': 1})
            self.assertEqual(import_cache(re_open, [self.root]), {'complete': 1})

    def test_empty_page_identity_mismatch_and_conflicting_map_do_not_succeed(self):
        self.write(120, '<html></html>')
        self.write(121, page(121, mismatched=True))
        self.assertEqual(import_cache(self.db, [self.root]), {'invalid': 2})
        self.write(121, page(121).replace('>7</span>', '>8</span>'))
        self.assertEqual(import_cache(self.db, [self.root], ids=[121]), {'invalid': 1})
        self.assertEqual(self.db.execute('SELECT count(*) FROM accepted_players').fetchone()[0], 0)
        with self.assertRaises(ValueError):
            open_store(self.archive)

    def test_reparse_invalid_cached_page_after_parser_fix(self):
        self.write(120, page(120, sweep=True))
        # The live pages have .team1/.team2 flags before the team-name containers.
        with patch('known_map_recent.hltv_backfill.store.parse_detail', side_effect=InvalidPage('old flags selector')):
            self.assertEqual(import_cache(self.db, [self.root]), {'invalid': 1})
        self.assertEqual(import_cache(self.db, [self.root]), {'cached': 1})
        self.assertEqual(import_cache(self.db, [self.root], reparse_invalid=True), {'complete': 1})
        self.assertEqual(self.db.execute('SELECT count(*) FROM observations').fetchone()[0], 1)
        self.assertEqual(status(self.db)['complete'], 1)

    def test_original_index_display_name_recovers_rejected_cache(self):
        self.write(120, page(120, sweep=True).replace('Alpha', 'Original Alpha'))
        self.assertEqual(import_cache(self.db, [self.root]), {'invalid': 1})
        entry = dict(match_id='120', slug='a-vs-b', date='2026-06-09', team_a='Original Alpha', team_b='Beta',
                     score_a=2, score_b=0)
        with closing(sqlite3.connect(self.archive)) as upstream, upstream:
            upstream.execute('ALTER TABLE matches ADD COLUMN raw TEXT')
            upstream.execute("UPDATE matches SET raw=? WHERE match_id='hltv:120'", (json.dumps(entry | {'match_id': '121'}),))
        self.assertEqual(seed(self.db, self.archive, ids=[120])['inserted'], 0)
        self.assertEqual(import_cache(self.db, [self.root], reparse_invalid=True), {'invalid': 1})
        with closing(sqlite3.connect(self.archive)) as upstream, upstream:
            upstream.execute("UPDATE matches SET raw=? WHERE match_id='hltv:120'", (json.dumps(entry),))
        self.assertEqual(seed(self.db, self.archive, ids=[120])['inserted'], 0)
        self.assertEqual(import_cache(self.db, [self.root], reparse_invalid=True), {'complete': 1})
        self.assertEqual(json.loads(self.db.execute('SELECT team_ids FROM accepted_scores').fetchone()[0]), [81, 82])

    def test_patch_does_not_backfill_earlier_dates_from_later_anchors(self):
        self.write(120, page(120, sweep=True))
        self.write(121, page(121))
        self.assertEqual(import_cache(self.db, [self.root]), {'complete': 2})
        matches = pd.DataFrame([dict(match_id=i, date=pd.Timestamp(d), team1_key='beta', team2_key='alpha', score1=s,
                                     score2=2, bo_reported=3) for i, d, s in [(41, '2026-06-09', 0), (42, '2026-06-10', 1)]])
        maps = pd.DataFrame([dict(match_id=i, map_name=m, score1=b, score2=a, player_stats_quarantined=False)
                             for i, group in ((41, [('dust2', (13, 7)), ('mirage', (13, 9))]),
                                              (42, list(STATS.items())[:2])) for m, (a, b) in group])
        # Only the second day has two independent HLTV/local anchors per player.
        players = pd.DataFrame([dict(match_id=42, map_name=m, side=3-s, player_id=s*10+i, kills=10+i, deaths=8+i,
                                     adr=70+i, kast=80.) for m in ('dust2', 'mirage') for s in (1, 2) for i in range(5)])
        output = build_patch(self.db, self.archive, matches, maps, players, {'x': 'hash'}, set())
        self.assertEqual(output['report']['unproven_or_conflicted_player_rows'], 20)
        self.assertEqual(len(output['rows']), 10)
        self.assertEqual({(r['match_id'], r['map_name'], tuple(r['scores'])) for r in output['rows']}, {(42, 'overpass', (5, 13))})
        f = self.root / 'patch.json'
        f.write_text(json.dumps(output), encoding='utf-8')
        self.assertEqual(len(read_patch(f, matches, maps, players, {'x': 'hash'}, set())), 10)
        output['rows'][0]['identity_as_of'] = '2026-06-11'
        f.write_text(json.dumps(output), encoding='utf-8')
        with self.assertRaises(ValueError):
            read_patch(f, matches, maps, players, {'x': 'hash'}, set())
        # Adding another copy of the same archive signature makes it unsafe even when unqueued.
        with closing(sqlite3.connect(self.archive)) as db, db:
            db.execute("INSERT INTO matches SELECT 'hltv:999',start_date,slug,format,team_a,team_b,score_a,score_b,source FROM matches WHERE match_id='hltv:121'")
        ambiguous = build_patch(self.db, self.archive, matches, maps, players, {'x': 'hash'}, set())
        self.assertEqual(ambiguous['rows'], [])
        self.assertEqual(ambiguous['report']['unmatched_or_ambiguous_maps'], 3)

    def test_extra_rows_change_only_future_day_player_features(self):
        days = pd.to_datetime(['2026-05-01', '2026-05-02', '2026-05-02', '2026-05-03'])
        m = pd.DataFrame(dict(match_id=[1, 2, 3, 4], date=days, team1_key='a', team2_key='b'))
        for i, col in enumerate(ROSTERS[0] + ROSTERS[1], 1):
            m[col] = i
        g = m[['match_id']].assign(map_name='overpass', score1=13, score2=9, team1_won_map=1)
        p = pd.DataFrame([dict(match_id=1, map_name='mirage', player_id=1, adr=70., kast=80., kills=11, deaths=10, rounds=22)])
        g = pd.concat([g, g.iloc[:1].assign(map_name='mirage')], ignore_index=True)
        extra = p.assign(match_id=2, map_name='overpass', day='2026-05-02', adr=125.)
        overall = m[['match_id']].assign(diff_elo=0.)
        for get, team in ((lambda e: known_map_history(m, g, p, extra_history=e), TEAM_MAP),
                          (lambda e: earlier_day_specialty(m, g, p, overall, extra_history=e), TEAM_FIT)):
            base, changed = get(None).set_index(KEY), get(extra).set_index(KEY)
            pd.testing.assert_frame_equal(base.iloc[:-1], changed.iloc[:-1])
            pd.testing.assert_frame_equal(base[team], changed[team])
        self.assertGreater(changed.loc[(4, 'overpass'), PLAYER_FIT[0]], base.loc[(4, 'overpass'), PLAYER_FIT[0]])
        self.assertEqual(known_map_history(m, g, p, extra_history=extra).iloc[-1].team1_player_coverage, .2)


if __name__ == '__main__':
    unittest.main()
