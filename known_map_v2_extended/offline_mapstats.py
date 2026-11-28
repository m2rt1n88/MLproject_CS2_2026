"""Offline import of browser-saved HLTV match + mapstats pages (no network access).

Only the two `totalstats` tables and their `traditional-data` cells are read; CT/T split tables and
eco-adjusted cells are ignored. Exports an independent Rating 3.0 metric history that passes
`metric_history.read_metrics`, merges it into an existing history without overwriting, and keeps the
structured scoreboard (incl. headshots) for a later, separately validated import.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory

from known_map_recent.hltv_backfill.history import fp, local_series, series_key
from known_map_recent.hltv_backfill.parse import Blocked, InvalidPage, MAP_NAMES, TEAM, key, parse_detail, text
from known_map_v2_extended.metric_history import KEY, VERSION, read_metrics

CAPTURE_VERSION = 'hltv-browser-mapstats-capture-v1'
SOURCE = 'hltv_accepted_map_scoreboard'
STATS_PLAYER = re.compile(r'^/stats/players/([1-9]\d*)/')
STATS_TEAM = re.compile(r'^/stats/teams/([1-9]\d*)/')
MATCH_LINK = re.compile(r'^/matches/([1-9]\d*)/')
MAPSTATS_LINK = re.compile(r'/stats/matches/mapstatsid/([1-9]\d*)/')
MATCH_URL = re.compile(r'https://www\.hltv\.org/matches/([1-9]\d*)/[^/?#]+')
MAP_URL = re.compile(r'https://www\.hltv\.org/stats/matches/mapstatsid/([1-9]\d*)/[^/?#]+')
DEC = r'\d+(?:\.\d+)?'
CHALLENGE_TITLES = {'just a moment...', 'access denied', 'attention required! | cloudflare',
                    '403 forbidden', '429 too many requests', 'too many requests'}


class Rejected(ValueError):
    """The saved pages are valid but cannot be linked safely to the local snapshot."""


def soup_of(html):
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, 'html.parser')
    if (key(text(soup.title)) in CHALLENGE_TITLES or soup.select_one(
            '#challenge-form, #cf-challenge-running, #challenge-stage, .cf-turnstile, '
            'script[src*="/cdn-cgi/challenge-platform/"]')):
        raise Blocked('challenge page')
    return soup


def one(nodes, what):
    nodes = list(nodes)
    if len(nodes) != 1:
        raise InvalidPage(f'expected exactly one {what}, found {len(nodes)}')
    return nodes[0]


def utc(node, what):
    epoch = node.get('data-unix', '') if node else ''
    if not re.fullmatch(r'\d{13}', epoch):
        raise InvalidPage(f'missing UTC epoch for {what}')
    return datetime.fromtimestamp(int(epoch) / 1000, tz=timezone.utc)


def cell(tr, cls):
    """One logical traditional value; responsive duplicates (e.g. two KAST cells) must agree."""
    cells = [td for td in tr.select(f'td.{cls}')
             if 'traditional-data' in td['class'] and 'eco-adjusted-data' not in td['class']]
    values = {text(td) for td in cells}
    if not cells or len(values) != 1:
        raise InvalidPage(f'missing or disagreeing traditional {cls}')
    return values.pop()


def pair(value, what):
    m = re.fullmatch(r'(\d+)\s*\((\d+)\)', value)
    if not m:
        raise InvalidPage(f'bad {what}: {value!r}')
    return int(m[1]), int(m[2])
def decimal(value, what, pct=False):
    if not re.fullmatch(DEC + ('%' if pct else ''), value):
        raise InvalidPage(f'bad {what}: {value!r}')
    return float(value[:-1] if pct else value)


def link_id(node, pattern, what):
    m = pattern.match(node.get('href', '')) if node else None
    if not m:
        raise InvalidPage(f'missing {what} link')
    return int(m[1])


def parse_scoreboard(table):
    """One team's combined (CT+T) table; never a split or eco-adjusted value."""
    classes = set(table.get('class', ()))
    if 'totalstats' not in classes or classes & {'ctstats', 'tstats'}:
        raise InvalidPage('not a combined totalstats table')
    version = text(one(table.select('thead th.st-rating .ratingDesc'), 'rating version'))
    if not re.fullmatch(r'\d+\.\d+', version):
        raise InvalidPage('unknown rating version')
    rows = []
    for tr in table.select('tbody tr'):
        pid = link_id(one(tr.select('td.st-player a[href^="/stats/players/"]'), 'player'), STATS_PLAYER, 'player')
        k, hs = pair(cell(tr, 'st-kills'), 'K(hs)')
        d, traded = pair(cell(tr, 'st-deaths'), 'D(t)')
        assists = re.fullmatch(r'(\d+)\s*\((\d+)\)', text(one(tr.select('td.st-assists'), 'assists')))
        adr, kast = decimal(cell(tr, 'st-adr'), 'ADR'), decimal(cell(tr, 'st-kast'), 'KAST', pct=True)
        rating = decimal(text(one(tr.select('td.st-rating'), 'rating')), 'rating')
        if hs > k or traded > d or k + d <= 0 or not 0 <= kast <= 100 or adr >= 1000:
            raise InvalidPage(f'implausible scoreboard row for player {pid}')
        rows.append(dict(hltv_player_id=pid, nickname=text(tr.select_one('td.st-player a')), kills=k,
                         headshot_kills=hs, deaths=d, traded_deaths=traded,
                         assists=int(assists[1]) if assists else None,
                         flash_assists=int(assists[2]) if assists else None,
                         adr=adr, kast=kast, rating=rating, rating_version=version))
    if len(rows) != 5 or len({r['hltv_player_id'] for r in rows}) != 5:
        raise InvalidPage('a team scoreboard must list exactly five distinct players')
    return text(one(table.select('thead th.st-teamname'), 'team name')), rows


def parse_mapstats(html, mapstats_id):
    soup = soup_of(html)
    box = one(soup.select('.match-info-box'), 'match info box')
    ids = {int(m[1]) for a in box.select('a[href*="/mapstatsid/"]') for m in [MAPSTATS_LINK.search(a['href'])] if m}
    if ids - {mapstats_id}:
        raise InvalidPage('mapstats page links to a different mapstats ID')
    names = [s.strip() for s in box.find_all(string=True, recursive=False) if s.strip()]
    if len(names) != 1 or key(names[0]) not in MAP_NAMES:
        raise InvalidPage('map name not identified')
    teams = []
    for cls in ('team-left', 'team-right'):
        node = one(box.select(f':scope > .{cls}'), cls)
        anchor = one(node.select('a[href^="/stats/teams/"]'), 'team link')
        score = text(one(node.select('.bold'), 'team score'))
        if not re.fullmatch(r'\d+', score):
            raise InvalidPage('bad map score')
        teams.append(dict(hltv_team_id=link_id(anchor, STATS_TEAM, 'team'), name=text(anchor), score=int(score)))
    match_id = link_id(one(soup.select('a.match-page-link[href^="/matches/"]'), 'match page link'), MATCH_LINK, 'match')
    tables = [t for t in soup.select('table.stats-table') if 'totalstats' in t.get('class', ())]
    if len(tables) != 2:
        raise InvalidPage('expected two totalstats tables')
    boards = dict(parse_scoreboard(t) for t in tables)
    by_name = {key(t['name']): t for t in teams}
    if len(by_name) != 2 or set(map(key, boards)) != set(by_name):
        raise InvalidPage('scoreboard teams disagree with the map header')
    for name, rows in boards.items():
        for r in rows:
            r['hltv_team_id'] = by_name[key(name)]['hltv_team_id']
    players = [r for rows in boards.values() for r in rows]
    if len({r['hltv_player_id'] for r in players}) != 10 or len({r['rating_version'] for r in players}) != 1:
        raise InvalidPage('scoreboard is not a complete five-versus-five box')
    a, b = (t['score'] for t in teams)
    if a == b or max(a, b) < 13 or any(r['kills'] > 5 * (a + b) or r['deaths'] > a + b for r in players):
        raise InvalidPage('implausible map score')
    when = utc(one(box.select('span[data-unix]'), 'map time'), 'map')
    return dict(mapstats_id=mapstats_id, match_id=match_id, map_name=key(names[0]), utc=when.isoformat(),
                teams=teams, rating_version=players[0]['rating_version'], players=players)
def parse_match(html, match_id):
    """Series header from the match page, cross-checked by the existing conservative parser."""
    soup = soup_of(html)
    canonical = soup.select_one('link[rel="canonical"]')
    m = MATCH_URL.fullmatch(canonical.get('href', '') if canonical else '')
    if not m or int(m[1]) != match_id:
        raise InvalidPage('missing or conflicting canonical match ID')
    teams = []
    for side in (1, 2):
        node = one(soup.select(f'.teamsBox .team{side}-gradient'), f'team{side}')
        score = one(node.select('div.won, div.lost, div.tie'), 'series score')
        anchor = one(node.select('a[href^="/team/"]'), 'team link')
        teams.append(dict(hltv_team_id=link_id(anchor, TEAM, 'team'), name=text(node.select_one('.teamName')),
                          score=int(text(score)) if re.fullmatch(r'\d+', text(score)) else None))
    formats = {int(x) for n in soup.select('.preformatted-text') for x in re.findall(r'Best of ([135])\b', text(n))}
    if len(formats) != 1 or None in (t['score'] for t in teams):
        raise InvalidPage('series format or score not identified')
    bo = formats.pop()
    when = utc(one(soup.select('.timeAndEvent .time[data-unix]'), 'match time'), 'match')
    probe = dict(id=match_id, team_a=teams[0]['name'], team_b=teams[1]['name'], day=when.date().isoformat(),
                 score_a=teams[0]['score'], score_b=teams[1]['score'], format=f'bo{bo}')
    detail = parse_detail(html, probe)
    if not detail['scores_ok'] or detail['team_ids'] != [t['hltv_team_id'] for t in teams]:
        raise InvalidPage('map scores do not reproduce the series score')
    return dict(match_id=match_id, utc=when.isoformat(), day=probe['day'], bo=bo, teams=teams,
                veto_ok=detail['veto_ok'], played=detail['played'])


def sha256(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read_manifest(folder, manifest):
    """Every listed page must exist, hash-match and carry its ID in both URL and file name."""
    blob = json.loads(Path(manifest).read_text(encoding='utf-8-sig'))
    pages = {}
    for e in blob['files']:
        name = re.fullmatch(r'(match|mapstats)_([1-9]\d*)\.html', str(e.get('file', '')))
        if not name:
            raise InvalidPage(f'unexpected capture file name {e.get("file")!r}')
        kind, ident = name.groups()
        url = (MATCH_URL if kind == 'match' else MAP_URL).fullmatch(e['url'])
        path = Path(folder) / e['file']
        digest = sha256(path)
        if not url or url[1] != ident or digest != e['sha256'] or (kind, int(ident)) in pages:
            raise InvalidPage(f'capture manifest does not verify {e["file"]}')
        captured = datetime.fromisoformat(e['captured_utc'].replace('Z', '+00:00'))
        if captured.utcoffset() is None or captured.utcoffset().total_seconds() != 0:
            raise InvalidPage('captured_utc must be UTC')
        pages[kind, int(ident)] = dict(file=e['file'], path=path, url=e['url'], sha256=digest,
                                       captured_utc=captured.astimezone(timezone.utc).isoformat())
    return dict(method=blob.get('method'), capture_time_evidence=blob.get('capture_time_evidence')), pages


def oriented(scores, flip):
    return list(scores[::-1] if flip else scores)
def capture_series(match_id, pages):
    """Parse one saved series and its saved maps; every played map must have a matching page."""
    entry = pages['match', match_id]
    series = parse_match(entry['path'].read_text(encoding='utf-8'), match_id)
    page_teams = [t['hltv_team_id'] for t in series['teams']]
    maps = []
    for played in series['played']:
        mid = played['mapstats_id']
        if mid is None or ('mapstats', mid) not in pages:
            raise Rejected(f'mapstats page for {played["name"]} ({mid}) was not captured')
        mp_entry = pages['mapstats', mid]
        box = parse_mapstats(mp_entry['path'].read_text(encoding='utf-8'), mid)
        flip = [t['hltv_team_id'] for t in box['teams']] == page_teams[::-1]
        if box['match_id'] != match_id or box['map_name'] != played['name'] or not (
                flip or [t['hltv_team_id'] for t in box['teams']] == page_teams):
            raise InvalidPage(f'mapstats {mid} does not belong to this series/map/teams')
        scores = oriented([t['score'] for t in box['teams']], flip)
        if scores != played['scores'] or box['utc'][:10] != series['day']:
            raise InvalidPage(f'mapstats {mid} score or UTC day disagrees with the match page')
        for r in box['players']:
            r['page_side'] = page_teams.index(r['hltv_team_id']) + 1
        if played['complete']:
            # The match page repeats the combined box; it must agree with the stats page exactly.
            ours = {(r['hltv_player_id'], r['kills'], r['deaths'], r['adr'], r['kast'], r['rating'])
                    for r in box['players']}
            theirs = {(r['player_id'], r['kills'], r['deaths'], r['adr'], r['kast'], r['rating'])
                      for r in played['rows']}
            if ours != theirs:
                raise InvalidPage(f'mapstats {mid} box disagrees with the match page box')
        maps.append(dict(box, scores=scores, source_url=mp_entry['url'], sha256=mp_entry['sha256'],
                         captured_utc=mp_entry['captured_utc'], match_page_box_checked=played['complete']))
    del series['played']
    return dict(series, source_url=entry['url'], sha256=entry['sha256'], captured_utc=entry['captured_utc'],
                maps=maps)


def link_local(series, matches, maps, excluded):
    """Unique local series by UTC day, normalized names, BO and series score; then every map score."""
    t1, t2 = series['teams']
    sig = series_key(series['day'], t1['name'], t2['name'], t1['score'], t2['score'], series['bo'])
    found = local_series(matches).get(sig, [])
    if len(found) != 1:
        raise Rejected(f'{len(found)} local series match {sig}')
    local = found[0]
    if key(t1['name']) not in (local.team1_key, local.team2_key):
        raise Rejected('team names disagree with local series')
    flip = key(t1['name']) != local.team1_key
    local_maps = {m.map_name: m for m in maps.itertuples(index=False) if int(m.match_id) == int(local.match_id)}
    page = {m['map_name']: oriented(m['scores'], flip) for m in series['maps']}
    if set(page) != set(local_maps) or any(page[n] != [m.score1, m.score2] for n, m in local_maps.items()):
        raise Rejected('per-map scores disagree with the local series')
    for n, m in local_maps.items():
        if m.player_stats_quarantined or (int(local.match_id), n) in excluded:
            raise Rejected(f'local map {n} is quarantined or excluded')
    return int(local.match_id), flip
def link_players(linked, players, prior=()):
    """Two distinct mapstats fingerprints (side,k,d,adr,kast), one-to-one, proven no later than the day.

    `linked` holds (series, local_match_id, flip). `prior` lists existing (hltv_player_id, local_id)
    links; it can only add conflicts, never evidence. Nicknames and slots are never used.
    """
    wanted = {(mid, m['map_name']) for _, mid, _ in linked for m in _maps(linked, mid)}
    local_fps = defaultdict(lambda: defaultdict(list))
    for p in players.itertuples(index=False):
        k = int(p.match_id), p.map_name
        f = fp(p.side, p.kills, p.deaths, p.adr, p.kast) if k in wanted else None
        if f is not None:
            local_fps[k][f].append(int(p.player_id))
    h_to_l, l_to_h = defaultdict(set), defaultdict(set)
    for h, l in prior:
        h_to_l[h].add(l)
        l_to_h[l].add(h)
    evidence, by_day = defaultdict(dict), defaultdict(list)
    for series, mid, flip in linked:
        for m in series['maps']:
            people = [(r, 3 - r['page_side'] if flip else r['page_side']) for r in m['players']]
            people = [(r, side, fp(side, r['kills'], r['deaths'], r['adr'], r['kast'])) for r, side in people]
            by_day[series['day']].append((series, mid, flip, m, people))
    accepted, skipped = [], []
    for day in sorted(by_day):
        for series, mid, flip, m, people in by_day[day]:
            counts = Counter(f for _, _, f in people)
            for r, _, f in people:
                found = local_fps[mid, m['map_name']].get(f, ())
                if f is None or counts[f] != 1 or len(found) != 1:
                    continue
                h, l = r['hltv_player_id'], found[0]
                h_to_l[h].add(l)
                l_to_h[l].add(h)
                links = evidence[h, l]
                if len(links) < 2:
                    links.setdefault(m['mapstats_id'], day)
        for series, mid, flip, m, people in by_day[day]:
            for r, side, f in people:
                h = r['hltv_player_id']
                l = next(iter(h_to_l[h])) if len(h_to_l[h]) == 1 else None
                links = evidence.get((h, l), {})
                if l is None or len(l_to_h[l]) != 1:
                    reason = 'no_local_fingerprint' if not h_to_l[h] else 'conflicting_identity'
                elif len(links) != 2:
                    reason = 'single_map_identity_evidence'
                elif max(links.values()) > day:
                    reason = 'identity_proven_after_event'
                elif local_fps[mid, m['map_name']].get(f) != [l]:
                    reason = 'no_matching_local_box_on_this_map'
                else:
                    accepted.append((series, mid, flip, m, r, side, l, links))
                    continue
                skipped.append(dict(hltv_match_id=series['match_id'], mapstats_id=m['mapstats_id'],
                                    map_name=m['map_name'], hltv_player_id=h, nickname=r['nickname'], reason=reason))
    return accepted, skipped


def _maps(linked, mid):
    return next(s['maps'] for s, m, _ in linked if m == mid)


def metric_row(series, mid, flip, m, r, side, local_id, links, imported_utc):
    scores_local = oriented(m['scores'], flip)
    return dict(match_id=mid, map_name=m['map_name'], day=series['day'], side=side, player_id=local_id,
                kills=r['kills'], deaths=r['deaths'], adr=r['adr'], kast=r['kast'],
                scores=scores_local, rounds=sum(m['scores']), hltv_id=series['match_id'],
                mapstats_id=m['mapstats_id'], hltv_player_id=r['hltv_player_id'],
                identity_proven_by=list(links), identity_as_of=max(links.values()),
                sha256=m['sha256'], source=m['source_url'], source_url=m['source_url'],
                match_source_url=series['source_url'], match_sha256=series['sha256'],
                captured_utc=m['captured_utc'], imported_utc=imported_utc,
                metrics={'rating': r['rating']}, rating_version=r['rating_version'], metric_source=SOURCE)
def merge(base, new):
    """Union by (match_id,map_name,player_id); identical re-imports are dropped, any disagreement fails."""
    seen = {}
    for r in base['rows']:
        idx = tuple(r[c] for c in KEY)
        if idx in seen:
            raise ValueError(f'Duplicate key in base metric history for {idx}')
        seen[idx] = r
    same = ('day', 'side', 'kills', 'deaths', 'adr', 'kast', 'rounds', 'mapstats_id', 'hltv_player_id',
            'metrics', 'rating_version', 'metric_source')
    added, duplicates = [], 0
    for r in new:
        idx = tuple(r[c] for c in KEY)
        old = seen.get(idx)
        if old is not None:
            if any(old.get(c) != r[c] for c in same):
                raise ValueError(f'Conflicting metric history row for {idx}')
            duplicates += 1
            continue
        seen[idx] = r
        added.append(r)
    return dict(base, rows=base['rows'] + added), added, duplicates


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def run(folder, manifest, base_history, out, history_patch=None):
    from known_map_recent.features import excluded_map_keys, load_source
    from known_map_recent.hltv_backfill.history import read_patch

    out = Path(out)
    if out.exists():
        raise FileExistsError(f'Refuse to overwrite {out}')
    imported = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    meta, pages = read_manifest(folder, manifest)
    matches, maps, players, hashes, _ = load_source()
    excluded = excluded_map_keys()
    extra = None if history_patch is None else read_patch(history_patch, matches, maps, players, hashes, excluded)
    base = json.loads(Path(base_history).read_text(encoding='utf-8-sig'))
    read_metrics(base_history, players, hashes, extra)  # base must already be valid for this snapshot
    captures, rejected, linked = [], [], []
    for kind, ident in sorted(pages):
        if kind != 'match':
            continue
        try:
            series = capture_series(ident, pages)
        except (Blocked, InvalidPage, Rejected) as exc:
            rejected.append(dict(hltv_match_id=ident, stage='parse', reason=f'{type(exc).__name__}: {exc}'))
            continue
        series['imported_utc'] = imported
        captures.append(series)
        try:
            mid, flip = link_local(series, matches, maps, excluded)
        except Rejected as exc:
            rejected.append(dict(hltv_match_id=ident, stage='local_series_link', reason=str(exc)))
            continue
        series['local_match_id'], series['local_team_order_flipped'] = mid, flip
        linked.append((series, mid, flip))
    prior = {(r['hltv_player_id'], r['player_id']) for r in base['rows']}
    accepted, skipped = link_players(linked, players, prior)
    rows = [metric_row(*a, imported) for a in accepted]
    merged, added, duplicates = merge(base, rows)
    report = dict(imported_utc=imported, capture=meta, series_captured=len(captures), series_linked=len(linked),
                  maps_captured=sum(len(s['maps']) for s in captures), rejected_series=rejected,
                  accepted_player_maps=len(rows), new_rating_rows=len(added), duplicate_existing_rows=duplicates,
                  skipped_player_maps=skipped, skipped_by_reason=dict(Counter(s['reason'] for s in skipped)),
                  per_series={s['match_id']: dict(local_match_id=s.get('local_match_id'),
                              accepted=sum(a[0] is s for a in accepted),
                              new=sum(r['hltv_id'] == s['match_id'] for r in added)) for s in captures},
                  headshots='captured in capture.json only; not exported to metric history in this pilot')
    with TemporaryDirectory(dir=out.parent) as tmp:
        stage = Path(tmp) / out.name
        stage.mkdir()
        write_json(stage / 'capture.json', dict(version=CAPTURE_VERSION, imported_utc=imported, capture=meta,
                                               series=captures, rejected=rejected))
        write_json(stage / 'rating_patch.json', dict(version=VERSION, source_sha256=hashes, rows=added))
        write_json(stage / 'metric_history.json', dict(merged, report=dict(
            base.get('report', {}), browser_pilot_v1=dict(base_sha256=sha256(base_history),
                                                          added_rows=len(added)))))
        # The written files must load through the unchanged validator.
        report['rating_patch_counts'] = read_metrics(stage / 'rating_patch.json', players, hashes, extra)[1]
        report['merged_history_counts'] = read_metrics(stage / 'metric_history.json', players, hashes, extra)[1]
        report['files_sha256'] = {f.name: sha256(f) for f in sorted(stage.iterdir())}
        write_json(stage / 'import_report.json', report)
        stage.rename(out)
    return report


def main():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder', type=Path, required=True)
    p.add_argument('--manifest', type=Path)
    p.add_argument('--base-history', type=Path, default=here / 'data/five_metrics_v1/metric_history.json')
    p.add_argument('--history-patch', type=Path, default=here / 'data/five_metrics_v1/hltv_history_patch.json',
                   help='Patch that validates the base history''s patched rows')
    p.add_argument('--out', type=Path, default=here / 'data/browser_pilot_v1/import')
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = run(args.folder, args.manifest or args.folder / 'capture_manifest.json', args.base_history, args.out,
                 args.history_patch)
    print(json.dumps({k: v for k, v in report.items() if k not in ('skipped_player_maps',)}, indent=2))
    return 0 if report['series_linked'] else 2


if __name__ == '__main__':
    sys.exit(main())
