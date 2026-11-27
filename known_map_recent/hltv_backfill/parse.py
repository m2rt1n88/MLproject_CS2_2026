"""Conservative parser for the match page (no per-map requests)."""
from datetime import datetime, timezone
import re
import unicodedata


MAP_NAMES = set('ancient anubis cache cobblestone dust2 inferno mirage nuke overpass train vertigo'.split())
TEAM = re.compile(r'^/team/([1-9]\d*)(?:/|$)')
PLAYER = re.compile(r'^/player/([1-9]\d*)(?:/|$)')
MAP_ID = re.compile(r'^/stats/matches/mapstatsid/([1-9]\d*)(?:/|$)')
DECIMAL = r'\d+(?:\.\d+)?'


class InvalidPage(ValueError):
    pass


class Blocked(InvalidPage):
    pass


def key(s):
    return ' '.join(unicodedata.normalize('NFKC', s).casefold().split())


def text(node):
    return node.get_text(' ', strip=True) if node else ''


def number_id(node, pattern):
    match = pattern.match(node.get('href', '')) if node else None
    return int(match[1]) if match else None


def parse_table(group, team_ids):
    # The match page places each team's combined T+CT table before its split tables.
    # Never use a later split table to replace a failed combined table.
    seen = set()
    result = []
    for table in group.select('table') if group else ():
        team = number_id(table.select_one('a[href^="/team/"]'), TEAM)
        if team not in team_ids or team in seen:
            continue
        seen.add(team)
        version = re.fullmatch(r'Rating\s+(\d+\.\d+)', text(table.select_one('td.rating, th.rating')), re.I)
        for tr in table.select('tr'):
            pid = number_id(tr.select_one('a[href^="/player/"]'), PLAYER)
            if pid is None:
                continue
            kd = re.fullmatch(r'(\d+)\s*-\s*(\d+)', text(tr.select_one('.kd.traditional-data')))
            adr = text(tr.select_one('.adr.traditional-data'))
            kast = text(tr.select_one('.kast.traditional-data'))
            if not (kd and re.fullmatch(DECIMAL, adr) and re.fullmatch(DECIMAL + '%', kast)):
                continue
            k, d = map(int, kd.groups())
            adr, kast = float(adr), float(kast[:-1])
            if k + d <= 0 or not 0 <= kast <= 100 or not 0 <= adr < 1000:
                continue
            rating = text(tr.select_one('td.rating'))
            result.append(dict(team_id=team, player_id=pid, kills=k, deaths=d, adr=adr, kast=kast,
                               rating=float(rating) if version and re.fullmatch(DECIMAL, rating) else None,
                               rating_version=version[1] if version else None))
    # Partial rows are retained for diagnosis but are not exportable.
    complete = (len(result) == 10 and len({r['player_id'] for r in result}) == 10
                and all(sum(p['team_id'] == team for p in result) == 5 for team in team_ids))
    return result, complete


def parse_detail(html, match, *, page_names=None):
    try:
        from bs4 import BeautifulSoup
    except ModuleNotFoundError as exc:
        # The embedded project interpreter ignores PYTHONPATH; reuse the existing isolated audit install.
        from pathlib import Path
        import sys

        deps = Path(__file__).resolve().parents[2] / '.venv/hltv-crawler-audit-deps'
        if exc.name != 'bs4' or not (deps / 'bs4').is_dir():
            raise
        sys.path.insert(0, str(deps))
        from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, 'html.parser')
    if (key(text(soup.title)) in {'just a moment...', 'access denied', 'attention required! | cloudflare',
                                 '403 forbidden', '429 too many requests', 'too many requests'}
            or soup.select_one('#challenge-form, #cf-challenge-running, script[src*="/cdn-cgi/challenge-platform/"]')):
        raise Blocked('challenge page; stop and retain the queue')
    canonical = soup.select_one('link[rel="canonical"], meta[property="og:url"]')
    url = (canonical.get('href') or canonical.get('content') or '') if canonical else ''
    ident = re.fullmatch(r'https://(?:www\.)?hltv\.org/matches/([1-9]\d*)/[^/?#]+/?', url)
    if not ident or int(ident[1]) != match['id']:
        raise InvalidPage('missing or conflicting HLTV canonical match ID')

    names, team_ids = [], []
    for side in (1, 2):
        node = (soup.select_one(f'.teamsBox .team{side}-gradient')
                or soup.select_one(f'.teamsBox .team{side}:has(.teamName)'))
        if node is None:
            raise InvalidPage('page team sides not identified')
        names.append(key(text(node.select_one('.teamName'))))
        team_ids.append(number_id(node if TEAM.match(node.get('href', '')) else node.select_one('a[href^="/team/"]'), TEAM))
    archive_names = list(map(key, (match['team_a'], match['team_b'])))
    side_by_name = {n: side for side, n in enumerate(archive_names)}
    for side, label in enumerate(page_names or ()):
        alias = key(label)
        if alias in side_by_name and side_by_name[alias] != side:
            raise InvalidPage('ambiguous archive display names')
        side_by_name[alias] = side
    page_sides = [side_by_name.get(n) for n in names]
    if set(page_sides) != {0, 1} or None in team_ids or len(set(team_ids)) != 2:
        raise InvalidPage('page team identities disagree with archive')
    flip = page_sides == [1, 0]
    if flip:
        team_ids.reverse()

    schedule = soup.select_one('.time[data-unix]')
    epoch = schedule.get('data-unix', '') if schedule else ''
    scheduled = datetime.fromtimestamp(int(epoch) / 1000, tz=timezone.utc) if re.fullmatch(r'\d{13}', epoch) else None
    if scheduled and scheduled.date().isoformat() != match['day']:
        raise InvalidPage('page scheduled UTC day disagrees with archive')

    played, order, bad_scores = [], [], False
    for holder in soup.select('.mapholder'):
        name = key(text(holder.select_one('.mapname')))
        scores = [text(holder.select_one(f'.results-{s} .results-team-score')) for s in ('left', 'right')]
        name_known = name in MAP_NAMES
        if name_known:
            order.append(name)
        node = holder.select_one('a[href^="/stats/matches/mapstatsid/"]')
        mid = number_id(node, MAP_ID)
        if scores in (['-', '-'], ['', ''], ['0', '0']) and mid is None:
            continue
        if not name_known or not all(re.fullmatch(r'\d+', n) for n in scores):
            bad_scores = True
            continue
        a, b = map(int, scores[::-1] if flip else scores)
        if a == b or min(a, b) < 0 or max(a, b) < 13:
            bad_scores = True
            continue
        group = soup.find(id=f'{mid}-content') if mid else None
        rows, complete = parse_table(group, team_ids)
        if any(r['kills'] > 5 * (a + b) or r['deaths'] > a + b for r in rows):
            complete = False
        played.append(dict(name=name, mapstats_id=mid, stats_path=node.get('href') if mid else None,
                           scores=[a, b], rows=rows, complete=complete))
    wins = [sum(p['scores'][i] > p['scores'][1-i] for p in played) for i in (0, 1)]
    ids = [p['mapstats_id'] for p in played if p['mapstats_id'] is not None]
    scores_ok = (bool(played) and not bad_scores and wins == [match['score_a'], match['score_b']]
                 and len(ids) == len(set(ids)) and len({p['name'] for p in played}) == len(played))

    steps = []
    veto_lines = [text(n) for n in soup.select('.veto-box .padding > div') if re.match(r'^\d+\.', text(n))]
    for line in veto_lines:
        step = re.fullmatch(r'(\d+)\.\s+(.+?)\s+(removed|picked)\s+(.+)', line)
        left = re.fullmatch(r'(\d+)\.\s+(.+?)\s+was left over', line)
        if step and key(step[2]) in side_by_name and key(step[4]) in MAP_NAMES:
            steps.append(dict(order=int(step[1]), team_id=team_ids[side_by_name[key(step[2])]],
                              action={'removed': 'ban', 'picked': 'pick'}[step[3]], name=key(step[4])))
        elif left and key(left[2]) in MAP_NAMES:
            steps.append(dict(order=int(left[1]), team_id=None, action='decider', name=key(left[2])))
    chosen = [r['name'] for r in steps if r['action'] != 'ban']
    fmt = re.fullmatch(r'bo([135])', match['format'])
    veto_ok = (bool(fmt) and len(chosen) == int(fmt[1]) and bool(scores_ok) and chosen == order
               and chosen[:len(played)] == [p['name'] for p in played] and len(veto_lines) == len(steps) == 7
               and [r['order'] for r in steps] == list(range(1, 8)) and steps[-1]['action'] == 'decider'
               and sum(r['action'] == 'decider' for r in steps) == 1 and len({r['name'] for r in steps}) == 7)
    return dict(team_ids=team_ids, scheduled_utc=scheduled.isoformat() if scheduled else None,
                played=played, scores_ok=scores_ok, veto=steps, veto_ok=veto_ok,
                veto_published_at=None, lineups_published_at=None)
