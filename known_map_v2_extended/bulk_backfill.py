"""Bulk offline HLTV backfill: candidate queue, capture manifest and batch import (no network access).

Pages are saved by the orchestrator's verified Chrome session; this module only reads them.
Two capture modes share one identity/metric pipeline (`offline_mapstats`):
- `mapstats`: every played map has a saved mapstats page (cross-checked against the match page box);
- `match_page`: the combined T+CT box embedded in the match page (`parse_detail`), same fields the
  earlier HLTV history patch used. The row source is then the match page URL/SHA, never a fake mapstats file.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
PLAN = HERE / 'data/backfill_plan_ranked.json'
BULK = ROOT / 'temp/hltv_bulk'
REPORTS = ROOT / 'reports/hltv_bulk'
MATCH_URL = re.compile(r'https://www\.hltv\.org/matches/([1-9]\d*)/[^/?#]+')
FILE = re.compile(r'(match|mapstats)_([1-9]\d*)\.html')
HREF_MAPSTATS = re.compile(r'href="(/stats/matches/mapstatsid/([1-9]\d*)/[^"?#]+)"')
BROWSER_METHOD = 'user Chrome Browser MCP download after manual Cloudflare verification'
TIME_EVIDENCE = 'filesystem LastWriteTimeUtc at download; not historical publication time'
STATUSES = ('pending', 'acquired', 'accepted', 'rejected', 'blocked', 'no_match', 'unsupported')
UNSUPPORTED_METRICS = {
    'headshot_kills': ('available on HLTV mapstats pages, but not implemented in this rating lane; '
                       'a separate lane (browser_headshots.py) owns its schema/validation'),
    'grenade_damage': ('not on per-map scoreboards; HLTV all-time leaderboards show GADR, which is not a '
                       '90-day past-only value and is not imported; left missing, never zero-filled'),
    'saved_by_teammate': 'not shown on HLTV scoreboards; left missing, never zero-filled',
    'saved_teammates': 'not shown on HLTV scoreboards; left missing, never zero-filled'}


def sha256(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def mtime_utc(path):
    return datetime.fromtimestamp(Path(path).stat().st_mtime, tz=timezone.utc).isoformat()


def write_json(path, value, *, replace=False):
    """Derived files only; raw HTML is never written by this module."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
    if path.exists() and not replace:
        tmp.unlink()
        raise FileExistsError(f'Refuse to overwrite {path}')
    tmp.replace(path)


def load_plan(plan=PLAN):
    """Series candidates exactly as listed in the plan (IDs/URLs are never constructed here)."""
    blob = json.loads(Path(plan).read_text(encoding='utf-8-sig'))
    series, unlinked = {}, []
    for r in blob['rows']:
        c = r.get('candidate')
        if not c:
            unlinked.append(dict(local_match_id=r['local_match_id'], map=r['map'], day=r['day']))
            continue
        m = MATCH_URL.fullmatch(c['match_url'])
        if not m or int(m[1]) != int(c['hltv_id']):
            raise ValueError(f'plan URL does not carry its HLTV ID: {c["match_url"]}')
        s = series.setdefault(int(c['hltv_id']), dict(
            hltv_id=int(c['hltv_id']), match_url=c['match_url'], day=r['day'], evidence=c['evidence'],
            local_match_ids=set(), maps=[], rating_windows=0, rating_completions=0))
        if s['match_url'] != c['match_url']:
            raise ValueError(f'plan lists two URLs for HLTV match {s["hltv_id"]}')
        s['local_match_ids'].add(int(r['local_match_id']))
        s['maps'].append(r['map'])
        s['rating_windows'] += int(r['possible_affected_team_windows'].get('rating', 0))
        s['rating_completions'] += int(r['possible_completions_if_only_this_map_added'].get('rating', 0))
    for s in series.values():
        s['local_match_ids'] = sorted(s['local_match_ids'])
        s['maps'] = sorted(s['maps'])
    return series, unlinked


def priority(s):
    # Two or more local maps let one series prove player identity on its own; then expected gain.
    return (len(s['maps']) < 2, -s['rating_completions'], -s['rating_windows'], -len(s['maps']), s['hltv_id'])


def saved_pages(folders):
    """(kind, id) -> path; the first folder wins, later duplicates must be byte-identical."""
    pages, dupes = {}, []
    for folder in folders:
        for path in sorted(Path(folder).glob('*.html')):
            m = FILE.fullmatch(path.name)
            if not m:
                continue
            k = m[1], int(m[2])
            if k in pages:
                if sha256(pages[k]) != sha256(path):
                    raise ValueError(f'two different saved copies of {path.name}')
                dupes.append(str(path))
                continue
            pages[k] = path
    return pages, dupes


def write_queue(series, pages, out_dir, take):
    pending = sorted((s for i, s in series.items() if ('match', i) not in pages), key=priority)
    batch = pending[:take]
    lines = [f'{s["match_url"]}\t{(BULK / f"match_{s["hltv_id"]}.html").resolve()}' for s in batch]
    if len({l.split('\t')[0] for l in lines}) != len(lines):
        raise ValueError('duplicate URL in queue')
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'next_browser_commands.txt').write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')
    write_json(out_dir / 'next_browser_batch.json', dict(
        generated_utc=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        format='one line per page: <plan match_url> TAB <absolute save path>; save as-is, never rename',
        pending_series=len(pending), batch=[dict(hltv_id=s['hltv_id'], url=s['match_url'], day=s['day'],
                                                 local_maps=s['maps'], rating_completions=s['rating_completions'],
                                                 rating_windows=s['rating_windows']) for s in batch]),
        replace=True)
    return pending, batch


PARSER_VERSION = 'bulk-match-parse-v1'
CHECKPOINT = BULK / 'checkpoint'
CANONICAL = re.compile(r'<link rel="canonical" href="([^"]+)"')
RATING_NOTE = ('HLTV Rating 3.0 as displayed at capture time; HLTV recomputed it retroactively, '
               'so it is not a snapshot of what was visible before the match')


def parse_one(job):
    """Worker: parse one saved match page. Returns JSON-safe data only."""
    from known_map_v2_extended.offline_mapstats import Blocked, InvalidPage, parse_match

    path, ident = job
    html = Path(path).read_text(encoding='utf-8', errors='strict')
    m = CANONICAL.search(html)
    try:
        return dict(status='ok', canonical=m[1] if m else None, series=parse_match(html, ident))
    except Blocked as exc:
        return dict(status='blocked', reason=str(exc))
    except InvalidPage as exc:
        return dict(status='invalid', reason=str(exc))
    except Exception as exc:  # malformed file: isolate it, never drop the batch
        return dict(status='invalid', reason=f'{type(exc).__name__}: {exc}')


def parse_cached(pages, workers, cache_dir=CHECKPOINT):
    """Parse match pages once per SHA256 (append-only JSONL checkpoint; resumable)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / 'match_parse_cache.jsonl'
    cache = {}
    if cache_file.exists():
        for line in cache_file.read_text(encoding='utf-8').splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn last line from an interrupted run
            if e.get('parser') == PARSER_VERSION:
                cache[e['sha256'], e['id']] = e['result']
    todo, out = [], {}
    for (kind, ident), path in pages.items():
        if kind != 'match':
            continue
        digest = sha256(path)
        if (digest, ident) in cache:
            out[ident] = cache[digest, ident]
        else:
            todo.append((digest, ident, path))
    if todo:
        from concurrent.futures import ProcessPoolExecutor

        with cache_file.open('a', encoding='utf-8') as f, ProcessPoolExecutor(max_workers=workers) as pool:
            for (digest, ident, _), res in zip(todo, pool.map(parse_one, [(str(p), i) for _, i, p in todo],
                                                               chunksize=4)):
                out[ident] = res
                f.write(json.dumps(dict(parser=PARSER_VERSION, sha256=digest, id=ident, result=res),
                                   ensure_ascii=False) + '\n')
                f.flush()
    return out, len(todo)


def folder_times(folder):
    """captured_utc per file: an existing verified manifest wins, else the download mtime."""
    manifest = Path(folder) / 'capture_manifest.json'
    known = {}
    if manifest.exists():
        blob = json.loads(manifest.read_text(encoding='utf-8-sig'))
        for e in blob['files']:
            known[e['file']] = (e['captured_utc'], e['sha256'], blob.get('method'), blob.get('capture_time_evidence'))
    return known


def capture_entries(pages, parsed, series):
    """Manifest of every page used: URL (plan or canonical, href for mapstats), SHA256, capture time."""
    times = {}
    for path in {p.parent for p in pages.values()}:
        times[path] = folder_times(path)
    hrefs = {}
    for ident, res in parsed.items():
        for p in (res.get('series') or {}).get('played', ()):
            if p['mapstats_id'] is not None:
                hrefs[p['mapstats_id']] = 'https://www.hltv.org' + p['stats_path']
    entries, problems = {}, []
    for (kind, ident), path in sorted(pages.items()):
        digest = sha256(path)
        if kind == 'match':
            res = parsed.get(ident, {})
            url = series[ident]['match_url'] if ident in series else res.get('canonical')
        else:
            url = hrefs.get(ident)
        if url is None:
            problems.append(dict(file=path.name, reason='no plan URL, canonical URL or referencing match page'))
            continue
        known = times[path.parent].get(path.name)
        if known and known[1] != digest:
            problems.append(dict(file=path.name, reason='SHA256 differs from the folder capture manifest'))
            continue
        captured, _, method, evidence = known or (mtime_utc(path), digest, BROWSER_METHOD, TIME_EVIDENCE)
        captured = datetime.fromisoformat(captured.replace('Z', '+00:00')).astimezone(timezone.utc).isoformat()
        entries[kind, ident] = dict(file=path.name, folder=str(path.parent.relative_to(ROOT)), path=path, url=url,
                                    sha256=digest, captured_utc=captured, method=method,
                                    capture_time_evidence=evidence)
    return entries, problems


def match_page_series(parsed_series, entry):
    """Capture structure from the boxes embedded in the match page (provenance_source='match_page')."""
    s = dict(parsed_series)
    page_teams = [t['hltv_team_id'] for t in s['teams']]
    maps, map_issues = [], []
    for p in s.pop('played'):
        players, issue = [], None
        versions = {r['rating_version'] for r in p['rows']}
        if p['mapstats_id'] is None:
            issue = 'no_mapstats_id_on_match_page'  # identity evidence is keyed by mapstats ID
        elif not p['complete']:
            issue = 'incomplete_match_page_box'
        elif versions != {'3.0'} or any(r['rating'] is None for r in p['rows']):
            issue = f'rating_version_not_3.0:{sorted(map(str, versions))}'
        if issue is None or issue.startswith('rating_version'):
            # Complete boxes count as identity evidence even when their rating is unusable.
            players = [dict(hltv_player_id=r['player_id'], nickname=None, kills=r['kills'], deaths=r['deaths'],
                            adr=r['adr'], kast=r['kast'], rating=r['rating'], rating_version=r['rating_version'],
                            page_side=page_teams.index(r['team_id']) + 1) for r in p['rows']]
        if issue:
            map_issues.append(dict(hltv_match_id=s['match_id'], mapstats_id=p['mapstats_id'], map_name=p['name'],
                                   reason=issue))
        maps.append(dict(mapstats_id=p['mapstats_id'], map_name=p['name'], scores=p['scores'], players=players,
                         rating_version='3.0' if issue is None else None, source_url=entry['url'],
                         sha256=entry['sha256'], captured_utc=entry['captured_utc'], provenance_source='match_page',
                         mapstats_url=None if p['stats_path'] is None else 'https://www.hltv.org' + p['stats_path']))
    return dict(s, source_url=entry['url'], sha256=entry['sha256'], captured_utc=entry['captured_utc'],
                provenance_source='match_page', maps=maps), map_issues


def mapstats_series(ident, entries):
    """Every played map saved: use the verified mapstats path (cross-checked against the match page)."""
    from known_map_v2_extended.offline_mapstats import capture_series

    pages = {k: dict(e, path=e['path']) for k, e in entries.items()}
    s = capture_series(ident, pages)
    for m in s['maps']:
        m['provenance_source'] = 'mapstats_page'
    s['provenance_source'] = 'mapstats_page'
    return s


def diagnose_link(series, plan_entry, matches, maps):
    """Why a page did not link to exactly one local series (reported only; no fuzzy linking)."""
    from known_map_recent.hltv_backfill.parse import key

    out = []
    for lid in (plan_entry or {}).get('local_match_ids', ()):
        rows = matches[matches.match_id.eq(lid)]
        if rows.empty:
            continue
        m = rows.iloc[0]
        page = {key(t['name']): t['score'] for t in series['teams']}
        local = {m.team1_key: m.score1, m.team2_key: m.score2}
        why = []
        if set(page) != set(local):
            why.append('team_name_differs')
        if str(m.date)[:10] != series['day']:
            why.append('utc_day_differs')
        if sorted(page.values()) != sorted(local.values()):
            why.append('series_score_differs')
        if m.bo_reported != series['bo']:
            why.append('bo_differs')
        out.append(dict(local_match_id=int(lid), local_teams=list(local), local_day=str(m.date)[:10],
                        page_teams=list(page), page_day=series['day'], causes=why or ['other']))
    return out


def local_fp_index(players, ids):
    from known_map_recent.hltv_backfill.history import fp

    idx = defaultdict(lambda: defaultdict(list))
    for p in players[players.match_id.isin(ids)].itertuples(index=False):
        f = fp(p.side, p.kills, p.deaths, p.adr, p.kast)
        if f is not None:
            idx[int(p.match_id), p.map_name][f].append(int(p.player_id))
    return idx


def full_box_match(m, flip, local_fps):
    """All ten page players hit exactly one distinct local player on their (oriented) side."""
    from known_map_recent.hltv_backfill.history import fp

    if len(m['players']) != 10:
        return False
    hits = []
    for r in m['players']:
        side = 3 - r['page_side'] if flip else r['page_side']
        found = local_fps.get(fp(side, r['kills'], r['deaths'], r['adr'], r['kast']), ())
        if len(found) != 1:
            return False
        hits.append(found[0])
    return len(set(hits)) == 10


def anchored_link(series, plan_entry, matches, maps, excluded, fps):
    """Fallback for pages whose exact signature failed: the plan's single local ID is the only candidate.

    Requires the same UTC day, BO and series score. Team orientation comes from exact names, or (for a
    name alias) from complete 5v5 fingerprints of both teams on at least two maps, with the other
    orientation failing. Page maps missing locally are dropped (labels are never changed); every local
    map must exist on the page with the same score. Returns (local_id, flip, kept_maps, dropped, method).
    """
    from known_map_recent.hltv_backfill.parse import key
    from known_map_v2_extended.offline_mapstats import Rejected, oriented

    ids = (plan_entry or {}).get('local_match_ids', ())
    if len(ids) != 1:
        raise Rejected(f'anchor: plan lists {len(ids)} local series')
    lid = ids[0]
    rows = matches[matches.match_id.eq(lid)]
    if len(rows) != 1:
        raise Rejected('anchor: plan local series not in snapshot')
    local = rows.iloc[0]
    if str(local.date)[:10] != series['day'] or local.bo_reported != series['bo']:
        raise Rejected('anchor: UTC day or BO differs')
    local_maps = {m.map_name: m for m in maps.itertuples(index=False) if int(m.match_id) == lid}
    page = {m['map_name']: m for m in series['maps']}
    names = [key(t['name']) for t in series['teams']]
    if set(names) == {local.team1_key, local.team2_key}:
        flips, method = [names[0] != local.team1_key], 'plan_anchor_exact_names'
    else:
        flips, method = [False, True], 'plan_anchor_fp_alias'
    good = []
    for flip in flips:
        sc = oriented([t['score'] for t in series['teams']], flip)
        if sc != [local.score1, local.score2] or set(local_maps) - set(page):
            continue
        if any(oriented(page[n]['scores'], flip) != [m.score1, m.score2] for n, m in local_maps.items()):
            continue
        boxes = sum(full_box_match(page[n], flip, fps[lid, n]) for n in local_maps)
        if method == 'plan_anchor_fp_alias' and boxes < 2:
            continue
        good.append(flip)
    if len(good) != 1:
        raise Rejected(f'anchor: {len(good)} orientations pass ({method}; needs score agreement'
                       + (' and two complete 5v5 fingerprint maps)' if method.endswith('alias') else ')'))
    flip = good[0]
    kept, dropped = [], []
    for n, m in page.items():
        if n not in local_maps:
            dropped.append(dict(map_name=n, reason='excluded_short_score_review' if (lid, n) in excluded
                                else 'map_absent_from_local_snapshot'))
        elif local_maps[n].player_stats_quarantined or (lid, n) in excluded:
            dropped.append(dict(map_name=n, reason='local_map_quarantined_or_excluded'))
        else:
            kept.append(m)
    return lid, flip, kept, dropped, method


def merge_isolated(base, new):
    """Union by key; identical re-imports are duplicates, disagreeing rows are isolated (not fatal)."""
    from known_map_v2_extended.metric_history import KEY

    seen = {tuple(r[c] for c in KEY): r for r in base['rows']}
    if len(seen) != len(base['rows']):
        raise ValueError('duplicate keys in base metric history')
    same = ('day', 'side', 'kills', 'deaths', 'adr', 'kast', 'rounds', 'mapstats_id', 'hltv_player_id', 'metrics',
            'rating_version', 'metric_source')
    added, duplicates, conflicts = [], 0, []
    for r in new:
        idx = tuple(r[c] for c in KEY)
        old = seen.get(idx)
        if old is None:
            seen[idx] = r
            added.append(r)
        elif all(old.get(c) == r[c] for c in same):
            duplicates += 1
        else:
            conflicts.append(dict(key=list(idx), hltv_id=r['hltv_id'], mapstats_id=r['mapstats_id'],
                                  fields=[c for c in same if old.get(c) != r[c]]))
    return dict(base, rows=base['rows'] + added), added, duplicates, conflicts


def build_series(ident, res, entries):
    """One saved series -> (capture or None, status, reason, map_issues)."""
    from known_map_v2_extended.offline_mapstats import Blocked, InvalidPage, Rejected

    if res['status'] == 'blocked':
        return None, 'blocked', res['reason'], []
    if res['status'] != 'ok':
        return None, 'rejected', f'parse: {res["reason"]}', []
    entry = entries.get(('match', ident))
    if entry is None:
        return None, 'rejected', 'manifest: page has no verifiable URL/SHA', []
    played = res['series']['played']
    ids = [p['mapstats_id'] for p in played]
    if played and None not in ids and all(('mapstats', i) in entries for i in ids):
        try:
            return mapstats_series(ident, entries), 'acquired', None, []
        except (Blocked, InvalidPage, Rejected) as exc:
            return None, 'rejected', f'mapstats_crosscheck: {type(exc).__name__}: {exc}', []
    s, issues = match_page_series(res['series'], entry)
    return s, 'acquired', None, issues


def run(folders, plan, base_history, history_patch, out, workers):
    from known_map_recent.features import excluded_map_keys, load_source
    from known_map_recent.hltv_backfill.history import read_patch
    from known_map_v2_extended.metric_history import VERSION, read_metrics
    from known_map_v2_extended.offline_mapstats import Rejected, link_local, link_players, metric_row

    out = Path(out)
    if out.exists():
        raise FileExistsError(f'Refuse to overwrite {out}')
    imported = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    series_plan, unlinked = load_plan(plan)
    pages, dupes = saved_pages(folders)
    parsed, newly_parsed = parse_cached(pages, workers)
    entries, manifest_problems = capture_entries(pages, parsed, series_plan)
    matches, maps, players, hashes, _ = load_source()
    excluded = excluded_map_keys()
    extra = read_patch(history_patch, matches, maps, players, hashes, excluded)
    base = json.loads(Path(base_history).read_text(encoding='utf-8-sig'))
    read_metrics(base_history, players, hashes, extra)

    status = {i: dict(hltv_id=i, url=s['match_url'], day=s['day'], plan_local_match_ids=s['local_match_ids'],
                      plan_maps=s['maps'], status='pending', reason='no saved match page')
              for i, s in series_plan.items()}
    captures, linked, map_issues, by_local, failed = [], [], [], defaultdict(list), []
    for ident in sorted(parsed):
        st = status.setdefault(ident, dict(hltv_id=ident, url=None, day=None, plan_local_match_ids=[],
                                           plan_maps=[], note='saved page not in plan'))
        s, st['status'], st['reason'], issues = build_series(ident, parsed[ident], entries)
        map_issues += issues
        if s is None:
            continue
        st.update(provenance_source=s['provenance_source'], page_day=s['day'], bo=s['bo'],
                  page_teams=[t['name'] for t in s['teams']])
        captures.append(s)
        try:
            mid, flip = link_local(s, matches, maps, excluded)
            s['link_method'] = 'exact_signature'
        except Rejected as exc:
            st['status'] = 'no_match' if str(exc).startswith('0 local series') else 'rejected'
            st['reason'] = f'local_series_link: {exc}'
            st['diagnosis'] = diagnose_link(s, series_plan.get(ident), matches, maps)
            failed.append(s)
            continue
        s['local_match_id'], s['local_team_order_flipped'] = mid, flip
        by_local[mid].append((s, mid, flip))
    anchors = {i for s in failed for i in series_plan.get(s['match_id'], {}).get('local_match_ids', ())}
    fps = local_fp_index(players, anchors)
    for s in failed:
        st = status[s['match_id']]
        try:
            mid, flip, kept, dropped, method = anchored_link(s, series_plan.get(s['match_id']), matches, maps,
                                                             excluded, fps)
        except Rejected as exc:
            st['anchor_reason'] = str(exc)
            continue
        s.update(maps=kept, dropped_maps=dropped, link_method=method, local_match_id=mid,
                 local_team_order_flipped=flip)
        st.update(status='acquired', reason=None, link_method=method, dropped_maps=dropped,
                  exact_link_failure=st.pop('reason', None))
        by_local[mid].append((s, mid, flip))
    for mid, group in by_local.items():
        if len(group) > 1:  # two saved pages claim one local series: keep neither
            for s, _, _ in group:
                status[s['match_id']].update(status='rejected', reason=f'two pages link local series {mid}')
            continue
        linked.append(group[0])
    prior = {(r['hltv_player_id'], r['player_id']) for r in base['rows']}
    accepted, skipped = link_players(linked, players, prior)
    rows, rating_skips = [], []
    for a in accepted:
        series, _, _, m, r = a[:5]
        if m['rating_version'] != '3.0' or r['rating_version'] != '3.0' or r['rating'] is None:
            rating_skips.append(dict(hltv_match_id=series['match_id'], mapstats_id=m['mapstats_id'],
                                     map_name=m['map_name'], hltv_player_id=r['hltv_player_id'],
                                     reason='identity_ok_but_rating_not_3.0'))
            continue
        row = metric_row(*a, imported)
        row.update(provenance_source=m['provenance_source'], mapstats_url=m.get('mapstats_url', m['source_url']),
                   rating_value_basis='hltv_display_at_capture_time')
        rows.append(row)
    merged, added, duplicates, conflicts = merge_isolated(base, rows)
    conflict_keys = {tuple(c['key']) for c in conflicts}
    if conflict_keys:  # drop every new row on a disputed key; base rows stay untouched
        from known_map_v2_extended.metric_history import KEY
        added = [r for r in added if tuple(r[c] for c in KEY) not in conflict_keys]
        merged = dict(base, rows=base['rows'] + added)
    new_keys = {(r['hltv_id'], r['mapstats_id'], r['hltv_player_id']) for r in added}
    per = defaultdict(Counter)
    for r in rows:
        per[r['hltv_id']]['accepted_player_maps'] += 1
        per[r['hltv_id']]['new_rows'] += (r['hltv_id'], r['mapstats_id'], r['hltv_player_id']) in new_keys
    for x in skipped + rating_skips:
        per[x['hltv_match_id']][x['reason']] += 1
    for s, mid, _ in linked:
        st = status[s['match_id']]
        st.update(local_match_id=mid, counts=dict(per[s['match_id']]))
        if per[s['match_id']]['accepted_player_maps']:
            st['status'], st['reason'] = 'accepted', None
        elif all(m['rating_version'] != '3.0' for m in s['maps']):
            st['status'], st['reason'] = 'unsupported', 'no complete Rating 3.0 box on any map'
        else:
            st['status'], st['reason'] = 'rejected', 'linked, but no player-map passed two-map identity'
    for i, st in status.items():
        if st['status'] == 'acquired':
            st['status'], st['reason'] = 'rejected', st.get('reason') or 'not linked'
    counts = Counter(st['status'] for st in status.values())
    reasons = Counter(f'{st["status"]}: {re.sub(r"[0-9]+", "#", st["reason"] or "")[:90]}'
                      for st in status.values() if st['status'] not in ('accepted',))
    report = dict(
        imported_utc=imported, parser=PARSER_VERSION, folders=[str(Path(f).relative_to(ROOT)) for f in folders],
        plan_series=len(series_plan), plan_unlinked_target_maps=len(unlinked),
        note_unlinked=('target maps without any plan HLTV link need a separate discovery step; they are '
                       'not covered by this import'),
        saved_match_pages=sum(k == 'match' for k, _ in pages), saved_mapstats_pages=sum(k == 'mapstats' for k, _ in pages),
        identical_duplicate_files=len(dupes), parsed_this_run=newly_parsed, manifest_problems=manifest_problems,
        series_status=dict(counts), series_reasons=dict(reasons.most_common()),
        series_by_provenance=dict(Counter(s['provenance_source'] for s, _, _ in linked)),
        series_by_link_method=dict(Counter(s['link_method'] for s, _, _ in linked)),
        accepted_by_link_method=dict(Counter(status[s['match_id']].get('link_method', 'exact_signature')
                                             for s, _, _ in linked if status[s['match_id']]['status'] == 'accepted')),
        anchor_failures=dict(Counter(re.sub(r'[0-9]+', '#', st['anchor_reason'])
                                     for st in status.values() if 'anchor_reason' in st)),
        dropped_page_maps=dict(Counter(d['reason'] for s, _, _ in linked for d in s.get('dropped_maps', ()))),
        linked_series=len(linked), linked_maps=sum(len(s['maps']) for s, _, _ in linked),
        map_box_issues=dict(Counter(x['reason'] for x in map_issues)),
        accepted_player_maps=len(rows), new_rating_rows=len(added), duplicate_existing_rows=duplicates,
        isolated_conflicts=conflicts, skipped_by_reason=dict(Counter(x['reason'] for x in skipped + rating_skips)),
        new_rows_by_year=dict(sorted(Counter(r['day'][:4] for r in added).items())),
        new_rows_by_provenance=dict(Counter(r['provenance_source'] for r in added)),
        new_rating_maps=len({(r['match_id'], r['map_name']) for r in added}),
        rating_note=RATING_NOTE, metrics_not_exported=UNSUPPORTED_METRICS)
    from tempfile import TemporaryDirectory

    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = dict(generated_utc=imported, note='captured_utc is file mtime at Browser download (or a prior '
                    'verified manifest); it is not import time and not historical publication time',
                    files=[{k: v for k, v in e.items() if k != 'path'} for _, e in sorted(entries.items())])
    with TemporaryDirectory(dir=out.parent) as tmp:
        stage = Path(tmp) / out.name
        stage.mkdir()
        write_json(stage / 'capture_manifest.json', manifest)
        write_json(stage / 'queue_status.json', dict(generated_utc=imported, statuses=STATUSES,
                                                     series=[status[i] for i in sorted(status)]))
        write_json(stage / 'unlinked_target_maps.json', dict(note=report['note_unlinked'], rows=unlinked))
        write_json(stage / 'skipped_player_maps.json', dict(rows=skipped + rating_skips, map_box_issues=map_issues))
        write_json(stage / 'rating_patch.json', dict(version=VERSION, source_sha256=hashes, rows=added))
        write_json(stage / 'metric_history.json', dict(merged, report=dict(
            base.get('report', {}), bulk_v1=dict(base_sha256=sha256(base_history), added_rows=len(added),
                                                 rating_note=RATING_NOTE))))
        report['rating_patch_counts'] = read_metrics(stage / 'rating_patch.json', players, hashes, extra)[1]
        report['merged_history_counts'] = read_metrics(stage / 'metric_history.json', players, hashes, extra)[1]
        report['files_sha256'] = {f.name: sha256(f) for f in sorted(stage.iterdir())}
        write_json(stage / 'import_report.json', report)
        stage.rename(out)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)
    q = sub.add_parser('queue', help='write the next Browser download batch (no network)')
    q.add_argument('--take', type=int, default=50)
    i = sub.add_parser('import', help='parse all saved pages and export a Rating 3.0 history (no network)')
    i.add_argument('--base-history', type=Path, default=HERE / 'data/browser_pilot_v1/import/metric_history.json')
    i.add_argument('--history-patch', type=Path, default=HERE / 'data/five_metrics_v1/hltv_history_patch.json')
    i.add_argument('--out', type=Path, default=HERE / 'data/bulk_v1/import')
    i.add_argument('--workers', type=int, default=8)
    for s in (q, i):
        s.add_argument('--plan', type=Path, default=PLAN)
        s.add_argument('--folder', type=Path, action='append', help='saved page folders (default temp/hltv_bulk)')
    args = p.parse_args(argv)
    folders = args.folder or [BULK, ROOT / 'temp/hltv_verified']
    series, unlinked = load_plan(args.plan)
    pages, dupes = saved_pages(folders)
    if args.cmd == 'queue':
        pending, batch = write_queue(series, pages, BULK, args.take)
        print(json.dumps(dict(plan_series=len(series), saved_match_pages=sum(k == 'match' for k, _ in pages),
                              pending=len(pending), batch=len(batch),
                              file=str(BULK / 'next_browser_commands.txt')), indent=2))
        return 0
    report = run(folders, args.plan, args.base_history, args.history_patch, args.out, args.workers)
    print(json.dumps({k: v for k, v in report.items() if k not in ('isolated_conflicts', 'manifest_problems')},
                     indent=2, ensure_ascii=False))
    return 0 if report['new_rating_rows'] else 2


if __name__ == '__main__':
    sys.exit(main())
