"""Read-only csgopred index; resumable, separate detail store."""
from contextlib import closing
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3

from .parse import Blocked, InvalidPage, MAP_NAMES, key, parse_detail

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / 'references/csgopred/data/csgopred.db'
STORE = Path(__file__).resolve().parent / '.state/backfill.sqlite3'
CACHE_DIRS = (ROOT / 'references/csgopred/data/raw/hltv/cache', ROOT / '.venv/hltv-crawler-audit-cache')
BASE = 'https://www.hltv.org'
SCHEMA = '''
CREATE TABLE IF NOT EXISTS jobs (
  id INTEGER PRIMARY KEY, day TEXT NOT NULL, slug TEXT NOT NULL, format TEXT NOT NULL,
  team_a TEXT NOT NULL, team_b TEXT NOT NULL, score_a INTEGER NOT NULL, score_b INTEGER NOT NULL,
  archive_scores TEXT, CHECK (team_a != team_b)
);
CREATE TABLE IF NOT EXISTS archive_page_names (
  id INTEGER PRIMARY KEY REFERENCES jobs(id), names TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS observations (
  id INTEGER NOT NULL REFERENCES jobs(id), sha TEXT NOT NULL,
  imported_utc TEXT NOT NULL, source TEXT NOT NULL, outcome TEXT NOT NULL,
  error TEXT, detail TEXT, PRIMARY KEY(id,sha)
);
CREATE TABLE IF NOT EXISTS accepted_scores (
  id INTEGER PRIMARY KEY REFERENCES jobs(id), sha TEXT NOT NULL, scores TEXT NOT NULL, team_ids TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS accepted_veto (
  id INTEGER PRIMARY KEY REFERENCES jobs(id), sha TEXT NOT NULL, veto TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS accepted_players (
  id INTEGER NOT NULL REFERENCES jobs(id), name TEXT NOT NULL, mapstats_id INTEGER NOT NULL UNIQUE,
  sha TEXT NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(id,name)
);
CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE VIEW IF NOT EXISTS progress AS
SELECT j.id, j.day, j.slug, j.score_a+j.score_b AS expected_maps,
       CASE WHEN s.id IS NOT NULL THEN 'page' WHEN j.archive_scores IS NOT NULL THEN 'archive' ELSE 'missing' END AS scores,
       CASE WHEN v.id IS NULL THEN 'missing' ELSE 'complete' END AS veto,
       count(p.name) AS player_maps,
       (SELECT o.outcome FROM observations o WHERE o.id=j.id ORDER BY o.rowid DESC LIMIT 1) AS last_outcome
FROM jobs j LEFT JOIN accepted_scores s ON s.id=j.id LEFT JOIN accepted_veto v ON v.id=j.id
LEFT JOIN accepted_players p ON p.id=j.id GROUP BY j.id;
'''


def dumps(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(',', ':'), sort_keys=True)


def open_store(path):
    path = Path(path).resolve()
    if path.is_file():
        with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)) as probe:
            tables = {r[0] for r in probe.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if tables and not {'jobs', 'observations', 'state'}.issubset(tables):
            raise ValueError('Refusing to modify a database that is not a backfill store')
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA foreign_keys=ON')
    db.executescript(SCHEMA)
    return db


def original_page_names(raw, ident, day, slug, a, b, sa, sb):
    """The upstream importer preserves side order when normalizing display names."""
    try:
        entry = json.loads(raw or 'null')
    except (TypeError, ValueError):
        return None
    if not isinstance(entry, dict) or (entry.get('match_id'), entry.get('date'), entry.get('slug'),
            entry.get('score_a'), entry.get('score_b')) != (ident.split(':')[1], day, slug, sa, sb):
        return None
    orig = [entry.get('team_a'), entry.get('team_b')]
    if not all(isinstance(n, str) and key(n) for n in orig):
        return None
    if list(map(key, orig)) == [key(a), key(b)] or {key(a), key(orig[0])} & {key(b), key(orig[1])}:
        return None
    return dumps(orig)


def seed(db, archive, *, since=None, until=None, ids=()):
    query = '''SELECT m.match_id, start_date, slug, format, team_a, team_b, m.score_a, m.score_b,
                 x.map_name, x.score_a, x.score_b, {raw}
               FROM matches m LEFT JOIN maps x ON m.match_id=x.match_id
               WHERE m.source='hltv' AND m.start_date >= ? AND m.start_date <= ?'''
    args = [date.fromisoformat(since).isoformat() if since else '0001-01-01',
            date.fromisoformat(until).isoformat() if until else '9999-12-31']
    selected = {f'hltv:{i}' for i in ids}
    rows, page_names = {}, {}
    with closing(sqlite3.connect(Path(archive).resolve().as_uri() + '?mode=ro', uri=True)) as old:
        columns = {r[1] for r in old.execute('PRAGMA table_info(matches)')}
        for *info, name, a, b, raw in old.execute(query.format(raw='m.raw' if 'raw' in columns else 'NULL'), args):
            ident, day, slug, fmt, ta, tb, sa, sb = info
            if (selected and ident not in selected) or not re.fullmatch(r'hltv:[1-9]\d*', ident):
                continue
            if (not slug or not re.fullmatch(r'[a-z0-9-]+', slug) or not re.fullmatch(r'bo[135]', fmt)
                    or not ta or not tb or key(ta) == key(tb)):
                continue
            if min(sa, sb) < 0 or min(sa, sb) >= (int(fmt[2]) + 1) // 2 or max(sa, sb) != (int(fmt[2]) + 1) // 2:
                continue
            entry = tuple(info)
            if entry not in rows:
                rows[entry] = []
                page_names[entry] = original_page_names(raw, ident, day, slug, ta, tb, sa, sb)
            if name and key(name) in MAP_NAMES and a != b and min(a, b) >= 0 and max(a, b) >= 13:
                rows[tuple(info)].append((key(name), [a, b]))
    found = {i[0] for i in rows}
    if selected - found:
        raise ValueError(f'IDs missing from the playable archive selection: {sorted(selected-found)}')
    inserted = 0
    with db:
        for info, maps in rows.items():
            ident, day, slug, fmt, ta, tb, sa, sb = info
            hltv_id = int(ident.split(':')[1])
            valid = (len({m[0] for m in maps}) == len(maps)
                     and [sum(s[i] > s[1-i] for _, s in maps) for i in (0, 1)] == [sa, sb])
            inserted += db.execute('INSERT OR IGNORE INTO jobs VALUES (?,?,?,?,?,?,?,?,?)',
                                   (hltv_id, day, slug, fmt, ta, tb, sa, sb, dumps(dict(maps)) if valid else None)).rowcount
            if page_names[info]:
                db.execute('''INSERT OR IGNORE INTO archive_page_names SELECT id,? FROM jobs
                              WHERE id=? AND day=? AND slug=? AND format=? AND team_a=? AND team_b=? AND score_a=? AND score_b=?''',
                           (page_names[info], hltv_id, day, slug, fmt, ta, tb, sa, sb))
    return {'selected': len(rows), 'inserted': inserted}


def page_path(job):
    return f"/matches/{job['id']}/{job['slug']}"


def existing_file(job, folders):
    archive_name = hashlib.sha1(page_path(job).encode()).hexdigest()[:24] + '.html'
    for folder in folders:
        for name in (f"{job['id']}.html", archive_name):
            path = Path(folder) / name
            if path.is_file():
                yield path


def ingest_page(db, job, raw, source, *, reparse_invalid=False):
    """Caller holds the transaction. A partial/error page cannot downgrade accepted fields."""
    sha = hashlib.sha256(raw).hexdigest()
    prior = db.execute('SELECT outcome FROM observations WHERE id=? AND sha=?', (job['id'], sha)).fetchone()
    if prior and not (reparse_invalid and prior[0] == 'invalid'):
        return 'cached'
    detail, error = None, None
    try:
        ident = job['id']
        alias = db.execute('SELECT names FROM archive_page_names WHERE id=?', (ident,)).fetchone()
        detail = parse_detail(raw.decode('utf-8-sig'), job, page_names=json.loads(alias[0]) if alias else None)
        scores = {p['name']: p['scores'] for p in detail['played']}
        teams = dumps(detail['team_ids'])
        old = db.execute('SELECT scores,team_ids FROM accepted_scores WHERE id=?', (ident,)).fetchone()
        old_maps = json.loads(old['scores'] if old else job['archive_scores'] or '{}')
        if ((old and old['team_ids'] != teams)
                or any(scores[m] != old_maps[m] for m in scores.keys() & old_maps.keys())
                or (detail['scores_ok'] and old_maps and scores != old_maps)):
            raise InvalidPage('played maps/team IDs conflict with accepted evidence')
        if detail['scores_ok']:
            old_veto = db.execute('SELECT veto FROM accepted_veto WHERE id=?', (ident,)).fetchone()
            if detail['veto_ok'] and old_veto and old_veto[0] != dumps(detail['veto']):
                raise InvalidPage('veto conflicts with accepted evidence')
            for p in detail['played']:
                if not p['mapstats_id']:
                    continue
                saved = db.execute('''SELECT id,name,mapstats_id,payload FROM accepted_players
                    WHERE (id=? AND name=?) OR mapstats_id=?''', (ident, p['name'], p['mapstats_id'])).fetchall()
                if any(tuple(row)[:3] != (ident, p['name'], p['mapstats_id'])
                       or (p['complete'] and json.loads(row['payload']) != p) for row in saved):
                    raise InvalidPage('mapstats ID or completed player table conflicts with accepted evidence')
            db.execute('INSERT OR IGNORE INTO accepted_scores VALUES (?,?,?,?)', (ident, sha, dumps(scores), teams))
            if detail['veto_ok']:
                db.execute('INSERT OR IGNORE INTO accepted_veto VALUES (?,?,?)', (ident, sha, dumps(detail['veto'])))
            for p in detail['played']:
                if p['complete']:
                    db.execute('INSERT OR IGNORE INTO accepted_players VALUES (?,?,?,?,?)',
                               (ident, p['name'], p['mapstats_id'], sha, dumps(p)))
        progress = db.execute('SELECT * FROM progress WHERE id=?', (ident,)).fetchone()
        outcome = 'complete' if (progress['scores'] == 'page' and progress['veto'] == 'complete'
                                 and progress['player_maps'] == progress['expected_maps']) else 'partial'
        if outcome == 'partial':
            error = 'check reconciled series scores, full veto and five players per team per played map'
    except Blocked as exc:
        outcome, error = 'blocked', str(exc)
        db.execute("INSERT OR REPLACE INTO state VALUES ('blocked',?)", (f"{job['id']}: {error}",))
    except (InvalidPage, UnicodeError) as exc:
        outcome, error = 'invalid', str(exc)
    # After a parser fix, replace only the failed attempt for this exact cached file.
    db.execute('INSERT OR REPLACE INTO observations VALUES (?,?,?,?,?,?,?)',
               (job['id'], sha, datetime.now(timezone.utc).isoformat(timespec='seconds'), source,
                outcome, error, dumps(detail) if detail is not None else None))
    return outcome


def import_cache(db, folders, *, ids=(), limit=100, resume=False, reparse_invalid=False):
    if limit < 1:
        raise ValueError('limit must be positive')
    counts = {}
    with db:
        if resume:
            db.execute("DELETE FROM state WHERE key='blocked'")
        elif db.execute("SELECT 1 FROM state WHERE key='blocked'").fetchone():
            return {'stopped': 'blocked; review the challenge before using --resume'}
    id_filter = set(ids)
    jobs = db.execute('''SELECT j.* FROM jobs j JOIN progress p ON p.id=j.id
        WHERE p.scores!='page' OR p.veto!='complete' OR p.player_maps != p.expected_maps ORDER BY j.day,j.id''').fetchall()
    attempted = 0
    for job in jobs:
        if id_filter and job['id'] not in id_filter:
            continue
        for path in existing_file(job, folders):
            if path.stat().st_size > 8_000_000:
                raise ValueError(f'Unexpected HTML file size: {path}')
            raw = path.read_bytes()
            with db:
                db.execute('BEGIN IMMEDIATE')
                if db.execute("SELECT 1 FROM state WHERE key='blocked'").fetchone():
                    return {**counts, 'stopped': 'blocked'}
                outcome = ingest_page(db, job, raw, str(path.resolve()), reparse_invalid=reparse_invalid)
            counts[outcome] = counts.get(outcome, 0) + 1
            attempted += outcome != 'cached'
            if outcome == 'blocked' or attempted >= limit:
                return counts
            if outcome == 'complete':
                break
    return counts


def status(db):
    fields = ('scores', 'veto', 'player_maps', 'last_outcome')
    total = db.execute('SELECT count(*) FROM jobs').fetchone()[0]
    done = db.execute("SELECT count(*) FROM progress WHERE scores='page' AND veto='complete' AND player_maps=expected_maps").fetchone()[0]
    return dict(total=total, complete=done, components={f: {str(k): n for k, n in db.execute(
        f'SELECT {f}, count(*) FROM progress GROUP BY {f}')} for f in fields},
        blocked=[r[0] for r in db.execute("SELECT value FROM state WHERE key='blocked'")])
