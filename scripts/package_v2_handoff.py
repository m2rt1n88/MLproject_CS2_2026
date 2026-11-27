"""Build an allowlisted, portable v2 partner handoff without copying local caches."""
import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
SNAP = Path('known_map_v2_extended/data/five_metrics_v1')
DB = Path('known_map_recent/hltv_backfill/.state/backfill.sqlite3')
ARCHIVE = Path('references/csgopred/data/csgopred.db')
UPSTREAM = Path('references/csgopred')


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def json_write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')


def json_read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def copy(rel, dest, target=None):
    out = dest / (target or rel)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / rel, out)
    return out


def read_db(path):
    return sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)


def portable_source(value):
    if not isinstance(value, str):
        raise ValueError('Unexpected source in handoff')
    if value.startswith('https://') or re.fullmatch(r'archived-hltv-html:[a-zA-Z0-9_.-]+', value):
        return value
    name = value.replace('\\', '/').rsplit('/', 1)[-1]
    if re.fullmatch(r'[a-zA-Z0-9_.-]+', name):
        return 'archived-hltv-html:' + name
    raise ValueError('Unrecognized source in handoff')


def slim_archive(dest):
    """Keep the inputs to seed()/archive_signatures(), not other upstream tables."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    assert not dest.exists()
    with closing(read_db(ROOT / ARCHIVE)) as src, closing(sqlite3.connect(dest)) as out, out:
        out.executescript('''
            CREATE TABLE matches(match_id TEXT PRIMARY KEY, source TEXT NOT NULL, start_date TEXT NOT NULL,
                slug TEXT, format TEXT NOT NULL, team_a TEXT NOT NULL, team_b TEXT NOT NULL,
                score_a INTEGER NOT NULL, score_b INTEGER NOT NULL, raw TEXT);
            CREATE TABLE maps(match_id TEXT NOT NULL, map_order INTEGER NOT NULL, map_name TEXT NOT NULL,
                score_a INTEGER NOT NULL, score_b INTEGER NOT NULL, PRIMARY KEY(match_id, map_order));
            CREATE INDEX matches_by_day ON matches(start_date);
        ''')
        rows = src.execute('''SELECT match_id, source, start_date, slug, format, team_a, team_b, score_a, score_b, raw
                            FROM matches WHERE source='hltv' ORDER BY match_id''')
        fields = ('match_id', 'date', 'slug', 'score_a', 'score_b', 'team_a', 'team_b')
        for *values, raw in rows:
            try:
                data = json.loads(raw or 'null')
                small = {k: data[k] for k in fields if k in data} if isinstance(data, dict) else None
            except (ValueError, TypeError):
                small = None
            out.execute('INSERT INTO matches VALUES (?,?,?,?,?,?,?,?,?,?)', (*values, json.dumps(small) if small else None))
        out.executemany('INSERT INTO maps VALUES (?,?,?,?,?)', src.execute('''
            SELECT x.match_id,x.map_order,x.map_name,x.score_a,x.score_b FROM maps x JOIN matches m ON x.match_id=m.match_id
            WHERE m.source='hltv' ORDER BY x.match_id,x.map_order'''))
        sizes = {t: out.execute(f'SELECT count(*) FROM {t}').fetchone()[0] for t in ('matches', 'maps')}
        if out.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
            raise ValueError('Invalid archive index')
    return sizes


def copy_state(dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    with closing(read_db(ROOT / DB)) as src, closing(sqlite3.connect(dest)) as out:
        src.backup(out)
        with out:
            rows = out.execute('SELECT id,sha,source FROM observations').fetchall()
            out.executemany('UPDATE observations SET source=? WHERE id=? AND sha=?',
                            [(portable_source(source), ident, sha) for ident, sha, source in rows])
        out.execute('PRAGMA journal_mode=DELETE')
        out.execute('VACUUM')  # Remove superseded local-path text from free pages in the copy.
        if out.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
            raise ValueError('Invalid copied backfill store')


def bundle_snapshot(dest):
    for name in ('maps.csv', 'row_provenance.csv', 'headshot_history.json'):
        copy(SNAP / name, dest)
    manifest = json_read(ROOT / SNAP / 'manifest.json')
    for label in ('hltv_history_patch', 'metric_history'):
        name = manifest[label]['file']
        blob = json_read(ROOT / SNAP / name)
        for row in blob['rows']:
            row['source'] = portable_source(row['source'])
        json_write(dest / SNAP / name, blob)
        updated = digest(dest / SNAP / name)
        manifest[label]['sha256'] = manifest['files_sha256'][name] = updated
    json_write(dest / SNAP / 'manifest.json', manifest)


VERIFY = '''"""Verify the files in the delivered package, before importing another batch."""
import hashlib
from pathlib import Path

root = Path(__file__).resolve().parent
items = (root / 'SHA256SUMS.txt').read_text(encoding='utf8').splitlines()
errors = []
for line in items:
    expected, name = line.split('  ', 1)
    path = root / name
    if not path.is_file():
        errors.append(name + ' (missing)')
        continue
    with path.open('rb') as f:
        if hashlib.file_digest(f, 'sha256').hexdigest() != expected:
            errors.append(name + ' (different)')
if errors:
    raise SystemExit('File verification failed: ' + ', '.join(errors))
print(f'Checked {len(items)} delivered files: OK')
'''


def build(dest, zip_path):
    if dest.exists() or zip_path.exists():
        raise FileExistsError('Choose a new handoff name; existing handoffs are not overwritten')
    dest.mkdir(parents=True)
    for source, target in (('V2_PARTNER_HANDOFF.md', 'README.md'), ('V2_PARTNER_SOURCES.md', 'SOURCES.md')):
        path = copy(f'handoff/{source}', dest, target)
        path.write_text(path.read_text(encoding='utf-8-sig'), encoding='utf8')
    for directory in ('known_map_v2_extended', 'known_map_recent/hltv_backfill'):
        for f in sorted((ROOT / directory).glob('*.py')):
            if f.name != 'audit_hltv.py':  # The existing audit JSON does not need the raw Kaggle archive.
                copy(f.relative_to(ROOT), dest)
    include = '''
    known_map_v2/__init__.py known_map_v2/runtime.py known_map_v2/data_adapter.py
    known_map_v2/data/maps.csv known_map_v2/data/manifest.json
    known_map_v2_extended/data/maps.csv known_map_v2_extended/data/manifest.json known_map_v2_extended/data/row_provenance.csv
    known_map_v2_extended/hltv_audit_verified.json known_map_v2_extended/data/backfill_plan_ranked.json
    known_map_recent/__init__.py known_map_recent/features.py known_map_recent/hltv_backfill/README.md
    known_map_recent/tests/test_hltv_backfill.py
    scripts/experiment_known_map.py scripts/basic_team_features.py scripts/match_splits_and_metrics.py
    scripts/roster_and_map_features.py scripts/team_and_opponent_strength.py scripts/train_m1_m2.py
    data/reference/victorpicinin-cs2-v2.json data/raw/players.csv
    data/processed/clean_v3/README.md data/processed/clean_v3/manifest.json data/processed/clean_v3/maps.csv
    data/processed/clean_v3/matches.csv data/processed/clean_v3/player_stats.csv
    '''
    for rel in include.split():
        copy(rel, dest)
    for rel in ('LICENSE', 'CITATION.cff', 'src/csgopred/__init__.py', 'src/csgopred/config.py',
                'src/csgopred/normalize.py', 'src/csgopred/ingest/__init__.py', 'src/csgopred/ingest/hltv.py'):
        copy(UPSTREAM / rel, dest)
    archive_sizes = slim_archive(dest / ARCHIVE)
    code_revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT / UPSTREAM, text=True).strip()
    json_write(dest / UPSTREAM / 'UPSTREAM.json', dict(repository='https://github.com/Aokua/csgopred',
        checkout_revision=code_revision, included='HLTV ingestion source and its dependencies; not the full project',
        archive='Local derived HLTV index; not the original upstream snapshot', archive_rows=archive_sizes))
    copy_state(dest / DB)
    bundle_snapshot(dest)
    copy('known_map_v2_extended/requirements.txt', dest, 'requirements.txt')
    (dest / 'requirements-optional.txt').write_text('''# For separately approved use of upstream client or public OpenCS2 parquets.
curl_cffi>=0.8,<1
duckdb>=1,<2
''', encoding='utf8')
    (dest / 'verify_delivery.py').write_text(VERIFY, encoding='utf8')
    files = sorted(f for f in dest.rglob('*') if f.is_file())
    manifest = dest / 'SHA256SUMS.txt'
    manifest.write_text(''.join(f'{digest(f)}  {f.relative_to(dest).as_posix()}\n' for f in files), encoding='utf8')
    with ZipFile(zip_path, 'x', compression=ZIP_DEFLATED, compresslevel=6) as out:
        for f in (*files, manifest):
            out.write(f, arcname=f.relative_to(dest.parent).as_posix())
    with ZipFile(zip_path) as out:
        if out.testzip() is not None:
            raise ValueError('ZIP CRC verification failed')
    print(f'{zip_path.resolve()} ({len(files) + 1} files, {zip_path.stat().st_size:,} bytes)')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name', default='CS2-v2-partner-handoff-2026-11-27')
    args = p.parse_args()
    if not re.fullmatch(r'CS2-v2-[A-Za-z0-9-]+', args.name):
        p.error('Use a single handoff folder name starting with CS2-v2-')
    dest = ROOT / 'handoff' / args.name
    build(dest, dest.with_suffix('.zip'))


if __name__ == '__main__':
    main()
