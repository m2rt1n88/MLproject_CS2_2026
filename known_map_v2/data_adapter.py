"""Adapt and validate the fixed raw snapshot without touching v9 artifacts."""
from contextlib import closing
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sqlite3
from zipfile import ZipFile

import numpy as np
import pandas as pd

from .runtime import HOME, ROOT

DATA = HOME / 'data'
ZIP = ROOT / 'data/raw/victorpicinin-cs2-v2.zip'
SOURCE = ROOT / 'data/reference/victorpicinin-cs2-v2.json'
UPSTREAM = ROOT / 'references/archotium-cs2-match-prediction'
DATABASE = ROOT / 'references/csgopred/data/csgopred.db'
KEY = ['matchID', 'map']
IDENTITY = ['matchID', 'team1', 'team2', 'map']
TARGET = 'team1_win'
TWO = ('Rating_2.0_mean', 'mapwinrate')
MAPS = ('ancient', 'anubis', 'inferno', 'mirage', 'nuke', 'overpass', 'vertigo')
MAP_ALIASES = {'de_' + name: name for name in MAPS}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf8')


def pairs(columns):
    sides = [sorted(c[3:] for c in columns if c.startswith(f'T{i}_')) for i in (1, 2)]
    if sides[0] != sides[1] or not set(TWO).issubset(sides[0]):
        raise ValueError('Unpaired or missing team statistics')
    return sides[0]


def normalize_map(col):
    return col.str.strip().str.lower().replace(MAP_ALIASES)


def archived_days(db):
    with closing(sqlite3.connect(Path(db).resolve().as_uri() + '?mode=ro&immutable=1', uri=True)) as conn:
        rows = pd.read_sql_query("SELECT match_id, start_date FROM matches WHERE source='hltv'", conn)
    rows = rows[rows.match_id.str.fullmatch(r'hltv:[0-9]+', na=False)].copy()
    rows['matchID'] = rows.match_id.str.split(':').str[1].astype('int64')
    rows['day'] = pd.to_datetime(rows.start_date, errors='raise', format='ISO8601').dt.normalize()
    return rows[['matchID', 'day']]


def prepare(archive=ZIP, upstream=UPSTREAM, database=DATABASE):
    source = json.loads(SOURCE.read_text(encoding='utf8'))
    if digest(archive) != source['archive_sha256']:
        raise ValueError('Wrong Kaggle v2 ZIP checksum')
    spec = importlib.util.spec_from_file_location('known_map_upstream', upstream / 'cs2_data.py')
    upstream_data = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream_data)
    with ZipFile(archive) as zipped, zipped.open(source['raw_member']) as stream:
        raw = pd.read_csv(stream, sep=';')
    with ZipFile(archive) as zipped, zipped.open(source['raw_member']) as stream:
        variants = upstream_data.clean_raw_data(stream, save=False)
    result = upstream_data.build_team_features(variants['all_nan_dropped'], save_path=None).reset_index(drop=True)
    reference = pd.read_csv(upstream / 'team_summary_statistics.csv')
    pd.testing.assert_frame_equal(result.sort_index(axis=1), reference.sort_index(axis=1),
                                  check_dtype=False, atol=1e-10, rtol=1e-10)
    stat_names = pairs(result.columns)
    numeric = [f'T{i}_{name}' for i in (1, 2) for name in stat_names]
    if set(result) != set(IDENTITY + [TARGET] + numeric) or len(numeric) != 142:
        raise ValueError('Unexpected input schema in the pinned snapshot')
    result['map'] = normalize_map(result['map'])
    days = archived_days(database)
    result = result.merge(days, on='matchID', how='left', validate='many_to_one')
    counts = len(raw), len(result), result.matchID.nunique()
    if counts != tuple(source[k] for k in ('raw_row_count', 'complete_map_rows', 'complete_series_count')):
        raise ValueError(f'Unexpected snapshot size: {counts}')
    if (result[IDENTITY + ['day']].isna().any().any() or result.duplicated(KEY).any()
            or not result['map'].isin(MAPS).all() or result[TARGET].value_counts().size != 2
            or not result[TARGET].isin([0, 1]).all() or not np.isfinite(result[numeric]).all().all()):
        raise ValueError('Invalid identity, date, target or statistics in prepared rows')
    DATA.mkdir(exist_ok=True)
    rows = DATA / 'maps.csv'
    result[['day', *IDENTITY, TARGET, *numeric]].sort_values(['day', *KEY]).to_csv(rows, index=False)
    metadata = dict(source, raw_archive_sha256=digest(archive), upstream_cleaner_sha256=digest(upstream / 'cs2_data.py'),
                    archive_db_sha256=digest(database), prepared_sha256=digest(rows),
                    raw_date_time_columns=[str(c) for c in raw if re.search('date|time', c, flags=re.I)],
                    actual_map_rows=len(variants['with_nan']), date_joined_series=counts[2],
                    date_min=str(result.day.min().date()), date_max=str(result.day.max().date()),
                    map_categories=list(MAPS), numerical_columns=numeric,
                    feature_time_provenance='Unavailable; archived day is for splitting and Elo, not snapshot verification.')
    write_json(DATA / 'manifest.json', metadata)
    return result


def load():
    manifest = json.loads((DATA / 'manifest.json').read_text(encoding='utf8'))
    rows = DATA / 'maps.csv'
    if digest(rows) != manifest['prepared_sha256']:
        raise ValueError('Prepared CSV digest mismatch; rerun prepare')
    return pd.read_csv(rows, parse_dates=['day']), manifest


def swap(rows):
    rename = {'team1': 'team2', 'team2': 'team1'}
    rename.update({c: f'T{3-int(c[1])}_{c[3:]}' for c in rows if c.startswith(('T1_', 'T2_'))})
    result = rows.rename(columns=rename).copy()
    if TARGET in result:
        result[TARGET] = 1 - result[TARGET]
    return result
