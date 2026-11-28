"""Load the hash-verified Rating 3.0 / headshot history and turn it into per-player-map events.

Only the three history files of the snapshot are read; the wide `maps.csv` (with its Kaggle prefix) is never used.
Every value is re-validated by the existing project validators (`read_patch`, `read_metrics`, `read_history`).
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from known_map_v2.runtime import ROOT

SNAPSHOT = ROOT / 'known_map_v2_extended/data/bulk_with_headshots_v1/dataset'
FILES = ('metric_history.json', 'headshot_history.json', 'hltv_history_patch.json', 'manifest.json')
# Pinned at integration time (HEAD 725b5cd). A different snapshot must be pinned explicitly, never silently.
PINNED_SHA256 = {
    'metric_history.json': '3a490885d90a99c314266b7dd3b8ba0b78af16c51d8c21dc28a1ac0c53388967',
    'headshot_history.json': '612a9e0b2c7f87ddd06f80e995cd01e460208a2a814928c6d5b0b2471b1552d7',
    'hltv_history_patch.json': '671ad55c54ddfc00e1f460eaeb39e147720be6bf2f6f641368c6c542d133643a',
}
ALLOWED_METRICS = {'rating', 'headshot_kills'}
EVENT_COLUMNS = ['date', 'match_id', 'map_name', 'player_id', 'rounds', 'kills', 'rating', 'headshot_kills',
                 'hs_source']


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot_fingerprint(folder=SNAPSHOT, pinned=PINNED_SHA256):
    """sha256 of each file; the three history files must match the manifest and the pinned values."""
    folder = Path(folder)
    found = {f: sha256(folder / f) for f in FILES}
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf8'))
    for f in pinned:
        if manifest['files_sha256'].get(f) != found[f]:
            raise ValueError(f'{f} does not match the snapshot manifest')
        if pinned[f] != found[f]:
            raise ValueError(f'{f} sha256 {found[f]} differs from the pinned snapshot; refusing to load')
    return dict(folder=str(folder.relative_to(ROOT) if folder.is_relative_to(ROOT) else folder),
                sha256=found, manifest_version=manifest['version'],
                manifest_clean_v3_sha256=manifest['source_sha256']['clean_v3'])


def assert_rating3_only(rows):
    """Reject any rating that is not explicitly Rating 3.0 and any metric this model does not accept."""
    for r in rows:
        metrics = r.get('metrics') or {}
        if set(metrics) - ALLOWED_METRICS:
            raise ValueError(f'unsupported metric fields {sorted(set(metrics) - ALLOWED_METRICS)}')
        if 'rating' in metrics and r.get('rating_version') != '3.0':
            raise ValueError(f'rating version {r.get("rating_version")!r} is not 3.0; ratings are never mixed')


def to_events(players, rating_by_key, hltv_hs, opencs2_hs):
    """One row per frozen player-map with any metric; missing fields stay NaN (never 0)."""
    hs = dict(hltv_hs)
    for k, v in opencs2_hs.items():
        if k in hs and hs[k] != v:
            raise ValueError(f'headshot sources disagree for {k}')
        hs.setdefault(k, v)
    source = {k: 'hltv' for k in hltv_hs} | {k: 'opencs2' for k in opencs2_hs if k not in hltv_hs}
    base = players.set_index(['match_id', 'map_name', 'player_id'])[['date', 'rounds', 'kills']]
    keys = sorted(set(rating_by_key) | set(hs))
    known = [k for k in keys if k in base.index]
    out = base.loc[known].reset_index() if known else base.iloc[:0].reset_index()
    out['rating'] = [rating_by_key.get(k, np.nan) for k in known]
    out['headshot_kills'] = [hs.get(k, np.nan) for k in known]
    out['hs_source'] = [source.get(k) for k in known]
    return out[EVENT_COLUMNS].sort_values(['date', 'match_id', 'map_name', 'player_id']).reset_index(drop=True), \
        len(keys) - len(known)


def load_metric_events(source, folder=SNAPSHOT, pinned=PINNED_SHA256):
    """Return (events, fingerprint). `source` is the tuple from known_map_recent.features.load_source()."""
    from known_map_recent.features import excluded_map_keys
    from known_map_recent.hltv_backfill.history import read_patch
    from known_map_v2_extended.metric_history import read_metrics
    from known_map_v2_extended.opencs2_headshots import read_history

    matches, maps, players, digests = source[:4]
    folder = Path(folder)
    fingerprint = snapshot_fingerprint(folder, pinned)
    if fingerprint['manifest_clean_v3_sha256'] != digests:
        raise ValueError('metric snapshot was built from a different clean_v3 snapshot')
    blob = json.loads((folder / 'metric_history.json').read_text(encoding='utf-8-sig'))
    assert_rating3_only(blob['rows'])
    # The 14-row basic patch is only needed so read_metrics can validate the metric rows on those keys.
    extra = read_patch(folder / 'hltv_history_patch.json', matches, maps, players, digests, excluded_map_keys())
    by_key, counts = read_metrics(folder / 'metric_history.json', players, digests, extra)
    ratings = {k: float(v['rating']) for k, v in by_key.items() if 'rating' in v}
    hltv_hs = {k: int(v['headshot_kills']) for k, v in by_key.items() if 'headshot_kills' in v}
    opencs2 = {k: int(v) for k, v in read_history(folder / 'headshot_history.json', players, digests).items()}
    events, off_frozen = to_events(players, ratings, hltv_hs, opencs2)
    fingerprint.update(
        validated_counts=dict(counts, opencs2_headshot_kills=len(opencs2)),
        basic_patch=dict(rows=len(extra), used_as_history=False,
                         days=sorted(set(extra.day)), reason='only validates metric rows; all rows are after '
                         'the development end and are not in the frozen clean_v3 player table'),
        events=dict(rows=len(events), rating=int(events.rating.notna().sum()),
                    headshot_hltv=int((events.hs_source == 'hltv').sum()),
                    headshot_opencs2=int((events.hs_source == 'opencs2').sum()),
                    skipped_not_in_frozen_player_table=off_frozen))
    return events, fingerprint
