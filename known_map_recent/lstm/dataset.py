"""Day-batched dataset: one row per played map, features come from strictly earlier days."""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .state import HistoryState, SEQ_LEN, SEQ_NAMES, norm_map, norm_team

DEV_END = pd.Timestamp('2026-04-01')        # never build rows on or after this date
WARMUP_END = pd.Timestamp('2024-01-21')     # earlier days only feed history
SPLITS = {'inner_train': ('2024-01-21', '2025-07-31'), 'early_stop': ('2025-08-01', '2025-09-30'),
          'val': ('2025-10-01', '2026-03-31')}


@dataclass
class Rows:
    meta: pd.DataFrame                       # date, match_id, map_name, team1, team2, y, split
    static: np.ndarray                       # [N, 2, S]
    seq: np.ndarray                          # [N, 2, T, F] left-aligned, zero padded
    seq_maps: list = field(default_factory=list)   # [N][2] list of map names
    lengths: np.ndarray = None               # [N, 2]


def day_blocks(matches, maps, players, end=None, metrics=None):
    """Yield (day, matches, maps, players[, metrics]) in date order; `end` is exclusive."""
    frames = [matches, maps, players] + ([] if metrics is None else [metrics])
    if end is not None:
        frames = [f[f.date < end] for f in frames]
    groups = [dict(tuple(f.groupby(f.date.dt.normalize(), sort=True))) for f in frames]
    empty = [f.iloc[:0] for f in frames]
    for stamp in sorted(set().union(*groups)):
        yield stamp.date(), *(g.get(stamp, e) for g, e in zip(groups, empty))


def ingest(state, block):
    day, m, mp, p, *rest = block
    state.ingest_day(day, m, mp, p, metrics=rest[0] if rest else None)


def build_rows(matches, maps, players, *, end=DEV_END, start=WARMUP_END, state=None, feature_set='base',
               metrics=None):
    state = state or HistoryState(feature_set)
    if state.feature_set != feature_set:
        raise ValueError('state and requested feature set differ')
    if feature_set != 'base' and metrics is None:
        raise ValueError(f'feature set {feature_set!r} needs metric events (load_metric_events)')
    if feature_set == 'base':
        metrics = None          # base never sees the new metrics, even if they are passed
    width = len(state.static_names)
    keys = matches.set_index('match_id')[['team1_key', 'team2_key']]
    meta, static, seq, names, lengths, cover = [], [], [], [], [], []
    for block in day_blocks(matches, maps, players, end, metrics):
        day, m_day, maps_day, p_day = block[:4]
        if pd.Timestamp(day) >= start:
            for r in maps_day.sort_values(['match_id', 'map_name']).itertuples(index=False):
                a, b = keys.loc[r.match_id]
                qa = state.query(a, r.map_name, day)       # proxy lineup only, never today's roster
                qb = state.query(b, r.map_name, day)
                meta.append((pd.Timestamp(day), r.match_id, norm_map(r.map_name), norm_team(a), norm_team(b),
                             int(r.team1_won_map)))
                static.append(np.stack([qa[0], qb[0]]))
                pair = np.zeros((2, SEQ_LEN, len(SEQ_NAMES)), dtype=np.float32)
                for s, q in enumerate((qa, qb)):
                    pair[s, :len(q[1])] = q[1]
                seq.append(pair)
                names.append([qa[2], qb[2]])
                lengths.append((len(qa[2]), len(qb[2])))
                cover.append(coverage_row(qa[3], qb[3]))
        ingest(state, block)   # only after every query of this day
    meta = pd.DataFrame(meta, columns=['date', 'match_id', 'map_name', 'team1', 'team2', 'y'])
    meta['split'] = assign_split(meta.date)
    meta = pd.concat([meta, pd.DataFrame(cover, index=meta.index, columns=COVERAGE_COLUMNS)], axis=1)
    n = len(meta)
    return Rows(meta, np.array(static, dtype=np.float32).reshape(n, 2, width),
                np.array(seq, dtype=np.float32).reshape(n, 2, SEQ_LEN, len(SEQ_NAMES)), names,
                np.array(lengths, dtype=np.int64).reshape(n, 2)), state


COVERAGE_COLUMNS = ['hs_cov1', 'hs_cov2', 'rating3_cov1', 'rating3_cov2']


def coverage_row(meta_a, meta_b):
    """Per-side fraction of the proxy roster with an observed metric history (NaN if not in feature set)."""
    out = []
    for kind in ('headshots', 'rating3'):
        for m in (meta_a, meta_b):
            c = m['metric_coverage'].get(kind)
            out.append(np.nan if c is None else c['coverage'])
    return out


def assign_split(dates):
    out = pd.Series('none', index=dates.index)
    for name, (a, b) in SPLITS.items():
        day = dates.dt.normalize()
        out[(day >= pd.Timestamp(a)) & (day <= pd.Timestamp(b))] = name
    return out


def map_vocab(rows, mask):
    """Index 0 = unknown/padding-safe unknown; vocab fitted on training rows only."""
    seen = set(rows.meta.map_name[mask])
    for i in np.flatnonzero(mask):
        for side in rows.seq_maps[i]:
            seen.update(side)
    return ['<unk>', *sorted(seen)]


def encode_maps(names_list, vocab):
    index = {m: i for i, m in enumerate(vocab)}
    out = np.zeros((len(names_list), 2, SEQ_LEN), dtype=np.int64)
    for i, pair in enumerate(names_list):
        for s, names in enumerate(pair):
            out[i, s, :len(names)] = [index.get(m, 0) for m in names]
    return out


def fit_preprocess(rows, fit_split='inner_train'):
    """Map vocabulary and scaler are fitted on `fit_split` rows only."""
    mask = rows.meta.split.eq(fit_split).to_numpy()
    if not mask.any():
        raise ValueError(f'no {fit_split} rows to fit preprocessing')
    vocab = map_vocab(rows, mask)
    scaler = Scaler().fit(rows.static[mask], rows.seq[mask], rows.lengths[mask])
    return vocab, scaler


def model_inputs(rows, idx, vocab, scaler):
    """Arrays (static, seq, seq_maps, lengths, target_map) for selected row indices."""
    idx = np.asarray(idx)
    static, seq = scaler.transform(rows.static[idx], rows.seq[idx], rows.lengths[idx])
    index = {m: i for i, m in enumerate(vocab)}
    target = np.array([index.get(m, 0) for m in rows.meta.map_name.to_numpy()[idx]], dtype=np.int64)
    seq_maps = encode_maps([rows.seq_maps[i] for i in idx], vocab)
    return static, seq, seq_maps, rows.lengths[idx], target


class Scaler:
    """Fit on training rows only; padding steps are excluded from sequence statistics."""

    def fit(self, static, seq, lengths):
        flat = static.reshape(-1, static.shape[-1])
        self.s_mean, self.s_std = flat.mean(0), flat.std(0) + 1e-6
        steps = np.arange(seq.shape[2])[None, None, :] < lengths[..., None]
        valid = seq[steps]
        if len(valid):
            self.q_mean, self.q_std = valid.mean(0), valid.std(0) + 1e-6
        else:
            self.q_mean, self.q_std = np.zeros(seq.shape[-1]), np.ones(seq.shape[-1])
        return self

    def transform(self, static, seq, lengths):
        steps = np.arange(seq.shape[-2]) < lengths[..., None]
        seq = np.where(steps[..., None], (seq - self.q_mean) / self.q_std, 0.0)
        return ((static - self.s_mean) / self.s_std).astype(np.float32), seq.astype(np.float32)

    def to_dict(self):
        return {k: getattr(self, k).astype(float).tolist() for k in ('s_mean', 's_std', 'q_mean', 'q_std')}

    @classmethod
    def from_dict(cls, d):
        obj = cls()
        for k, v in d.items():
            setattr(obj, k, np.asarray(v, dtype=np.float32))
        return obj
