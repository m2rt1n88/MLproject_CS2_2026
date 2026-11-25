"""Auditable v9 cleaning into clean_v3; never overwrite clean_v2 or fit a model.

Run: python -B -m scripts.build_clean_v3
Use --tables-only to omit chronological train/validation feature rebuilding.
All quality flags describe observed data quality, not pre-match predictors.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import unicodedata

import numpy as np
import pandas as pd

from scripts.build_clean_match_dataset import clean, classify, SLOTS, roster_flags
from scripts.match_splits_and_metrics import ROOT, csv_checksum, verify_snapshot

STATS = ('kills', 'deaths', 'assists', 'adr', 'kast', 'kddiff')
AUDIT_COLUMNS = ['source_row', 'match_id', 'game_id', 'side', 'slot', 'field',
                 'reason', 'action', 'before', 'after']


def team_key(value):
    if pd.isna(value):
        return None
    return ' '.join(unicodedata.normalize('NFKC', str(value)).split()).casefold() or None


def integer(value, positive=False):
    try:
        return bool(np.isfinite(value) and value % 1 == 0 and value >= (1 if positive else 0))
    except (TypeError, ValueError):
        return False


def refine(raw, tiers):
    """Return tables, history events and a cell/row audit; no feature/model fitting."""
    d = raw.copy().reset_index(drop=True)
    d['_source_row'] = np.arange(len(d)) + 2  # header is CSV line 1
    audit = []

    def note(index, reason, action, field='', before='', after='', side='', slot=''):
        r = d.loc[index]
        audit.append([int(r['_source_row']), r.match_id, r.game_id, side, slot,
                      field, reason, action, str(before), str(after)])

    def blank(index, columns, reason, side='', slot=''):
        for col in columns:
            if pd.notna(d.at[index, col]):
                note(index, reason, 'set_missing', col, d.at[index, col], '', side, slot)
                d.at[index, col] = np.nan

    if not pd.api.types.is_bool_dtype(d.is_total) or d.is_total.isna().any():
        raise ValueError('is_total must be nonmissing boolean; do not coerce strings with bool()')
    bad_ids = ~d.match_id.map(lambda x: integer(x, True))
    if bad_ids.any():
        raise ValueError('Invalid match IDs require source correction')
    parsed = pd.to_datetime(d.datetime, utc=True, errors='coerce')
    bad_dates = parsed.isna() | parsed.ge(pd.Timestamp('2026-11-01', tz='UTC'))
    # A broken map is isolated; a broken aggregate also removes its dependent maps.
    invalid_matches = set(d.loc[bad_dates & d.is_total, 'match_id'])
    remove = set(d.index[bad_dates])
    for i in d.index[bad_dates]:
        note(i, 'invalid_or_out_of_snapshot_date', 'quarantine_row', 'datetime', d.at[i, 'datetime'])

    for col in ('team1', 'team2'):
        for i in d.index:
            old = d.at[i, col]; new = team_key(old)
            if pd.notna(old) and new != old:
                note(i, 'normalized_team_name', 'normalize', col, old, new)
            d.at[i, col] = new

    for i, row in d[d.is_total & ~d.index.isin(remove)].iterrows():
        reason = None
        if row.team1 is None or row.team2 is None:
            reason = 'missing_team_name'
        elif row.team1 == row.team2:
            reason = 'same_team_on_both_sides'
        elif not integer(row.score1_match) or not integer(row.score2_match):
            reason = 'missing_or_invalid_series_score'
        elif row.score1_match == row.score2_match:
            reason = 'draw_not_binary_target'
        elif pd.isna(tiers.get(row.match_id)):
            reason = 'missing_tier'
        if reason:
            invalid_matches.add(row.match_id)
            note(i, reason, 'quarantine_match')

    core = ['team1', 'team2', 'team1_id', 'team2_id', 'score1_match', 'score2_match', 'bestOf']
    aggregates = d[d.is_total & ~d.match_id.isin(invalid_matches)]
    for mid, group in aggregates.groupby('match_id', sort=False):
        if len(group) < 2:
            continue
        conflict = group[core].nunique(dropna=False).gt(1).any()
        time_conflict = parsed.loc[group.index].nunique() > 1
        if conflict or time_conflict:
            invalid_matches.add(mid)
            for i in group.index:
                note(i, 'conflicting_aggregate_identity_or_result', 'quarantine_match')
            continue
        first = group.index[0]
        # Same result does not imply that competing roster payloads are reliable.
        for side, slots in enumerate(SLOTS, 1):
            values = list(group[slots].itertuples(index=False, name=None))
            rosters = [frozenset(v for v in row if pd.notna(v)) for row in values]
            most = max(range(len(rosters)), key=lambda n: len(rosters[n]))
            if not all(roster.issubset(rosters[most]) for roster in rosters):
                blank(first, slots, 'conflicting_duplicate_aggregate_roster', side)
            elif len(rosters[most]) > len(rosters[0]):
                # A superset is observed in another duplicate of this same match.
                # Never reconstruct a roster from other dates or team identity.
                for col, value in zip(slots, values[most]):
                    previous = d.at[first, col]
                    if pd.notna(value) and (pd.isna(previous) or previous != value):
                        note(first, 'recovered_duplicate_aggregate_roster', 'fill_from_duplicate',
                             col, previous, value, side)
                        d.at[first, col] = value
        for i in group.index[1:]:
            note(i, 'duplicate_aggregate', 'drop_duplicate')
            remove.add(i)
        if group.tournament.nunique(dropna=False) > 1:
            note(first, 'tournament_alias_unverified', 'flag_review', 'tournament',
                 json.dumps(group.tournament.tolist(), ensure_ascii=False))

    active_ids = set(d.loc[d.is_total, 'match_id']) - invalid_matches
    for i in d.index[~d.is_total & ~d.match_id.isin(active_ids)]:
        note(i, 'no_accepted_aggregate', 'quarantine_row')
        remove.add(i)
    remove.update(d.index[d.match_id.isin(invalid_matches)])

    # Invalid player identifiers are never silently truncated with int().
    for i in d.index[~d.index.isin(remove)]:
        for side, slots in enumerate(SLOTS, 1):
            for slot, col in enumerate(slots, 1):
                pid = d.at[i, col]
                if pd.notna(pid) and not integer(pid, True):
                    blank(i, [col] + [col[:-3] + '_' + s for s in STATS],
                          'invalid_player_id', side, slot)
            values = d.loc[i, slots].dropna()
            if values.duplicated().any():
                blank(i, slots, 'duplicate_player_within_team', side)
        a, b = (set(d.loc[i, slots].dropna()) for slots in SLOTS)
        if a & b:
            blank(i, SLOTS[0] + SLOTS[1], 'player_on_both_teams')

    agg = d[d.is_total & ~d.index.isin(remove)].set_index('match_id')
    metadata = ['datetime', 'tournament', 'team1_id', 'team2_id']
    for i in d.index[~d.is_total & ~d.index.isin(remove)]:
        r = d.loc[i]; parent = agg.loc[r.match_id]
        if (r.team1, r.team2) != (parent.team1, parent.team2):
            note(i, 'map_team_orientation_mismatch', 'quarantine_row')
            remove.add(i)
            continue
        if pd.isna(r.map_name) or not str(r.map_name).strip():
            note(i, 'missing_map_name', 'quarantine_row')
            remove.add(i)
            continue
        if not integer(r.score1_game) or not integer(r.score2_game) or r.score1_game == r.score2_game:
            note(i, 'missing_invalid_or_draw_map_score', 'quarantine_row')
            remove.add(i)
            continue
        d.at[i, 'map_name'] = str(r.map_name).strip().casefold()
        if parsed.loc[i] != pd.to_datetime(parent.datetime, utc=True):
            note(i, 'map_date_differs_from_aggregate', 'quarantine_row')
            remove.add(i)
            continue
        for col in metadata[1:]:
            if r[col] != parent[col]:
                note(i, 'map_metadata_aligned_to_aggregate', 'replace_metadata', col, r[col], parent[col])
                d.at[i, col] = parent[col]

    # Quarantine every conflicting candidate, rather than selecting a convenient winner.
    candidates = d[~d.is_total & ~d.index.isin(remove)]
    for _, g in candidates.groupby(['match_id', 'map_name'], sort=False):
        if len(g) < 2:
            continue
        if g[['score1_game', 'score2_game']].nunique(dropna=False).gt(1).any():
            for i in g.index:
                note(i, 'conflicting_duplicate_map_score', 'quarantine_row')
                remove.add(i)
        else:
            first = g.index[0]
            payload = [c for c in d if c.startswith(('team1_player', 'team2_player'))]
            if g[payload].nunique(dropna=False).gt(1).any():
                blank(first, [c for c in payload if c.endswith(tuple('_' + s for s in STATS))],
                      'conflicting_duplicate_map_statistics')
            for i in g.index[1:]:
                note(i, 'duplicate_map', 'drop_duplicate')
                remove.add(i)

    bad_stat_games = set()
    for i in d.index[~d.is_total & ~d.index.isin(remove)]:
        r = d.loc[i]; rounds = r.score1_game + r.score2_game
        contradictions = []
        for side in (1, 2):
            for slot in range(1, 6):
                p = f'team{side}_player{slot}'
                for stat in STATS:
                    col = p + '_' + stat; value = d.at[i, col]
                    if pd.isna(value):
                        continue
                    valid = np.isfinite(value)
                    if stat in ('kills', 'deaths', 'assists'):
                        valid = valid and integer(value)
                    elif stat == 'adr':
                        valid = valid and value >= 0
                    elif stat == 'kast':
                        valid = valid and 0 <= value <= 100
                    elif stat == 'kddiff':
                        valid = valid and value % 1 == 0
                    if not valid:
                        blank(i, [col], 'invalid_player_stat_range', side, slot)
                death = d.at[i, p + '_deaths']
                if pd.notna(death) and death > rounds:
                    contradictions.append(p + '_deaths')
                k, de, kd = (d.at[i, p + '_' + s] for s in ('kills', 'deaths', 'kddiff'))
                if all(pd.notna(v) for v in (k, de, kd)) and k - de != kd:
                    contradictions.append(p + '_kddiff')
        if contradictions:
            bad_stat_games.add(r.game_id)
            note(i, 'map_statistics_internal_contradiction', 'exclude_map_player_statistics',
                 ','.join(contradictions), 'cannot identify which source field is wrong')
            # Do not guess whether the scoreboard or individual statistics are wrong.
            blank(i, [f'team{s}_player{p}_{stat}' for s in (1, 2)
                      for p in range(1, 6) for stat in STATS], 'unreliable_map_statistics')

    accepted = d.loc[~d.index.isin(remove)].copy()
    matches, maps, raw_maps, _ = clean(accepted.drop(columns='_source_row'), tiers)
    # Preserve evidence that was masked before the legacy cleaner saw the rows.
    original_aggregate_flags = {}
    for row in raw.loc[raw.is_total].itertuples(index=False):
        flags = roster_flags(*[[getattr(row, c) for c in slots] for slots in SLOTS])
        previous = original_aggregate_flags.get(row.match_id, (False, False, False))
        original_aggregate_flags[row.match_id] = tuple(a or b for a, b in zip(previous, flags))
    for position, col in enumerate(('team1_repeat', 'team2_repeat', 'roster_cross')):
        matches[col] |= matches.match_id.map(lambda mid: original_aggregate_flags[mid][position])
    for side, slots in enumerate(SLOTS, 1):
        matches.loc[matches[f'team{side}_repeat'] | matches.roster_cross, slots] = np.nan
    conflict_ids = {row[1] for row in audit if row[6] == 'conflicting_duplicate_map_score'}
    matches['ambiguous_map'] |= matches.match_id.isin(conflict_ids)
    matches['maps_complete'] &= ~matches.ambiguous_map
    matches['team1_key'] = matches.team1
    matches['team2_key'] = matches.team2
    events = defaultdict(list); player_rows = []
    ix = matches.set_index('match_id')
    for r in raw_maps.itertuples(index=False):
        sides = [[getattr(r, c) for c in slots] for slots in SLOTS]
        rep1, rep2, cross = roster_flags(*sides)
        for side, (slots, repeat) in enumerate(zip(SLOTS, (rep1, rep2)), 1):
            parent = ix.loc[r.match_id]
            if repeat or cross or parent.roster_cross or parent[f'team{side}_repeat']:
                continue
            for slot, col in enumerate(slots, 1):
                prefix = col[:-3]; pid = getattr(r, col)
                adr, kast = (getattr(r, prefix + '_' + s) for s in ('adr', 'kast'))
                if integer(pid, True) and np.isfinite(adr) and adr >= 0 and np.isfinite(kast) and 0 <= kast <= 100:
                    events[r.match_id].append((int(pid), adr, kast))
                    player_rows.append([r.match_id, r.game_id, r.date, r.map_name, side, slot, int(pid),
                                        *[getattr(r, prefix + '_' + s) for s in STATS]])
    player_stats = pd.DataFrame(player_rows, columns=['match_id', 'game_id', 'date', 'map_name',
                               'side', 'slot', 'player_id', *STATS])
    # Flags are kept out of design()/training inputs.
    quality = matches[['match_id', 'split', 'bo_reported', 'bo_score_conflict', 'map_score_verified',
                       'maps_complete', 'n_maps', 'team1_repeat', 'team2_repeat', 'roster_cross']].copy()
    issues = pd.DataFrame(audit, columns=AUDIT_COLUMNS)
    for reason in ('duplicate_player_within_team', 'player_on_both_teams',
                   'map_statistics_internal_contradiction', 'conflicting_duplicate_map_score'):
        quality[reason] = quality.match_id.isin(issues.loc[issues.reason.eq(reason), 'match_id'])
    quality['needs_review'] = quality.bo_score_conflict | quality.match_id.isin(
        issues.loc[issues.action.isin(['flag_review', 'exclude_map_player_statistics', 'quarantine_row']), 'match_id'])
    maps['player_stats_quarantined'] = pd.MultiIndex.from_frame(maps[['match_id', 'map_name']]).isin(
        pd.MultiIndex.from_frame(raw_maps.loc[raw_maps.game_id.isin(bad_stat_games), ['match_id', 'map_name']]))
    maps['short_score_needs_review'] = maps[['score1', 'score2']].max(axis=1).lt(13)
    quality['needs_review'] |= quality.match_id.isin(maps.loc[maps.short_score_needs_review, 'match_id'])
    quarantine = raw.reset_index(drop=True).loc[sorted(remove)].copy()
    quarantine.insert(0, '_source_row', np.array(sorted(remove), dtype=int) + 2)
    identities = pd.concat([raw[[f'team{s}_id', f'team{s}']].rename(
        columns={f'team{s}_id': 'source_team_id', f'team{s}': 'source_team_name'}) for s in (1, 2)])
    identities['team_key'] = identities.source_team_name.map(team_key)
    identities = identities.groupby(['source_team_id', 'source_team_name', 'team_key'],
                                    dropna=False).size().reset_index(name='source_rows')
    return dict(matches=matches, maps=maps, player_stats=player_stats, match_quality=quality,
                audit_log=issues, quarantined_rows=quarantine, team_identity_audit=identities), events


def main(tables_only=False):
    verify_snapshot()
    # Guard the complete published release, including files outside its manifest.
    protected = [ROOT/'data/v9_manifest.json', ROOT/'data/processed/clean_v2/manifest.json']
    for folder in ('data/raw', 'data/processed/clean_v2', 'models/m1', 'models/m2', 'reports/release'):
        protected.extend(p for p in (ROOT/folder).rglob('*') if p.is_file())
    before = {p: csv_checksum(p) for p in protected}
    raw = pd.read_csv(ROOT/'data/raw/cs2_all_tiers_games.csv', low_memory=False)
    tiers = pd.concat([pd.Series(t, index=pd.read_csv(ROOT/f'data/raw/cs2_{t}_games.csv',
                       usecols=['match_id']).match_id.unique()) for t in ('tier1', 'tier2', 'tier3')], verify_integrity=True)
    tables, events = refine(raw, tiers)
    out = ROOT/'data/processed/clean_v3'
    out.mkdir(parents=True, exist_ok=True)
    matches, maps = tables['matches'], tables['maps']
    if not tables_only:
        from scripts.basic_team_features import build_features
        from scripts.roster_and_map_features import past_features
        from scripts.team_and_opponent_strength import build
        from threadpoolctl import threadpool_limits
        history = matches[matches.split.isin(['train', 'val'])].copy()
        history_maps = maps[maps.match_id.isin(history.match_id)]
        print('Rebuilding train/validation history; no test predictions or model fitting.', flush=True)
        p0 = build_features(history, history_maps)
        p1 = past_features(history, history_maps, events)
        with threadpool_limits(limits=1):
            strength = build(history, history_maps)
        features = p0.merge(p1, on='match_id', validate='one_to_one').merge(strength, on='match_id', validate='one_to_one')
        for split in ('train', 'val'):
            tables['m2/' + split] = features[features.split.eq(split)]
    paths = {}
    for name, table in tables.items():
        path = out/(name + '.csv'); path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False)
        paths[name + '.csv'] = csv_checksum(path)
    old = pd.read_csv(ROOT/'data/processed/clean_v2/matches.csv')
    summary = {'version': 'v9-clean-v3', 'source_manifest_sha256': before[ROOT/'data/v9_manifest.json'],
               'raw_rows': len(raw), 'counts': matches.split.value_counts().to_dict(),
               'map_rows': len(maps), 'accepted_player_stat_rows': len(tables['player_stats']),
               'map_statistics_quarantined': int(maps.player_stats_quarantined.sum()),
               'short_map_scores_for_review': int(maps.short_score_needs_review.sum()),
               'identity_counts': {'source_team_ids': int(tables['team_identity_audit'].source_team_id.nunique()),
                   'normalized_names': int(tables['team_identity_audit'].team_key.nunique()),
                   'names_with_multiple_source_ids': int(tables['team_identity_audit'].groupby('team_key').source_team_id.nunique().gt(1).sum())},
               'quarantined_source_rows': len(tables['quarantined_rows']),
               'actions': dict(Counter(tables['audit_log'].action)),
               'reason_counts': tables['audit_log'].groupby('reason').agg(
                   audit_entries=('source_row', 'size'), source_rows=('source_row', 'nunique'),
                   matches=('match_id', 'nunique')).to_dict(orient='index'),
               'removed_v2_match_ids': sorted(set(old.match_id) - set(matches.match_id)),
               'added_v2_match_ids': sorted(set(matches.match_id) - set(old.match_id)),
               'bo_policy': 'unknown for all pre-match rows; never infer from final score',
               'identity_policy': 'NFKC + whitespace + casefold name; source team IDs are not global identities',
               'feature_policy': 'train/val only; earlier-day history; no model fitting or test evaluation',
               'files': paths}
    if not tables_only:
        from scripts.train_m1_m2 import design
        changes = {}
        for split in ('train', 'val'):
            current = tables['m2/' + split].set_index('match_id', drop=False)
            baseline = pd.read_csv(ROOT/f'data/processed/clean_v2/m2/{split}.csv').set_index('match_id', drop=False)
            overlap = current.index.intersection(baseline.index)
            delta = ~np.isclose(design(current.loc[overlap]).to_numpy(),
                                design(baseline.loc[overlap]).to_numpy(), rtol=1e-10, atol=1e-12)
            changes[split] = {'compared_matches': len(overlap), 'changed_matches': int(delta.any(axis=1).sum()),
                             'changed_columns': {c: int(n) for c, n in zip(design(current).columns,
                                                                       delta.sum(axis=0)) if n}}
        summary['feature_changes_vs_v2'] = changes
    # Labels and partitions must stay unchanged for every retained match.
    common = matches.merge(old, on='match_id', suffixes=('_v3', '_v2'), validate='one_to_one')
    for col in ('date', 'team1_won', 'split'):
        if not common[col+'_v3'].astype(str).eq(common[col+'_v2'].astype(str)).all():
            raise ValueError(f'Unexpected change in {col}')
    if any(csv_checksum(p) != checksum for p, checksum in before.items()):
        raise RuntimeError('Protected input/release changed during run')
    (out/'manifest.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False)+'\n', encoding='utf8')
    verify()
    print(json.dumps({k: v for k, v in summary.items() if k not in ('files', 'reason_counts')}, indent=2))


def verify():
    """Check candidate artifacts and feature schema without loading any model weights."""
    from scripts.match_splits_and_metrics import split_matches
    out = ROOT/'data/processed/clean_v3'
    manifest = json.loads((out/'manifest.json').read_text(encoding='utf8'))
    if manifest['version'] != 'v9-clean-v3':
        raise ValueError('Unexpected candidate version')
    verify_snapshot()
    if manifest['source_manifest_sha256'] != csv_checksum(ROOT/'data/v9_manifest.json'):
        raise ValueError('Source manifest changed')
    for name, checksum in manifest['files'].items():
        if csv_checksum(out/name) != checksum:
            raise ValueError(f'Candidate fingerprint changed: {name}')
    matches = pd.read_csv(out/'matches.csv')
    original_split = matches.set_index('match_id').split
    assigned = split_matches(matches).set_index('match_id').split
    if not original_split.sort_index().equals(assigned.sort_index()):
        raise ValueError('Candidate partition mismatch')
    maps = pd.read_csv(out/'maps.csv')
    if maps.duplicated(['match_id', 'map_name']).any() or not maps.match_id.isin(matches.match_id).all():
        raise ValueError('Invalid candidate map identity')
    players = pd.read_csv(out/'player_stats.csv')
    joined = players.merge(maps[['match_id', 'map_name', 'score1', 'score2']],
                           on=['match_id', 'map_name'], validate='many_to_one')
    if (len(joined) != len(players) or players.duplicated(['match_id', 'map_name', 'player_id']).any()
            or not players.player_id.map(lambda x: integer(x, True)).all()
            or not np.isfinite(players[['adr', 'kast']]).all().all()
            or not players.adr.ge(0).all() or not players.kast.between(0, 100).all()
            or joined.deaths.gt(joined.score1 + joined.score2).any()):
        raise ValueError('Invalid candidate player statistics')
    for split in ('train', 'val'):
        if f'm2/{split}.csv' not in manifest['files']:
            continue
        from scripts.train_m1_m2 import design
        rows = pd.read_csv(out/f'm2/{split}.csv')
        expected = set(matches.loc[matches.split.eq(split), 'match_id'])
        if (rows.match_id.duplicated().any() or set(rows.match_id) != expected
                or not rows.split.eq(split).all() or rows.is_swapped.any()
                or not rows.bo_missing.eq(1).all()):
            raise ValueError(f'Invalid candidate features: {split}')
        design(rows)  # finite 41-column contract
    print('clean_v3 fingerprints, identities, statistics and feature schema verified.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tables-only', action='store_true')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        verify()
    else:
        main(args.tables_only)
