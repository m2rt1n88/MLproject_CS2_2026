"""Build the offline Singapore Major dashboard data using the released M2 model."""
import hashlib
from itertools import combinations
import json

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import tzdata  # noqa: F401  (timezone data for Windows)

from scripts.basic_team_features import build_features, normalize_team
from scripts.build_clean_match_dataset import clean, events_for_history
from scripts.match_splits_and_metrics import ROOT, verify_snapshot
from scripts.predict_matchup import load_tiers
from scripts.roster_and_map_features import past_features
from scripts.team_and_opponent_strength import build as build_strength
from scripts.train_m1_m2 import predict

DASH = ROOT / 'dashboard'
MODEL = ROOT / 'models/m2/m2_clean_retuned.joblib'


def build_dashboard_data(event, matches, maps, events, bundle):
    """Evaluate every pairing at the same opening day, without Major results."""
    day = pd.Timestamp(event['start'])
    if matches.date.max() >= day:
        raise ValueError('History overlaps the Major: refusing to use future results')
    roster = event['roster']
    names = [normalize_team(t['history']) for t in roster]
    ids = [t['id'] for t in roster]
    if len(ids) != 32 or len(set(ids)) != len(ids) or len(set(names)) != len(names):
        raise ValueError('Expected 32 distinct teams and history aliases')
    fixtures = event['fixtures']
    if len({f['id'] for f in fixtures}) != len(fixtures) or any(
        f['a'] not in ids or f['b'] not in ids or f['a'] == f['b'] for f in fixtures
    ):
        raise ValueError('Invalid fixtures')

    pairs = list(combinations(roster, 2))
    targets = pd.DataFrame([
        dict(match_id=int(matches.match_id.max()) + i + 1, date=day,
             team1=a['history'], team2=b['history'], split='future', tier='tier1',
             bo=np.nan, team1_won=1, score1=1, score2=0)
        for i, (a, b) in enumerate(pairs)
    ])
    # The three builders emit the day's features before updating history.
    # Placeholders are processed only after all target features have been emitted;
    # no later rows use their outcomes. BO1/BO3/BO5 remain disabled as in training.
    all_rows = pd.concat([matches, targets], ignore_index=True)
    target_ids = targets.match_id.to_list()
    with threadpool_limits(limits=1):
        blocks = [build_features(all_rows, maps), past_features(all_rows, maps, events),
                  build_strength(all_rows, maps)]
        base, roster_features, strength = [b.set_index('match_id').loc[target_ids] for b in blocks]
        features = base.join(roster_features, validate='one_to_one').join(strength, validate='one_to_one')
        probabilities = predict(bundle, features)
    if not (np.isfinite(probabilities).all() and ((0 <= probabilities) & (probabilities <= 1)).all()):
        raise ValueError('Invalid M2 probabilities')

    outcomes = pd.concat([matches[['team1', 'date']].rename(columns={'team1': 'team'}),
                          matches[['team2', 'date']].rename(columns={'team2': 'team'})])
    summary = outcomes.groupby(outcomes.team.map(normalize_team)).date.agg(['size', 'max'])
    team_history = {}
    for team, name in zip(roster, names):
        key = team['id']
        team_history[key] = {'matches': int(summary.at[name, 'size']),
                             'lastMatch': str(summary.at[name, 'max'].date())} if name in summary.index else {
                             'matches': 0, 'lastMatch': None}
    predictions = {}
    for i, (a, b) in enumerate(pairs):
        left, right = a['id'], b['id']
        covered = team_history[left]['matches'] and team_history[right]['matches']
        predictions[left + '|' + right] = round(float(probabilities[i]), 8) if covered else None
    return {**event, 'historyThrough': str(matches.date.max().date()), 'predictionDay': str(day.date()),
            'model': 'M2', 'modelSha256': hashlib.sha256(MODEL.read_bytes()).hexdigest(),
            'teamHistory': team_history, 'predictions': predictions}


def main():
    event = json.loads((DASH / 'event.json').read_text(encoding='utf-8'))
    verify_snapshot()
    raw = pd.read_csv(ROOT / 'data/raw/cs2_all_tiers_games.csv', low_memory=False)
    matches, maps, raw_maps, _ = clean(raw, load_tiers())
    bundle = joblib.load(MODEL)
    if bundle['stage'] != 'm2' or bundle['dataset'] != 'v9-clean-v2':
        raise ValueError('Expected the frozen M2 release')
    data = build_dashboard_data(event, matches, maps, events_for_history(raw_maps, matches), bundle)
    output = json.dumps(data, ensure_ascii=True, separators=(',', ':'), allow_nan=False)
    (DASH / 'predictions.js').write_text('window.SINGAPORE_DATA = ' + output + ';\n', encoding='utf-8')
    count = sum(p is not None for p in data['predictions'].values())
    print(f'Generated {count} probabilities from {len(data["roster"])} teams and {len(data["fixtures"])} published fixtures.')


if __name__ == '__main__':
    main()
