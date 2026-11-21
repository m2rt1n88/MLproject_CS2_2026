"""Generate the common pre-match P0 features and train-only mirrored rows.

History is updated after each UTC day, since match end times are unavailable.
Earlier validation/test outcomes become history only on subsequent days;
no fitted model parameters are updated. No test metrics are computed here.
"""

from bisect import bisect_left
from collections import defaultdict, deque
import json

import numpy as np
import pandas as pd

try:
    from .training_protocol import ROOT, augment_training, load_matches
except ImportError:
    from training_protocol import ROOT, augment_training, load_matches

TEAM_FEATURES = [
    "elo", "prior_matches", "win_rate10", "opponent_weighted_form10",
    "streak", "rest_days", "matches7d", "map_elo",
]
TEAM_PAIRS = [("team1", "team2")] + [
    (f"team1_{name}", f"team2_{name}") for name in TEAM_FEATURES
]
DIFFERENCES = [f"diff_{name}" for name in TEAM_FEATURES]
CONTEXT_FEATURES = ["bo1", "bo3", "bo5", "bo_missing", "tier1", "tier2", "tier3"]
FEATURE_COLUMNS = [column for pair in TEAM_PAIRS[1:] for column in pair] + DIFFERENCES + CONTEXT_FEATURES
ELO_K = 16.0
ELO_HALF_LIFE_DAYS = 365.0


def normalize_team(value: str) -> str:
    return " ".join(value.split()).casefold()


def build_features(matches: pd.DataFrame, maps: pd.DataFrame | None = None) -> pd.DataFrame:
    """Allowlist pre-match inputs; never copy scores or map completion metadata."""
    ratings = defaultdict(lambda: 1500.0)
    history = defaultdict(list)
    recent = defaultdict(lambda: deque(maxlen=10))
    streaks = defaultdict(int)
    rows = []
    # Each map has a separate rating. Pre-match aggregation never uses the current veto.
    map_ratings = defaultdict(dict)
    maps_by_match = defaultdict(list)
    if maps is not None:
        for row in maps.itertuples(index=False):
            maps_by_match[row.match_id].append((row.map_name, row.team1_won_map))

    def decayed_map_rating(team, name, day):
        previous, last_day = map_ratings[team].get(name, (1500.0, day))
        return 1500 + (previous - 1500) * 0.5 ** ((day - last_day).days / ELO_HALF_LIFE_DAYS)

    def team_values(team: str, day: pd.Timestamp) -> dict:
        games = history[team]
        rest = (day - games[-1]).days if games else np.nan
        rating = 1500.0 + (ratings[team] - 1500.0) * (
            0.5 ** (rest / ELO_HALF_LIFE_DAYS) if games else 1.0
        )
        last10 = list(recent[team])
        return {
            "elo": rating,
            "map_elo": np.mean([decayed_map_rating(team, name, day) for name in map_ratings[team]]) if map_ratings[team] else 1500.0,
            "prior_matches": len(games),
            "win_rate10": np.mean([won for won, _ in last10]) if last10 else 0.5,
            "opponent_weighted_form10": np.mean([won - expected for won, expected in last10]) if last10 else 0.0,
            "streak": streaks[team],
            "rest_days": rest,
            "matches7d": len(games) - bisect_left(games, day - pd.Timedelta(days=7)),
        }

    ordered = matches.sort_values(["date", "match_id"])
    for day, daily in ordered.groupby(ordered["date"].dt.normalize(), sort=True):
        pending = []
        map_deltas = defaultdict(float)
        for match in daily.itertuples(index=False):
            a, b = normalize_team(match.team1), normalize_team(match.team2)
            if a == b:
                raise ValueError(f"Same normalized team in match {match.match_id}")
            left, right = team_values(a, day), team_values(b, day)
            expected = 1 / (1 + 10 ** ((right["elo"] - left["elo"]) / 400))
            row = {
                "match_id": match.match_id, "date": match.date, "split": match.split,
                "team1": a, "team2": b, "team1_won": int(match.team1_won), "is_swapped": False,
            }
            for name in TEAM_FEATURES:
                row[f"team1_{name}"] = left[name]
                row[f"team2_{name}"] = right[name]
                row[f"diff_{name}"] = left[name] - right[name]
            for bo in (1, 3, 5):
                row[f"bo{bo}"] = int(match.bo == bo)
            row["bo_missing"] = int(match.bo not in (1, 3, 5))
            for tier in ("tier1", "tier2", "tier3"):
                row[tier] = int(match.tier == tier)
            rows.append(row)
            for name, map_won in maps_by_match.get(match.match_id, ()):
                ra, rb = decayed_map_rating(a, name, day), decayed_map_rating(b, name, day)
                change = ELO_K * (map_won - 1 / (1 + 10 ** ((rb - ra) / 400)))
                map_deltas[a, name] += change
                map_deltas[b, name] -= change
            pending.append((a, b, int(match.team1_won), expected, left["elo"], right["elo"]))

        # All same-day predictions precede every same-day result update.
        for (team, name), delta in map_deltas.items():
            map_ratings[team][name] = (decayed_map_rating(team, name, day) + delta, day)
        daily_base = {}
        daily_delta = defaultdict(float)
        for a, b, won, expected, ra, rb in pending:
            for team, outcome, exp, rating in ((a, won, expected, ra), (b, 1 - won, 1 - expected, rb)):
                daily_base[team] = rating
                daily_delta[team] += ELO_K * (outcome - exp)
                history[team].append(day)
                recent[team].append((outcome, exp))
                sign = 1 if outcome else -1
                streaks[team] = streaks[team] + sign if streaks[team] * sign > 0 else sign
        for team, base in daily_base.items():
            ratings[team] = base + daily_delta[team]

    return pd.DataFrame(rows)


def main() -> None:
    matches = load_matches()
    maps = pd.read_csv(ROOT / "data/processed/maps.csv", usecols=["match_id", "map_name", "team1_won_map"])
    features = build_features(matches, maps)
    output = ROOT / "data/processed/training"
    output.mkdir(parents=True, exist_ok=True)
    features[["match_id", "date", "split"]].to_csv(output / "split_manifest.csv", index=False)
    for split in ("train", "val", "test"):
        partition = features[features["split"].eq(split)]
        partition.to_csv(output / f"{split}.csv", index=False)
        print(f"{split}: {len(partition):,} original matches")
    train = features[features["split"].eq("train")]
    augmented = augment_training(train, TEAM_PAIRS, DIFFERENCES)
    augmented.to_csv(output / "train_augmented.csv", index=False)
    schema = {
        "dataset_version": "v9", "feature_version": "p0-v2",
        "feature_columns": FEATURE_COLUMNS,
        "team_pairs": TEAM_PAIRS, "difference_columns": DIFFERENCES,
        "history_policy": "strictly earlier UTC days; observed results roll forward across splits",
        "elo_k": ELO_K, "elo_half_life_days": ELO_HALF_LIFE_DAYS,
        "missing_policy": "rest_days is NaN for unseen teams; fit imputation on each training fold only",
        "cv_policy": "chronological_folds on original train.csv; augment each fold training partition only",
        "prediction_columns": ["match_id", "y_true", "p_team1", "model"],
    }
    (output / "feature_schema.json").write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    print(f"train_augmented: {len(augmented):,} rows; {len(FEATURE_COLUMNS)} features")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
