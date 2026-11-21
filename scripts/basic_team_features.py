"""根据更早比赛日的记录，为双方生成 Elo、近期战绩和休息天数等基础特征。"""

from bisect import bisect_left
from collections import defaultdict, deque

import numpy as np
import pandas as pd

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
    # 各张地图分别维护 Elo；赛前只汇总历史地图，不看本场选图。
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

        # 同一天的比赛先全部生成特征，再用当天赛果更新历史。
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
