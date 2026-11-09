"""Build clean match-level and map-level tables from the raw Kaggle CS2 dataset.

Outputs (data/processed/):
  matches.csv  one row per match   (date, teams, tournament, bo, score, winner, tier, lineups)
  maps.csv     one row per map     (match fields + map_name, map score, map winner)

Design decisions, all forced by what the raw data actually contains:
  * `team1_win` in the raw file is a *row-level* flag and is unreliable on
    `is_total` rows (a 2-1 win can carry team1_win=0), so the match winner is
    derived from score1_match / score2_match.
  * ~46% of matches only exist as an aggregate row (no per-map detail), so the
    match table keeps them and `n_maps` / `maps_complete` record what is missing.
  * Impossible results (draws in BO1/BO3, missing scores) are dropped and counted.
"""

from pathlib import Path

import pandas as pd

RAW = Path("data/raw")
OUT = Path("data/processed")
MAIN = RAW / "cs2_all_tiers_games.csv"
TIERS = ["cs2_tier1_games.csv", "cs2_tier2_games.csv", "cs2_tier3_games.csv"]


def load_tier_map() -> pd.Series:
    """match_id -> tier label (the three tier files partition the match set)."""
    parts = []
    for name in TIERS:
        p = RAW / name
        if p.exists():
            ids = pd.read_csv(p, usecols=["match_id"])["match_id"].unique()
            parts.append(pd.Series(name.replace("cs2_", "").replace("_games.csv", ""), index=ids))
    return pd.concat(parts) if parts else pd.Series(dtype="object")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(MAIN, low_memory=False)
    df["date"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)

    lineup_cols = [f"team{t}_player{i}_id" for t in (1, 2) for i in range(1, 6)]

    # ---- match level -------------------------------------------------------
    agg = df[df["is_total"].astype(bool)].drop_duplicates("match_id")
    matches = agg[
        ["match_id", "date", "team1", "team2", "team1_id", "team2_id",
         "tournament", "bestOf", "score1_match", "score2_match", *lineup_cols]
    ].rename(columns={"bestOf": "bo", "score1_match": "score1", "score2_match": "score2"})
    matches = matches.reset_index(drop=True)

    matches["team1_won"] = (matches["score1"] > matches["score2"]).astype("Int64")
    matches.loc[matches["score1"].isna() | matches["score2"].isna(), "team1_won"] = pd.NA
    matches["_draw"] = matches["score1"].eq(matches["score2"])

    # ---- map level ---------------------------------------------------------
    maps = df[~df["is_total"].astype(bool)]
    maps = maps.sort_values(["match_id", "game_id"]).drop_duplicates(["match_id", "map_name"])
    maps = maps.rename(columns={"score1_game": "score1", "score2_game": "score2"})
    maps["team1_won_map"] = (maps["score1"] > maps["score2"]).astype("Int64")
    maps = maps[["match_id", "date", "team1", "team2", "tournament", "bestOf",
                 "map_name", "score1", "score2", "team1_won_map"]].rename(columns={"bestOf": "bo"})
    maps = maps.dropna(subset=["map_name", "score1", "score2"])

    # ---- attach derived info ----------------------------------------------
    n_maps = maps.groupby("match_id").size().rename("n_maps")
    matches = matches.merge(n_maps, on="match_id", how="left")
    matches["n_maps"] = matches["n_maps"].fillna(0).astype(int)
    matches["maps_complete"] = matches["n_maps"].eq(matches["bo"].astype("Int64"))
    matches["tier"] = matches["match_id"].map(load_tier_map())

    before = len(matches)
    matches = matches[~matches["_draw"] & matches["team1_won"].notna()].drop(columns="_draw").reset_index(drop=True)
    print(f"matches kept {len(matches):,} / {before:,} (dropped {before - len(matches)} draws/NaN)")
    print(f"map rows     {len(maps):,}")
    print(f"date range   {matches['date'].min()} -> {matches['date'].max()}")
    print(f"team1 win rate {matches['team1_won'].mean():.3f}")
    print(f"tier split   {matches['tier'].value_counts(dropna=False).to_dict()}")
    print(f"with map data {matches['n_maps'].gt(0).mean():.1%}")

    matches.to_csv(OUT / "matches.csv", index=False)
    maps.to_csv(OUT / "maps.csv", index=False)
    print(f"wrote {OUT / 'matches.csv'} and {OUT / 'maps.csv'}")

    print("\nhead of matches:")
    print(matches.head(6)[["date", "team1", "team2", "score1", "score2", "bo", "n_maps", "tier"]]
          .to_string(index=False))
    print("\nhead of maps:")
    print(maps.head(6).to_string(index=False))


if __name__ == "__main__":
    main()
