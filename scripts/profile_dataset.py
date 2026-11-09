"""Profile the Kaggle "Counter-Strike Pro Matches" dataset.

Run from the project root:  .venv\\Scripts\\python.exe scripts\\profile_dataset.py

Prints a readable report to stdout (tee it into reports/ if you want to keep it):
    .venv\\Scripts\\python.exe scripts\\profile_dataset.py > reports\\dataset_profile.txt
"""

from pathlib import Path

import pandas as pd

RAW = Path("data/raw")
MAIN = RAW / "cs2_all_tiers_games.csv"


def hr(title: str, n: int = 78) -> None:
    print("\n" + "=" * n)
    print(title)
    print("=" * n)


def main() -> None:
    if not MAIN.exists():
        raise SystemExit(f"missing {MAIN} - unzip the Kaggle archive into data/raw first")

    hr("0. files")
    for f in sorted(RAW.glob("*.csv")):
        head = pd.read_csv(f, nrows=1)
        print(f"{f.name:<28} {f.stat().st_size / 1e6:6.2f} MB  {len(head.columns):3d} cols")

    df = pd.read_csv(MAIN, low_memory=False)

    hr("1. shape / dtypes")
    print(f"rows={len(df):,}  cols={df.shape[1]}")
    print(df.dtypes.to_string())

    hr("2. is_total (mixed granularity)")
    print(df["is_total"].value_counts(dropna=False).to_string())

    hr("3. datetime coverage")
    dt = pd.to_datetime(df["datetime"], errors="coerce", utc=True).dt.tz_localize(None)
    df["_dt"] = dt
    print(f"min={dt.min()}  max={dt.max()}  unparsed={dt.isna().sum()}")
    quarter = dt.dt.year.astype(str) + "Q" + dt.dt.quarter.astype(str)
    print("\nrows per quarter:")
    print(quarter.value_counts().sort_index().to_string())

    hr("4. matches vs map-level rows")
    maps = df[df["is_total"].eq(False)]
    print(f"map-level rows          {len(maps):,}")
    print(f"unique match_id         {maps['match_id'].nunique():,}")
    print(f"unique (match, map)     {maps.drop_duplicates(['match_id', 'map_name']).shape[0]:,}")
    print("\nmaps per match:")
    print(maps.groupby("match_id").size().value_counts().sort_index().to_string())

    hr("5. team coverage")
    appearances = pd.concat([maps["team1"], maps["team2"]]).dropna()
    vc = appearances.value_counts()
    print(f"distinct team names     {vc.size:,}")
    print("\nmaps played per team (describe):")
    print(vc.describe().to_string())
    print(f"\nteams with <5 maps: {(vc < 5).sum():,}  ({(vc < 5).mean():.1%})")
    print(f"teams with <20 maps: {(vc < 20).sum():,}  ({(vc < 20).mean():.1%})")
    print("\ntop 15 by maps played:")
    print(vc.head(15).to_string())

    hr("6. missing values on key columns")
    key = [
        "datetime", "tournament", "bestOf", "map_name",
        "score1_game", "score2_game", "score1_match", "score2_match",
        "team1_player1_id", "team1_player1_adr", "team1_player1_kast",
    ]
    key = [c for c in key if c in df.columns]
    print(df[key].isna().mean().round(4).to_string())

    hr("7. match-level sanity")
    win_col = "team1_win" if "team1_win" in df.columns else "team1_win1"
    g = df.groupby("match_id").agg(
        n_maps=("map_name", "nunique"),
        s1=("score1_match", "max"),
        s2=("score2_match", "max"),
        win_values=(win_col, "nunique"),
        bo=("bestOf", "first"),
        t1=("team1", "first"),
        t2=("team2", "first"),
        date=("_dt", "first"),
    )
    print(f"match_id column used    {win_col}")
    print(f"matches                 {len(g):,}")
    print(f"inconsistent win flag   {int((g['win_values'] > 1).sum()):,}")
    print(f"missing map wins (s1/s2) {int(g[['s1', 's2']].isna().any(axis=1).sum()):,}")
    print(f"draws (s1 == s2)        {int((g['s1'] == g['s2']).sum()):,}")
    print("\nbestOf distribution:")
    print(g["bo"].value_counts(dropna=False).to_string())

    print("\nrecorded maps vs BO expectation (crosstab):")
    print(pd.crosstab(g["bo"], g["n_maps"]).to_string())

    hr("7b. sample BO3 as stored")
    pick = g[(g["bo"] == 3) & (g["n_maps"] == 3)]
    if len(pick):
        sample_id = pick.index[0]
        cols = [
            "match_id", "game_id", "is_total", "team1", "team2",
            "score1_match", "score2_match", "score1_game", "score2_game",
            "map_name", "bestOf", win_col, "datetime",
        ]
        cols = [c for c in cols if c in df.columns]
        print(df[df["match_id"] == sample_id][cols].to_string(index=False))

    hr("7c. monthly match count")
    print(g["date"].dt.to_period("M").value_counts().sort_index().to_string())

    hr("7d. tier files vs all")
    tier_ids = {}
    for name in ["cs2_all_tiers_games.csv", "cs2_tier1_games.csv",
                 "cs2_tier2_games.csv", "cs2_tier3_games.csv"]:
        p = RAW / name
        if p.exists():
            t = pd.read_csv(p, usecols=["match_id"])
            tier_ids[name] = set(t["match_id"])
            print(f"{name:<28} matches={len(tier_ids[name]):,}")
    union = set().union(*tier_ids.values()) if tier_ids else set()
    print(f"\nunion of tier files     {len(union):,} matches")
    print(f"all vs union diff       {len(tier_ids['cs2_all_tiers_games.csv'] ^ union):,}")


    hr("8. map / tournament distribution")
    print("maps:")
    print(maps["map_name"].value_counts(dropna=False).head(15).to_string())
    print("\ntop 12 tournaments:")
    print(maps["tournament"].value_counts().head(12).to_string())

    hr("9. player stat coverage (map rows)")
    pcols = [c for c in maps.columns if "_player" in c]
    stats = ["kills", "deaths", "assists", "adr", "kast", "kddiff"]
    present = [c for c in pcols if any(s in c for s in stats)]
    print(f"player columns: {len(pcols)} (stat columns: {len(present)})")
    if present:
        cov = maps[present].notna().mean()
        print(f"non-null rate: min={cov.min():.3f} median={cov.median():.3f} max={cov.max():.3f}")
        print(f"rows with all 60 stat fields filled: {maps[present].notna().all(axis=1).mean():.1%}")

    hr("10. score sanity")
    rng = pd.concat([
        maps["score1_game"].describe().rename("score1_game"),
        maps["score2_game"].describe().rename("score2_game"),
    ], axis=1)
    print(rng.round(2).to_string())
    print(f"\nnegative game scores: {(maps['score1_game'] < 0).sum()} / {(maps['score2_game'] < 0).sum()}")
    print(f"15-15 map draws: {((maps['score1_game'] == 15) & (maps['score2_game'] == 15)).sum():,}")

    hr("11. match table QA (what a modelling table would look like)")
    agg = df[df["is_total"].astype(bool)]
    print(f"is_total=True rows      {len(agg):,} (one per match, {agg['match_id'].nunique():,} unique)")
    lineup_cols = [c for c in df.columns if c.endswith(("player1_id", "player5_id"))]
    stat_cols = [c for c in df.columns if c.endswith(("_adr", "_kast"))]
    if lineup_cols:
        print(f"agg rows with lineup    {agg[lineup_cols].notna().all(axis=1).mean():.1%}")
    if stat_cols:
        print(f"agg rows with adr/kast  {agg[stat_cols].notna().all(axis=1).mean():.1%}")

    # team-name hygiene
    names = pd.Series(sorted(set(appearances.unique())))
    norm = names.str.strip().str.lower()
    collisions = norm[norm.duplicated(keep=False)]
    print(f"\nteam names              {len(names):,}")
    print(f"case/space collisions   {collisions.nunique():,}")
    if len(collisions):
        print("  e.g. " + ", ".join(sorted(set(collisions))[:6]))

    # same team pair twice on the same day (rematches / duplicates)
    pair_key = (
        df[["match_id", "team1", "team2", "_dt"]]
        .drop_duplicates("match_id")
        .assign(pair=lambda d: [
            " | ".join(sorted([str(a), str(b)])) for a, b in zip(d["team1"], d["team2"])
        ])
    )
    pair_key["day"] = pair_key["_dt"].dt.date
    dup_days = pair_key.groupby(["day", "pair"]).size()
    print(f"\nsame pairing on same day: {int((dup_days > 1).sum()):,} groups")
    print(f"matches with >=1 map    {df[df['is_total'].eq(False)]['match_id'].nunique():,} "
          f"({df[df['is_total'].eq(False)]['match_id'].nunique() / df['match_id'].nunique():.1%})")

    hr("12. odd rows")
    odd_date = df[df["_dt"] < "2023-06-01"][["match_id", "datetime", "team1", "team2", "tournament"]]
    print(f"rows before CS2 launch: {len(odd_date)}")
    if len(odd_date):
        print(odd_date.head(5).to_string(index=False))
    draw_ids = g[(g["s1"] == g["s2"]) & g["s1"].notna()].index
    print(f"\ndraw matches: {len(draw_ids)}")
    if len(draw_ids):
        print(g.loc[draw_ids, ["t1", "t2", "s1", "s2", "bo", "date"]].to_string())
    dups = maps[maps.duplicated(["match_id", "map_name"], keep=False)]
    print(f"\nduplicate (match, map) rows: {len(dups):,}")
    if len(dups):
        print(dups[["match_id", "game_id", "map_name", "score1_game", "score2_game"]].head(6).to_string(index=False))



if __name__ == "__main__":
    main()
