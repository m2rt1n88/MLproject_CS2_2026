"""Quick feasibility check: how far does a plain Elo get on this dataset?

No features, no model - just a chronological Elo rating updated after every
match, evaluated on the last ~7 months. This is the floor any real model has
to beat, together with the trivial "always pick team1" baseline.

Run:  .venv\\Scripts\\python.exe scripts/elo_baseline.py
"""

import numpy as np
import pandas as pd

TEST_START = pd.Timestamp("2026-04-01")


def load() -> pd.DataFrame:
    m = pd.read_csv("data/processed/matches.csv", parse_dates=["date"])
    m = m.sort_values("date").reset_index(drop=True)
    # team names are the only stable key in this dataset (ids drift on rebrand)
    m["t1"] = m["team1"].str.strip()
    m["t2"] = m["team2"].str.strip()

    seen: dict[str, int] = {}
    prior = []
    for a, b in zip(m["t1"], m["t2"]):
        prior.append((seen.get(a, 0), seen.get(b, 0)))
        seen[a] = seen.get(a, 0) + 1
        seen[b] = seen.get(b, 0) + 1
    m[["p1", "p2"]] = pd.DataFrame(prior, index=m.index)
    return m


def elo_run(train: pd.DataFrame, test: pd.DataFrame, k: float = 32.0) -> dict:
    elo: dict[str, float] = {}

    def get(team: str) -> float:
        return elo.setdefault(team, 1500.0)

    for a, b, won in zip(train["t1"], train["t2"], train["team1_won"]):
        ra, rb = get(a), get(b)
        exp_a = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
        elo[a] = ra + k * (won - exp_a)
        elo[b] = rb + k * ((1 - won) - (1 - exp_a))

    p = np.array([1.0 / (1.0 + 10 ** ((get(b) - get(a)) / 400.0))
                  for a, b in zip(test["t1"], test["t2"])])
    y = test["team1_won"].to_numpy()
    return {
        "p": p,
        "y": y,
        "acc": float(((p > 0.5) == y).mean()),
        "logloss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "brier": float(np.mean((p - y) ** 2)),
    }


def main() -> None:
    m = load()
    train, test = m[m["date"] < TEST_START], m[m["date"] >= TEST_START]
    print(f"train {len(train):,}  ({train['date'].min().date()} -> {train['date'].max().date()})")
    print(f"test  {len(test):,}  ({test['date'].min().date()} -> {test['date'].max().date()})")
    print(f"test 'always team1' accuracy: {test['team1_won'].mean():.3f}")
    print(f"matches where both teams have >=10 prior matches: "
          f"{((m['p1'] >= 10) & (m['p2'] >= 10)).mean():.1%}")

    print("\nK sweep (test set):")
    best = None
    for k in (16, 24, 32, 48, 64):
        r = elo_run(train, test, k)
        print(f"  K={k:<3} acc={r['acc']:.3f}  logloss={r['logloss']:.4f}  brier={r['brier']:.4f}")
        if best is None or r["logloss"] < best[1]["logloss"]:
            best = (k, r)

    k, r = best
    print(f"\nbest K={k}")
    p, y = r["p"], r["y"]

    print("\nby tier:")
    for tier, idx in test.groupby("tier").groups.items():
        tt = test.loc[idx]
        pp = p[test.index.get_indexer(idx)]
        yy = y[test.index.get_indexer(idx)]
        print(f"  {tier:<6} n={len(tt):<5} acc={((pp > .5) == yy).mean():.3f} "
              f"logloss={-np.mean(yy * np.log(pp) + (1 - yy) * np.log(1 - pp)):.4f}")

    print("\nby BO:")
    for bo, idx in test.groupby("bo").groups.items():
        tt = test.loc[idx]
        pp = p[test.index.get_indexer(idx)]
        yy = y[test.index.get_indexer(idx)]
        print(f"  BO{int(bo) if bo == bo else '?'} n={len(tt):<5} acc={((pp > .5) == yy).mean():.3f}")

    print("\ncalibration (test):")
    bins = pd.cut(p, [0, .2, .4, .5, .6, .8, 1.0])
    cal = pd.DataFrame({"p": p, "y": y, "bin": bins}).groupby("bin", observed=True).agg(
        n=("y", "size"), pred=("p", "mean"), actual=("y", "mean"))
    print(cal.round(3).to_string())


if __name__ == "__main__":
    main()
