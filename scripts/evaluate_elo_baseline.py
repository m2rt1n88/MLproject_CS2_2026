"""评估独立 Elo 基线：在验证集选择 K，只有显式指定 --final-test 才评估测试集。"""

import argparse
import json
import numpy as np
import pandas as pd

try:
    from .match_splits_and_metrics import ROOT, load_matches, prediction_table, probability_metrics
    from .basic_team_features import normalize_team
except ImportError:
    from match_splits_and_metrics import ROOT, load_matches, prediction_table, probability_metrics
    from basic_team_features import normalize_team


def load() -> pd.DataFrame:
    m = load_matches()
    # 队伍更名可能导致 ID 变化，这里用统一格式的队名关联历史记录。
    m["t1"] = m["team1"].map(normalize_team)
    m["t2"] = m["team2"].map(normalize_team)

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

    # 进入评估期后固定双方评分，不再用评估期赛果更新 Elo。
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-test", action="store_true", help="Evaluate test after validation selection")
    args = parser.parse_args()
    m = load()
    train = m[m["split"].eq("train")]
    val = m[m["split"].eq("val")]
    print(f"train {len(train):,}; val {len(val):,}")
    print("K sweep (validation only):")
    best = None
    candidates = []
    for k in (16, 24, 32, 48, 64):
        r = elo_run(train, val, k)
        print(f"  K={k:<3} acc={r['acc']:.3f}  logloss={r['logloss']:.4f}  brier={r['brier']:.4f}")
        candidates.append({"k": k, **{key: r[key] for key in ("acc", "logloss", "brier")}})
        if best is None or r["logloss"] < best[1]["logloss"]:
            best = (k, r)

    k, r = best
    print(f"Selected K={k} by validation log loss")
    model_name = f"elo_frozen_k{k}"
    prediction_root = ROOT / "reports/predictions"
    val_output = prediction_root / "val"
    val_output.mkdir(parents=True, exist_ok=True)
    predictions = prediction_table(val, r["p"], model_name)
    predictions.to_csv(val_output / "elo.csv", index=False)
    report = {
        "dataset_version": "v9", "protocol": "frozen ratings within each evaluation partition",
        "selection_split": "val", "selected_k": k, "candidates": candidates,
        "validation": probability_metrics(predictions), "final_test_evaluated": args.final_test,
    }
    if args.final_test:
        test = m[m["split"].eq("test")]
        fitted = m[m["split"].isin(["train", "val"])]
        result = elo_run(fitted, test, k)
        final_predictions = prediction_table(test, result["p"], model_name)
        test_output = prediction_root / "test"
        test_output.mkdir(parents=True, exist_ok=True)
        final_predictions.to_csv(test_output / "elo.csv", index=False)
        report["test"] = probability_metrics(final_predictions)
        (ROOT / "reports/elo_final_test.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Final test: {report['test']}")
    else:
        print("Validation predictions exported; final test was not evaluated.")
    (ROOT / "reports/elo_validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
