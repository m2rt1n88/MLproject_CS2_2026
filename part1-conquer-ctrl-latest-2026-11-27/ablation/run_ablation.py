"""固定逻辑回归，每次删除一组特征。只在训练期做五折向前评估。"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# 允许直接运行本文件；只导入共用的数据检查，不调用三个模型的留出评估。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.data_utils import DATA_FILE, TEST_START, load_data, make_features

OUT = Path(__file__).resolve().parent / "results"
N_FOLDS = 5
SEED = 2026
LABELS = {
    "full": "全部特征",
    "rating": "Rating 综合评分",
    "map_win_rate": "队伍同图胜率",
    "combat": "交战表现",
    "support": "支援表现",
    "experience": "样本经验",
}
PREFIXES = {
    "rating": ("Rating_2.0_",),
    "combat": ("Total_kills_", "Total_deaths_", "K/D_Ratio_", "Kills_/_round_",
               "Deaths_/_round_", "Damage_/_Round_", "Headshot_%_"),
    "support": ("Assists_/_round_", "Grenade_dmg_/_Round_",
                "Saved_by_teammate_/_round_", "Saved_teammates_/_round_"),
    "experience": ("Maps_played_", "Rounds_played_"),
}


def feature_groups(columns):
    """把 71 个差值特征分成五组，检查没有遗漏或重复。"""
    groups = {name: [] for name in ("rating", "map_win_rate", "combat", "support", "experience")}
    for col in columns:
        owners = [name for name, prefixes in PREFIXES.items() if col.startswith(prefixes)]
        if col == "mapwinrate":
            owners.append("map_win_rate")
        if len(owners) != 1:
            raise ValueError(f"特征没有明确分组：{col}")
        groups[owners[0]].append(col)
    if any(not cols for cols in groups.values()):
        raise ValueError("出现空的特征组，需要检查数据列名。")
    return groups


def forward_folds(rows):
    """按比赛日分为六段，第一段只用于起始训练，后五段轮流留出。"""
    days = np.sort(rows.day.unique())
    if len(days) < N_FOLDS + 1:
        raise ValueError("比赛日不足，无法进行五折向前验证。")
    for block in np.array_split(days, N_FOLDS + 1)[1:]:
        train = np.flatnonzero(rows.day.lt(block[0]))
        held = np.flatnonzero(rows.day.isin(block))
        if not len(train) or not len(held):
            raise ValueError("时间折为空。")
        if set(rows.matchID.iloc[train]) & set(rows.matchID.iloc[held]):
            raise ValueError("同一系列赛跨越了训练和留出。")
        yield train, held


def fit_predict(x_train, y_train, x_held):
    # 分类器和参数固定；只改变输入列，不根据消融成绩调参。
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.01, fit_intercept=False, max_iter=1500),
    )
    # 先分折，再只对训练侧交换队伍、翻转标签。
    model.fit(np.vstack([x_train, -x_train]), np.r_[y_train, 1 - y_train])
    return (model.predict_proba(x_held)[:, 1] + 1 - model.predict_proba(-x_held)[:, 1]) / 2


def scores(y, p):
    return dict(accuracy=accuracy_score(y, p > 0.5), log_loss=log_loss(y, p, labels=[0, 1]),
                brier=brier_score_loss(y, p))


def loss_per_row(y, p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def paired_day_interval(y, p, baseline, days):
    """同一天的比赛一起抽样，估计相对基线的损失差区间。"""
    delta = loss_per_row(y, p) - loss_per_row(y, baseline)
    grouped = pd.DataFrame({"day": days, "delta": delta}).groupby("day").delta.agg(["sum", "size"])
    rng = np.random.default_rng(SEED)
    index = rng.integers(0, len(grouped), size=(2000, len(grouped)))
    means = grouped["sum"].to_numpy()[index].sum(axis=1) / grouped["size"].to_numpy()[index].sum(axis=1)
    return np.quantile(means, [0.025, 0.975])


def main():
    # 1. 只取 2024-01-16 之前的训练期，不使用 220 张最终留出地图评分。
    rows = load_data()
    rows = rows.loc[rows.day < TEST_START].reset_index(drop=True)
    x = make_features(rows)
    y = rows.team1_win.to_numpy()
    groups = feature_groups(x.columns)
    folds = list(forward_folds(rows))
    configurations = {"full": list(x.columns)}
    for name, removed in groups.items():
        configurations[name] = [col for col in x if col not in removed]

    # 2. 所有配置使用完全相同的时间折；删除列后重新训练，不是将列填零。
    fold_scores, predictions, boundaries = [], {}, []
    for fold, (train, held) in enumerate(folds, 1):
        boundaries.append(dict(fold=fold, train_maps=len(train), held_maps=len(held),
                               train_last_day=str(rows.day.iloc[train].max().date()),
                               held_first_day=str(rows.day.iloc[held].min().date()),
                               held_last_day=str(rows.day.iloc[held].max().date())))
    for name, columns in configurations.items():
        p = np.full(len(rows), np.nan)
        for fold, (train, held) in enumerate(folds, 1):
            values = x[columns].to_numpy()
            p[held] = fit_predict(values[train], y[train], values[held])
            fold_scores.append(dict(config=name, fold=fold, n_features=len(columns), n_held=len(held),
                                    **scores(y[held], p[held])))
        predictions[name] = p
        print(f"Completed {name}: {len(columns)} features", flush=True)

    # 3. 比较同一批训练期留出预测。正的 Δ loss 表示删掉后更差，该组有贡献。
    mask = np.isfinite(predictions["full"])
    held_rows = rows.loc[mask, ["day", "matchID", "team1", "team2", "map", "team1_win"]].copy()
    baseline = predictions["full"][mask]
    baseline_loss = scores(y[mask], baseline)["log_loss"]
    results = []
    for name, p in predictions.items():
        if not np.array_equal(np.isfinite(p), mask):
            raise ValueError("不同配置的预测覆盖不一致。")
        lo, hi = paired_day_interval(y[mask], p[mask], baseline, held_rows.day.to_numpy())
        result = dict(config=name, description=LABELS[name], n_features=len(configurations[name]),
                      n_oof=int(mask.sum()), **scores(y[mask], p[mask]),
                      delta_log_loss=scores(y[mask], p[mask])["log_loss"] - baseline_loss,
                      delta_ci_low=lo, delta_ci_high=hi)
        results.append(result)
        held_rows[name + "_p_team1"] = p[mask]

    # 4. 单独保存到 ablation/results，不覆盖 models 中的任何结果。
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(results)
    metrics.to_csv(OUT / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_scores).to_csv(OUT / "fold_metrics.csv", index=False)
    held_rows.to_csv(OUT / "predictions.csv", index=False)
    pd.DataFrame([dict(group=name, description=LABELS[name], feature=col)
                  for name, cols in groups.items() for col in cols]).to_csv(
                      OUT / "feature_groups.csv", index=False, encoding="utf-8-sig")
    protocol = dict(model="LogisticRegression(C=0.01, fit_intercept=False)", training_maps=len(rows),
                    final_holdout_evaluated=False, n_oof=int(mask.sum()), seed=SEED,
                    bootstrap="2000 paired resamples of entire days", folds=boundaries,
                    data_sha256=hashlib.sha256(DATA_FILE.read_bytes()).hexdigest())
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf8")
    lines = ["# 特征消融结果", "", "固定逻辑回归，删除一组列后重新训练。以下为训练期五折向前留出结果，不是最终测试成绩。", ""]
    for r in results:
        label = "全部特征基线" if r["config"] == "full" else "去掉「" + r["description"] + "」"
        lines.append(f"- {label}：{r['n_features']} 列，准确率 {r['accuracy']:.2%}，log loss {r['log_loss']:.4f}，Δ loss {r['delta_log_loss']:+.4f}，95% 区间 [{r['delta_ci_low']:+.4f}, {r['delta_ci_high']:+.4f}]。")
    lines += ["", "Δ loss 为正表示删除后更差；为负表示删除后更好。区间跨零时证据不明确。", "",
              "这些是条件贡献，不是因果作用。相关特征可互相替代。区间按日重采样，但没有校正多重比较，也不覆盖跨日依赖。",
              "第一段日期只训练、不评分，所以 OOF 场次数少于训练总量。不能与 models/results 的 220 张地图成绩直接比较。",
              "统计快照时间仍未核验，不能宣称已排除原数据的未来信息泄漏。"]
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    print(metrics[["config", "n_features", "accuracy", "log_loss", "delta_log_loss"]].to_string(index=False))
    print(f"Results saved to: {OUT}")


if __name__ == "__main__":
    main()
