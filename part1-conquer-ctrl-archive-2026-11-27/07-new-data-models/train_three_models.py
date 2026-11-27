"""新数据上的三个基础模型。直接运行本文件即可，不需要导入项目其他模块。

任务：地图已知时预测队1是否赢下这一张地图，不是预测整场系列赛。
数据统计的赛前快照时间尚未核验，本实验只能作为离线数据集评估。
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# 可以先从这里看懂：输入路径、输出路径、时间边界和随机种子。
ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "known_map_v2/data/maps.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "results"
TEST_START = pd.Timestamp("2024-01-16")
SEED = 2026


def load_data():
    """读入同学已经整理好的地图表，检查基本错误。"""
    data = pd.read_csv(DATA_FILE, parse_dates=["day"])
    required = ["day", "matchID", "team1", "team2", "map", "team1_win"]
    if data[required].isna().any().any():
        raise ValueError("日期、比赛信息或标签存在缺失值，请先检查数据。")
    if not data.team1_win.isin([0, 1]).all():
        raise ValueError("team1_win 只能是 0 或 1。")
    if data.duplicated(["matchID", "map"]).any():
        raise ValueError("同一场比赛的同一张地图出现了多次。")
    if data.groupby("matchID").day.nunique().gt(1).any():
        raise ValueError("同一系列赛出现不同日期，需要先核对。")
    return data.sort_values(["day", "matchID", "map"]).reset_index(drop=True)


def make_features(data):
    """每个输入都是双方差值，例如队1平均 Rating 减去队2平均 Rating。"""
    stat_names = sorted(col[3:] for col in data.columns if col.startswith("T1_"))
    other_names = sorted(col[3:] for col in data.columns if col.startswith("T2_"))
    if not stat_names or stat_names != other_names:
        raise ValueError("双方统计列不完整或不对应。")

    x = pd.DataFrame(index=data.index)
    for name in stat_names:
        x[name] = data["T1_" + name] - data["T2_" + name]
    if not np.isfinite(x.to_numpy()).all():
        raise ValueError("统计存在缺失或无穷值，请先检查，不能直接填成 0。")
    for side in (1, 2):
        if not data[f"T{side}_mapwinrate"].between(0, 1).all():
            raise ValueError("地图胜率应在 0 到 1 之间，例如 0.65，不是 65。")
    return x


def create_models():
    """三个模型都使用相同的输入；参数预先固定，不根据留出成绩调参。"""
    return {
        # 根据各类中的均值、方差计算概率；平滑可以减少极端概率。
        "naive_bayes": make_pipeline(
            StandardScaler(), GaussianNB(var_smoothing=1.0)
        ),
        # 对输入加权求和后转换为概率；C 越小，正则化越强。
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.01, fit_intercept=False, max_iter=1500),
        ),
        # 训练 128 棵树，平均它们的预测；限制叶子大小以减少过拟合。
        "random_forest": make_pipeline(
            StandardScaler(),
            RandomForestClassifier(
                n_estimators=128, min_samples_leaf=8,
                max_features="sqrt", random_state=SEED, n_jobs=1,
            ),
        ),
    }


def main():
    # 1. 准备特征 X 和标签 y。日期、比赛 ID、队名、赛果不放入 X。
    data = load_data()
    x = make_features(data)
    y = data["team1_win"].to_numpy()

    # 2. 按日期切分，较早的用于训练，较晚的用于评估，不随机打乱。
    is_train = data.day < TEST_START
    x_train = x.loc[is_train].to_numpy()
    x_test = x.loc[~is_train].to_numpy()
    y_train = y[is_train]
    y_test = y[~is_train]
    if len(y_train) == 0 or len(y_test) == 0:
        raise ValueError("训练集或留出集为空，请检查日期边界。")
    train_ids = set(data.loc[is_train, "matchID"])
    test_ids = set(data.loc[~is_train, "matchID"])
    if train_ids & test_ids:
        raise ValueError("同一系列赛跨越了训练集和留出集。")

    # 3. 仅扩增训练集：两队交换后，差值取负，胜负标签翻转。
    x_fit = np.vstack([x_train, -x_train])
    y_fit = np.concatenate([y_train, 1 - y_train])
    print(f"Training maps: {len(y_train)}; held-out maps: {len(y_test)}")
    print(f"Features: {x.shape[1]}; training rows after mirroring: {len(y_fit)}")

    # 4. 依次训练三个模型。Pipeline 保证标准化只在训练数据上拟合。
    predictions = data.loc[~is_train, ["day", "matchID", "team1", "team2", "map", "team1_win"]].copy()
    results = []
    for name, model in create_models().items():
        model.fit(x_fit, y_fit)

        # 同时预测原方向和交换方向，取平均，让换队后的胜率互补。
        p_forward = model.predict_proba(x_test)[:, 1]
        p_reverse = model.predict_proba(-x_test)[:, 1]
        p = (p_forward + 1 - p_reverse) / 2
        y_pred = (p > 0.5).astype(int)
        accuracy = accuracy_score(y_test, y_pred)
        loss = log_loss(y_test, p, labels=[0, 1])
        results.append({
            "model": name, "n_test": len(y_test),
            "correct": int((y_pred == y_test).sum()),
            "accuracy": accuracy, "log_loss": loss,
            "brier": brier_score_loss(y_test, p),
        })
        predictions[name + "_p_team1"] = p
        print(f"{name:22s} accuracy={accuracy:.2%}  log_loss={loss:.4f}")

    # 5. 将汇总指标、逐图概率、特征名存到本目录，不覆盖同学的报告。
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "predictions.csv", index=False)
    pd.Series(x.columns, name="feature").to_csv(OUTPUT_DIR / "features.csv", index=False)
    print(f"Results saved to: {OUTPUT_DIR}")
    print("Offline known-map evaluation; pre-match feature timestamps remain unverified.")


if __name__ == "__main__":
    main()
