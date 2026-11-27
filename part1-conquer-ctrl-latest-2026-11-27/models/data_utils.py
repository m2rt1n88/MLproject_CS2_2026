"""三个模型共用的数据处理和评估。先看各模型文件，再看这里的五步流程。"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data/maps.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "results"
TEST_START = pd.Timestamp("2024-01-16")


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


def run_model(name, model):
    # 1. 读数据，构造双方差值。日期、ID、队名和标签不作为模型输入。
    data = load_data()
    x = make_features(data)
    y = data.team1_win.to_numpy()

    # 2. 按日期切分。保持较早的 790 张图训练、较晚的 220 张图评估。
    is_train = data.day < TEST_START
    x_train = x.loc[is_train].to_numpy()
    x_test = x.loc[~is_train].to_numpy()
    y_train, y_test = y[is_train], y[~is_train]
    if len(y_train) == 0 or len(y_test) == 0:
        raise ValueError("训练集或留出集为空，请检查日期边界。")
    if set(data.loc[is_train, "matchID"]) & set(data.loc[~is_train, "matchID"]):
        raise ValueError("同一系列赛跨越了训练集和留出集。")

    # 3. 只扩增训练集：交换双方，差值取负，标签翻转。
    x_fit = np.vstack([x_train, -x_train])
    y_fit = np.concatenate([y_train, 1 - y_train])
    print(f"Training maps: {len(y_train)}; held-out maps: {len(y_test)}; features: {x.shape[1]}")

    # 4. 训练和预测。Pipeline 中的标准化只在训练数据上拟合。
    model.fit(x_fit, y_fit)
    p_forward = model.predict_proba(x_test)[:, 1]
    p_reverse = model.predict_proba(-x_test)[:, 1]
    p = (p_forward + 1 - p_reverse) / 2  # 让交换双方后的概率互补
    y_pred = (p > 0.5).astype(int)
    result = {
        "model": name, "n_test": len(y_test),
        "correct": int((y_pred == y_test).sum()),
        "accuracy": accuracy_score(y_test, y_pred),
        "log_loss": log_loss(y_test, p, labels=[0, 1]),
        "brier": brier_score_loss(y_test, p),
    }

    # 5. 每个模型分别保存自己的结果，不覆盖其他模型的结果。
    predictions = data.loc[~is_train, ["day", "matchID", "team1", "team2", "map", "team1_win"]].copy()
    predictions[name + "_p_team1"] = p
    out = OUTPUT_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_csv(out / "metrics.csv", index=False)
    predictions.to_csv(out / "predictions.csv", index=False)
    pd.Series(x.columns, name="feature").to_csv(out / "features.csv", index=False)
    print(f"{name}: accuracy={result['accuracy']:.2%}  log_loss={result['log_loss']:.4f}")
    print(f"Results saved to: {out}")
    print("Offline known-map evaluation; pre-match feature timestamps remain unverified.")
    return result, predictions

