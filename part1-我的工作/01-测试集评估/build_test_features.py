"""为最终测试期生成赛前特征，并用已冻结的 M1/M2 权重各预测一次。

设计原则：
  * 不改任何冻结产物（clean_v2 的 train/val、manifest、模型权重都不碰）。
  * 复用仓库里现有的特征函数，只是把输入从「train+val」换成「全部比赛」，
    这样 2026-04 之后的 Elo / 阵容 / 实力评分能沿着历史自然的延续下去，
    而不是每支队都从 1500 重新开始。
  * 生成后强制与已发布的 train/val 特征逐列比对；不一致就报错退出，
    避免在错误输入上产生一份看起来正常、实际全错的测试集预测。

用法（项目根目录）：
    python -B -m scripts.build_test_features

输出：
    reports/release/m1_test.csv
    reports/release/m2_test.csv
    reports/release/test_features.csv   （供 selective_report 取 tier）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.basic_team_features import build_features
from scripts.build_clean_match_dataset import SLOTS, clean, events_for_history, roster_flags  # noqa: F401
from scripts.match_splits_and_metrics import (PREDICTION_COLUMNS, ROOT, csv_checksum,
    prediction_table, probability_metrics, validate_predictions, verify_snapshot)
from scripts.roster_and_map_features import past_features
from scripts.team_and_opponent_strength import build as build_strength
from scripts.train_m1_m2 import MODELS, predict

RAW = ROOT / "data/raw"
CLEAN = ROOT / "data/processed/clean_v2"
REPORT = ROOT / "reports/release"


def load_tiers() -> pd.Series:
    parts = []
    for tier in ("tier1", "tier2", "tier3"):
        ids = pd.read_csv(RAW / f"cs2_{tier}_games.csv", usecols=["match_id"]).match_id.unique()
        parts.append(pd.Series(tier, index=ids))
    return pd.concat(parts, verify_integrity=True)


def check_against_release(features: pd.DataFrame) -> None:
    """本次重算的 train/val 特征必须与已发布版本完全一致。"""
    for split in ("train", "val"):
        released = pd.read_csv(CLEAN / "m2" / f"{split}.csv", parse_dates=["date"])
        rebuilt = features[features.split.eq(split)].reset_index(drop=True)
        if list(rebuilt.match_id) != list(released.match_id):
            raise SystemExit(f"{split}: 行顺序或比赛集合与已发布版本不一致")
        common = [c for c in released.columns if c in rebuilt.columns and c != "date"]
        bad = []
        for c in common:
            a, b = rebuilt[c], released[c]
            if a.dtype.kind in "fi" and b.dtype.kind in "fi":
                if not np.allclose(a.to_numpy(float), b.to_numpy(float), equal_nan=True, rtol=0, atol=1e-5):
                    bad.append(c)
            elif not a.astype(str).equals(b.astype(str)):
                bad.append(c)
        if bad:
            raise SystemExit(f"{split}: 重算特征与已发布版本不一致，列出前 10 列 {bad[:10]}")
        print(f"  {split}: {len(rebuilt)} 行 / {len(common)} 列与已发布特征完全一致", flush=True)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(REPORT), help="输出目录，默认 reports/release")
    out_dir = Path(ap.parse_args().out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    verify_snapshot()
    print("重建 clean_v2（内存中，不写盘）…", flush=True)
    m, maps, raw_maps, _ = clean(
        pd.read_csv(RAW / "cs2_all_tiers_games.csv", low_memory=False), load_tiers())
    print(f"  全部比赛 {len(m):,} 场；split = {m.split.value_counts().to_dict()}", flush=True)

    # 关键：喂全部比赛，测试期的历史才能接上
    print("生成赛前特征（覆盖全部比赛）…", flush=True)
    p0 = build_features(m, maps)
    p1 = past_features(m, maps, events_for_history(raw_maps, m))
    with threadpool_limits(limits=1):
        extra = build_strength(m, maps)
    features = (p0.merge(p1, on="match_id", validate="one_to_one")
                  .merge(extra, on="match_id", validate="one_to_one"))
    if len(features) != len(m):
        raise SystemExit("特征行数与比赛数不一致")

    print("与已发布的 train/val 特征比对：", flush=True)
    check_against_release(features)

    test = features[features.split.eq("test")].reset_index(drop=True)
    if len(test) != 2159:
        raise SystemExit(f"测试集应有 2159 场，实际 {len(test)}")
    print(f"测试期 {len(test):,} 场  {test.date.min().date()} → {test.date.max().date()}", flush=True)

    for stage, name in MODELS.items():
        bundle = joblib.load(ROOT / "models" / stage / (name + ".joblib"))
        if bundle.get("train_sha256") != csv_checksum(CLEAN / "m2/train.csv"):
            raise SystemExit(f"{stage}: 权重不是基于当前 clean_v2 训练的，拒绝预测")
        p = predict(bundle, test)
        table = prediction_table(test, p, name)
        if list(table.columns) != PREDICTION_COLUMNS:
            raise SystemExit("预测表列名不符合协议")
        validate_predictions(table, test)
        out = out_dir / f"{stage}_test.csv"
        table.to_csv(out, index=False)
        mt = probability_metrics(table)
        print(f"  {stage}  测试集 n={len(table)}  acc={mt['acc']:.4f}  "
              f"logloss={mt['logloss']:.4f}  brier={mt['brier']:.4f}  -> {out.name}", flush=True)

    keep = ["match_id", "date", "split", "team1", "team2", "team1_won",
            "tier1", "tier2", "tier3", "bo_missing", "bo1", "bo3", "bo5"]
    test[[c for c in keep if c in test.columns]].to_csv(out_dir / "test_features.csv", index=False)
    (out_dir / "test_manifest.json").write_text(json.dumps({
        "source": "v9-clean-v2",
        "n_test": int(len(test)),
        "date_range": [str(test.date.min()), str(test.date.max())],
        "note": "特征由全部比赛在线重建后切片；train/val 部分与已发布版本逐列一致。",
    }, indent=2) + "\n", encoding="utf8")
    print("完成。测试集只应评估这一次，之后不要再据此调参。", flush=True)


if __name__ == "__main__":
    main()
