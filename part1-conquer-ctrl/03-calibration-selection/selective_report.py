"""覆盖度–准确率曲线 / 校准 / 分层报告（临时查看工具）。

注意：正式口径以 reports/M4.md 为准（含 bootstrap 置信区间与显著性判断）。
本脚本保留为「对任意预测文件快速看一眼」的工具，不要和 M4 同时引用同一张表。

用法（在项目根目录）：
    python -B -m scripts.selective_report
    python -B -m scripts.selective_report --pred reports/release/m2_test.csv \
        --feat data/processed/clean_v2/m2/test.csv --out reports/release/m2_test_selective.md

输入：
    --pred  每行一场：match_id, y_true, p_team1, model（M1/M2 已有这种文件）
    --feat  clean_v2 的 m2/{val,test}.csv，只用来取 tier / date / bo_missing

不写任何文件除非给了 --out。所有统计都是逐场二分类。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def md_table(df: pd.DataFrame) -> str:
    """不依赖 tabulate 的 markdown 表。"""
    cols = [str(c) for c in df.columns]
    rows = [[str(v) for v in r] for r in df.itertuples(index=False, name=None)]
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def metrics(p: np.ndarray, y: np.ndarray) -> dict:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return {
        "n": int(len(y)),
        "acc": float(((p > .5) == y).mean()),
        "logloss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "brier": float(np.mean((p - y) ** 2)),
        "mean_conf": float(np.mean(np.abs(p - .5))),
    }


def coverage_table(p: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """按「把握程度」筛场次：只留 |p-0.5| >= t 的比赛。"""
    rows = []
    for t in (0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
        keep = np.abs(p - .5) >= t
        if keep.sum() == 0:
            continue
        m = metrics(p[keep], y[keep])
        rows.append({"阈值": f"{t:.2f}", "覆盖场次": m["n"],
                     "覆盖率": f"{keep.mean():.1%}", "子集准确率": f"{m['acc']:.2%}",
                     "子集 logloss": f"{m['logloss']:.4f}", "_acc": m["acc"]})
    return pd.DataFrame(rows)


def calibration_table(p: np.ndarray, y: np.ndarray) -> tuple[pd.DataFrame, float]:
    edges = np.array([0., .35, .45, .50, .55, .60, .65, 1.])
    b = np.clip(np.digitize(p, edges) - 1, 0, len(edges) - 2)
    rows = []
    ece = 0.0
    for k in range(len(edges) - 1):
        sel = b == k
        if sel.sum() == 0:
            continue
        pred, act = float(p[sel].mean()), float(y[sel].mean())
        ece += sel.mean() * abs(pred - act)
        rows.append({"预测区间": f"[{edges[k]:.2f},{edges[k+1]:.2f})", "场次": int(sel.sum()),
                     "平均预测": f"{pred:.3f}", "实际胜率": f"{act:.3f}",
                     "偏差": f"{pred - act:+.3f}"})
    return pd.DataFrame(rows), ece


def tier_table(p: np.ndarray, y: np.ndarray, tier: pd.Series) -> pd.DataFrame:
    d = pd.DataFrame({"tier": tier.to_numpy(), "p": p, "y": y})
    rows = []
    for name, g in d.groupby("tier", dropna=False):
        m = metrics(g.p.to_numpy(), g.y.to_numpy())
        rows.append({"分区": str(name), "场次": m["n"], "占比": f"{len(g)/len(d):.1%}",
                     "准确率": f"{m['acc']:.2%}", "logloss": f"{m['logloss']:.4f}"})
    return pd.DataFrame(rows)


def target_arithmetic(p: np.ndarray, y: np.ndarray, tier: pd.Series) -> str:
    """要全场 70%，每个分区得各自做到多少。"""
    d = pd.DataFrame({"tier": tier.to_numpy(), "p": p, "y": y}).dropna(subset=["tier"])
    if d.empty:
        return "（缺 tier，跳过）"
    share = d.groupby("tier").size() / len(d)
    acc = d.groupby("tier").apply(lambda g: ((g.p > .5) == g.y).mean(), include_groups=False)
    now = float((share * acc).sum())
    lines = [f"当前按分区加权 = {now:.2%}（各分区："
             + "、".join(f"{k} {share[k]:.1%}×{acc[k]:.2%}" for k in share.index) + "）",
             f"距全场 70% 还差 {0.70 - now:+.1%}。"]
    lowest = acc.idxmin()
    rest_share = float(1 - share[lowest])
    need_rest = (0.70 - float(share[lowest] * acc[lowest])) / rest_share
    if need_rest > 1.0:
        lines.append(f"若 {lowest} 原地不动（{acc[lowest]:.2%}），其余分区即便做到 100% 也到不了 70% —— "
                     f"这条目标是数学上不可达的。")
    else:
        lines.append(f"若 {lowest} 原地不动（{acc[lowest]:.2%}），其余分区平均要做到 {need_rest:.2%}；"
                     f"若其余只做到 65%，{lowest} 需要 "
                     f"{(0.70 - rest_share * 0.65) / float(share[lowest]):.2%}。")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred", default=str(ROOT / "reports/release/m2_val.csv"))
    ap.add_argument("--feat", default=str(ROOT / "data/processed/clean_v2/m2/val.csv"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    pred = pd.read_csv(a.pred)
    y = pred["y_true"].to_numpy().astype(int)
    p = pred["p_team1"].to_numpy().astype(float)
    if not (0 <= p.min() and p.max() <= 1):
        raise ValueError("p_team1 不在 [0,1] 内")

    tier = pd.Series([np.nan] * len(pred))
    feat_path = Path(a.feat)
    if feat_path.exists():
        feat = pd.read_csv(feat_path, usecols=lambda c: c in
                           ("match_id", "tier", "date", "bo_missing", "tier1", "tier2", "tier3"))
        joined = pred[["match_id"]].merge(feat, on="match_id", how="left")
        if "tier" in joined:
            tier = joined["tier"]
        elif {"tier1", "tier2", "tier3"} <= set(joined.columns):
            # clean_v2 只有 one-hot，没有 tier 标签
            tier = np.select(
                [joined["tier1"].eq(1), joined["tier2"].eq(1), joined["tier3"].eq(1)],
                ["tier1", "tier2", "tier3"], default="unknown")
            tier = pd.Series(tier, index=joined.index)

    out = []
    m = metrics(p, y)
    out.append(f"# 逐场报告：{Path(a.pred).name}\n")
    out.append(f"全场 {m['n']} 场 · 准确率 {m['acc']:.2%} · logloss {m['logloss']:.4f} "
               f"· Brier {m['brier']:.4f}\n")

    ct = coverage_table(p, y)
    out.append("\n## 1. 覆盖度–准确率（这才是「70%」该出现的地方）\n")
    out.append(md_table(ct.drop(columns="_acc")))

    cal, ece = calibration_table(p, y)
    out.append(f"\n## 2. 校准（ECE = {ece:.4f}，越小越好）\n")
    out.append(md_table(cal))

    if tier.notna().any():
        tt = tier_table(p, y, tier)
        out.append("\n## 3. 按赛事分区\n")
        out.append(md_table(tt))
        out.append("\n## 4. 「全场 70%」需要每个分区各做到多少\n")
        out.append(target_arithmetic(p, y, tier))

    text = "\n".join(out)
    print(text)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text + "\n", encoding="utf8")
        print(f"\n[written] {a.out}", flush=True)


if __name__ == "__main__":
    main()
